# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for :mod:`pproc.climate._logging`.

Verifies the verbosity-count → root-logger configuration contract:
level mapping, format selection, stdout streaming, and idempotency
across repeated calls within the same process.
"""

from __future__ import annotations

import logging
import sys

import pytest

from pproc.climate._logging import _HANDLER_TAG, configure_logging


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Snapshot and restore root logger handlers around each test.

    ``configure_logging`` mutates the root logger; we save and restore
    so tests don't leak handlers into each other or the rest of the
    pytest session.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    # Strip our handler if one was left from a previous test.
    root.handlers = [h for h in root.handlers if not getattr(h, _HANDLER_TAG, False)]
    try:
        yield
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)


def _our_handler():
    """Return the handler we tagged, or ``None`` if not installed."""
    for h in logging.getLogger().handlers:
        if getattr(h, _HANDLER_TAG, False):
            return h
    return None


class TestLevelMapping:
    def test_zero_is_warning(self):
        configure_logging(0)
        assert logging.getLogger().level == logging.WARNING

    def test_one_is_info(self):
        configure_logging(1)
        assert logging.getLogger().level == logging.INFO

    def test_two_is_debug(self):
        configure_logging(2)
        assert logging.getLogger().level == logging.DEBUG

    def test_high_count_clamps_to_debug(self):
        configure_logging(99)
        assert logging.getLogger().level == logging.DEBUG


class TestFormatter:
    def test_info_format_has_no_timestamp(self):
        configure_logging(1)
        handler = _our_handler()
        assert handler is not None
        # Format a fake record and check the rendered output.
        record = logging.LogRecord(
            name="x",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        rendered = handler.formatter.format(record)
        assert rendered == "[x] hello"
        # No colon-separated time prefix (HH:MM:SS would contain digits + ":").
        assert not rendered[:8].count(":") == 2

    def test_debug_format_has_timestamp(self):
        configure_logging(2)
        handler = _our_handler()
        assert handler is not None
        record = logging.LogRecord(
            name="x",
            level=logging.DEBUG,
            pathname=__file__,
            lineno=1,
            msg="hi",
            args=(),
            exc_info=None,
        )
        rendered = handler.formatter.format(record)
        # ``HH:MM:SS`` prefix → first 8 chars contain two colons.
        prefix = rendered[:8]
        assert prefix.count(":") == 2, rendered
        assert "[x] hi" in rendered


class TestStream:
    def test_handler_streams_to_stdout(self):
        configure_logging(1)
        handler = _our_handler()
        assert handler is not None
        # ``StreamHandler.stream`` is the underlying file-like.
        assert handler.stream is sys.stdout


class TestIdempotency:
    def test_repeated_calls_do_not_duplicate_handlers(self):
        configure_logging(1)
        configure_logging(1)
        configure_logging(2)
        configure_logging(0)
        tagged = [
            h for h in logging.getLogger().handlers if getattr(h, _HANDLER_TAG, False)
        ]
        assert len(tagged) == 1

    def test_level_is_updated_on_second_call(self):
        configure_logging(1)
        assert logging.getLogger().level == logging.INFO
        configure_logging(2)
        assert logging.getLogger().level == logging.DEBUG
        configure_logging(0)
        assert logging.getLogger().level == logging.WARNING

    def test_format_is_updated_on_second_call(self):
        configure_logging(1)
        h1 = _our_handler()
        # Switch to DEBUG; handler should remain the same instance but with
        # a new formatter that emits timestamps.
        configure_logging(2)
        h2 = _our_handler()
        assert h1 is h2
        record = logging.LogRecord(
            name="x",
            level=logging.DEBUG,
            pathname=__file__,
            lineno=1,
            msg="m",
            args=(),
            exc_info=None,
        )
        rendered = h2.formatter.format(record)
        assert rendered[:8].count(":") == 2
