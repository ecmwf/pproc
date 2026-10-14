# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from unittest.mock import patch

import pytest
from conflator import Conflator
from pydantic import ValidationError

from pproc.config.log import LoggingConfig


def test_logging_model_default():
    """Test that the logging configuration is applied after validation."""
    model = LoggingConfig()
    assert model.level == "INFO"

    # Check logging level
    assert logging.getLogger().getEffectiveLevel() == logging.INFO


def test_logging_app_default():
    """Test that the default values are correctly set."""

    with patch("sys.argv", ["test_script.py"]):
        conflator = Conflator("pproc", LoggingConfig, nested={})
        config = conflator.load()

        assert config.level == "INFO"
        assert config.format == "%(asctime)s; %(name)s; %(levelname)s - %(message)s"

        # Check logging level
        assert logging.getLogger().getEffectiveLevel() == logging.INFO


def test_logging_model_set():
    """Test that the logging configuration is applied after validation."""
    model = LoggingConfig(level="DEBUG")
    assert model.level == "DEBUG"

    # Check logging level
    assert logging.getLogger().getEffectiveLevel() == logging.DEBUG


def test_logging_model_env_var(monkeypatch):
    """Test that environment variable overrides are applied."""

    monkeypatch.setenv("PPROC_LOG", "WARNING")
    with patch("sys.argv", ["test_script.py"]):
        conflator = Conflator("pproc", LoggingConfig, nested={})
        config = conflator.load()

        assert config.level == "WARNING"

    # Check logging level
    assert logging.getLogger().getEffectiveLevel() == logging.WARNING


def test_logging_model_cli_arg():
    """Test that CLI arguments override defaults."""

    with patch("sys.argv", ["test_script.py"] + ["--log", "ERROR"]):
        conflator = Conflator("pproc", LoggingConfig, nested={})
        config = conflator.load()

        assert config.level == "ERROR"

        # Check logging level
        assert logging.getLogger().getEffectiveLevel() == logging.ERROR


def test_invalid_logging_level():
    """Test that invalid logging levels raise an error."""
    with pytest.raises(ValidationError):
        LoggingConfig(level="INVALID")
