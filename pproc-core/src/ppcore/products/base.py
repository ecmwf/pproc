# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class Product(ABC):
    input_overrides: dict
    output_overrides: dict

    def action(self, *args, **kwargs):
        raise NotImplementedError

    def in_mars(self, sources: Optional[list[str]] = None) -> Iterator[dict]:
        raise NotImplementedError

    def out_mars(self, targets: Optional[list[str]] = None) -> Iterator[dict]:
        raise NotImplementedError
