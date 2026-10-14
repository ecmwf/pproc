# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional

import yaml

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
SCHEMA = os.path.join(TEST_DIR, "schema.yaml")


def schema(section: Optional[str] = None) -> dict:
    with open(SCHEMA, "r") as f:
        schema = yaml.safe_load(f)
    return schema if section is None else schema[section]
