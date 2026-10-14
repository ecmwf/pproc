# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0


from conflator import ConfigModel


class ExecutionConfig(ConfigModel):
    hosts: int = 1
    workers: int = 1
