# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict


class PProcBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
