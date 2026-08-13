# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

from earthkit.data import FieldList
from earthkit.data.core.metadata import Metadata as ekdMetadata


def standardise_output(data):
    # Also, nest the data to avoid problems with not finding geography attribute
    if len(data.shape) == 1:
        data = data.reshape((1, *data.shape))
    assert len(data.shape) == 2
    return data


def new_fieldlist(data, metadata: list[ekdMetadata], overrides: dict = {}):
    if len(overrides) > 0:
        try:
            new_metadata = [
                metadata[x].override(overrides) for x in range(len(metadata))
            ]
            return FieldList.from_array(
                standardise_output(data),
                new_metadata,
            )
        except Exception as e:
            print(
                "Error setting metadata",
                overrides,
                "On data with:",
                list(map(lambda x: x.dump(), metadata)),
            )
            print(e)
    return FieldList.from_array(standardise_output(data), metadata)
