#!/usr/bin/env python3
# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

#
# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.
import sys
import logging

from conflator import Conflator

from earthkit.workflows.compilers import graph2job
from cascade.low.core import JobInstanceRich
from cascade.main import run_locally
from ppcore.configs.entrypoint.base import EntrypointConfig
from ppcore.products import graph_from_configs

logger = logging.getLogger(__name__)


def main():
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore
    cfg: EntrypointConfig = Conflator(
        app_name="pproc-product", model=EntrypointConfig
    ).load()  # type: ignore
    logger.info(cfg.dump())
    graph = graph_from_configs(cfg.products, cfg.input_overrides, cfg.output_overrides)
    run_locally(
        JobInstanceRich(jobInstance=graph2job(graph), checkpointSpec=None),
        hosts=cfg.execution.hosts,
        workers=cfg.execution.workers,
    )


if __name__ == "__main__":
    sys.exit(main())
