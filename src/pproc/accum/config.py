import argparse
from datetime import datetime
from typing import Any, Dict

from pproc.common.config import Config
from pproc.common.param_requester import ParamConfig


class AccumParamConfig(ParamConfig):
    def __init__(self, name, options: Dict[str, Any], overrides: Dict[str, Any] = {}):
        super().__init__(name, options, overrides)
        self.vmin = options.get("vmin", None)
        self.vmax = options.get("vmax", None)


class AccumConfig(Config):
    def __init__(self, args: argparse.Namespace, verbose: bool = True):
        super().__init__(args, verbose=verbose)

        self.num_members = self.options.get("num_members", 51)
        self.total_fields = self.options.get("total_fields", self.num_members)

        self.out_keys = self.options.get("out_keys", {})

        self.params = [
            AccumParamConfig(pname, popt, overrides=self.override_input)
            for pname, popt in self.options["params"].items()
        ]
        self.steps = self.options.get("steps", [])
        self.windows = self.options.get("windows", [])

        self.sources = self.options.get("sources", {})

        date = self.options.get("date")
        self.date = None if date is None else datetime.strptime(str(date), "%Y%m%d%H")
        self.root_dir = self.options.get("root_dir", None)

        self.n_par_read = self.options.get("n_par_read", 1)
        self.n_par_compute = self.options.get("n_par_compute", 1)
        self.window_queue_size = self.options.get("queue_size", self.n_par_compute)
