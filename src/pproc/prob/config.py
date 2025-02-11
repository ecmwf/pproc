from pproc import common
from pproc.common import parallel


class BaseProbConfig(common.Config):
    def __init__(self, args, target_types):
        super().__init__(args)
        self.members = self.options.get("num_members", 51)
        self.total_fields = self.options.get("total_fields", self.members)
        self.out_keys = self.options.get("out_keys", {})
        self.n_par_read = self.options.get("n_par_read", 1)
        self.n_par_compute = self.options.get("n_par_compute", 1)
        self.window_queue_size = self.options.get("queue_size", self.n_par_compute)
        self.sources = self.options.get("sources", {})

        self.steps = self.options.get("steps", [])
        self.windows = self.options.get("windows", [])

        for attr in target_types:
            location = getattr(args, attr)
            target = common.io.target_from_location(
                location, overrides=self.override_output
            )
            if self.n_par_compute > 1:
                target.enable_parallel(parallel)
            if args.recover:
                target.enable_recovery()
            self.__setattr__(attr, target)
