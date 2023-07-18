from pproc import common
from pproc.common import parallel


class ProbConfig(common.Config):
    def __init__(self, args, target_types):
        super().__init__(args)
        self.n_ensembles = int(self.options.get("number_of_ensembles", 50))
        self.global_input_cfg = self.options.get("global_input_keys", {})
        self.global_output_cfg = self.options.get("global_output_keys", {})
        self.n_par_read = self.options.get("n_par_read", 1)
        self.n_par_compute = self.options.get("n_par_compute", 1)
        self.window_queue_size = self.options.get("queue_size", self.n_par_compute)

        for attr in target_types:
            location = getattr(args, attr)
            target = common.io.target_from_location(location)
            if self.n_par_compute > 1:
                target.enable_parallel(parallel)
            if args.recover:
                target.enable_recovery()
            self.__setattr__(attr, target)
