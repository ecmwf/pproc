from typing import Optional

from pproc.configs.base import BaseConfig
from pproc.configs.request import Request, write_requests


class Config(BaseConfig):
    fc_date: Optional[str] = None
    type_em: Optional[str] = "em"
    type_es: Optional[str] = "es"

    def outputs(self, output_file: str):
        output_reqs = []
        for param, param_config in self.params.items():
            req = {**param_config.in_keys}
            req.update({**self.out_keys, "param": param_config.out or param_config.in_})
            req.update(param_config.out_keys)

            for dim, dim_config in param_config.accumulations.items():
                if dim_config.get("operation", None) is None:
                    print(dim, dim_config)
                    req[dim] = [x[0] for x in dim_config["coords"]]

            steps = [
                str(steps[0]) if len(steps) == 1 else f"{steps[0]}-{steps[-1]}"
                for steps in param_config.accumulations["step"]["coords"]
            ]
            req["step"] = steps
            for tp in [
                self.type_em,
                self.type_es,
            ]:
                req["type"] = tp
                output_reqs.append(Request(**req))

        write_requests(output_file, output_reqs)
