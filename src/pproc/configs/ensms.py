from typing import Optional

from typing_extensions import Self

from pproc.configs.base import BaseConfig
from pproc.configs.request import Request, write_requests


class Config(BaseConfig):
    fc_date: str = "{date}"
    root_dir: str
    out_keys_em: Optional[dict] = None
    out_keys_es: Optional[dict] = None

    @classmethod
    def from_inputs(cls, outputs, template, schema) -> Self:
        config = cls._from_inputs(outputs, template, schema)
        datetime = None
        for _, param_config in config.params.items():
            pdatetime = f"{param_config.in_keys['date']}{param_config.in_keys['time']}"
            if datetime is None:
                datetime = pdatetime
            elif datetime != pdatetime:
                raise ValueError("Date/time in requests must match")
        config.fc_date = config.fc_date.format_map({"date": datetime})
        return config

    def outputs(self, output_file: str):
        output_reqs = []
        for param, param_config in self.params.items():
            req = {**param_config.in_keys}
            req.update({**self.out_keys, "param": param_config.out or param_config.in_})
            req.update(param_config.out_keys)

            for dim, dim_config in param_config.accumulations.items():
                if dim_config.get("operation", None) is None:
                    req[dim] = [x[0] for x in dim_config["coords"]]

            steps = []
            for accum_steps in param_config.accumulations["step"]["coords"]:
                if isinstance(accum_steps, dict):
                    steps.append(f"{accum_steps['from']}-{accum_steps['to']}")
                else:
                    steps.append(
                        (
                            str(accum_steps[0])
                            if len(accum_steps) == 1
                            else f"{accum_steps[0]}-{accum_steps[-1]}"
                        )
                    )
            req["step"] = steps
            for grib_sets, default in [
                (self.out_keys_em, "em"),
                (self.out_keys_es, "es"),
            ]:
                req["type"] = grib_sets.get("type", default) if grib_sets else default
                output_reqs.append(Request(**req))

        write_requests(output_file, output_reqs)
