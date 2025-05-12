import yaml

from pproc.schema.base import BaseSchema


class ConfigSchema(BaseSchema):
    def config(self, output_request: dict) -> dict:
        config = self.traverse(output_request)
        if output_request["type"] in ["pb", "cd"]:
            quantiles = []
            for quantile in output_request["quantile"]:
                number, total = map(int, quantile.split(":"))
                quantiles.append(number / total)
            config["quantiles"] = quantiles
        if output_request["type"] == "sot":
            config["sot"] = output_request["number"]
        return config
