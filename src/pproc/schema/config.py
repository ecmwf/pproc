import numpy as np

from pproc.schema.base import BaseSchema


class ConfigSchema(BaseSchema):
    def config(self, output_request: dict) -> dict:
        config = self.traverse(output_request)
        if output_request["type"] in ["pb", "cd"]:
            numbers = np.zeros(len(output_request["quantile"]))
            totals = np.zeros(len(output_request["quantile"]))
            for index, quantile in enumerate(output_request["quantile"]):
                number, total = map(int, quantile.split(":"))
                numbers[index] = number
                totals[index] = total
            if np.all(totals == totals[0]) and np.all(np.diff(numbers) == 1):
                quantiles = int(totals[0])
            else:
                quantiles = numbers / totals
            config["quantiles"] = quantiles
        if output_request["type"] == "sot":
            config["sot"] = output_request["number"]
        return config
