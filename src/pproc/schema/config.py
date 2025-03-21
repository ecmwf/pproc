import yaml

from pproc.schema.base import Schema


class ConfigSchema(Schema):
    def config(self, output_request: dict) -> dict:
        config = self.traverse(output_request)
        out = yaml.load(
            yaml.dump(config).format_map(output_request),
            Loader=yaml.SafeLoader,
        )
        return out
