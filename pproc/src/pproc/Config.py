# coding: utf-8


def postproc_keys(metkit_share_dir: str = ""):
    from yaml import safe_load
    from os import path

    fn = path.expanduser(path.join(metkit_share_dir, "language.yaml"))
    with open(fn, "r") as f:
        y = safe_load(f)
        return set(y['_postproc'].keys())
    return set()



class ParamId:
    def __init__(self, metkit_share_dir: str = ""):
        from yaml import safe_load
        from os import path

        self._id_name = dict()

        fn = path.expanduser(path.join(metkit_share_dir, "paramids.yaml"))
        with open(fn, "r") as f:
            y = safe_load(f)
            for id, lst in y.items():
                # keeps only the first space-free name
                for n in filter(lambda n: " " not in n, lst):
                    n = n.lower()
                    if n not in self._id_name:
                        self._id_name[n] = id

        self._id_wind_uv = dict()
        self._id_wind_vod = dict()

        fn = path.expanduser(path.join(metkit_share_dir, "param-matching.yaml"))
        with open(fn, "r") as f:
            y = safe_load(f)
            for u, v, vo, d in y["wind"]:
                u, v, vo, d = map(self.id, (u, v, vo, d))
                self._id_wind_vod[u, v] = (vo, d)
                self._id_wind_uv[vo, d] = (u, v)

    def id(self, key) -> int:
        # from PARAM.TABLE
        try:
            p, t = map(int, "{:.3f}".format(float(key)).split("."))
            assert 0 <= p < 1000 and t < 1000
            return p if t == 128 else (1000 * t + p)
        except:
            pass

        # from shortName
        if isinstance(key, str):
            k = key.lower()
            if k in self._id_name:
                return self._id_name[k]

        return int(key)

    def vod(self, key) -> int:
        assert len(key) == 2
        u, v = map(self.id, (key[0], key[1]))
        return self._id_wind_vod[u, v]

    def uv(self, key) -> int:
        assert len(key) == 2
        vo, d = map(self.id, (key[0], key[1]))
        return self._id_wind_uv[vo, d]


class VariableTree:
    def __init__(self, yaml: str):
        from yaml import safe_load
        from os import path

        fn = path.expanduser(yaml)
        with open(fn, "r") as y:
            self._root = safe_load(y)

    def variables(self, *path) -> dict:
        return self._vars(*path, root=self._root)

    @staticmethod
    def _vars(*path, root: dict) -> dict:
        d = {k: v for k, v in root.items() if not isinstance(v, dict)}
        if path:
            d.update(VariableTree._vars(*path[1:], root=root[path[0]]))
        return d


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Configuration tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("file", help="Variables configuration file", metavar="CONFIG_FILE")
    parser.add_argument("--config-node", help="Variables configuration node", nargs="*", type=str)
    # parser.add_argument(
    #     "--metkit-share-dir",
    #     help="Metkit configuration directory",
    #     default="~/git/metkit/share/metkit",
    # )

    args = parser.parse_args()

    tree = VariableTree(args.file)
    node = [int(p) if p.isdigit() else p for p in args.config_node]
    print(tree.variables(*node))


if __name__ == "__main__":
    from sys import exit

    exit(main())
