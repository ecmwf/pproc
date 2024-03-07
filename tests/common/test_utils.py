from pproc.common import utils


def test_dictprod():
    assert list(utils.dict_product({})) == [{}]

    assert list(utils.dict_product({"x": range(3), "empty": []})) == []

    assert list(utils.dict_product({"foo": [1, 2, 3]})) == [
        {"foo": n} for n in [1, 2, 3]
    ]

    dic = {"a": [5, 12], "b": range(3), "c": ("a", "b")}
    assert list(utils.dict_product(dic)) == [
        {"a": 5, "b": 0, "c": "a"},
        {"a": 5, "b": 0, "c": "b"},
        {"a": 5, "b": 1, "c": "a"},
        {"a": 5, "b": 1, "c": "b"},
        {"a": 5, "b": 2, "c": "a"},
        {"a": 5, "b": 2, "c": "b"},
        {"a": 12, "b": 0, "c": "a"},
        {"a": 12, "b": 0, "c": "b"},
        {"a": 12, "b": 1, "c": "a"},
        {"a": 12, "b": 1, "c": "b"},
        {"a": 12, "b": 2, "c": "a"},
        {"a": 12, "b": 2, "c": "b"},
    ]


def test_delayedmap(capsys):
    def myfunc(x):
        print(f"p {x}")
        return x

    for x in utils.delayed_map(5, myfunc, []):
        print(f"r {x}")
    assert capsys.readouterr().out == ""

    for x in utils.delayed_map(2, myfunc, range(4)):
        print(f"r {x}")
    assert capsys.readouterr().out == (
        "\n".join(
            [
                "p 0",
                "p 1",
                "r 0",
                "p 2",
                "r 1",
                "p 3",
                "r 2",
                "r 3",
            ]
        )
        + "\n"
    )

    for x in utils.delayed_map(10, myfunc, "abc"):
        print(f"r {x}")
    assert capsys.readouterr().out == (
        "\n".join(["p a", "p b", "p c", "r a", "r b", "r c"]) + "\n"
    )

    for x in utils.delayed_map(0, myfunc, (False, True)):
        print(f"r {x}")
    assert capsys.readouterr().out == (
        "\n".join(["p False", "r False", "p True", "r True"]) + "\n"
    )
