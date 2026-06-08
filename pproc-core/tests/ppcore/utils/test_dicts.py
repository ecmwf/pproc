from ppcore.utils.dicts import dict_product


def test_dictprod():
    assert list(dict_product({})) == [{}]

    assert list(dict_product({"x": range(3), "empty": []})) == []

    assert list(dict_product({"foo": [1, 2, 3]})) == [{"foo": n} for n in [1, 2, 3]]

    dic = {"a": [5, 12], "b": range(3), "c": ("a", "b")}
    assert list(dict_product(dic)) == [
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
