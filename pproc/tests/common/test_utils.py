# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pproc.common import utils


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
