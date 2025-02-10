import argparse

import pytest

from pproc.common import config


@pytest.mark.parametrize(
    "inp,exp",
    [
        ([], {}),
        (["foo=bar"], {"foo": "bar"}),
        (
            ["working=true", "pi=3.14", "mode=production"],
            {"working": "true", "pi": "3.14", "mode": "production"},
        ),
    ],
)
def test_parse_vars(inp, exp):
    assert config.parse_vars(inp) == exp


@pytest.mark.parametrize(
    "inp,exp",
    [
        ([], {}),
        ([""], {}),
        (["foo=bar"], {"foo": "bar"}),
        (["foo=bar", ""], {"foo": "bar"}),
        (
            ["working=true", "pi=3.14", "mode=production"],
            {"working": "true", "pi": "3.14", "mode": "production"},
        ),
        (
            ["working=true,pi=3.14", "mode=production", "parallel=true,nprocs=12"],
            {
                "working": "true",
                "pi": "3.14",
                "mode": "production",
                "parallel": "true",
                "nprocs": "12",
            },
        ),
    ],
)
def test_parse_var_strs(inp, exp):
    assert config.parse_var_strs(inp) == exp


@pytest.mark.parametrize(
    "inp",
    [
        [],
        ["class=rd"],
        ["class=rd,expver=abcd"],
        ["stream=lwda", "class=rd,expver=abcd"],
    ],
)
def test_parser_override_input(inp):
    in_args = ["-c", "config.yaml"]
    for arg in inp:
        in_args.append("--override-input")
        in_args.append(arg)
    parser = config.default_parser("test")
    args = parser.parse_args(in_args)
    assert args.override_input == inp


@pytest.mark.parametrize(
    "inp",
    [
        [],
        ["edition=2"],
        ["edition=2, paramId=123456"],
        ["edition=2", "class=rd,expver=abcd"],
    ],
)
def test_parser_override_output(inp):
    in_args = ["-c", "config.yaml"]
    for arg in inp:
        in_args.append("--override-output")
        in_args.append(arg)
    parser = config.default_parser("test")
    args = parser.parse_args(in_args)
    assert args.override_output == inp


@pytest.fixture
def dummy_configfile(tmp_path):
    cfgfile = tmp_path / "config.yaml"
    cfgfile.write_text("foo: bar")
    return cfgfile


@pytest.mark.parametrize(
    "inp,exp",
    [
        ([], {}),
        (["class=rd"], {"class": "rd"}),
        (["class=rd,expver=abcd"], {"class": "rd", "expver": "abcd"}),
        (
            ["stream=lwda", "class=rd,expver=abcd"],
            {"class": "rd", "expver": "abcd", "stream": "lwda"},
        ),
    ],
)
def test_override_input(inp, exp, dummy_configfile):
    args = argparse.Namespace(
        config=dummy_configfile,
        set=[],
        recover=False,
        override_input=inp,
        override_output=[],
    )
    cfg = config.Config(args, verbose=False)
    assert cfg.override_input == exp


@pytest.mark.parametrize(
    "inp,exp",
    [
        ([], {}),
        (["edition=2"], {"edition": "2"}),
        (["edition=2,paramId=123456"], {"edition": "2", "paramId": "123456"}),
        (
            ["edition=2", "class=rd,expver=abcd"],
            {"class": "rd", "expver": "abcd", "edition": "2"},
        ),
    ],
)
def test_override_output(inp, exp, dummy_configfile):
    args = argparse.Namespace(
        config=dummy_configfile,
        set=[],
        recover=False,
        override_input=[],
        override_output=inp,
    )
    cfg = config.Config(args, verbose=False)
    assert cfg.override_output == exp
