# coding: utf-8

import pytest
import argparse
from pproc.Hardcode import Hardcode


TESTS = ((range(0, 48 + 1, 3), ("mx2t6",), (850, 250)),)


@pytest.mark.parametrize("test", TESTS)
def test_ensms(test):
    step_range, list_param, list_lev = test
    for a in (step_range, list_param, list_lev):
        print(a, type(a))
    h = Hardcode()
    for p in list_param:
        print(h[p])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="wind850.ecf test")
    # parser.add_argument('fc_date', help='Forecast date')
    # parser.add_argument('clim_date', help='climatology date')
    # parser.add_argument('efi', help='EFI file')

    args = parser.parse_args()
    # clim_date = args.clim_date
    # fc_date = args.fc_date
    # efi_file = args.efi

    for test in TESTS:
        try:
            test_ensms(test)
        except Exception as e:
            print(e)
