import calendar
from datetime import datetime
import pytest

from pproc.clustereps.season import Season, SeasonConfig


SEASONS = [(5, 9), (10, 4)]


def id_tests(val):
    if isinstance(val, datetime):
        return val.strftime("%Y%m%d")
    elif isinstance(val, Season):
        return f"{val.name}{val.baseYear}"
    elif val is SEASONS:
        return "ondjfma/mjjas"


@pytest.mark.parametrize(
    "start, end, year, ename, estart, eend, endays",
    [
        (10, 4, 2019, "ondjfma", datetime(2018, 10, 1), datetime(2019, 4, 30), 212),
        (5, 9, 2019, "mjjas", datetime(2019, 5, 1), datetime(2019, 9, 30), 153),
        (10, 4, 2020, "ondjfma", datetime(2019, 10, 1), datetime(2020, 4, 30), 212),
        (5, 9, 2020, "mjjas", datetime(2020, 5, 1), datetime(2020, 9, 30), 153),
    ],
    ids=id_tests,
)
def test_season(start, end, year, ename, estart, eend, endays):
    season = Season(start, end, year)
    assert season.name == ename
    assert season.start == estart
    assert season.end == eend
    assert season.ndays == endays


@pytest.mark.parametrize("year", [2019, 2020], ids=["non-leap", "leap"])
def test_season_dos(year):
    winter = Season(10, 4, year)
    assert winter.dos(datetime(year - 1, 11, 1)) == 31
    assert winter.dos(datetime(year, 2, 28)) == 150
    assert winter.dos(datetime(year, 3, 1)) == 151
    if calendar.isleap(year):
        assert winter.dos(datetime(year, 2, 29)) == 150

    summer = Season(5, 9, year)
    assert summer.dos(datetime(year, 6, 1)) == 31


@pytest.mark.parametrize(
    "date, seasons, eseason",
    [
        (datetime(2022, 4, 11), SEASONS, Season(10, 4, 2022)),
        (datetime(2022, 4, 30), SEASONS, Season(10, 4, 2022)),
        (datetime(2022, 5, 1), SEASONS, Season(5, 9, 2022)),
        (datetime(2022, 7, 20), SEASONS, Season(5, 9, 2022)),
        (datetime(2022, 9, 30), SEASONS, Season(5, 9, 2022)),
        (datetime(2022, 10, 1), SEASONS, Season(10, 4, 2023)),
        (datetime(2022, 12, 14), SEASONS, Season(10, 4, 2023)),
        (datetime(2020, 1, 1), SEASONS, Season(10, 4, 2020)),
        (datetime(2020, 2, 29), SEASONS, Season(10, 4, 2020)),
        (datetime(2020, 4, 30), SEASONS, Season(10, 4, 2020)),
        (datetime(2020, 5, 1), SEASONS, Season(5, 9, 2020)),
        (datetime(2020, 7, 20), SEASONS, Season(5, 9, 2020)),
        (datetime(2020, 9, 30), SEASONS, Season(5, 9, 2020)),
        (datetime(2020, 10, 1), SEASONS, Season(10, 4, 2021)),
        (datetime(2020, 12, 14), SEASONS, Season(10, 4, 2021)),
    ],
    ids=id_tests,
)
def test_get_season(date, seasons, eseason):
    config = SeasonConfig(seasons)
    season = config.get_season(date)
    assert date in season
    assert season.name == eseason.name
    assert season.start == eseason.start
    assert season.end == eseason.end


@pytest.mark.parametrize(
    "date, seasons, edos",
    [
        (datetime(2022, 10, 1), SEASONS, 0),
        (datetime(2022, 10, 31), SEASONS, 30),
        (datetime(2022, 11, 12), SEASONS, 42),
        (datetime(2023, 3, 20), SEASONS, 170),
        (datetime(2023, 4, 30), SEASONS, 211),
        (datetime(2023, 5, 1), SEASONS, 0),
        (datetime(2023, 5, 31), SEASONS, 30),
        (datetime(2023, 8, 17), SEASONS, 108),
        (datetime(2023, 9, 30), SEASONS, 152),
        (datetime(2024, 2, 28), SEASONS, 150),
        (datetime(2024, 2, 29), SEASONS, 150),
        (datetime(2024, 4, 30), SEASONS, 211),
        (datetime(2024, 5, 1), SEASONS, 0),
        (datetime(2024, 5, 31), SEASONS, 30),
        (datetime(2024, 8, 17), SEASONS, 108),
        (datetime(2024, 9, 30), SEASONS, 152),
        (datetime(2024, 10, 1), SEASONS, 0),
        (datetime(2024, 10, 31), SEASONS, 30),
        (datetime(2024, 11, 12), SEASONS, 42),
        (datetime(2025, 3, 20), SEASONS, 170),
        (datetime(2025, 4, 30), SEASONS, 211),
    ],
    ids=id_tests,
)
def test_dos(date, seasons, edos):
    config = SeasonConfig(seasons)
    season = config.get_season(date)
    assert date in season
    assert season.dos(date) == edos
