# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from datetime import datetime, timedelta
from typing import List, Tuple
import calendar


MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


class Season:

    def __init__(self, startMonth, endMonth, baseYear):

        self.baseYear = baseYear
        jumpYear = 0
        if startMonth > endMonth:
            startYear = self.baseYear - 1
            jumpYear = 1
        else:
            startYear = self.baseYear
        first_day = datetime.strptime(f"{startYear:04d}{startMonth:02d}01", "%Y%m%d")
        end_day = datetime.strptime(f"{self.baseYear:04d}{endMonth:02d}{MONTH_DAYS[endMonth-1]}", "%Y%m%d")

        self.start = first_day
        self.end = end_day

        all_months = list(range(1, 13)) * 2
        self.months = all_months[startMonth - 1 : (endMonth + jumpYear*12)]

        self.leapday = None
        if startMonth > endMonth and endMonth >= 2:
            if calendar.isleap(baseYear):
                self.leapday = datetime(baseYear, 2, 29)
        elif startMonth <= 2 and endMonth >= startMonth:
            if calendar.isleap(startYear):
                self.leapday = self.leapday = datetime(baseYear, 2, 29)

        self.name = ''.join([
            datetime.strptime(f"1970{mon:02d}01", "%Y%m%d").strftime("%b").lower()[0] for mon in self.months
        ])

    @property
    def ndays(self) -> int:
        count = (self.end - self.start).days + 1
        if self.leapday is not None:
            return count - 1
        return count

    @property
    def doys(self) -> List[int]:
        return [(self.start + timedelta(days=i)).timetuple().tm_yday - 1 for i in range(self.ndays)]

    def dos(self, date: datetime) -> int:
        if self.leapday is not None and date >= self.leapday:
            return (date - self.start).days - 1
        return (date - self.start).days

    def __len__(self) -> int:
        return len(self.months)

    def __contains__(self, date: datetime) -> bool:
        return self.start.date() <= date.date() <= self.end.date()

    def __repr__(self) -> str:
        return f"Season ({self.name}) - {self.start:%d/%m/%Y}: {self.end:%d/%m/%Y} ({self.ndays} days)"


class SeasonConfig:
    def __init__(self, months: List[Tuple[int]]):
        self.months = months

    def get_season(self, date: datetime) -> Season:
        month = date.month
        for start, end in self.months:
            if start <= month <= end:
                return Season(start, end, date.year)
            if end < start:
                if month <= end:
                    return Season(start, end, date.year)
                if start <= month:
                    return Season(start, end, date.year + 1)
        raise ValueError(f"No season containing month {month}")