import os
import sys
import wget
import argparse

import numpy as np
import pandas as pd


class IBtracksSummary:
    """
    Handles IBTracks data
    """

    noaa_url = "https://www.ncei.noaa.gov/data"
    ibtracks_sub_url = "international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv"
    ibtracks_file = "ibtracs.last3years.list.v04r00.csv"

    ibtracks_url = os.path.join(noaa_url, ibtracks_sub_url, ibtracks_file)
    
    winds = [
        "WMO_WIND",
        "USA_WIND",
        "TOKYO_WIND",
        "CMA_WIND",
        "NEWDELHI_WIND",
        "REUNION_WIND",
        "BOM_WIND",
        "WELLINGTON_WIND",
        "TD9636_WIND",
        "TD9635_WIND",
        "NEUMANN_WIND",
        "MLC_WIND",
    ]
    
    pres = [
        "WMO_PRES",
        "USA_PRES",
        "TOKYO_PRES",
        "CMA_PRES",
        "HKO_PRES",
        "NEWDELHI_PRES",
        "REUNION_PRES",
        "BOM_PRES",
        "BOM_PRES_METHOD",
        "NADI_PRES",
        "WELLINGTON_PRES",
        "DS824_PRES",
        "TD9636_PRES",
        "TD9635_PRES",
        "NEUMANN_PRES",
        "MLC_PRES",
    ]

    summary_fields = [
        "ISO_TIME",
        "SID",
        "NUMBER",
        "BASIN",
        "SUBBASIN",
        "NAME",
        "NATURE",
        "LAT",
        "LON",
        "wind",
        "pres"
    ]
    
    def __init__(self, df):
        self.df = df

    @classmethod
    def download(cls, download_path):
        """
        Download the IBtracks file
        """
        if not os.path.exists(download_path):
            wget.download(cls.ibtracks_url, out=download_path)
        else:
            print(f"File {download_path} already exists, not downloading.")

    @classmethod
    def from_file(cls, file_name):
        df = cls.readIBTracs(file_name)
        return cls(df)
    
    @classmethod
    def readIBTracs(cls, file):
        """
        Read ibtracks CSV file
        """

        df = pd.read_csv(file, sep=",", header=0, skiprows=[1, 2], low_memory=False)
        df = df.replace(-999.0, np.nan)

        # find max sustained wind and min pressure
        df[["wind"]] = df.apply(cls.bestWind, axis=1)
        df[["pres"]] = df.apply(cls.bestPres, axis=1)

        # extract only needed columns
        df = df[cls.summary_fields]

        # df = df[df["wind"] > 34]

        # sort fields by time
        df.sort_values(by=["ISO_TIME"])
        
        return df
    
    @classmethod
    def has_numbers(cls, input_string):
        return any(char.isdigit() for char in input_string)

    @classmethod
    def bestWind(cls, row):
        """
        Determine wind, could be from three
        different columns depending on who is reporting
        """

        result = np.nan

        for r in cls.winds:
            value = row[r]
            if cls.has_numbers(value):
                result = int(value)
                break

        return pd.Series(dict(wind=result))
    
    @classmethod
    def bestPres(cls, row):
        """
        Determine wind, could be from three
        different columns depending on who is reporting
        """

        result = np.nan

        for r in cls.pres:
            value = row[r]
            if cls.has_numbers(value):
                result = int(value)
                break

        return pd.Series(dict(pres=result))
    
    def save(self, csv_filename, only_time=None):
        """
        Saves to CSV
        """

        if only_time:
            time_ = pd.to_datetime(only_time)
            self.df.loc[pd.to_datetime(self.df['ISO_TIME']) == time_].to_csv(csv_filename, index=None, header=True)
        else:
            self.df.to_csv(csv_filename, index=None, header=True)

    
def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ibtracks_file", help=f"Path to downloaded CSV file", default="ibtracks.csv")
    parser.add_argument("--output", help=f"Path to summary CSV file", default="ibtracks-summary.csv")
    parser.add_argument("--only_time", help=f"Data only at date/time (format 'YYYYMMDD TT', e.g. '20211004 00')")
    args = parser.parse_args()
    
    # Make summary file
    IBtracksSummary.download(args.ibtracks_file)
    IBtracksSummary.from_file(args.ibtracks_file).save(args.output, args.only_time)


if __name__ == "__main__":
    sys.exit(main())
