# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
import sys
import argparse
import numpy as np
import pandas as pd

from pproc.tcycl.tcycl_tools import read_oper_fc


class Experiment:
    """
    An experiment (e.g. HRES, cf, pf..)
    for which we need to retrieve TC data
    """

    basin_list = [
        'wnp',
        'spc',
        'sin',
        'nin',
        'enp',
        'cnp',
        'aus',
        'atl'
    ]
    
    name = None
    exp_version = ""
    exp_class = None
    exp_type = None
    exp_stream = None
    max_member = None
    max_step = None
    
    # tar file retrieved from ECFS
    ecfs_tar_file = "tcyc.tar"
    
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.tmp_dir = os.path.join(self.out_dir, "tmp")
        self.outdir = os.path.join(self.out_dir, self.exp_version)
    
    def get_data(self, date_from, date_to):
        """
        Get Experiment data
        """
        
        date_a = pd.to_datetime(date_from)
        date_b = pd.to_datetime(date_to)
        by_day = 0.5
        
        dates = pd.date_range(date_a, date_b, freq=str(by_day) + 'D')
        
        if not os.path.exists(self.outdir):
            print(f"Creating dir: {self.outdir}")
            os.system(f"mkdir -p {self.outdir}")
        
        # get data
        self._get_tar_data(dates)
        
        # write a summary CSV file
        self._write_csv(date_from, date_to)
    
    def _get_tar_data(self, dates):
        """
        Retrieve the tar file from ECFS
        """
        
        for _date in dates:
            yyyy = _date.strftime('%Y')
            mm = _date.strftime('%m')
            dd = _date.strftime('%d')
            hh = _date.strftime('%H')
            
            # ECFS path
            ecfs_dir = '/emos/tc/' + self.exp_version + '/' + yyyy + mm + '/' + dd + hh
            ecfs_path = os.path.join(ecfs_dir, self.ecfs_tar_file)
            
            # target path
            target_path = os.path.join(self.tmp_dir, self.ecfs_tar_file)
            
            print(f"Copying {ecfs_path} from ECFS..")
            os.system(f"ecp -n ec:{ecfs_path} {self.tmp_dir + '/'}")
            
            print(f"Un-tarring {target_path} in {self.tmp_dir}")
            os.system(f"tar -xf {target_path} -C {self.tmp_dir + '/'}")
            
            print(f"Tar finished. Cleaning up {target_path}")
            # os.system(f"rm -f {target_path}")
            
            self._extract_inner_tar(_date)
    
    def _extract_inner_tar(self, date):
        """
        Un-tar the data file within the ecfs tar
        """
        
        data_tar = self._data_tar_file(date)
        
        print(f"Un-tarring {data_tar} in {self.outdir}")
        os.system(f"tar -xf {data_tar} -C {self.outdir + '/'}")
    
    def _data_tar_file(self, date):
        """
        Path of datafile tar (inside the retrieved tar)
        Each experiment type has its own path/names..
        """
        
        raise NotImplementedError

    def _write_csv(self, date_from, date_to):
        
        print(f"Generating summary CSV..")
    
        dd_from = pd.to_datetime(date_from)
        dd_to = pd.to_datetime(date_to)
        by_day = 1
        
        datelist_all = pd.date_range(dd_from, dd_to, freq=str(by_day) + 'D')
    
        df = pd.DataFrame()
        for basin in self.basin_list:
            for date1 in datelist_all:
            
                filename = self.out_dir + "/" \
                           + self.exp_version + "/" \
                           + self.exp_version + \
                           date1.strftime('%Y%m%d%H') + \
                           "_" + self.exp_stream + \
                           "_000" + str(self.max_step) + \
                           "_" + basin

                # NB obsfile must be set to "none"
                # => no tc-name matching allowed ["vitals", "bt", "ib"]
                df1 = read_oper_fc(filename, date1, basin, True, obsfile="none")
            
                for ind in df1:
                    # df = df.append(df1[ind])
                    df = pd.concat([df, df1[ind]])

        csv_summary_path = os.path.join(self.out_dir, f"tc_tracker_data_{self.name}.csv")
        
        print(f"Writing summary CSV file {csv_summary_path}")
        df.to_csv(csv_summary_path)


class HRES_Experiment(Experiment):
    """
    HRES experiment
    """
    
    name = "HRES"
    exp_version = "0001"
    exp_class = "od"
    exp_type = "da"
    exp_stream = "da"
    max_member = 1
    max_step = 240
    
    def __init__(self, out_dir):
        super().__init__(out_dir)
    
    def _data_tar_file(self, date):
        yyyy = date.strftime('%Y')
        mm = date.strftime('%m')
        dd = date.strftime('%d')
        hh = date.strftime('%H')
        
        tar_dir = self.tmp_dir + \
                  '/tcyc_' + str(self.max_step) + \
                  '/' + self.exp_version + \
                  '/' + yyyy + mm + dd + hh + \
                  '/' + self.exp_type
        
        tar_file = self.exp_version + yyyy + mm + dd + hh + \
                   '_' + self.exp_type + \
                   '_' + '000' + str(self.max_step)
        
        return os.path.join(tar_dir, tar_file)


class CF_Experiment(Experiment):
    """
    CF experiment
    """
    
    name = "CF"
    exp_version = "0001"
    exp_class = "od"
    exp_type = "cf"
    exp_stream = "000"
    max_member = 1
    max_step = 240
    
    def __init__(self, out_dir):
        super().__init__(out_dir)
    
    def _data_tar_file(self, date):
        yyyy = date.strftime('%Y')
        mm = date.strftime('%m')
        dd = date.strftime('%d')
        hh = date.strftime('%H')
        
        tar_dir = self.tmp_dir + \
                  '/tcyc_' + str(self.max_step) + \
                  '/' + self.exp_version + \
                  '/' + yyyy + mm + dd + hh + \
                  '/' + self.exp_type
        
        tar_file = self.exp_version + \
                   yyyy + mm + dd + hh + \
                   '_' + "000" + \
                   '_' + '000' + \
                   str(self.max_step)
        
        return os.path.join(tar_dir, tar_file)


class PF_Experiment(Experiment):
    """
    PF experiment
    """
    
    name = "PF"
    exp_version = "0001"
    exp_class = "od"
    exp_type = "pf"
    exp_stream = "001"
    max_member = 50
    max_step = 240
    
    def __init__(self, out_dir):
        super().__init__(out_dir)
    
    def _extract_inner_tar(self, date):
        yyyy = date.strftime('%Y')
        mm = date.strftime('%m')
        dd = date.strftime('%d')
        hh = date.strftime('%H')
        
        for number in np.arange(1, self.max_member + 1):
            ens_number = '{:03}'.format(number)
            
            tar_dir = self.tmp_dir + \
                      '/tcyc_' + str(self.max_step) + \
                      '/' + self.exp_version + \
                      '/' + yyyy + mm + dd + hh + \
                      '/' + self.exp_type + \
                      '/' + ens_number + '/'
            
            tar_file = self.exp_version + \
                       yyyy + mm + dd + hh + \
                       '_' + ens_number + \
                       '_000' + str(self.max_step)
            
            tar_file_path = os.path.join(tar_dir, tar_file)
            
            print(f"Un-tarring {tar_file_path} in {self.outdir}")
            os.system(f"tar -xf {tar_file_path} -C {self.outdir + '/'}")


exp_types = {
    "hres": HRES_Experiment,
    "pf": PF_Experiment,
    "cf": CF_Experiment
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("date_from", help=f"Requested data from (format 'YYYYMMDD TT', e.g. '20211004 00')")
    parser.add_argument("date_to", help=f"Requested data to (format 'YYYYMMDD TT', e.g. '20211004 12')")
    parser.add_argument("--output_dir", help=f"Output directory", default="/var/tmp/tc_tracker_data")
    parser.add_argument("--data_types", help=f"Data types requested (e.g. hres,pf,cf)", default="hres")
    args = parser.parse_args()
    
    for t in args.data_types.split(","):
        print(f"Retrieving data for {t} - from {args.date_from} to {args.date_to}")
        exp = exp_types[t](args.output_dir)
        exp.get_data(args.date_from, args.date_to)


if __name__ == "__main__":
    sys.exit(main())
