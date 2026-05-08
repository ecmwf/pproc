# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pandas as pd

# NOTE: the following functions are taken directly from "tctools".


def calc_dist(latdiff, londiff, lat):
    
    radius = 6371000
    dlat = np.radians(latdiff)
    dlon = np.radians(londiff)
    a = (np.sin(dlat / 2) * np.sin(dlat / 2) +
         np.cos(np.radians(lat)) * np.cos(np.radians(lat+latdiff)) *
         np.sin(dlon / 2) * np.sin(dlon / 2))
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = radius * c
    
    return dist


def calc_speed3(my_pd):
    
    latdiff = my_pd["lat_p"].diff()
    londiff = my_pd["lon_p"].diff()
    lat = my_pd["lat_p"]
    
    dist = calc_dist(latdiff, londiff, lat)
    speed = dist / (my_pd["step"].diff() * 3600)  # in m/s
    speedcum = dist.cumsum() / 1000.  # km

    return [speed, speedcum]


def read_besttrack_date(date1, obsfile):
    print(f"ERROR: {__name__} not used for tcycl validation..")
    raise NotImplementedError


def check_name(pd_fc, pd_bt, n_tc, tol=5):
    print(f"ERROR: {__name__} not used for tcycl validation..")
    raise NotImplementedError


def read_oper_fc(filename, date1, basin, Pall, obsfile="none"):
    
    # here "read_oper_fc" can only be used with no obsfile!
    assert obsfile == "none", "Error, for tcycl validation purposes " \
                              "obsfile must be set to 'none'"
    
    dict_of_tc = {}
    recon = date1.strftime('%Y')
    
    if obsfile != "none":
        df_bt = read_besttrack_date(date1, obsfile)
    
    if (basin == "aus") or (basin == "spc") or (basin == "sin"):
        lat_mult = -1.
    else:
        lat_mult = 1
    fp = open(filename, 'r')
    line = fp.readline()
    cnt = 1
    ar = []  # Array for properties from file
    ard = []  # Array for variables related to the datestamp
    n_tc = 1
    while line:
        s1 = line.split(" ")
        if s1[1][0:4] == recon:
            year = line[6:10]
            month = line[11:13]
            day = line[14:16]
            time = line[17:19]
            lat1 = lat_mult * float(line[20:23]) / 10. + 0.05
            lon1 = float(line[23:27]) / 10. + 0.05
            wind = float(line[28:31])
            pres = float(line[32:36])
            lat2 = lat_mult * float(line[37:40]) / 10. + 0.05
            lon2 = float(line[40:44]) / 10. + 0.05
            # print([year,month,day,time])
            # if time=="00" or time=="12":
            ard.append([year, month, day, time])
            ar.append([lat1, lon1, wind, pres, lat2, lon2])
        elif s1[0][4] != "0" and len(ard) > 0:
            # print("Finish TC",s1[0])
            np_ard = np.asarray(ard)
            np_ar = np.asarray(ar)
            my_dates = pd.to_datetime({'year': np_ard[:, 0],
                                       'month': np_ard[:, 1],
                                       'day': np_ard[:, 2],
                                       'hour': np_ard[:, 3]})  #
            d = {"datetime": my_dates, "lat_p": np_ar[:, 0], "lon_p": np_ar[:, 1], "wind": np_ar[:, 2],
                 "pres": np_ar[:, 3], "lat_w": np_ar[:, 4], "lon_w": np_ar[:, 5]}
            df1 = pd.DataFrame(d)
            # print(df1)
            
            # ==============================================================
            # Check if the cyclone is observed at the initialisation time,
            # and extract name from observation file in that case
            # if (obsfile != "none" and (df_bt.empty == False)):
            #     # print(df_bt)
            #     name1 = check_name(df1, df_bt, n_tc, 5)
            #
            # else:
            #     name1 = str(n_tc)

            # No name matching used
            name1 = str(n_tc)
            # ==============================================================
            
            # Calculate additional diagnostics
            df1["step"] = (df1["datetime"] - date1).values.astype("timedelta64[h]").astype(int)
            df1["windrad"] = calc_dist(df1.lat_p - df1.lat_w, df1.lon_p - df1.lon_w, df1.lat_p) / 1000.
            df1["wind"] = df1["wind"] * 0.514444
            
            speed = calc_speed3(df1)
            df1["speed"] = speed[0]
            df1["accdist"] = speed[1]
            # df1["accdist"]=calc_speed2(df1)
            df1["dpdt"] = df1["pres"].diff() / df1["step"].diff()
            # Fill a column with the cyclone name
            # print(str(n_tc),name1)
            if str(n_tc) != name1:
                df1["name"] = name1
                # dict_of_tc[name1] = df1
                dict_of_tc["tc" + str(n_tc)] = df1
            else:
                nameTmp = "unnamed_" + basin + str(n_tc)
                df1["name"] = nameTmp
                if Pall:
                    dict_of_tc[nameTmp] = df1
                    
                    # Add DataFrame to Dictionary only if the cyclone is in observation file. Can be changed.
            
            # if str(n_tc)!=name1:
            #    dict_of_tc[name1] = df1
            
            n_tc += 1
            ar = []
            ard = []
        
        line = fp.readline()
        cnt += 1
    
    return dict_of_tc
