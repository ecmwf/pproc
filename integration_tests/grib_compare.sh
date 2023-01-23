#!/usr/bin/env bash

# Assumes existence of data in fdb for comparison. Takes in date, time and mars request for data to be checked as 
# first three arguments. By default, the script does not check data values. If fourth argument is provided then 
# checking of data values is turned on.

DATE=$1
TIME=$2
MARS_REQUEST_FILE=$3

cp $MARS_REQUEST_FILE temp_request.mars
echo ",date=$DATE,time=$TIME" >> temp_request.mars
pproc-bundle/install/bin/fdb read temp_request.mars test.grib
echo ",target=forecast.grib" >> temp_request.mars
mars temp_request.mars
if [ -z $4 ]
then 
    OUTPUT=$(pproc-bundle/install/bin/grib_compare -b values,codedValues test.grib forecast.grib)
else
    OUTPUT=$(pproc-bundle/install/bin/grib_compare test.grib forecast.grib)
fi 

# Tidy 
rm temp_request.mars test.grib forecast.grib

if [ -z "$OUTPUT" ]
then 
    exit 0
else
    printf "Error in comparison for file %s: \n $OUTPUT \n" $MARS_REQUEST_FILE
    exit 1
fi

