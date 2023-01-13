#!/usr/bin/env bash

# Assumes existence of data in fdb for comparison. Takes in mars request for data to be checked as 
# first argument. By default, the script does not check data values. If second argument is provided then 
# checking of data values is turned on.

MARS_REQUEST_FILE=$1

pproc-bundle/install/bin/fdb read $MARS_REQUEST_FILE test.grib
cp $MARS_REQUEST_FILE temp_request.mars
echo ",target=forecast.grib" >> temp_request.mars
mars temp_request.mars
if [ -z $2 ]
then 
    OUTPUT=$(pproc-bundle/install/bin/grib_compare -b values test.grib forecast.grib)
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

