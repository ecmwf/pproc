cat >> mars_req_cf <<EOF
retrieve,
stream  =   enfo,
levtype =   sfc,
expver  =   0001,
date    =   20210812,
time    =   12,
type    =   cf,
Grid   =   O640, Gaussian=   'reduced',
accuracy = av,
class   =   od,
param   =   167.128,
step    =   6/to/240/by/6,
field   =   fld_167.128

write,
field   =   fld_167.128,
target  =   fld_167.128.grib
EOF

cat >> mars_req_pf <<EOF
retrieve,
stream  =   enfo,
levtype =   sfc,
expver  =   0001,
date    =   20210812,
time    =   12,
type    =   pf,
Grid   =   O640, Gaussian=   'reduced',
accuracy = av,
class   =   od,
param   =   167.128,
step    =   6/to/240/by/6,
number  =   1/to/50,
field   =   fld_167.128

write,
field   =   fld_167.128,
target  =   fld_167.128.grib
EOF

mars mars_req_cf
mars mars_req_pf

ls -lh fld_167.128.grib

export FDB5_CONFIG_FILE=$FDB_DIR/config.yaml

fdb-write fld_167.128.grib

cp fld_167.128.grib $DATA_DIR/
