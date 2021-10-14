cat >> mars_req_cf <<EOF
retrieve,
stream  =   enfo,
levtype =   sfc,
expver  =   0001,
date    =   $YMD,
time    =   $HOUR,
type    =   cf,
Grid   =   O640, Gaussian=   'reduced',
accuracy = av,
class   =   od,
param   =   167.128,
step    =   6/to/240/by/6,
field   =   fld_167.128

write,
field   =   fld_167.128,
target  =   cf_fld_167.128.grib
EOF

cat >> mars_req_pf <<EOF
retrieve,
stream  =   enfo,
levtype =   sfc,
expver  =   0001,
date    =   $YMD,
time    =   $HOUR,
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
target  =   pf_fld_167.128.grib
EOF

mars mars_req_cf
fdb-write cf_fld_167.128.grib

mars mars_req_pf
fdb-write pf_fld_167.128.grib

ls -lh *_fld_167.128.grib
cp *_fld_167.128.grib $DATA_DIR/

fdb-write /sc1/tcwork/emos/emos_data/0001/efi_clim/clim/$CLIM_YMD/clim_2t024_${CLIM_YMD}_000h_024h_perc.grib

cp /sc1/tcwork/emos/emos_data/0001/efi_clim/clim/$CLIM_YMD/clim_2t024_${CLIM_YMD}_000h_024h_perc.grib $DATA_DIR/
