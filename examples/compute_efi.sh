
module load ifs/CY47R2.20210609

test_data=/perm/ma/macw/ppop/efi_sot

set -ex

date_eps=2021062300
date_clim=20210621

cat > fort.75 <<EOF
&IN_LIST
type_num1=0,
type_num2=0,
sort=.FALSE.,
epsilon=-1e-4
epsfile="${test_data}/input/eps_2t024_${date_eps}_000h_024h.grib",
climfile="${test_data}/clim/clim_2t024_${date_clim}_000h_024h_perc.grib",
indexfile="efi0_2t024_000_024.grib",
/
EOF

produce_efi

grib_set -rs numberOfBitsContainingEachPackedValue=12 efi0_2t024_000_024.grib efi0_2t024_000_024.grib2
mv efi0_2t024_000_024.grib2 efi0_2t024_000_024.grib

cat > grib.filter << EOF
set timeRangeIndicator=3;
set stepRange="000-024";
set numberIncludedInAverage=4;
set numberMissingFromAveragesOrAccumulations=0;
set expver="0075";
set date=20210623;
set gribTablesVersionNo=132;
set indicatorOfParameter=167;
# set identificationOfOriginatingGeneratingSubCentre=0;
set subCentre=0;
set localDefinitionNumber=19;
set marsType=27;
set marsStream="enfo";
set totalNumber=51;
set powerOfTenUsedToScaleClimateWeight=75;
set weightAppliedToClimateMonth1=20990630;
set firstMonthUsedToBuildClimateMonth1=20;
set lastMonthUsedToBuildClimateMonth1=31;
set firstMonthUsedToBuildClimateMonth2=1980;
set lastMonthUsedToBuildClimateMonth2=0;
set efiOrder=0;
write;
EOF

grib_filter -f -o efi0_2t024_000_024.grib2 grib.filter efi0_2t024_000_024.grib
mv efi0_2t024_000_024.grib2 efi0_2t024_000_024.grib

cat > grib.filter << EOF
set number=0;
write;
EOF
grib_filter -f -o efi0_2t024_000_024.grib2 grib.filter efi0_2t024_000_024.grib
mv efi0_2t024_000_024.grib2 efi0_2t024_000_024.grib

diff efi0_2t024_000_024.grib ${test_data}/efi/efi0_50_2t024_2021062300_000h_024h.grib
