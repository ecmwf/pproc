#! /bin/ksh
# (C) ECMWF, Nils Wedi, grib_api version, 01122009
#
set -axe

ROOT=/fwsm/lb/user/naw/spectrum

mkdir -p -m 755 $ROOT/spc555
cd $ROOT/spc555

export MARS_GRIB_API=1

ARCH=${ARCH:-cray}

#DIR=accumulated_237_240
#DIR=accumulated_03_06
#DIR=accumulated_21_24_L64
#DIR=accumulated_21_24_L49
#DIR=accumulated_21_24_L39
DIR="orog"
LEV=1
#EXP=gajn
#EXP=gajp
EXP=orog_cmp
DATE=20140201
TIME=00

# Radius adjust in km orig = 6371.229 !!!!
RDS=6371.229
#RDS=637.1229
#RDS=318.56145

CLASS=RD
# lev 18 == 10hPa
# lev 39 == 100hPa
# lev 49 == 200hPa
# lev 64 == 500hPa
# lev 71 == 700hPa
# lev 77 == 850hPa

#60 lev

# lev 39 == 500
# lev 30 == 200

LT=ml

for PARAM in 129 ; do
# IFS
MARK=orography_cubic_old

istep=0
while [ $istep -le 0 ] ; do

STEP=$istep
#istep=$((istep+3))
#istep=$((istep+12))
#istep=$((istep+12))
#istep=$((istep+48))
istep=$((istep+12))
#istep=$((istep+60))

cat >tcpack.F90 << EOF
      PROGRAM TCPACK
      USE GRIB_API
      INTEGER iparam
      REAL,ALLOCATABLE :: ZT(:)

      REAL,ALLOCATABLE :: znorm1(:), zw(:)

      character*30 ytitle1,ytitle2,ytitle3

      ! open file
      CALL GRIB_OPEN_FILE(NFILE1,'targx1','R')
      ! get handle
      CALL GRIB_NEW_FROM_FILE(NFILE1,IGRIB_H, IRET)

      CALL GRIB_GET_SIZE(IGRIB_H,'values',ISIZE)
      IF(.NOT.ALLOCATED(ZT)) ALLOCATE(ZT(ISIZE))

      CALL GRIB_GET(IGRIB_H,'paramId',iparam)
      CALL GRIB_GET(IGRIB_H,'values',ZT)
      print *, 'FIELD: ',iparam
      CALL GRIB_GET(IGRIB_H,'pentagonalResolutionParameterJ',NSMAX)
      print *, 'NSMAX: ',NSMAX

      ALLOCATE(znorm1(0:NSMAX))
      ALLOCATE(zw(0:NSMAX))

      CALL GRIB_RELEASE(IGRIB_H)
      ! close output file
      CALL GRIB_CLOSE_FILE(NFILE1)

      nmin=1
      nmax=nsmax
      n=nmax-nmin+1
      do 900 jn=nmin,nmax
      zw(jn)=1./float(jn-nmin+1)
 900  continue     
      do jn=0,nsmax
        znorm1(jn)=0.0
      enddo
    
! only need 0.5 due to eq in Lambert, is eq(3)*2 in IFS 
      zlam=0.5 
 
      zmax=-1e10
      zmin=1e10
      zspmax=-1e10
      zspmin=1e10
! temperature
      IJ=0
      DO JM=0,NSMAX
        if(jm.eq.0) then
          zmet=1.0*zlam
        else
          zmet=2.0*zlam
        endif      
        DO JN=JM,NSMAX
          IJ=IJ+1
          zfact=1.
! M=N          
!         IF(JN.EQ.JM) THEN
          znorm1(jn)=znorm1(jn)+zmet*zt(ij)**2*zfact
!         ENDIF
          IJ=IJ+1
          IF(JM.NE.0) THEN
! M=N
!           IF(JN.EQ.JM) THEN
            znorm1(jn)=znorm1(jn)+zmet*zt(ij)**2*zfact
!           ENDIF
          ENDIF
        ENDDO
      ENDDO
        write(ytitle1,'(a)') 'T Lev $LEV H+$STEP $EXP '
        write(ytitle2,'(a)') 'T Lev $LEV H+$STEP $EXP '
        write(ytitle3,'(a)') 'T Lev $LEV H+$STEP $EXP '
      open(7,file='outx2',form='formatted')  
      write(7,'(a)') ytitle1
      write(7,'(a)') ytitle2
      write(7,'(a)') ytitle3
      do 213 jn=0,nsmax
      if(jn.ne.0) then
         write(7,'(4(f10.5))') &
     &    log10(float(jn)),&
     &    log10(znorm1(jn)),log10(znorm1(jn)),log10(znorm1(jn)*float(jn)**1.6666)
      endif
 213  continue      
 10   continue      

      DEALLOCATE(znorm1)
      DEALLOCATE(zw)

      DEALLOCATE(ZT)

      end
EOF
/bin/rm outx2 || true

GRIB_VERSION="1.11.0"
module switch grib_api grib_api/$GRIB_VERSION

\rm -f tcpack.o targx1
~rdx/bin/crayftn_wrapper -c OPTIMIZATION_STRING -r am -emf $GRIB_API_INCLUDE tcpack.F90
ftn -o spect tcpack.o -L/home/rd/rdx/liblinks/emos/000395/CRAY/83/lib -lemos.R64.D64.I32 $GRIB_API_LIB || exit 1
cat >dir<< EOF
retrieve,date=$DATE,class=$CLASS,expver="$EXP",levtype=ml,type=fc,
   stream=lwda,anoffset=9,level=1,param=129,step=$STEP,time=$TIME,target="targx1"
EOF
#mars dir
\rm -f xx.grib targx1
#ln -s /home/rd/nag/work/orog_testing/sporog_1279_cubic_lapl.0.01 targx1
#ln -s /home/rd/nag/work/orog_testing/sporog_2047_linear_lapl.0.01_filt5_32 targx1
#ln -s /home/rd/nag/work/orog_testing/sporog_2047_linear_lapl.0.02raw targx1
#ln -s /home/rd/nag/work/orog_testing/sporog_2047_linear_lapl.0.02_filt5_32 targx1
#ln -s /home/rd/nag/work/orog_testing/sporog_1279_cubic_lapl.0.02_filt5_32 targx1
#ln -s /home/rd/nag/work/orog_testing/sporog_639_cubic_lapl.0.01_filt5_32 targx1
#ln -s /home/rd/nag/work/orog_testing/sporog_1279_cubic_lapl.0.01_filt5_32 targx1
#ln -s /home/rd/nag/work/orog_testing/sporog_1279_cubic_lapl.0.01_filt5_16 targx1
#ln -s /home/rd/nag/work/tmp/40r3/climate/1279_3/sporog_cubic targx1
#ln -s /home/rd/nag/work/tmp/40r3/climate/639_3/sporog_cubic targx1
#ln -s /home/rd/nag/work/orog_testing/sporog_639_cubic_lapl.0.0025_filt5_32 targx1
#ln -s /home/rd/nag/work/tmp/40r3/climate/639_4/sporog targx1
#ln -s /home/rd/nag/work/tmp3/tmp/41r1/climate/3999_4/sporog targx1
#ln -s /home/rd/nag/work/tmp3/tmp/42r1/climate/319_4/sporog targx1
#ln -s /home/rd/naw/spectrum/sporog_319_4_0.0001 targx1

#ln -s /home/rd/naw/spectrum/sporog_TCo1279_v015 targx1
#ln -s /home/rd/naw/spectrum/sporog_2559_filtered_v015 targx1
#ln -s /home/rd/naw/spectrum/sporog_2559_not_smoothed_with_5km_pre_average targx1
#ln -s /home/rd/rdx/data/42r1/climate.v015/7999_4/sporog targx1
#ln -s /home/rd/rdx/data/42r1/climate.v015/1999_4/sporog targx1
#ln -s /home/rd/rdx/data/42r1/climate.v015/255l_2/sporog targx1
#ln -s /home/rd/rdx/data/42r1/climate.v015/799l_2/sporog targx1
#ln -s /home/rd/rdx/data/42r1/climate.v015/399_4/sporog targx1
ln -s /home/rd/naw/spectrum/sporog_1999_5 targx1

#ln -s /home/rd/rdx/data/42r1/climate.v015/319_4/sporog targx1
#ln -s /home/rd/rdx/data/41r1/climate.v012/1999_4/sporog targx1
#ln -s /home/rd/rdx/data/41r1/climate.v011/1023_4/sporog targx1
#ln -s /home/rd/rdx/data/41r1/climate.v011/639_4/sporog targx1
#ln -s /home/rd/rdx/data/41r1/climate.v011/1279_4/sporog targx1
#ln -s /home/rd/rdx/data/41r1/climate.v011/3999_4/sporog targx1

#ln -s /home/rd/rdx/data/41r1/climate.v011/1023_4/sporog targx1
#ln -s /home/rd/rdx/data/41r1/climate.v011/639_4/sporog targx1
#ln -s /home/rd/nag/work/tmp/40r3/climate/2047l_2/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate.v010/1279l_2/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate.v010/1279_3/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate.v010/639l_2/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate.v010/639_3/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate.v007/2047l_2/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate.v008/639l_2/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate/639l_2/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate/511l_2/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate.v008/511l_2/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate.v008/319l_2/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate/319l_2/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate/399l_2/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate.v008/799l_2/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate/799l_2/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate.v008/399l_2/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate.v008/255l_2/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate/255l_2/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate/159l_2/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate.v007/159l_2/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate.v008/95l_2/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate/95l_2/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate.v008/1279l_2/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate/1279l_2/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate.v008/1364_2/sporog targx1
#ln -s /home/rd/rdx/data/40r1/climate.v007/1364_2/sporog targx1
#ln -s /perm/rd/naw/data_scripts/sporog_1364 targx1
#ln -s /perm/rd/naw/data_scripts/sporog_1364_filt1 targx1
#ln -s /home/rd/rdx/data/40r1/climate.v008/2047l_2/sporog targx1
#ln -s /home/rd/rdx/data/39r1/climate/3999l_2/sporog targx1
#ln -s /home/rd/rdx/data/39r1/climate.v007/3999l_2/sporog targx1
#ln -s /home/rd/rdx/data/39r1/climate.v007/7999l_2/sporog targx1
./spect
mv outx2 $ROOT/curv.${MARK}.${LEV}.${EXP}.${STEP}.scalar.variance.dat

done
done

#/home/rd/naw/own/curvplot.nils
#ghostview curv.ps
#\rm targ1 
exit 0 
