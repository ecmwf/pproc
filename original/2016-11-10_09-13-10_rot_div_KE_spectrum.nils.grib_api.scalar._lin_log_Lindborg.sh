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
DIR="Lindborg"
LEV=1
#EXP=gajn
#EXP=gajp
EXP=aircraft_cmp
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
PARAM=130
MARK=aircraft_Lindborg

for lev in 9 10 11 20 21 22 23 24 25 ; do
# IFS
LEV=$lev
# IFS

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
         write(7,'(2(f10.5),2X,2(1X,e15.10))') &
     &    log10(float(jn)),log10(znorm1(jn)*float(jn)**1.6666),&
     &    znorm1(jn)*float(jn)**1.666666,znorm1(jn)
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
#ln -s /home/rd/rdx/data/39r1/climate.v007/7999l_2/sporog targx1
ln -s /fwsm/lb/user/naw/HRES_7999/spectra_T_$LEV targx1
./spect
mv outx2 $ROOT/curv.${MARK}.${LEV}.${EXP}.${STEP}.scalar.variance.dat

done
done

#/home/rd/naw/own/curvplot.nils
#ghostview curv.ps
#\rm targ1 
exit 0 
