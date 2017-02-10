#! /bin/ksh
# (C) ECMWF, Nils Wedi, grib_api version, 01122009
#
set -axe
mkdir -p -m 755 /fwsm/lb/user/naw/spc53_4
cd /fwsm/lb/user/naw/spc53_4

# need this for T7999
export MARS_READANY_BUFFER_SIZE=167772160
export MARS_GRIB_API=1

ARCH=${ARCH:-cray}

# ctrl
#EXP=fsi5
#EXP=fsi4
#EXP=fshz
#EXP=fsi3

# flt
#EXP=fsi2
#EXP=fsi0
#EXP=fsi1
#EXP=fsi6
#EXP=fsi7

# flt lossy
#EXP=fsjb
#EXP=fsjz
#EXP=fsk4

# 1279
# ctrl FLT 1.e-10 thresh=128
#EXP=fxzq
# ctrl FLT=false
#EXP=fy0m

# FLT 1.e-3 thresh=128
#EXP=fxzr
# FLT 1.e-3 thresh=512 inv only
#EXP=fy05
# FLT 1.e-2 inv/ 1.e-3 dir thresh=512
#EXP=fy1f

# T2047 ctrl
#EXP=fy2n
# T2047 flt
#EXP=fy2m

# T2047 vs T1364
#EXP=fyu0
#EXP=fyp8
#EXP=fywa
#EXP=fyxp

#EXP=fzzn
#EXP=fyzb
#EXP=fyzc
#EXP=fz4u
#EXP=fz2i

#EXP=fzy7
#EXP=fypc

# T639 linear
#EXP=g0xk
# T639 cubic
#EXP=g0xi
# T639 diff from linear
#EXP=g29e
# T639 diff from linear not T diff
#EXP=g2es

#EXP=g3l6
#EXP=g0xk
# T1279 cubic diff
#EXP=g6a9
#EXP=g6ad
#  6th order diff
#EXP=g5zi
#EXP=g6bv

# T639 cubic diff
#EXP=g5ze
#  6th order diff
#EXP=g5zg

# c2q
#EXP=g6lk
# c2c
EXP=g6lb
# c2l
#EXP=g6la

# exps

TIME=00
#DATE=20130701
DATE=20131210
#DATE=20121203
#DATE=20130616
#DATE=20130715
#DATE=20121025
#DATE=20120820
#DATE=20070401
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

# piotr
#for lev in 9 ; do
#for lev in 10 100 200 500 700 850 ; do
# 137 levels ~ 200 + 500
for lev in 137 96 75 55 36 10 ; do
#for lev in 137 96 75 ; do
#for lev in 91 64 49 ; do
# IFS
LEV=$lev

#istep=420
istep=120
while [ $istep -le 120 ] ; do

STEP=$istep
#LEV=10
#LEV=18
#LEV=7
#LEV=49
#LEV=30
#istep=$((istep+3))
#istep=$((istep+12))
istep=$((istep+12))
#istep=$((istep+24))
#istep=$((istep+60))

cat >tcpack.F90 << EOF
      PROGRAM TCPACK
      USE GRIB_API
      INTEGER iparam
      REAL,ALLOCATABLE :: ZVO(:),ZDIV(:)

      REAL,ALLOCATABLE :: znorm1(:), znorm2(:), znorm3(:), zw(:)
      REAL,ALLOCATABLE :: ZPARC1(:,:),ZPARC2(:,:),ZPARC3(:,:) 

      character*30 ytitle1,ytitle2,ytitle3


      ! open file
      CALL GRIB_OPEN_FILE(NFILE1,'targx1','R')
      ! get handle
      CALL GRIB_NEW_FROM_FILE(NFILE1,IGRIB_H, IRET)

      ! check order of fields !!!!!!!!
      CALL GRIB_GET_SIZE(IGRIB_H,'values',ISIZE)
      IF(.NOT.ALLOCATED(ZDIV)) ALLOCATE(ZDIV(ISIZE))

      CALL GRIB_GET(IGRIB_H,'paramId',iparam)
      CALL GRIB_GET(IGRIB_H,'values',ZDIV)
      print *, 'FIELD: ',iparam
      CALL GRIB_GET(IGRIB_H,'pentagonalResolutionParameterJ',NSMAX)

      ALLOCATE(znorm1(0:NSMAX))
      ALLOCATE(znorm2(0:NSMAX))
      ALLOCATE(znorm3(0:NSMAX))
      ALLOCATE(zw(0:NSMAX))

      ALLOCATE(ZPARC1(0:NSMAX,0:NSMAX))
      ALLOCATE(ZPARC2(0:NSMAX,0:NSMAX))
      ALLOCATE(ZPARC3(0:NSMAX,0:NSMAX))

      CALL GRIB_RELEASE(IGRIB_H)
      CALL GRIB_NEW_FROM_FILE(NFILE1,IGRIB_H, IRET)

      CALL GRIB_GET_SIZE(IGRIB_H,'values',ISIZE)
      IF(.NOT.ALLOCATED(ZVO)) ALLOCATE(ZVO(ISIZE))
      CALL GRIB_GET(IGRIB_H,'paramId',iparam)
      CALL GRIB_GET(IGRIB_H,'values',ZVO)
      print *, 'FIELD: ',iparam

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
        znorm2(jn)=0.0
        znorm3(jn)=0.0
      enddo
!     2*pi*a
!      zra=2.*3.14159*${RDS}*1000.
! don't need 2pi
      zra=${RDS}*1000.
! only need 0.5 due to eq in Lambert, is eq(3)*2 in IFS
      zlam=0.5

      zmax=-1e10
      zmin=1e10
      zspmax=-1e10
      zspmin=1e10
! vorticity
      IJ=0
      DO JM=0,NSMAX
        if(jm.eq.0) then
          zmet=1.0*zlam
        else
          zmet=2.0*zlam
        endif      
        DO JN=JM,NSMAX
          IJ=IJ+1
          if(jn.ne.0) then
            zfact=zra**2/(jn*(jn+1))
          else
            zfact=1.
          endif
          ZPARC1(JM,JN)= zmet*(zvo(ij)**2)*zfact
! M=N          
!         IF(JN.EQ.JM) THEN
          znorm1(jn)=znorm1(jn)+zmet*(zvo(ij)**2)*zfact
!         ENDIF
          IJ=IJ+1
          IF(JM.NE.0) THEN
            ZPARC1(JM,JN)=ZPARC1(JM,JN)+&
     &                            zmet*(zvo(ij)**2)*zfact
! M=N
!           IF(JN.EQ.JM) THEN     
            znorm1(jn)=znorm1(jn)+zmet*(zvo(ij)**2)*zfact
!           ENDIF
          ENDIF
        ENDDO
      ENDDO
! divergence
      IJ=0
      DO JM=0,NSMAX
        if(jm.eq.0) then
          zmet=1.0*zlam
        else
          zmet=2.0*zlam
        endif      
        DO JN=JM,NSMAX
          IJ=IJ+1
          if(jn.ne.0) then
            zfact=zra**2/(jn*(jn+1))
          else
            zfact=1.
          endif
          ZPARC2(JM,JN)= zmet*(zdiv(ij)**2)*zfact
! M=N          
!         IF(JN.EQ.JM) THEN
          znorm2(jn)=znorm2(jn)+zmet*(zdiv(ij)**2)*zfact
!         ENDIF
          IJ=IJ+1
          IF(JM.NE.0) THEN
            ZPARC2(JM,JN)=ZPARC2(JM,JN)+ &
     &                            zmet*(zdiv(ij)**2)*zfact
! M=N          
!           IF(JN.EQ.JM) THEN
            znorm2(jn)=znorm2(jn)+zmet*(zdiv(ij)**2)*zfact
!           ENDIF
          ENDIF
        ENDDO
      ENDDO
! kin energy
! NOTE1: kinetic energy is derived from vorticity and divergence
!        nabla^-1 ~ a/sqrt(n*(n+1)), u = nabla^-1 D, v = \nabla^-1 \zeta
! NOTE2: spectral coefficients are normalized by 2*pi to have the 
!        0th coefficient represent the mean of the field, hence ra=a*2*pi
      IJ=0
      DO JM=0,NSMAX
        if(jm.eq.0) then
          zmet=1.0*zlam
        else
          zmet=2.0*zlam
        endif      
        DO JN=JM,NSMAX
          IJ=IJ+1
          if(jn.ne.0) then
            zfact=zra**2/(jn*(jn+1))
          else
            zfact=1.
          endif
          ZPARC3(JM,JN)= zmet*(zvo(ij)**2+zdiv(ij)**2)*zfact
! M=N          
!         IF(JN.EQ.JM) THEN
          znorm3(jn)=znorm3(jn)+zmet*(zvo(ij)**2+zdiv(ij)**2)*zfact
!         ENDIF
          IJ=IJ+1
          IF(JM.NE.0) THEN
            ZPARC3(JM,JN)=ZPARC3(JM,JN)+ &
     &                            zmet*(zvo(ij)**2+zdiv(ij)**2)*zfact
! M=N          
!           IF(JN.EQ.JM) THEN
            znorm3(jn)=znorm3(jn)+zmet*(zvo(ij)**2+zdiv(ij)**2)*zfact
!           ENDIF
          ENDIF
        ENDDO
      ENDDO
        write(ytitle1,'(a)') 'rotKE Lev $LEV H+$STEP $EXP '
        write(ytitle2,'(a)') 'divKE Lev $LEV H+$STEP $EXP '
        write(ytitle3,'(a)') 'KE    Lev $LEV H+$STEP $EXP '
      open(7,file='outx2',form='formatted')  
      write(7,'(a)') ytitle1   
      write(7,'(a)') ytitle2
      write(7,'(a)') ytitle3
      do 213 jn=0,nsmax
      if(jn.ne.0) then
         zmax1=max(log10(znorm1(jn)),log10(znorm2(jn)))
         zmax=max(zmax1,zmax)
         zmin1=min(log10(znorm1(jn)),log10(znorm2(jn)))
         zmin=min(zmin1,zmin)
         write(7,'(4(f10.5))') &
     &    log10(float(jn)),&
     &    log10(znorm1(jn)),log10(znorm2(jn)),log10(znorm3(jn)*float(jn)**1.6666)
      endif
 213  continue      
      print *,'  zmax=',zmax,' zmin=',zmin
 10   continue      

      DEALLOCATE(znorm1)
      DEALLOCATE(znorm2)
      DEALLOCATE(znorm3)
      DEALLOCATE(zw)

      DEALLOCATE(ZPARC1)
      DEALLOCATE(ZPARC2)
      DEALLOCATE(ZPARC3)

      DEALLOCATE(ZVO)
      DEALLOCATE(ZDIV)

      end
EOF
/bin/rm outx2 || true

GRIB_VERSION="1.11.0"
module switch grib_api grib_api/$GRIB_VERSION

\rm -f tcpack.o
crayftn_wrapper -c OPTIMIZATION_STRING -r am -emf $GRIB_API_INCLUDE tcpack.F90
ftn -o spect tcpack.o -L/home/rd/rdx/liblinks/emos/000395/CRAY/83/lib -lemos.R64.D64.I32 $GRIB_API_LIB || exit 1

cat >dir<< EOF
retrieve,date=$DATE,class=$CLASS,expver="$EXP",levtype=ml,type=fc,
   level=$LEV,param=D/VO,step=$STEP,time=$TIME,target="targx1"
EOF
mars dir
#cp /fdb/naw/spectra/targ2.2047.lev.${LEV}.vor.d.grib targx1
#cat >dir2<<EOF2
#EOF2
#mars dir2
./spect
rcp outx2 cormac:/scratch/rd/naw/curv.${LEV}.${EXP}.${STEP}.rot.div.dat

done
done

#/home/rd/naw/own/curvplot.nils
#ghostview curv.ps
#\rm targ1 
exit 0 
