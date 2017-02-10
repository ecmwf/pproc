#! /bin/ksh
# (C) ECMWF, Nils Wedi, grib_api version, 01122009
#
set -axe
mkdir -p -m 755 /fwsm/lb/user/naw/spc53_5
cd /fwsm/lb/user/naw/spc53_5

# need this for T7999
export MARS_READANY_BUFFER_SIZE=768896196
MARS_READANY_BUFFER_SIZE=768896196
#export MARS_READANY_BUFFER_SIZE=167772160
export MARS_GRIB_API=1

ARCH=${ARCH:-cray}

#EXP=gjq2
EXP=giyv
#EXP=gk1n
#EXP=gkda
#EXP=gk0n

TIME=00
#DATE=20130701
DATE=20160328
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
#for lev in 137 96 75 55 36 10 ; do
#for lev in 137 96 75 ; do
#for lev in 91 64 49 ; do
#for lev in 62 35 21 ; do
#for lev in 62 35 21 16 9 ; do
for lev in 9 10 11 20 21 22 23 24 25 ; do
# IFS
LEV=$lev

#istep=420
istep=12
while [ $istep -le 12 ] ; do

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
! don't need 2pi, from Mike Blackburn
!The normalisation of the ECMWF spherical harmonics when we used the
!model in the 1990s was such that the m=0,n=0 spectral coefficient was
!the global average and the m=0 Fourier coefficient was the zonal
!average.  If this is still the case, you shouldn't need the factor 2*pi
!in zra.
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
         write(7,'((f10.5),2X,3(1X,e15.10))') &
     &    log10(float(jn)),&
     &    znorm1(jn),&
     &    znorm2(jn),&
     &    znorm3(jn)
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

#cat >dir<< EOF
#retrieve,stream=lwda,anoffset=9,date=$DATE,class=$CLASS,expver="$EXP",levtype=ml,type=fc,
#   level=$LEV,param=D/VO,step=$STEP,time=$TIME,source="/fwsm/lb/user/naw/HRES_3999/ICMSHgjq2+000360",target="targx1"
#EOF
#mars dir
#   stream=lwda,anoffset=9,
#cp /fdb/naw/spectra/targ2.2047.lev.${LEV}.vor.d.grib targx1
#cat >dir2<<EOF2
#EOF2
#mars dir2

\rm targx1
#ln -s /fwsm/lb/user/naw/HRES_1279NH/spectra_$LEV targx1
#ln -s /fwsm/lb/user/naw/HRES_7999H_FPOS_OROG_1279_no_decentering/spectra_$LEV targx1
#ln -s /fwsm/lb/user/naw/HRES_7999NH/spectra_$LEV targx1
ln -s /fwsm/lb/user/naw/HRES_7999/spectra_$LEV targx1
#ln -s /fwsm/lb/user/naw/HRES_1999H/spectra_$LEV targx1
./spect
scp outx2 cormac:/scratch/rd/naw/curv.${LEV}.${EXP}.${STEP}_Lindborg_aircraft_rot.div.dat

done
done

#/home/rd/naw/own/curvplot.nils
#ghostview curv.ps
#\rm targ1 
exit 0 
