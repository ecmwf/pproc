#!/usr/bin/env python

import sys

from eccodes import GribMessage
from eccodes import GribFile

def main():
    if len(sys.argv) == 1: # no arguments, so print help message
        print('Usage: good luck!')
        return

    for f in sys.argv[1:]:
        with GribFile(f) as gribFile:
            for i in range(len(gribFile)):
                msg = GribMessage(gribFile)
                print(msg["pentagonalResolutionParameterJ"], len(msg["values"]))


if __name__ == "__main__":
    sys.exit(main())


scalar = """
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

      zra=${RDS}*1000.

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
          if(jn.ne.0) then
            zfact=zra**2/(jn*(jn+1))
          else
            zfact=1.
          endif

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
     &    log10(znorm1(jn)),&
     &    log10(znorm1(jn)),&
     &    log10(znorm1(jn)*float(jn)**1.6666)
      endif
 213  continue
 10   continue

      DEALLOCATE(znorm1)
      DEALLOCATE(zw)

      DEALLOCATE(ZT)

      end
"""


vod = """
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

      CALL GRIB_GET_SIZE(IGRIB_H,'values',ISIZE)
      IF(.NOT.ALLOCATED(ZDIV)) ALLOCATE(ZDIV(ISIZE))

      CALL GRIB_GET(IGRIB_H,'paramId',iparam)
      CALL GRIB_GET(IGRIB_H,'values',ZDIV)
      print *, 'FIELD: ',iparam
      CALL GRIB_GET(IGRIB_H,'pentagonalResolutionParameterJ',NSMAX)
      print *, 'NSMAX: ',NSMAX

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
     &    znorm1(jn),&  ! log10(znorm1(jn))
     &    znorm2(jn),&  ! log10(znorm2(jn))
     &    znorm3(jn)    ! log10(znorm3(jn)*float(jn)**1.6666)
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
"""


