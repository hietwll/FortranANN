! @Date    : 2018-04-11 16:22:52
! @Author  : etwll (hietwll@gmail.com)
MODULE ANN
	IMPLICIT NONE
	integer :: nlayer
	integer , allocatable :: layers(:)
	real*8, allocatable :: InputMean(:),InputVar(:),OutputMean(:),OutPutVar(:)
	character(len=200) :: basename

	TYPE :: net
	type(net),pointer :: next,pre
	real*8, allocatable :: weights(:,:)
	real*8, allocatable :: biases(:)
	real*8, allocatable :: neuro(:,:)
	END TYPE net
	
	type(net) , pointer :: phead,ptail,pcur
END MODULE ANN

!==================================================
 SUBROUTINE InitANN
	USE ANN
	IMPLICIT NONE
	integer i,j,k,status
	character* 200 :: wname,bname,sname
	character* 1 :: istr
	character :: a

	open(101,file='ANN.prm',status='unknown',form='formatted')
	read (101,*) a !
	read (101,*) a ! ANN parameters
	read (101,*) a !
	read (101,*) nlayer
	allocate(layers(nlayer))
	read (101,*) layers(:)
	read (101,"(A100)") basename
	close(101)

	! Read ANN Weights and Biases
        phead => NULL()
        ptail => NULL()
        pcur => NULL()

	DO i=1,nlayer-1
                allocate(pcur, STAT = status)
                if (status > 0) STOP 'Fail to allocate a new node'
		allocate(pcur%weights(layers(i),layers(i+1)))
		allocate(pcur%biases(layers(i+1)))
		write(istr,"(i1.1)") i 
		wname = trim(basename) // '/weights'//istr//'.dat'
		bname = trim(basename) // '/biases'//istr//'.dat'

		OPEN(102,file=wname,status='unknown',form='formatted')
		DO j = 1, layers(i)
			READ(102,*) pcur%weights(j,:)
		ENDDO
		close(102)

		OPEN(103,file=bname,status='unknown',form='formatted')
		DO k = 1, layers(i+1)
			READ(103,*) pcur%biases(k)
		ENDDO
		close(103)

		pcur%next => NULL() 

		if (.NOT. ASSOCIATED(phead))then
			phead=>pcur
			phead%pre => NULL()
		else
			pcur%pre => ptail
			ptail%next=>pcur
		endif

		ptail=>pcur
	ENDDO

	! Read Scalers
	allocate(InputMean(layers(1)))
	allocate(InputVar(layers(1)))
	allocate(OutputMean(layers(nlayer)))
	allocate(OutputVar(layers(nlayer)))

	sname =  trim(basename) // '/scaler_input_mean.dat'
	OPEN(104,file=sname,status='unknown',form='formatted')
	DO i = 1, layers(1)
		READ(104,*) InputMean(i)
	ENDDO
	close(104)

	sname =  trim(basename) // '/scaler_input_var.dat'
	OPEN(104,file=sname,status='unknown',form='formatted')
	DO i = 1, layers(1)
		READ(104,*) InputVar(i)
	ENDDO
	close(104)

	sname =  trim(basename) // '/scaler_output_mean.dat'
	OPEN(104,file=sname,status='unknown',form='formatted')
	DO i = 1, layers(nlayer)
		READ(104,*) OutputMean(i)
	ENDDO
	close(104)

	sname =  trim(basename) // '/scaler_output_var.dat'
	OPEN(104,file=sname,status='unknown',form='formatted')
	DO i = 1, layers(nlayer)
		READ(104,*) OutputVar(i)
	ENDDO
	close(104)

	RETURN
 END SUBROUTINE InitANN
!==================================================

!==================================================
 SUBROUTINE NormInput(XNorm,XInput,row,column)
	USE ANN
	IMPLICIT NONE
	integer :: row,column,i
	real*8 :: XNorm(row,column),XInput(row,column)

	XNorm = 0.0
	DO i = 1,row
		XNorm(i,:) = (XInput(i,:) - InputMean) / InputVar**0.5
	ENDDO

	RETURN
 END SUBROUTINE NormInput
!==================================================
!==================================================
 SUBROUTINE InverOutput(YInver,YInput,row,column)
	USE ANN
	IMPLICIT NONE
	integer :: row,column,i,j
	real*8 :: YInver(row,column),YInput(row,column)

	YInver = 0.0
	DO i = 1,row
	DO j = 1,column  
		YInver(i,j) = YInput(i,j) * OutputVar(j)**0.5 + OutputMean(j)
	ENDDO
	ENDDO

	RETURN
 END SUBROUTINE InverOutput
!==================================================
!==================================================
 SUBROUTINE TestScaler
	USE ANN
	IMPLICIT NONE
	real*8 :: testx(4,12),testx_norm(4,12),testx_norm_read(4,12)
        real*8 :: testy(4,1), testy_inver(4,1),testy_inver_read(4,1)
        integer :: i

	OPEN(105,file='../TestScaler/testx.dat',status='unknown',form='formatted')
	DO i = 1, 4
		READ(105,*) testx(i,:)
	ENDDO
	close(105)

	OPEN(105,file='../TestScaler/testx_norm.dat',status='unknown',form='formatted')
	DO i = 1, 4
		READ(105,*) testx_norm_read(i,:)
	ENDDO
	close(105)

	OPEN(105,file='../TestScaler/testy.dat',status='unknown',form='formatted')
	DO i = 1, 4
		READ(105,*) testy(i,:)
	ENDDO
	close(105)

	OPEN(105,file='../TestScaler/testy_inver.dat',status='unknown',form='formatted')
	DO i = 1, 4
		READ(105,*) testy_inver_read(i,:)
	ENDDO
	close(105)

	CALL NormInput(testx_norm,testx,4,12)
	CALL InverOutput(testy_inver,testy,4,1)

	write(*,*) testx_norm - testx_norm_read
	write(*,*) testy_inver - testy_inver_read

	RETURN
 END SUBROUTINE TestScaler
!==================================================

!==================================================
 SUBROUTINE Inference(XIn,YOut,Ndata)
	USE ANN
	IMPLICIT NONE
	integer :: Ndata,i,sflag
	real*8 :: Xin(Ndata,layers(1)),YOut(Ndata,layers(nlayer))
	type(net) , pointer :: pzero

	allocate(pzero, STAT = sflag)
	if (sflag > 0) STOP 'Fail to allocate a new node'
	allocate(pzero%neuro(Ndata,layers(1)))
	pzero%pre => NULL()
	pzero%next => NULL()


	call NormInput(pzero%neuro,Xin,Ndata,layers(1))


	phead%pre => pzero
	
	IF (ASSOCIATED(phead)) THEN
		pcur => phead
	ENDIF



	DO i = 1,nlayer-1
		allocate(pcur%neuro(Ndata,layers(i+1)))
		pcur => pcur%next
	ENDDO


	IF (ASSOCIATED(phead)) THEN
		pcur => phead
	ENDIF

	DO i = 1,nlayer-1
		pcur%neuro = matmul(pcur%pre%neuro,pcur%weights)
		
		call AddBia(pcur%neuro,pcur%biases,Ndata,layers(i+1))

		IF (i.lt.nlayer-1) THEN
			call ReLu(pcur%neuro,Ndata,layers(i+1))
			pcur => pcur%next
		ENDIF
	ENDDO

	call InverOutput(YOut,pcur%neuro,Ndata,layers(nlayer))

	DEALLOCATE(pzero%neuro)
	DEALLOCATE(pzero)


	RETURN
 END SUBROUTINE Inference
!==================================================

!==================================================
 Subroutine ReLu(RInput,rows,cols)
	IMPLICIT NONE
	integer rows,cols,i,j
	real*8 RInput(rows,cols)

	DO i = 1,rows
	DO j = 1,cols
		IF (RInput(i,j).le.0.0) THEN
			RInput(i,j) = 0.0
		ENDIF
	ENDDO
	ENDDO
	RETURN
 END Subroutine ReLu
!==================================================

!==================================================
 Subroutine AddBia(AInput,Bia,rw,clm)
	IMPLICIT NONE
	integer rw,clm,i
	real*8 AInput(rw,clm),Bia(clm)

	DO i=1,rw
		AInput(i,:) = AInput(i,:) + Bia
	ENDDO
	
	RETURN
 END Subroutine AddBia
!==================================================

!==================================================
 SUBROUTINE TestANN
        USE ANN
	IMPLICIT NONE
	integer,parameter :: patch = 8
        real*8 :: XInput(patch,12),YOut(patch,1),YOut_TF(patch,1)
        integer i,j

	OPEN(105,file='../TestTensor/xinput_tf.dat',status='unknown',form='formatted')
	DO i = 1, patch
		READ(105,*) XInput(i,:)
	ENDDO
	close(105)

	OPEN(105,file='../TestTensor/yinver_tf.dat',status='unknown',form='formatted')
	DO i = 1, patch
		READ(105,*) YOut_TF(i,:)
	ENDDO
	close(105)


        call Inference(XInput,YOut,patch)

        write(*,*) YOut-YOut_TF


	RETURN
 END SUBROUTINE TestANN
!==================================================

program main
use ANN
implicit none

call InitANN
call TestANN


end program main