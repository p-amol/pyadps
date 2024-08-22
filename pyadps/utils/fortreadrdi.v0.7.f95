!File: readrdi.f95
!Version: 0.7
! 19/08/2024
!Corrupt file detection

subroutine fileheader(filename, datatype, byte, byteskip, address_offset, dataid, ensemble, error_code)

! FileHeader subroutine is for obtaining header information from an
! RDI ADCP File. 
! INPUT:
!       filename: RDI ADCP binary file. The subroutine can currently extract Workhorse, Ocean Surveyor, and DVS files. 
!
! OUTPUT
!       datatype: No. of data types for each ensemble
!       byte: number of bytes for each ensembles
!       byteskip: number of bytes to skip to reach and ensemble
!       address_offset: internal memory address offset for each data type
!       ensemble: Total ensembles in the file
!       error_code: Diplays error code if the file is corrupted
!
! Error code
! 0 = Healthy
! 1 = End of file
! 2 = File corrupted (ID's not recognized)
! 3 = Wrong file type
! 4 = Data type mismatch
! 9 = Unknown error

   IMPLICIT NONE

   integer, parameter :: ensmax = 65536
   character(len=100), intent(in) :: filename
   integer(1), dimension(ensmax), intent(out) :: datatype
   integer, intent(out) :: ensemble     
   integer(1), dimension(ensmax) :: header, source, spare
   integer(2), dimension(ensmax), intent(out) :: byte
   integer(2), dimension(ensmax, 20), intent(out) :: address_offset, dataid
   integer, dimension(ensmax), intent(out) :: byteskip
   integer :: i, j, k, n, bskip
   integer :: ioerr
   integer, intent(out) :: error_code

   open(5, file=filename, status='old', form='unformatted', access='stream')

   i = 1
   n = 1
   bskip = 1
   error_code = 0
   ioerr = 0

   do
        ! Read and check header & Source id
        read(5, pos=n, END=200) header(i), source(i)
        n = n + 2
        if (i == 1) then
           if (header(i).ne.127.or.source(i).ne.127) then
              write(*,*) "ERROR: File type not recognized. Enter a valid RDI file."
              error_code = 3
              go to 200
           endif
        else
          if (header(i).ne.127.or.source(i).ne.127) then
              write(*,*) "WARNING: The file appears corrupted. Ensemble no. ", i, " does not have a valid RDI file ID"
              error_code = 2 
              go to 200
          endif
        endif     

        ! Read byte, spare, datatype
        read(5,pos=n, END=900, IOSTAT=ioerr) byte(i)
        n = n + 2
        read(5, pos=n, END=900, IOSTAT=ioerr) spare(i), datatype(i)
        ! Pick address offset for each data type
        do j = 1, datatype(i)        
           n = n + 2
           read(5, pos=n, END=900, IOSTAT=ioerr) address_offset(i, j)
           k = bskip + address_offset(i, j)
           read(5, pos=k, END=900,  IOSTAT=ioerr) dataid(i, j)

        enddo
        bskip = bskip + byte(i) + 2        
        byteskip(i) = bskip 
        n = bskip
        i = i + 1
   enddo     

900 if (ioerr > 0) then
      write(*, *) "ERROR: Unknown (FileHeader)"
      error_code = 9
    else if (ioerr < 0) then
      write(*, *) "WARNING: End of File (FileHeader)"
      error_code = 1
    else
      error_code = 0
    endif
   
200   close(5)   
   ensemble = i-1
   return
end subroutine


subroutine fixedleader(filename, output, ensemble, error_code)
   IMPLICIT NONE
   integer, parameter :: ensmax = 65536     
   character(len=100), intent(in) :: filename
   integer(1), dimension(ensmax) :: datatype
   integer(2), dimension(ensmax) :: byte
   integer(2), dimension(ensmax, 20) :: offset, dataid
   integer, dimension(ensmax) :: byteskip
   integer, intent(out) :: ensemble     
   ! Maximum allowed variables in fixed leader is 36 
   integer(8), dimension(ensmax, 36), intent(out) :: output
 
   integer(1) :: onebyte
   integer(2) :: twobyte
   integer(4) :: fourbyte
   integer(8) :: eightbyte
   integer :: i, n
   integer, intent(out) :: error_code
   integer :: ioerr
  
   call fileheader(filename, datatype, byte, byteskip, offset, dataid, ensemble, error_code)
   open(5, file=filename, status='old', form='unformatted', access='stream')
  
   do i = 1, ensemble, 1
        if (i.eq.1) then
           n = offset(i, 1) + 1
        else
           n = offset(i, 1) + byteskip(i-1)
        endif
        ! 1. Fixed Leader ID
        read(5, pos=n) twobyte
        if ((twobyte.ne.0).and.(twobyte.ne.1)) then
           write(*, *) "Fixed Leader ID not found for Ensemble " , i    
           error_code = 2 
           go to 200
        endif
        output(i, 1) = twobyte; n = n + 2
        ! 2. CPU firmware version
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 2) = onebyte; n = n + 1
        ! 3. CPU firmware revision
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte  
        output(i, 3) = onebyte; n = n + 1
        ! 4. System Configuration
        read(5, pos=n, END=900, IOSTAT=ioerr) twobyte  
        output(i, 4) = twobyte; n = n + 2
        ! 5. Flag
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte  
        output(i, 5) = onebyte; n = n + 1
        ! 6. Lag length
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte  
        output(i, 6) = onebyte; n = n + 1
        ! 7. Number of beams
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte  
        output(i, 7) = onebyte; n = n + 1
        ! 8. WN/Number of cells
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte  
        output(i, 8) = onebyte; n = n + 1
        ! 9. WP/Pings per ensemble
        read(5, pos=n, END=900, IOSTAT=ioerr) twobyte  
        output(i, 9) = twobyte; n = n + 2
        ! 10. WS/Depth Cell length
        read(5, pos=n, END=900, IOSTAT=ioerr) twobyte  
        output(i, 10) = twobyte; n = n + 2
        ! 11. WF/Blank after transmit
        read(5, pos=n, END=900, IOSTAT=ioerr) twobyte  
        output(i, 11) = twobyte; n = n + 2
        ! 12. Signal processing mode
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte  
        output(i, 12) = onebyte; n = n + 1
        ! 13. WC/Low correlation threshold
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte  
        output(i, 13) = onebyte; n = n + 1
        ! 14. No. of code repetition in transmit pulse
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte  
        output(i, 14) = onebyte; n = n + 1
        ! 15. WG/Percent good minimum
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte  
        output(i, 15) = onebyte; n = n + 1
        ! 16. WE/Error velocity threshold
        read(5, pos=n, END=900, IOSTAT=ioerr) twobyte  
        output(i, 16) = twobyte; n = n + 2
        ! 17. Minutes 
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte  
        output(i, 17) = onebyte; n = n + 1
        ! 18. Seconds
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte  
        output(i, 18) = onebyte; n = n + 1
        ! 19. Hundredths
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte  
        output(i, 19) = onebyte; n = n + 1
        ! 20. EX/Coord transform
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte  
        output(i, 20) = onebyte; n = n + 1
        ! 21. EA/Heading alignment
        read(5, pos=n, END=900, IOSTAT=ioerr) twobyte  
        output(i, 21) = twobyte; n = n + 2
        ! 22. EB/Heading bias
        read(5, pos=n, END=900, IOSTAT=ioerr) twobyte  
        output(i, 22) = twobyte; n = n + 2
        ! 23. EZ/Sensor source
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte  
        output(i, 23) = onebyte; n = n + 1
        ! 24. Sensor available
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte  
        output(i, 24) = onebyte; n = n + 1
        ! 25. Bin 1 distance
        read(5, pos=n, END=900, IOSTAT=ioerr) twobyte  
        output(i, 25) = twobyte; n = n + 2
        ! 26. Xmit pulse length
        read(5, pos=n, END=900, IOSTAT=ioerr) twobyte  
        output(i, 26) = twobyte; n = n + 2
        ! 27. WL/ Reference layer
        read(5, pos=n, END=900, IOSTAT=ioerr) twobyte  
        output(i, 27) = twobyte; n = n + 2
        ! 28. False target threshold
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte  
        output(i, 28) = onebyte; n = n + 1
        ! 29. Spare
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte  
        output(i, 29) = onebyte; n = n + 1
        ! 30. Transmit lag distance
        read(5, pos=n, END=900, IOSTAT=ioerr) twobyte  
        output(i, 30) = twobyte; n = n + 2
        ! 31. CPU board serial no.
        read(5, pos=n, END=900, IOSTAT=ioerr) eightbyte  
        output(i, 31) = eightbyte; n = n + 8
        ! 32. WB/ System bandwidth
        read(5, pos=n, END=900, IOSTAT=ioerr) twobyte  
        output(i, 32) = twobyte; n = n + 2
        ! 33. System power
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte  
        output(i, 33) = onebyte; n = n + 1
        ! 34. Spare
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte  
        output(i, 34) = onebyte; n = n + 1
        ! 35. Serial number
        read(5, pos=n, END=900, IOSTAT=ioerr) fourbyte  
        output(i, 35) = fourbyte; n = n + 4
        ! 36. Beam angle
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte  
        output(i, 36) = onebyte; n = n + 1
   enddo           

900 if (ioerr > 0) then
      write(*, *) "ERROR: Unknown (Fixed Leader)"
      error_code = 9
    else if (ioerr < 0) then
      write(*, *) "WARNING: End of File (Fixed Leader)"
      error_code = 1
    else
      error_code = 0
    endif
200 close(5)
   return     
end subroutine

subroutine variableleader(filename, output, ensemble, error_code)
   IMPLICIT NONE
   integer, parameter :: ensmax = 65536
   character(len=100), intent(in) :: filename
   integer(1), dimension(ensmax) :: datatype
   integer(2), dimension(ensmax) :: byte
   integer(2), dimension(ensmax, 20) :: offset, dataid
   integer, dimension(ensmax) :: byteskip
   integer, intent(out) :: ensemble
   ! Maximum allowed variables in variable leader is 48 
   integer, dimension(ensmax, 48), intent(out) :: output
   integer, intent(out) :: error_code
   integer :: ioerr

   integer(1) :: onebyte
   integer(2) :: twobyte
   integer(4) :: fourbyte
   integer :: i, n

   call fileheader(filename, datatype, byte, byteskip, offset, dataid, ensemble, error_code)
   open(5, file=filename, status='old', form='unformatted', access='stream')

   do i = 1, ensemble, 1
        if (i.eq.1) then
           n = offset(i, 2) + 1
        else
           n = offset(i, 2) + byteskip(i-1)
        endif
        ! 1. Variable Leader ID
        read(5, pos=n, END=900, IOSTAT=ioerr) twobyte
        if ((twobyte.ne.128).and.(twobyte.ne.129)) then
           write(*, *) "WARNING: Variable Leader ID not found for Ensemble " , i
           error_code = 2 
           go to 200
        endif        
        output(i, 1) = twobyte; n = n + 2
        ! 2. Ensemble no.
        read(5, pos=n, END=900, IOSTAT=ioerr) twobyte
        output(i, 2) = twobyte; n = n + 2
        ! 3. RTC Year
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 3) = onebyte; n = n + 1
        ! 4. RTC Month
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 4) = onebyte; n = n + 1
        ! 5. RTC Day
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 5) = onebyte; n = n + 1
        ! 6. RTC Hour
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 6) = onebyte; n = n + 1
        ! 7. RTC Minute
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 7) = onebyte; n = n + 1
        ! 8. RTC Second
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 8) = onebyte; n = n + 1
        ! 9. RTC Hundredth
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 9) = onebyte; n = n + 1
        ! 10. Ensemble increment
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 10) = onebyte; n = n + 1
        ! 11. Bit result
        read(5, pos=n, END=900, IOSTAT=ioerr) twobyte
        output(i, 11) = twobyte; n = n + 2
        ! 12. EC/Speed of sound
        read(5, pos=n, END=900, IOSTAT=ioerr) twobyte
        output(i, 12) = twobyte; n = n + 2
        ! 13. ED/Depth of transducer
        read(5, pos=n, END=900, IOSTAT=ioerr) twobyte
        output(i, 13) = twobyte; n = n + 2
        ! 14. EH/Heading
        read(5, pos=n, END=900, IOSTAT=ioerr) twobyte
        output(i, 14) = twobyte; n = n + 2
        ! 15. EP/Pitch
        read(5, pos=n, END=900, IOSTAT=ioerr) twobyte
        output(i, 15) = twobyte; n = n + 2
        ! 16. ER/Roll
        read(5, pos=n, END=900, IOSTAT=ioerr) twobyte
        output(i, 16) = twobyte; n = n + 2
        ! 17. ES/Salinity
        read(5, pos=n, END=900, IOSTAT=ioerr) twobyte
        output(i, 17) = twobyte; n = n + 2
        ! 18. ET/Temperature
        read(5, pos=n, END=900, IOSTAT=ioerr) twobyte
        output(i, 18) = twobyte; n = n + 2
        ! 19. MPT Minute
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 19) = onebyte; n = n + 1
        ! 20. MPT Second
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 20) = onebyte; n = n + 1
        ! 21. MPT Hundredth
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 21) = onebyte; n = n + 1
        ! 22. Heading std dev
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 22) = onebyte; n = n + 1
        ! 23. Pitch std dev
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 23) = onebyte; n = n + 1
        ! 24. Roll std dev
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 24) = onebyte; n = n + 1
        ! 25. ADC Channel 0
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 25) = onebyte; n = n + 1
        ! 26. ADC Channel 1
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 26) = onebyte; n = n + 1
        ! 27. ADC Channel 2
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 27) = onebyte; n = n + 1
        ! 28. ADC Channel 3
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 28) = onebyte; n = n + 1
        ! 29. ADC Channel 4
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 29) = onebyte; n = n + 1
        ! 30. ADC Channel 5
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 30) = onebyte; n = n + 1
        ! 31. ADC Channel 6
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 31) = onebyte; n = n + 1
        ! 32. ADC Channel 7
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 32) = onebyte; n = n + 1
        ! 33. Error status word 1
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 33) = onebyte; n = n + 1
        ! 34. Error status word 2
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 34) = onebyte; n = n + 1
        ! 35. Error status word 3
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 35) = onebyte; n = n + 1
        ! 36. Error status word 4
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 36) = onebyte; n = n + 1
        ! 37. Reserve
        read(5, pos=n, END=900, IOSTAT=ioerr) twobyte
        output(i, 37) = twobyte; n = n + 2
        ! 38. Pressure
        read(5, pos=n, END=900, IOSTAT=ioerr) fourbyte
        output(i, 38) = fourbyte; n = n + 4
        ! 39. Pressure variance
        read(5, pos=n, END=900, IOSTAT=ioerr) fourbyte
        output(i, 39) = fourbyte; n = n + 4
        ! 40. Spare
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 40) = onebyte; n = n + 1
        ! 41. RTC Century
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 41) = onebyte; n = n + 1
        ! 42. RTC Year
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 42) = onebyte; n = n + 1
        ! 43. RTC Month
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 43) = onebyte; n = n + 1
        ! 44. RTC Day
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 44) = onebyte; n = n + 1
        ! 45. RTC Hour
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 45) = onebyte; n = n + 1
        ! 46. RTC Minute
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 46) = onebyte; n = n + 1
        ! 47. RTC Second
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 47) = onebyte; n = n + 1
        ! 48. RTC Hundredth
        read(5, pos=n, END=900, IOSTAT=ioerr) onebyte
        output(i, 48) = onebyte; n = n + 1
   enddo

900 if (ioerr > 0) then
      write(*, *) "ERROR: Unknown (Variable Leader)"
      error_code = 9
    else if (ioerr < 0) then
      write(*, *) "WARNING: End of File (Variable Leader)"
      error_code = 1
    else
      error_code = 0
    endif
200 close(5)     
   return     
end subroutine


subroutine velocity(filename, output, ensemble, cell, beam, error_code)
   IMPLICIT NONE
   integer, parameter :: ensmax = 65536
   integer, parameter :: cellmax = 200
   integer, parameter :: beammax = 4
   character(len=100), intent(in) :: filename
   integer(1), dimension(ensmax) :: datatype
   integer(2), dimension(ensmax) :: byte
   integer(2), dimension(ensmax, 20) :: offset, dataid
   integer, dimension(ensmax) :: byteskip
   integer, intent(out) :: ensemble
   ! Maximum allowed variables in variable leader is 48 
   integer(8), dimension(ensmax, 36) :: flout
   integer, dimension(beammax, cellmax, ensmax), intent(out) :: output
   integer(8), intent(out) :: cell, beam
   integer, intent(out) :: error_code
   integer :: ioerr

   integer(2) :: twobyte
   integer :: i, j, k, n

   ! Get data from File Header
   call fileheader(filename, datatype, byte, byteskip, offset, dataid, ensemble, error_code)
   ! Get data from Fixed Leader
   call fixedleader(filename, flout, ensemble, error_code)  

   cell = abs(maxval(flout(:,8)))
   beam = abs(maxval(flout(:,7)))

   open(5, file=filename, status='old', form='unformatted', access='stream')     
   do i = 1,ensemble,1
      if (datatype(i).lt.3) then
          write(*, *) "WARNING: data type not found for velocity"
          write(*, *) "Ensemble no: ", i
          write(*, *) "Available data types for the ensemble:", datatype(i)
          write(*, *) "Input data type index: 3"
          ensemble = i - 1
          error_code = 4
          go to 200 
      endif
      if (i.eq.1) then
         n = offset(i, 3) + 1
      else
         n = offset(i, 3) + byteskip(i-1)
      endif
      ! Velocity ID
      read(5, pos=n, END=900, IOSTAT=ioerr) twobyte; n = n + 2
      if ((twobyte.ne.256).and.(twobyte.ne.257)) then
         write(*, *) "WARNING: Velocity ID not found for Ensemble " , i
         ensemble = i - 1
         error_code = 2
         go to 200
      endif
      do j = 1,int(cell),1
        do k = 1,int(beam),1
           read(5, pos=n, END=900, IOSTAT=ioerr) twobyte; n = n+2 
           output(k, j, i)  = twobyte      
        enddo
      enddo
   enddo

900 if (ioerr > 0) then
      write(*, *) "ERROR: Unknown (Velocity)"
      error_code = 9
    else if (ioerr < 0) then
      write(*, *) "WARNING: End of File (Velocity)"
      error_code = 1
    else
      error_code = 0
    endif
200 close(5)
   return
end subroutine


subroutine othervar(filename, varname, output, ensemble, cell, beam, error_code)
   IMPLICIT NONE
   integer, parameter :: ensmax = 65536
   integer, parameter :: cellmax = 200
   integer, parameter :: beammax = 4

   character(len=100), intent(in) :: filename
   character(len=100), intent(in) :: varname
   integer, dimension(beammax, cellmax, ensmax), intent(out) :: output
   integer, intent(out) :: ensemble
   integer(8), intent(out) :: cell, beam

   integer(1), dimension(ensmax) :: datatype
   integer(2), dimension(ensmax) :: byte
   integer(2), dimension(ensmax, 20) :: offset, dataid
   integer, dimension(ensmax) :: byteskip
   ! Maximum allowed variables in fixed leader is 36 
   integer(8), dimension(ensmax, 36) :: flout
  
   integer :: vid1, vid2, ind, onebyteint
   integer(1) :: onebyte
   integer(2) :: twobyte
   integer :: i, j, k, n
   integer, intent(out) :: error_code
   integer :: ioerr

   select case(varname)
        case ('correlation')
            vid1 = 512
            vid2 = 513
            ind  = 4
        case ('echo')
            vid1 = 768
            vid2 = 769    
            ind  = 5
        case ('pgood')
            vid1 = 1024
            vid2 = 1025   
            ind  = 6
        case ('status')
            vid1 = 1024
            vid2 = 1025   
            ind  = 7
        case DEFAULT
            write(*, *) 'ERROR: Invalid variable name'
            write(*, *) 'Available options: correlation, echo, pgood, status'
            stop
   end select    
     

   ! Get data from File Header
   call fileheader(filename, datatype, byte, byteskip, offset, dataid, ensemble, error_code)
   ! Get data from Fixed Leader
   call fixedleader(filename, flout, ensemble, error_code)  

   cell = abs(maxval(flout(:,8)))
   beam = abs(maxval(flout(:,7)))

   open(5, file=filename, status='old', form='unformatted', access='stream')     
   do i = 1,ensemble,1
      ! Check if data type is available
      if (ind.gt.datatype(i)) then
        if (i.eq.1) then
          write(*, *) "INPUT ERROR: Data type not found for ", varname 
          stop
        else
          write(*, *) "WARNING: Data type not found for ", varname 
          write(*, *) "Ensemble no: ", i
          write(*, *) "Available data types for the ensemble:", datatype(i)
          write(*, *) "Input data type index: ", ind
          ensemble = i - 1
          error_code = 4
          go to 200 
        endif
      endif
      if (i.eq.1) then
         n = offset(i, ind) + 1
      else
         n = offset(i, ind) + byteskip(i-1)
      endif
      read(5, pos=n, END=900, IOSTAT=ioerr) twobyte; n = n + 2
      if ((twobyte.ne.vid1).and.(twobyte.ne.vid2)) then
         write(*, *) "WARNING: Data ID not found for Ensemble " , i
         ensemble = i - 1
         error_code = 2
         go to 200
      endif
      do j = 1,int(cell),1
        do k = 1,int(beam),1
           read(5, pos=n, END=900, IOSTAT=ioerr) onebyte; n = n+1
           ! Convert signed int to unsigned int
           if (onebyte < 0) then
             onebyteint = int(onebyte) + 256
           else 
             onebyteint = onebyte
           endif
           output(k, j, i)  = onebyteint
        enddo
      enddo
   enddo

900 if (ioerr > 0) then
      write(*, *) "ERROR: Unknown (Data Type)"
      error_code = 9
    else if (ioerr < 0) then
      write(*, *) "WARNING: End of File (Data Type)"
      error_code = 1
    else
      error_code = 0
    endif
200   close(5)
   return
end subroutine

subroutine variables(filename, varname, output, ensemble, cell, beam, error_code)
   IMPLICIT NONE
   integer, parameter :: ensmax = 65536
   integer, parameter :: cellmax = 200
   integer, parameter :: beammax = 4
   character(len=15) :: namearray(4) = [character(len=15) :: 'correlation', 'echo', 'pgood', 'status']

   character(len=100), intent(in) :: filename
   character(len=100), intent(in) :: varname
   integer, dimension(beammax, cellmax, ensmax), intent(out) :: output
   integer, intent(out) :: ensemble
   integer(8), intent(out) :: cell, beam
   integer, intent(out) :: error_code


   if (varname == 'velocity') then
        call velocity(filename, output, ensemble, cell, beam, error_code)
   elseif (any(namearray == varname)) then
        call othervar(filename, varname, output, ensemble, cell, beam, error_code)
   else
        write(*, *) 'ERROR: Invalid variable name'
        write(*, *) 'Available options: velocity, correlation, echo, pgood, status'
        stop
   endif
end subroutine
