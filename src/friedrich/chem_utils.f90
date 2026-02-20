! This module contains utilities routines for the calculation of the thermal and chemistry evolution.

module chem_utils

  !use precision, only: real64
  use, intrinsic :: iso_fortran_env, only: real64
  
  ! Boltzmann constant
  real(kind=real64),parameter :: k_B=1.381d-16     ! value from astropy==6.0.0

  ! Calculate the electron density
  elemental function electrondens(ndens,xhi,xheii,xheiii,abu_c)
  
    real(kind=real64),intent(in) :: ndens           ! gas number density
    real(kind=real64),intent(in) :: xhi             ! HI ionization fractions
    real(kind=real64),intent(in) :: xheii           ! HeI ionization fractions
    real(kind=real64),intent(in) :: xheiii          ! HeII ionization fractions
    real(kind=real64),intent(in) :: abu_c           ! Carbon abundance

    real(kind=real64),intent(out) :: electrondens   ! electron number density

    electrondens=ndens*(xhi*(1.0-abu_he)+abu_he*(xheii+2.0*xheiii)+abu_c)

  end function electrondens

  ! Calculate pressure from temperature
  elemental function pressr2temper(pressr,ndens,eldens)

    real(kind=real64),intent(in) :: pressr ! pressure
    real(kind=real64),intent(in) :: ndens ! gas number density
    real(kind=real64),intent(in) :: eldens ! electron density  

    real(kind=real64),intent(out) :: temper   ! electron number density

    temper=pressr/(k_B*(ndens+eldens))
        
  end function pressr2temper

end module chem_utils
