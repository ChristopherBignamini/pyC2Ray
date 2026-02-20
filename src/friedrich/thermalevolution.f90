! This module contains routines having to do with the calculation of the thermal evolution of a single point/cell. 

module thermalevolution

  !use precision, only: real64
  use, intrinsic :: iso_fortran_env, only: real64
  
  !> Thermal: minimum temperature [K]
  real(kind=real64),parameter :: minitemp=1.0
  !> Thermal: fraction of the cooling time step below which no iteration is done
  real(kind=real64),parameter :: relative_denergy=0.1
  !> adiabatic index
  real(kind=real64),public,parameter :: gamma = 5.0/3.0

  !use cosmology
  !use radiation, only: photrates
  !use radiation_photoionrates, only: photrates
  !use material, only: ionstates

  implicit none

contains
    ! TODO: this function below that loop over the voxels is not necessary. As the thermal subroutine will be directly callded inside the doric subroutine. Kept it for testing
    ! thermal evolution of the box
    subroutine thermal_evolve(dt,ndens,temp,xh,phi_ion,abu_c,m1,m2,m3)
        ! Subroutine Arguments
        real(kind=real64), intent(in) :: dt                          ! time step
        real(kind=real64), intent(in) :: temp(m1,m2,m3)             ! Temperature field
        real(kind=real64), intent(in) :: ndens(m1,m2,m3)            ! Hydrogen Density Field
        real(kind=real64), intent(inout) :: xh(m1,m2,m3)             ! HI ionization fractions of the cells
        real(kind=real64), intent(in) :: phi_ion(m1,m2,m3)           ! H Photo-ionization rate for the whole grid (called phih_grid in original c2ray)
        real(kind=real64), intent(in) :: bh00                        ! Hydrogen recombination parameter (value at 10^4 K)
        real(kind=real64), intent(in) :: albpow                      ! Hydrogen recombination parameter (power law index)
        real(kind=real64), intent(in) :: colh0                       ! Hydrogen collisional ionization parameter
        real(kind=real64), intent(in) :: temph0                      ! Hydrogen ionization energy expressed in K
        real(kind=real64), intent(in) :: abu_c                       ! Carbon abundance
        integer, intent(in) :: m1                                   ! mesh size x (hidden by f2py)
        integer, intent(in) :: m2                                   ! mesh size y (hidden by f2py)
        integer, intent(in) :: m3                                   ! mesh size z (hidden by f2py)

        integer,intent(out) :: conv_flag

        integer :: i,j,k  ! mesh position
        ! Mesh position of the cell being treated
        integer,dimension(3) :: pos

        conv_flag = 0
        do k=1,m3
            do j=1,m2
                do i=1,m1
                    pos=(/ i,j,k /)
                    call thermal(dt,pos,ndens,temp,xh,xh_av,xh_intermed,phi_ion, &
                        clump,bh00,albpow,colh0,temph0,abu_c,conv_flag,m1,m2,m3)
                enddo
            enddo
        enddo

    end subroutine thermal_evolve


  ! TODO: this function read some tables
  !> Calculate the cooling rate
  function coolin(nucldens,eldens,xh,temp0)
    
    real(kind=real64),intent(out) :: coolin
    
    real(kind=real64),intent(in) :: nucldens !< number density
    real(kind=real64),intent(in) :: eldens !< electron density
    real(kind=real64),dimension(0:1),intent(in) :: xh !< H ionization fractions
    real(kind=real64),intent(in) :: temp0 !< temperature
    
    real(kind=real64) :: tpos, dtpos
    integer :: itpos,itpos1
    
    tpos=(log10(temp0)-mintemp)/dtemp+1.0d0
    itpos=min(temppoints-1,max(1,int(tpos)))
    dtpos=tpos-real(itpos)
    itpos1=min(temppoints,itpos+1)
    
    ! Cooling curve
    coolin=nucldens*eldens* &
         (cie_cool(itpos)+(cie_cool(itpos1)-cie_cool(itpos))*dtpos)
    
  end function coolin
  
  ! TODO: this function below could be a variable in python passed to the thermal_evolve subroutine. Using astropy for the dz/dt will also assure consistency with the cosmology in the C2Ray class.
  !> Calculates the cosmological adiabatic cooling
  function cosmo_cool (e_int,H0,Omega0,zred)
    real(kind=real64),intent(out) :: cosmo_cool

    real(kind=real64),intent(in) :: e_int
    real(kind=real64),intent(in) :: H0
    real(kind=real64),intent(in) :: Omega0
    real(kind=real64),intent(in) :: zred
    real(kind=real64) :: dzdt

    ! dz/dt (for flat LambdaCDM)
    dzdt=H0*(1.+zred)*sqrt(Omega0*(1.+zred)**3+1.-Omega0)

    !Cooling rate
    cosmo_cool=e_int*2.0/(1.0+zred)*dzdt

  end function cosmo_cool

  ! calculates the thermal evolution of one grid point
  subroutine thermal (dt,end_temper,avg_temper,ndens_electron_av,ndens_atom,ion,phi,heat)!,pos)

    ! The time step
    real(kind=real64), intent(in) :: dt
    ! end time temperature of the cell
    real(kind=real64), intent(inout) :: end_temper
    ! average temperature of the cell
    real(kind=real64), intent(out) :: avg_temper
    ! Electron density of the cell
    real(kind=real64), intent(in) :: ndens_electron_av
    ! electron density TODO: not sure why we have to calculate these two quantities     
    real(kind=real64) :: ndens_electron_old
    ! Number density of atoms of the cell
    real(kind=real64), intent(in) :: ndens_atom
    ! Photoionization rate
    real(kind=real64), intent(in) :: phi
    ! Heating rate
    real(kind=real64), intent(in) :: heat
    ! Hydrogen ionized fraction (HI) of the cell
    real(kind=real64), intent(in) :: ion_hi_old
    real(kind=real64), intent(in) :: ion_hi_av
    real(kind=real64), intent(in) :: ion_hi
    ! Helium first ionized fraction (HeII) of the cell
    real(kind=real64), intent(in) :: ion_heii_old
    real(kind=real64), intent(in) :: ion_heii_av
    real(kind=real64), intent(in) :: ion_heii
    ! Helium second ionized fraction (HeIII) of the cell
    real(kind=real64), intent(in) :: ion_heiii_old
    real(kind=real64), intent(in) :: ion_heiii_av
    real(kind=real64), intent(in) :: ion_heiii
    ! mesh position of cell 
    !integer, intent(in) :: pos
 
    ! initial temperature
    real(kind=real64) :: initial_temp
    ! timestep taken to solve the ODE
    real(kind=real64) :: dt_ODE
    ! timestep related to thermal timescale
    real(kind=real64) :: dt_thermal
    ! record the time elapsed
    real(kind=real64) :: cumulative_time
    ! internal energy of the cell
    real(kind=real64) :: internal_energy
    ! thermal timescale, used to calculate the thermal timestep
    real(kind=real64) :: thermal_timescale
    ! heating rate
    real(kind=real64) :: heating
    ! cooling rate
    real(kind=real64) :: cooling
    ! difference of heating and cooling rate
    real(kind=real64) :: thermal_rate
    ! cosmological cooling rate
    real(kind=real64) :: cosmo_cool_rate
    ! Counter of number of thermal timesteps taken
    integer :: i_heating
    
    ndens_electron_old = electrondens(ndens_atom,ion_hi_old,ion_heii_old,ion_heiii_old)

    ! heating rate
    heating = heat

    ! Find initial internal energy
    internal_energy = temper2pressr(end_temper,ndens_atom, ndens_electron_old)/(gamma-1.0)

    ! TODO: the variable cosmo_cool_rate can 
    ! Set the cosmological cooling rate
    if (cosmological) then
       ! Disabled for testing
       cosmo_cool_rate=cosmo_cool(internal_energy)
    else
       cosmo_cool_rate=0.0
    endif

    ! Thermal process is only done if the temperature of the cell is larger than the minimum temperature requirement
    if (end_temper > minitemp) then

       ! stores the time elapsed is done
       cumulative_time = 0.0 
   
       ! initialize the counter
       i_heating = 0

       ! initialize time averaged temperature
       avg_temper = 0.0 

       ! initial temperature
       initial_temp = end_temper  

       ! thermal process begins
       do
          ! update counter              
          i_heating = i_heating+1 
         
          ! update cooling rate from cooling tables
          cooling = coolin(ndens_atom,ndens_electron,ion_h_av,ion_he_av, &
               end_temper)+cosmo_cool_rate

          ! Find total energy change rate
          thermal_rate = max(1d-50,abs(cooling-heating))

          ! Calculate thermal time scale
          thermal_timescale = internal_energy/abs(thermal_rate)

          ! Calculate time step needed to limit energy change to a fraction relative_denergy
          dt_thermal = relative_denergy*thermal_timescale

          ! Time step to large, change it to dt_thermal. Make sure we do not integrate for longer than the total time step
          dt_ODE = min(dt_thermal,dt-cumulative_time)

          ! Find new internal energy density
          internal_energy = internal_energy+dt_ODE*(heating-cooling)

          ! Update avg_temper sum (first part of dt_thermal sub time step)
          avg_temper = avg_temper+0.5*end_temper*dt_ODE

          ! Find new temperature from the internal energy density
          end_temper = pressr2temper(internal_energy*(gamma-1.0),ndens_atom, &
               electrondens(ndens_atom,ion_h_av,ion_he_av))
                    
          ! Take measures if temperature drops below minitemp
          if (end_temper < minitemp) then
             internal_energy = temper2pressr(minitemp,ndens_atom, &
                  electrondens(ndens_atom,ion_h_av,ion_he_av))
             end_temper = minitemp
          endif

          ! Update avg_temper sum (second part of dt_thermal sub time step)
          avg_temper = avg_temper+0.5*end_temper*dt_ODE
                    
          ! Update fractional cumulative_time
          cumulative_time = cumulative_time+dt_ODE
  
          ! Exit if we reach dt
          if (cumulative_time >= dt.or.abs(cumulative_time-dt) < 1e-6*dt) exit

          ! In case we spend too much time here, we exit
          if (i_heating > 10000) exit
       
       enddo
              
       ! Calculate the averaged temperature
       if (dt > 0.0) then
          avg_temper = avg_temper/dt
       else
          avg_temper = initial_temp
       endif
       
       ! Calculate the final temperature 
       end_temper = pressr2temper(internal_energy*(gamma-1.0),ndens_atom, &
            electrondens(ndens_atom,ion_h,ion_he))
       
    endif
    
  end subroutine thermal
  
end module thermalevolution
