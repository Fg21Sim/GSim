# Copyright (c) 2020-2022 Chenxi SHAN <cxshan@hey.com>
# Galactic Simulation Accommodating Tools
__version__ = "0.1.0"
__date__    = "2022-06-15"

# Basic modules
import numpy as np
import math

# PKGs related to healpix & SHT
import healpy as hp
import ducc0 as dc
    
# =================== Unit conversion ===================

def pixsize( nside ):
    r"""
        *** nside >> pixel size in rad *** # Basic function
        !!! Inverse of rad2nside() !!! # Important note
        !!! For healpix usage !!! # Important note
        +++  +++ # Improvements
        :Params nside: nside of the map
        :Output pix_rad: pixel size in rad
    """
    pix_rad = math.sqrt(math.pi / 3 / nside / nside)
    return pix_rad

def rad2nside( rad ):
    r"""
        *** nside >> pixel size in rad *** # Basic function
        !!! Inverse of pixsize() !!! # Important note
        !!! For healpix usage !!! # Important note
        +++  +++ # Improvements
        :Params pix_rad: pixel size in rad;
        :Output nside: nside of the map;
    """
    nside = int( math.sqrt(math.pi / 3 / rad / rad) )
    return nside

def rad2arcsec( rad ):
    r"""
        *** convert rad to arcsec *** # Basic function
        !!! Inverse of arcsec2rad() !!! # Important note
        !!! Verified by REF !!! # Important note
        # REF: https://www.advancedconverter.com/unit-conversions/angle-conversion/radians-to-arcseconds
        +++ Maybe use astropy? +++ # Improvements
        :Params rad: pixel size in rad
        :Output arcsec: pixel size in arcsec
    """
    arcsec = rad / math.pi * 180 * 3600
    return arcsec

def arcsec2rad( arcsec ):
    r"""
        *** convert arcsec to rad *** # Basic function
        !!! Inverse of rad2arcsec() !!! # Important note
        !!! Verified by REF !!! # Important note
        # REF: https://www.advancedconverter.com/unit-conversions/angle-conversion/arcseconds-to-radians
        +++ Maybe use astropy? +++ # Improvements
        :Params arcsec: pixel size in arcsec
        :Output rad: pixel size in rad
    """
    rad = arcsec / 3600 / 180 * math.pi
    return rad

def arcsec2nside( arcsec ):
    r"""
        *** convert arcsec to correponding nside *** # Basic function
        !!! Inverse of nside2arcsec() !!! # Important note
        +++  +++ # Improvements
        :Params arcsec: pixel size in arcsec
        :Output nside: nside of the map
    """
    # convert arcsec to correponding nside
    rad = arcsec2rad(arcsec)
    nside = int( math.sqrt(math.pi / 3 / rad / rad) )
    return nside

def nside2arcsec( nside ):
    r"""
        *** nside >> pixel size in arcsec *** # Basic function
        !!! Inverse of arcsec2nside() !!! # Important note
        +++  +++ # Improvements
        :Params nside: nside of the map
        :Output arcsec: pixel size in arcsec
    """
    rad = pixsize( nside )
    arcsec = rad2arcsec( rad )
    return arcsec

def nside2npix( nside ):
    r"""
        *** Cal number of pixels of a Nside grid *** # Basic function
        !!! Inverse of npix2nside() !!! # Important note
        !!! A HEALPix map has N_pix = 12N^2 side pixels; !!! # Important note
        * Górski,+ 2005: [HEALPix: A Framework for High-Resolution Discretization and Fast Analysis of Data
        Distributed on the Sphere](https://ui.adsabs.harvard.edu/abs/2005ApJ...622..759G)
        +++  +++ # Improvements
        :Params nside: nside of the map;
        :Output npix: number of pixels of a Nside grid;
    """
    npix = 12 * nside ** 2
    return npix

def npix2nside( npix ):
    r"""
        *** Cal nside from number of pixels of a Nside grid *** # Basic function
        !!! Inverse of nside2npix() !!! # Important note
        +++  +++ # Improvements
        :Params npix: number of pixels of a Nside grid;
        :Output nside: nside of the map;
    """
    nside = int( np.sqrt( npix / 12 ) )
    return  nside

def nside2lmax( nside ):
    r"""
        *** Cal largest multipole lmax of a Nside grid *** # Basic function
        !!!  Inverse of lmax2nside() !!! # Important note
        * Using the Nyquist multipole def from * Das,+ 2008: [A Large Sky Simulation of the Gravitational 
        Lensing of the Cosmic Microwave Background](https://ui.adsabs.harvard.edu/abs/2008ApJ...682....1D)
        * ℓmax ≃ π/sqrt(Ω_pix)
        +++  +++ # Improvements
        :Params nside: nside of the map;
        :Output lmax: largest multipole;
    """
    sr = nside2sr( nside )
    sqr_sr = np.sqrt( sr )
    lmax = int( math.pi / sqr_sr )
    return lmax

def lmax2nside( lmax ):
    r"""
        *** Cal nside from largest multipole lmax of a Nside grid *** # Basic function
        !!! Inverse of nside2lmax() !!! # Important note
        * Using the Nyquist multipole def from * Das,+ 2008: [A Large Sky Simulation of the Gravitational 
        Lensing of the Cosmic Microwave Background](https://ui.adsabs.harvard.edu/abs/2008ApJ...682....1D)
        * ℓmax ≃ π/sqrt(Ω_pix)
        +++  +++ # Improvements
        :Params lmax: largest multipole;
        :Output nside: nside of the map;
    """
    sqr_sr = math.pi / lmax
    sr = sqr_sr ** 2
    nside = int( sr2nside( sr ) )
    return nside

def arcsec2lmax( arcsec ):
    r"""
        *** Cal largest multipole lmax of a given scale *** # Basic function
        !!!  Inverse of lmax2arcsec() !!! # Important note
        * Using the Nyquist multipole def from * Das,+ 2008: [A Large Sky Simulation of the Gravitational 
        Lensing of the Cosmic Microwave Background](https://ui.adsabs.harvard.edu/abs/2008ApJ...682....1D)
        * ℓmax ≃ π/sqrt(Ω_pix)
        +++  +++ # Improvements
        :Params arcsec: aimed scale of a healpix map;
        :Output lmax: largest multipole;
    """
    deg2 = arcsec ** 2 / 3600 / 3600
    sr = deg22sr( deg2 )
    lmax = math.pi / np.sqrt( sr )
    return int( lmax )

def lmax2arcsec( lmax ):
    r"""
        *** Cal the scale from the given largest multipole lmax *** # Basic function
        !!!  Inverse of arcsec2lmax() !!! # Important note
        * Using the Nyquist multipole def from * Das,+ 2008: [A Large Sky Simulation of the Gravitational 
        Lensing of the Cosmic Microwave Background](https://ui.adsabs.harvard.edu/abs/2008ApJ...682....1D)
        * ℓmax ≃ π/sqrt(Ω_pix)
        +++  +++ # Improvements
        :Params lmax: largest multipole;
        :Output arcsec: aimed scale of a healpix map;
    """
    sqr_sr = math.pi / lmax
    sr = sqr_sr ** 2
    deg2 = sr2deg2( sr )
    arcsec = np.sqrt( deg2 * 3600 * 3600 )
    return arcsec

def defaultlmax( nside ):
    r"""
        *** Default est nside >> largest multipole of a Nside grid *** # Basic function
        !!! Inverse of defaultnside() !!! # Important note
        +++ Add reference +++ # Improvements
        :Params nside: nside of the map;
        :Output lmax: Largest multipole;
    """
    lmax = 3 * nside - 1
    return lmax

def defaultnside( lmax ):
    r"""
        *** Default est nside from largest multipole of a Nside grid *** # Basic function
        !!! Inverse of defaultlmax()!!! # Important note
        +++ Add reference +++ # Improvements
        :Params lmax: Largest multipole;
        :Output nside: nside of the map;
    """
    nside = int( ( lmax + 1 ) / 3 )
    return nside

def defaultarcsec2lmax( arcsec ):
    r"""
        *** Cal the scale from the given largest multipole lmax *** # Basic function
        !!!  Inverse of arcsec2lmax() !!! # Important note
        * Use default lmax & nside *
        +++  +++ # Improvements
        :Params lmax: largest multipole;
        :Output arcsec: aimed scale of a healpix map;
    """
    nside = arcsec2nside( arcsec )
    lmax = defaultlmax( nside )
    return lmax

def defaultlmax2arcsec( lmax ):
    r"""
        *** Cal the scale from the given largest multipole lmax *** # Basic function
        !!!  Inverse of arcsec2lmax() !!! # Important note
        * Use default lmax & nside *
        +++  +++ # Improvements
        :Params lmax: largest multipole;
        :Output arcsec: aimed scale of a healpix map;
    """
    nside = defaultnside( lmax )
    arcsec = nside2arcsec( nside )
    return arcsec

# Areas below 

def nside2sr(nside):
    r"""
        *** Cal pixel_area in steradian of a Nside grid *** # Basic function
        !!!  Inverse of sr2nside() !!! # Important note
        !!! A HEALPix map has pixels of the same area Ω_pix = π / 3N_side^2 !!! # Important note
        * Using the Nyquist multipole def from * Das,+ 2008: [A Large Sky Simulation of the Gravitational 
        Lensing of the Cosmic Microwave Background](https://ui.adsabs.harvard.edu/abs/2008ApJ...682....1D)
        * Ωpix = 4π / Npix steradians.
        +++  +++ # Improvements
        :Params nside: nside of the map;
        :Output sr: pixel_area in steradian of a Nside grid;
    """
    npix = nside2npix( nside )
    sr = 4 * math.pi / npix
    return sr

def sr2nside( sr ):
    r"""
        *** Cal nside from pixel_area in steradian of a Nside grid *** # Basic function
        !!!  Inverse of nside2sr() !!! # Important note
        !!! A HEALPix map has pixels of the same area Ω_pix = π / 3N_side^2 !!! # Important note
        * Using the Nyquist multipole def from * Das,+ 2008: [A Large Sky Simulation of the Gravitational 
        Lensing of the Cosmic Microwave Background](https://ui.adsabs.harvard.edu/abs/2008ApJ...682....1D)
        * Ωpix = 4π / Npix steradians.
        +++  +++ # Improvements
        :Params sr: pixel_area in steradian of a Nside grid;
        :Output nside: nside of the map;
    """
    npix = 4 * math.pi / sr
    nside = int( npix2nside( npix ) )
    return nside

def sr2deg2( sr ):
    r"""
        *** convert steradian(sr) to deg^2 *** # Basic function
        !!! Inverse of deg22sr() !!! # Important note
        !!! Verified by Astropy.units !!! # Important note
        (1 * au.sr).to(au.deg**2) >> 3282.8064deg2
        +++ Maybe use astropy? +++ # Improvements
        :Params sr: pixel_area in steradian
        :Output deg2: pixel_area in deg^2
    """
    deg2 = sr * ( 180 / math.pi ) ** 2
    return deg2

def deg22sr( deg2 ):
    r"""
        *** convert deg^2 to steradian(sr) *** # Basic function
        !!! Inverse of sr2deg2() !!! # Important note
        !!! Verified by Astropy.units !!! # Important note
        ( 3282.8064 * au.deg**2).to(au.sr) >> 1sr
        +++ Maybe use astropy? +++ # Improvements
        :Params deg2: pixel_area in deg^2
        :Output sr: pixel_area in steradian
    """
    sr = deg2 / ( 180 / math.pi ) ** 2
    return sr

# =================== Misc tools ===================

def log2( number ):
    r"""
        *** Cal log2 *** # Basic function
        !!!  !!! # Important note
        +++  +++ # Improvements
        :Params number: number to be logged
        :Output n: order
    """
    x = int( number )
    n = math.log( x,2 )
    n = int( n )
    return n

def power2( n ):
    r"""
        *** Gen list of 2^j, j=[0:n]*** # Basic function
        !!!  !!! # Important note
        +++  +++ # Improvements
        :Params n: max order
        :Output p2list: aimed list
    """
    p2list = [ 2**j for j in range( 1,n+1 ) ] 
    return p2list

def steplist( nmin, nmax, nstep ):
    r"""
        *** Gen list based on min, max, & step *** # Basic function
        !!! Calls np.arange !!! # Important note
        +++  +++ # Improvements
        :Params nmin: min range
        :Params nmax: max range
        :Params nstep: step
        :Output steplist: aimed list
    """
    steplist = np.arange( nmin, nmax, nstep, dtype=int )
    return steplist

def nlist2llist( nlist, flag="default" ):
    r"""
    *** Return lmax list from a nside list *** # Basic function
    !!!  !!! # Important note
    +++  +++ # Improvements
    :Params nlist: list of nside from `power2` or `steplist`;
    :Output llist: list of lmax coresponding to each nside from nlist;
    """
    llist = []
    for n in nlist:
        if flag == "default":
            l = defaultlmax( n )
        else:
            l = nside2lmax( n )
        llist.append( l )
    return llist