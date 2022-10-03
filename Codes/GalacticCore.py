# Copyright (c) 2020-2022 Chenxi SHAN <cxshan@hey.com>
# Galactic Simulation Core Functions
""" Notes:
Version 0.1: replacing hp.synalm with cl2alm; replacing hp.alm2cl with alm2cl
"""

__version__ = "0.3"
__date__    = "2022-06-27"

# Basic modules
import numpy as np
import math
import time
from scipy.stats import linregress

import matplotlib.pyplot as plt

# PKGs related to healpix & SHT
import healpy as hp
import ducc0 as dc
import ducc0.healpix as dhp
import ducc0.sht as dsh

# cl2alm pkg
from cl2alm import cl2alm
from alm2cl import alm2cl

# Custom modules
from Gtools import *
from Gplot import *

# =================== I/O Section ===================

def loadmap( server='sgi' ):
    r"""
    *** Load the default files on the computers ***
    
    >> All healpy routines assume RING ordering, 
    in fact as soon as you read a map with read_map, 
    even if it was stored as NESTED, 
    it is transformed to RING. 
    However, you can work in NESTED ordering passing 
    the nest=True argument to most healpy routines.
    
    > https://healpy.readthedocs.io/en/latest/tutorial.html#:~:text=was%20stored%20as-,NESTED,-%2C%20it%20is%20transformed
    """
    if server == 'sgi':
        datadir = "/mnt/ddnfs/data_users/cxshan/radiodata/"
        fn = "haslam408_dsds_Remazeilles2014_ns512.fits"
    elif server == 'gravity':
        datadir = "/home/cxshan/radiodata/"
        fn = "haslam408_dsds_Remazeilles2014_ns512.fits"
    else:
        print('Map location unknow! You should use gopen()')
    fname = datadir + fn
    R14ns512 = hp.read_map(fname)
    return R14ns512


def gopen( fdir, fn ):
    r"""
    *** Open the file map w/ hp.READ_MAP ***
    !!! Check out the hp.READ_MAP instructuion !!!
    :Params fdir: path to the file;
    :Params fn: file name;
    :Output gmap: map in healpix `RING` scheme;
    """
    file = fdir + fn
    gmap = hp.read_map(file)
    return gmap


def pklsave( data, file ):
    r"""
    *** Save intermediate files w/ pkl ***
    Params data: data to be stored;
    Params file: file name to be stored;
    """
    #Save skymodel
    pickle_file = open(file, 'wb')
    pickle.dump(data, pickle_file,  protocol=4)
    pickle_file.close()
    print(file, "is saved!")

    
def loadpkl( file ):
    r"""
    *** Load the intermediate from pkls ***
    Check the encoding here 
    https://stackoverflow.com/questions/4281619/unpicking-data-pickled-in-python-2-5-in-python-3-1-then-uncompressing-with-zlib
    :Params file: input file path + name;
    :Output pkl: loaded data;
    """
    infile = open(file,'rb')
    pkl = pickle.load(infile, encoding='latin1')
    return pkl

# =================== Cl & Alm Utilities ===================

def genGauss_cl( lmax, lmin=1, type_='Gsync' ):
    r"""
    *** Generate the angular power spectrum of the Gaussian random field ***
    !!! The Guassian Random Field should be in Radian !!!
    Check the reference: https://ui.adsabs.harvard.edu/abs/2013A%26A...553A..96D/abstract
    :Params lmax: multipole max;
    :Params lmin: multipole min;
    :Params type_: type handle, "Gsync" & "Gff"
    :Output cl: return power spectrum;
    """

    # angular power spectrum of the Gaussian random field
    ell = np.arange(lmax+1, dtype=int)
    cl = np.zeros(ell.shape)
    ell_idx = ell >= lmin
    if type_=='Gsync':
        # Parameters to extrapolate the angular power spectrum
        gamma = -2.703  # index of the power spectrum between l [30, 90]
        _sigma_tp = 56  # original beam resolution of the template [ arcmin ]
        sigma_tp = arcsec2rad( _sigma_tp * 60 ) # original beam resolution of the template [radian]
        cl[ell_idx] = ell[ell_idx] ** gamma * ( 1.0 - np.exp(-ell[ell_idx]**2 * sigma_tp**2) )
    elif type_=='Gff':
        # Parameters to extrapolate the angular power spectrum
        gamma = -3.136  # index of the power spectrum between l [30, 90]
        _sigma_tp = 6  # original beam resolution of the template [ arcmin ]
        sigma_tp = arcsec2rad( _sigma_tp * 60 )
        _sigma_sim = 6  # simulated beam resolution [ arcsec ]
        sigma_sim = arcsec2rad( _sigma_sim )
        cl[ell_idx] = ell[ell_idx] ** gamma * ( np.exp(-ell[ell_idx]**2 * sigma_sim**2) - np.exp(-ell[ell_idx]**2 * sigma_tp**2) )
    cl[ell < lmin] = cl[lmin]
    return cl


def genRNG_cl( lmax ):
    r"""
    *** Generate cl using RNG.normal ***
    :Params lmax: multipole max;
    :Params lmin: multipole min;
    :Output cl: return power spectrum;
    """
    rng = np.random.default_rng(42)
    
    ell = np.arange(lmax+1, dtype=int)
    cl = rng.normal(0, 0.1, ell.shape)
    return cl

def fitcl( cl, lmin, lmax ):
    r"""
        *** Linear fitting of cls w/ l range *** # Basic function
        !!! Using log - log 1st & fit linearly !!! # Important note
        +++ Fitting in a log log space instead +++ # Improvements
        :Params cl: cl to be fitted
        :Params lmin: l range min
        :Params lmax: l range max
        :Output output from linregress: slope, intercept, r_value, p_value, std_err
    """
    ell = np.arange(len(cl))
    data_ = cl.copy()
    data = data_[lmin:lmax]
    l = ell[lmin:lmax]
    
    xfid = np.linspace(1.3,2) 
    slope, intercept, r_value, p_value, std_err = linregress(np.log10(l), np.log10(data))
    plt.plot(np.log10(l), np.log10(data))
    plt.plot(xfid, xfid*slope+intercept)
    plt.xlabel('Log(l)')
    plt.ylabel('Log(Cl)')
    plt.show()
    return slope, intercept, r_value, p_value, std_err

# =================== Alm Utilities ===================

def genGauss_alm( nside, lmin=1 ):
    r"""
    *** Generate the alms from angular power spectrum of the Gaussian random field ***
    *** Calls cl2alm instead of synalm ***
    :Params nside: nside of the hp geometery;
    :Output almcl: dict contains alm & cl; 
                   almcl['alm']: return alm of a Gaussian random field;
                   almcl['cl']: return the generated angular power spectrum;
    """
    almcl = {}
    # Set the lmax & mmax
    lmax = defaultlmax( nside )
    mmax = lmax
    nalm = hp.Alm.getsize(lmax)
    # Parameters to extrapolate the angular power spectrum
    gamma = -2.703  # index of the power spectrum between l [30, 90]
    sigma_tp = 56  # original beam resolution of the template [ arcmin ]
    alpha = 0.0599
    beta = 0.782
    
    # Gen angular power spectrum of the Gaussian random field
    ell = np.arange(lmax+1, dtype=int)
    cl = np.zeros(ell.shape)
    ell_idx = ell
    ell_idx = ell >= lmin
    cl[ell_idx] = (ell[ell_idx] ** gamma *
                   1.0 - np.exp(-ell[ell_idx]**2 * sigma_tp**2))
    cl[ell < lmin] = cl[lmin]
    
    # Gen the alms
    alm = cl2alm(cls=cl,lmax=lmax, mmax=mmax)
    almcl['alm'] = alm
    almcl['cl'] = cl
    
    return almcl

def genGauss_alm_pair( nside, lmin=1 ):
    r"""
    !!! Using cl2alm & healpy.synalm !!!
    *** Generate the a pair of alms from angular power spectrum of the Gaussian random field ***
    :Params nside: nside of the hp geometery;
    :Output almcl: dict;
    """
    almcl = {}
    # Set the lmax & mmax
    lmax = 3 * nside -1
    mmax = lmax
    nalm = hp.Alm.getsize(lmax)
    # Parameters to extrapolate the angular power spectrum
    gamma = -2.703  # index of the power spectrum between l [30, 90]
    sigma_tp = 56  # original beam resolution of the template [ arcmin ]
    alpha = 0.0599
    beta = 0.782
    
    # Gen angular power spectrum of the Gaussian random field
    ell = np.arange(lmax+1, dtype=int)
    cl = np.zeros(ell.shape)
    ell_idx = ell
    ell_idx = ell >= lmin
    cl[ell_idx] = (ell[ell_idx] ** gamma *
                   1.0 - np.exp(-ell[ell_idx]**2 * sigma_tp**2))
    cl[ell < lmin] = cl[lmin]
    
    # Gen the alms
    alm_old = hp.synalm(cls=cl,lmax=lmax, mmax=mmax)
    alm_new = cl2alm(cls=cl,lmax=lmax, mmax=mmax)
    alm_diff = alm_new - alm_old
    
    almcl["nside"] = nside
    almcl["alm_old"] = alm_old
    almcl["alm_new"] = alm_new
    almcl["alm_diff"] = alm_diff
    almcl["cl"] = cl
    
    return almcl

def genRNG_alm( nside ):
    r"""
    *** Generate the alms using cl from genRNG_cl() ***
    *** Calls cl2alm instead of synalm ***
    *** Calls defaultlmax = 3 * nside - 1 ***
    :Params nside: nside;
    :Output almcl: dict contains alm & cl; 
                   almcl['alm']: return alm of RNG;
                   almcl['cl']: return the generated angular power spectrum;
    """
    
    lmax = defaultlmax( nside )
    cl = genRNG_cl( lmax )
    mmax = lmax
    alm = cl2alm(cls=cl,lmax=lmax, mmax=mmax)
    return alm, cl


def nalm( lmax, mmax ):
    r"""
    *** output the size of an alm given the lmax & mmax ***
    The alm is in the order:
    (0,0), (1,0), (2,0), … (lmax,0), (1,1), (2,1), …, (lmax, lmax)
    > https://mtr.pages.mpcdf.de/ducc/sht.html#:~:text=Parameters-,alm,-(numpy.ndarray(((lmax
    '//' > https://www.educative.io/edpresso/floor-division
    :Params lmax: max l;
    :Parmas mmax: max m;
    """
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def random_alm( lmax, mmax, spin, ncomp, rng ):
    r"""
    *** Generate a_lm from a RNG random distribution ***
    :Params lmax: max l;
    :Parmas mmax: max m;
    :Parmas spin: spin of the alm, default should be 0;
    :Params ncomp: number of components of the alm;
    :Params rng: random number generator;
    > https://numpy.org/doc/stable/reference/random/generator.html
    :Output res: the (ncomp, nalm) shaped a_lm with m==0 real-valued;
    > Libsharp requires only real-valued complex a_lm;
    """
    rng = np.random.default_rng(42)
    res = rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax))) \
     + 1j*rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax)))
    # make a_lm with m==0 real-valued
    res[:, 0:lmax+1].imag = 0.
    ofs=0
    for s in range(spin):
        res[:, ofs:ofs+spin-s] = 0.
        ofs += lmax+1-s
    return res

# =================== SHT Utilities ===================

def alm2hpmap_dc( alm, nside, nthreads, spin=0, test=0 ):
    r"""
    *** alm2map using ducc0 in a healpix grid ***
    :Params alm: spherical harmonics coefficients; 
    :Params nside: nside of the extended map;
    :Params nthreads: number of threads;
    :Params spin: spin of the alm, default spin = 0;
    :Params test: test type;
    ::::::::::::: test == 0 using a RNG distribution alm;
    ::::::::::::: test == 1 using a custom alm;
    :Output hpmap: healpix map;
    """
    lmax = 3 * nside -1
    mmax = lmax
    if test == 0:
        #alm0
        rng = np.random.default_rng(48)
        ncomp = 1 if spin == 0 else 2
        alm0 = random_alm(lmax, mmax, spin, ncomp, rng)
        print('test = 0, generating random alm', alm0.shape)
        inalm = alm0
    else:
        print('alm1', alm.shape)
        rsalm = alm.reshape((1,-1))
        print('test = 1, using custom alm', rsalm.shape)
        inalm = rsalm
    base = dhp.Healpix_Base(nside, "RING")
    geom = base.sht_info()

    # test adjointness between synthesis and adjoint_synthesis
    _hpmap = dsh.experimental.synthesis(alm=inalm, lmax=lmax, spin=spin, nthreads=nthreads, **geom)
    shape = _hpmap.shape
    hpmap = _hpmap.reshape(shape[1],)
    return hpmap


def test_hpmap2alm_dc( hpmap, nside, spin, nthreads, test=0 ):
    r"""
    *** map2alm using ducc0 in a healpix grid ***
    !!! Still testing !!!
    :Params hpmap: custom healpix map;
    :Params nside: nside of the extended map;
    :Params nthreads: number of threads;
    :Params spin: spin of the alm, default spin = 0;
    :Params test: test type;
    ::::::::::::: test == 0 using a RNG distribution alm;
    ::::::::::::: test == 1 using a custom alm;
    :Output alm: spherical harmonics coefficients;
    :Output inmap: input healpix map;
    """
    lmax = 3 * nside - 1
    mmax = lmax
    if test == 0:
        rng = np.random.default_rng(48)
        ncomp = 1 if spin == 0 else 2
        alm0 = random_alm(lmax, mmax, spin, ncomp, rng)
        print(alm0.shape[0])
        map0 = rng.uniform(0., 1., (alm0.shape[0], 12*nside**2))
        print(map0.shape)
        inmap = map0
        print(map0)
    elif test == 1:
        rng = np.random.default_rng(48)
        ncomp = 1 if spin == 0 else 2
        alm0 = random_alm(lmax, mmax, spin, ncomp, rng)
        print(alm0.shape[0])
        map0 = rng.uniform(0., 1., (alm0.shape[0], 12*nside**2))
        print(map0.shape)
        print(map0)
        hpmap64 = np.float64(hpmap)
        map0[0,:] = hpmap64
        print(map0)
        inmap = map0
    else:
        hpmap64 = np.float64(hpmap)
        rsmap = hpmap64.reshape((1,-1))
        inmap = rsmap
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    geom = base.sht_info()
    alm1 = ducc0.sht.experimental.adjoint_synthesis(lmax=lmax, spin=spin, map=inmap, nthreads=nthreads, **geom)
    return alm1, inmap


def alm2hpmap_sh( alm, nside, nthreads, spin=0, test=0 ):
    r"""
    !!! This is a backup to alm2hpmap_dc, please always use alm2hpmap_dc first !!!
    *** alm2map using libsharp in a healpix grid ***
    :Params alm: spherical harmonics coefficients; 
    :Params nside: nside of the extended map;
    :Params nthreads: number of threads;
    :Params spin: spin of the alm, default spin = 0;
    :Params test: test type;
    ::::::::::::: test == 0 using a RNG distribution alm;
    ::::::::::::: test == 1 using a custom alm;
    :Output hpmap: healpix map;
    """
    lmax = 3 * nside -1
    mmax = lmax
    
    if test == 0:
        #alm0
        rng = np.random.default_rng(48)
        ncomp = 1 if spin == 0 else 2

        alm0 = random_alm(lmax, mmax, spin, ncomp, rng)
        print('test = 0, generating random alm', alm0.shape)
        rsalm0 = alm0.reshape(alm0.shape[1],)
        inalm = rsalm0
    else:
        print('test = 1, using custom alm', alm.shape)
        inalm = alm
    inalm[0:lmax+1].imag = 0. #job only receive alm with imag part equals zero! 
    print('inalm', inalm.shape)
    job = dsh.sharpjob_d()
    job.set_healpix_geometry(nside)
    job.set_triangular_alm_info(lmax, mmax)
    job.set_nthreads( nthreads )
    sharpmap = job.alm2map(inalm)
    return sharpmap

# =================== Map Utilities ===================

def ncheck( inmap ):
    r"""
    *** Check the nside of the map ***
    """
    ncheck = hp.get_nside(inmap)
    return ncheck


def nchecks( map1, map2 ):
    r"""
    *** Check if the nside of two maps matches ***
    """
    n1 = ncheck( map1 )
    n2 = ncheck( map2 )
    if n1 == n2:
        status = True
    else:
        status = False
    return status


def prograde( inmap, ns ):
    r"""
    *** Check the map nside & upgrade to a higher nside ***
    :Params inmap: healpix map;
    :Params ns: aimed nside;
    :Output outmap: prgraded map;
    """
    n = ncheck(inmap)
    if n < ns:
        outmap = hp.ud_grade(inmap, nside_out=ns)
    elif n == ns:
        outmap = inmap
    else:
        print('Nside of inmap is larger than the assigned nside')
        outmap = inmap
    return outmap


def whiten( gss ):
    r"""
    *** Whiten a guass map ***
    """
    gss = (gss - gss.mean()) / gss.std()
    return gss


def gen_smallscales( gss, hpmap ):
    r"""
    *** Check the gss & hpmap and generate smallscales temp ***
    :Params gss: whitened GRF map;
    :Params hpmap: original healpix map;
    :Output hpmap_smallscales: small scale map;
    """
    # Parameters to extrapolate the angular power spectrum
    gamma = -2.703  # index of the power spectrum between l [30, 90]
    sigma_tp = 56  # original beam resolution of the template [ arcmin ]
    alpha = 0.0599
    beta = 0.782
    
    match = nchecks( gss, hpmap )
    if match == True:
        hpmap_smallscales = alpha * gss * hpmap**beta
    else:
        print( "!!! Nside Error, please check the Nside" )
        print( "gss:", ncheck(gss) )
        print( "hpmap:", ncheck(hpmap) )
        hpmap_smallscales = gss
    return hpmap_smallscales


def add_smallscales( smallscales, hpmap ):
    r"""
    *** Check the smallscales & hpmap and add smallscales to hpmap ***
    :Params smallscales: small scale map from a whitened GRF map;
    :Params hpmap: original healpix map;
    :Output addedmap: small scale added hpmap;
    """
    match = nchecks( smallscales, hpmap )
    if match == True:
        addedmap = smallscales + hpmap
    else:
        print( "!!! Nside Error, please check the Nside" )
        print( "smallscales:", ncheck(smallscales) )
        print( "hpmap:", ncheck(hpmap) )
        addedmap = smallscales
    return addedmap