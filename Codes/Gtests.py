# Copyright (c) 2020-2022 Chenxi SHAN <cxshan@hey.com>
# Galactic Simulation Test Utilities
__version__ = "0.0.3"
__date__    = "2022-03-21"

# Basic modules
import numpy as np
import math
import time

import matplotlib.pyplot as plt

# PKGs related to healpix & SHT
import healpy as hp
import ducc0 as dc
import ducc0.healpix as dhp
import ducc0.sht as dsh

# Custom modules
from Gtools import *
from Gplot import *
from Galactic import *

# =================== Tests Utilities: alm2map =================== 

def st_alm2map_order2( n, nthreads, pkg="dc", test=0 ):
    r"""
    *** stress testing alm2hpmap_dc() or alm2hpmap_sh() w/ 2 order ***
    :Params n: order of 2;
    :Params nthreads: number of threads;
    :Params test: checking alm2hpmap_dc/alm2hpmap_sh for more info;
    :Output datalist: data dict block;
    ::::::::::::::::: datalist[i]["alm"] stores alm;
    ::::::::::::::::: datalist[i]["cl"] stores cl;
    ::::::::::::::::: datalist[i]['dchpmap'] stores hpmap from alm2hpmap_dc();
    """
    nlist = power2(n)
    datalist = {}
    for i in nlist:
        try:
            datalist[i] = {}
            nside = i
            print("#============ running nside:", nside, "===============#")
            t0 = time.time()
            print('{}'.format(time.ctime()))
            print("#============ running genGuass_alm ===============#")
            inalm, incl = genGuass_alm( nside )
            datalist[i]["alm"] = inalm
            datalist[i]["cl"] = incl
            print('{}'.format(time.ctime()))
            t1 = time.time()
            datalist[i]['almtime'] = t1-t0
            print("nside:", i, "alm time: %.2f s"%( t1-t0 ) )
            print('incl.shape:', incl.shape)
            print('inalm.shape:', inalm.shape)
            print("#============ running alm2hpmap ===============#")
            t0 = time.time()
            print('{}'.format(time.ctime()))
            if pkg == "dc":
                hpmap = alm2hpmap_dc( alm=inalm, nside=nside, nthreads=nthreads, spin=0, test=test )
            elif pkg == "sh":
                hpmap = alm2hpmap_sh( alm=inalm, nside=nside, nthreads=nthreads, spin=0, test=test )
            else:
                print("Check the PKG info!")
            datalist[i]['dchpmap'] = hpmap
            print('{}'.format(time.ctime()))
            t1 = time.time()
            datalist[i]['maptime'] = t1-t0
            print("nside:", i, "map time: %.2f s"%( t1-t0) )
            view(hpmap)
            plt.show()
            print("#============ ending nside:", nside, "===============#")
        except Exception as e:
            print("!!!!!!!!!=========== Error ===============!!!!!!!!!")
            print("nside:", i, e)
    return datalist


def st_alm2map( nmin, nmax, nstep, nthreads, pkg="dc", test=0 ):
    r"""
    *** stress testing alm2hpmap_dc() or alm2hpmap_sh() ***
    :Params nmin: nside min;
    :Params nmax: nside max;
    :Params nstep: nside step;
    :Params nthreads: number of threads;
    :Params pkg: which pkg are used:
    :::::::::::: "dc": using `alm2hpmap_dc`
    :::::::::::: "sh": using `alm2hpmap_sh`
    :Params test: checking alm2hpmap_dc/alm2hpmap_sh for more info;
    :Output datalist: data dict block;
    ::::::::::::::::: datalist[i]["alm"] stores alm;
    ::::::::::::::::: datalist[i]["cl"] stores cl;
    ::::::::::::::::: datalist[i]['dchpmap'] stores hpmap from alm2hpmap_dc();
    """
    nlist = steplist(nmin, nmax, nstep)
    datalist = {}
    for i in nlist:
        try:
            datalist[i] = {}
            nside = i
            print("#============ running nside:", nside, "===============#")
            t0 = time.time()
            print('{}'.format(time.ctime()))
            print("#============ running genGuass_alm ===============#")
            inalm, incl = genGuass_alm( nside )
            datalist[i]["alm"] = inalm
            datalist[i]["cl"] = incl
            print('{}'.format(time.ctime()))
            t1 = time.time()
            datalist[i]['almtime'] = t1-t0
            print("nside:", i, "alm time: %.2f s"%( t1-t0 ) )
            print('incl.shape:', incl.shape)
            print('inalm.shape:', inalm.shape)
            print("#============ running alm2hpmap ===============#")
            t0 = time.time()
            print('{}'.format(time.ctime()))
            if pkg == "dc":
                hpmap = alm2hpmap_dc( alm=inalm, nside=nside, nthreads=nthreads, spin=0, test=test )
            elif pkg == "sh":
                hpmap = alm2hpmap_sh( alm=inalm, nside=nside, nthreads=nthreads, spin=0, test=test )
            else:
                print("Check the PKG info!")
            datalist[i]['dchpmap'] = hpmap
            print('{}'.format(time.ctime()))
            t1 = time.time()
            datalist[i]['maptime'] = t1-t0
            print("nside:", i, "map time: %.2f s"%( t1-t0) )
            view(hpmap)
            plt.show()
            print("#============ ending nside:", nside, "===============#")
        except Exception as e:
            print("!!!!!!!!!=========== Error ===============!!!!!!!!!")
            print("nside:", i, e)
    
    return datalist

# =================== Tests Utilities: cl2alm ===================

def st_cl2alm_order2( n, test="Guass", flag="default" ):
    r"""
    *** stress testing gen_alm functions w/ 2 order ***
    :Params n: order of 2;
    :Params test: which func are used:
    :::::::::::: "Guass": using `genGuass_alm`
    :::::::::::: "RNG": using `genRNG_alm`
    :Params flag: check from nlist2llist
    :Output datalist: alm & cl dict;
    """
    nlist = power2(n)
    lmaxlist = nlist2llist( nlist, flag=flag )
    datalist = {}
    for i in lmaxlist:
        try:
            datalist[i] = {}
            print("#============ running lmax:", i, "===============#")
            if test == "Guass":
                alm, cl = genGuass_alm( i )
            elif test == "RNG":
                alm, cl = genRNG_alm( i )
            else:
                print("Check the alm type info!")
            datalist[i]['alm'] = alm
            datalist[i]['cl'] = cl
            plotcl(cl)
            plt.show()
            print("#============ running finished:", i, "===============#")
        except Exception as e:
            print("!!!!!!!!!=========== Error ===============!!!!!!!!!")
            print("lmax:", i, e)
    return datalist


def st_cl2alm( nmin, nmax, nstep, test="Guass" ):
    r"""
    *** stress testing gen_alm functions ***
    :Params nmin: nside min;
    :Params nmax: nside max;
    :Params nstep: nside step;
    :Params test: which func are used:
    :::::::::::: "Guass": using `genGuass_alm`
    :::::::::::: "RNG": using `genRNG_alm`
    :Output datalist: alm & cl dict;
    """
    lmaxlist = steplist( nmin, nmax, nstep )
    datalist = {}
    for i in lmaxlist:
        try:
            datalist[i] = {}
            print("#============ running lmax:", i, "===============#")
            if test == "Guass":
                alm, cl = genGuass_alm( i )
            elif test == "RNG":
                alm, cl = genRNG_alm( i )
            else:
                print("Check the alm type info!")
            datalist[i]['alm'] = alm
            datalist[i]['cl'] = cl
            plotcl(cl)
            plt.show()
            print("#============ running finished:", i, "===============#")
        except Exception as e:
            print("!!!!!!!!!=========== Error ===============!!!!!!!!!")
            print("lmax:", i, e)
    return datalist

