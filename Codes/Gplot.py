# Copyright (c) 2020-2022 Chenxi SHAN <cxshan@hey.com>
# Galactic Simulation Plotting Tools
__version__ = "0.0.2"
__date__    = "2022-06-26"

import matplotlib.pyplot as plt

# Basic modules
import numpy as np
import math
from scipy.stats import linregress

# PKGs related to healpix & SHT
import healpy as hp
import ducc0 as dc

# =================== Map plotting tools ===================
def view( inmap ):
    r"""
    *** Plot an healpix map in a mollweide projection ***
    """
    hp.mollview(inmap)
    hp.graticule()
    
def viewgl( inmap ):
    r"""
    *** Plot an GLmap in a mollweide projection ***
    :Params inmap: input GLmap
    """
    GL_map = inmap
    ax = plt.axes(projection='mollweide')
    ax.grid()
    ax.imshow(GL_map[0], extent=[0, 1, 0, 1], aspect=ax.get_aspect(), transform=ax.transAxes)
    
def show( inmap ):
    r"""
    *** show map via calling plt.imshow ***
    """
    plt.imshow(inmap)
    
# =================== Old Cl plotting tools ===================
"""
Need to update the anafast to something else, avoid the MAX_NSIDE = 8196 issue

* Solution: map to alm & then alm to cl;
* map to alm: 
    - ducc0.sht.sharpjob_d.map2alm
"""
def plotmap2cl(inmap, LMAX, log=True):
    """
    *** Covert inmap to cl & plot w/ log option ***
    Main pkg: healpy.anafast, limited by `MAX_NSIDE=8192`
    
    :Params inmap: healpix map;
    :Params LMAX: largest l;
    :Params log: True for setting matplotlib.pyplot using log scale;
    """
    cl = hp.anafast(inmap, lmax = LMAX)
    ell = np.arange(len(cl))
    plt.figure(figsize=(10, 5))
    plt.plot(ell, ell * (ell + 1) * cl)
    plt.xlabel("$\ell$")
    plt.ylabel("$\ell(\ell+1)C_{\ell}$")
    if log == True:
        plt.xscale("log")
        plt.yscale("log")
    plt.grid()

def plot2map2cl(inmap1, inmap2, LMAX, log=True):
    """
    *** Covert 2inmaps to cl & plot w/ log option ***
    Main pkg: healpy.anafast, limited by `MAX_NSIDE=8192`
    
    :Params inmap1: healpix map1;
    :Params inmap2: healpix map2;
    :Params LMAX: largest l;
    :Params log: True for setting matplotlib.pyplot using log scale;
    """
    n1 = hp.get_nside(inmap1)
    n2 = hp.get_nside(inmap2)
    LMAX1 = min(3*n1, LMAX)
    LMAX2 = min(3*n2, LMAX)
    cl1 = hp.anafast(inmap1, lmax = LMAX1)
    cl2 = hp.anafast(inmap2, lmax = LMAX2)
    ell1 = np.arange(len(cl1))
    ell2 = np.arange(len(cl2))
    plt.figure(figsize=(10, 5))
    plt.plot(ell1, ell1 * (ell1 + 1) * cl1 / 2 / np.pi, label=get_vname(inmap1))
    plt.plot(ell2, ell2 * (ell2 + 1) * cl2 / 2 / np.pi, '--', label=get_vname(inmap2))
    plt.xlabel("$\ell$")
    plt.ylabel("$\ell(\ell+1)C_{\ell}$")
    plt.legend()
    if log == True:
        plt.xscale("log")
        plt.yscale("log")
    plt.grid()

    
def plot2map2cllim(inmap1, inmap2, LMAX, xlim, ylim, log=True):
    """
    *** Covert 2inmaps to cl & plot w/ log option w/ xylim option ***
    
    :Params inmap1: healpix map1;
    :Params inmap2: healpix map2;
    :Params LMAX: largest l;
    :Params xlim: set plt.xlim;
    :Params ylim: set plt.ylim;
    :Params log: True for setting matplotlib.pyplot using log scale;
    """
    n1 = hp.get_nside(inmap1)
    n2 = hp.get_nside(inmap2)
    LMAX1 = min(3*n1, LMAX)
    LMAX2 = min(3*n2, LMAX)
    cl1 = hp.anafast(inmap1, lmax = LMAX1)
    cl2 = hp.anafast(inmap2, lmax = LMAX2)
    ell1 = np.arange(len(cl1))
    ell2 = np.arange(len(cl2))
    plt.figure(figsize=(10, 5))
    plt.plot(ell1, ell1 * (ell1 + 1) * cl1 / 2 / np.pi, label=get_vname(inmap1))
    plt.plot(ell2, ell2 * (ell2 + 1) * cl2 / 2 / np.pi, '--', label=get_vname(inmap2))
    plt.xlabel("$\ell$")
    plt.ylabel("$\ell(\ell+1)C_{\ell}$")
    plt.legend()
    plt.xlim(xlim)
    plt.ylim(ylim)
    if log == True:
        plt.xscale("log")
        plt.yscale("log")
    plt.grid()
    

def get_vname(v):
    vnames = [name for name in globals() if globals()[name] is v]
    return vnames[0]

# =================== Cl plotting tools ===================

def plotcl(cl, cl_=False, log=True):
    r"""
    *** plot cls w/ log option & cl option ***
    
    :Params cl: input cl;
    :Params cl_: cl handle, if True only plot Cl on y-axis;
    :Params log: log handle, True for setting matplotlib.pyplot using log scale;
    """
    ell = np.arange(len(cl))
    plt.figure(figsize=(10, 5))
    if cl_:
        plt.plot(ell, cl)
        plt.xlabel("$\ell$")
        plt.ylabel("$C_{\ell}$")
    else:
        plt.plot(ell, ell * (ell + 1) * cl)
        plt.xlabel("$\ell$")
        plt.ylabel("$\ell(\ell+1)C_{\ell}$")
    if log:
        plt.xscale("log")
        plt.yscale("log")
    plt.grid()
    
    
def plot2cl(cl1, cl2, log=True):
    r"""
    *** plot two cls w/ log option ***
    
    :Params cl1: input cl1;
    :Params cl2: input cl2;
    :Params log: True for setting matplotlib.pyplot using log scale;
    """
    ell1 = np.arange(len(cl1))
    ell2 = np.arange(len(cl2))
    plt.figure(figsize=(10, 5))
    plt.plot(ell1, ell1 * (ell1 + 1) * cl1, '--', label='original')
    plt.plot(ell2, ell2 * (ell2 + 1) * cl2,  label='reproduced')
    plt.xlabel("$\ell$")
    plt.ylabel("$\ell(\ell+1)C_{\ell}$")
    if log == True:
        plt.xscale("log")
        plt.yscale("log")
    plt.legend(loc='best')
    plt.grid() 

    
def plotcldiff(oricl, recl, log=False):    
    r"""
    *** plot cl difference w/ log option, default log=False ***
    *** Two subplot will show up: cl difference & difference ratio ***
    
    :Params oricl: original cl;
    :Params recl: reproduced cl;
    :Params log: True for subplot setting axs.set_xscale("log");
    """
    cl = oricl
    ell = np.arange(len(cl))
    
    diff = oricl - recl
    ratio = np.absolute(oricl - recl) / oricl
    
    fig, ax = plt.subplots(2, figsize=(10, 10))
    
    ax[0].plot(ell, ell * (ell + 1) * diff, label='Difference')
    ax[0].set_title('cl difference')
    ax[0].set(xlabel="$\ell$", ylabel="$\ell(\ell+1)C_{\ell}$")

    ax[1].plot(ell, ratio, label='Ratio')
    ax[1].set_title('cl difference ratio')
    ax[1].set(xlabel="$\ell$", ylabel="Ratio")

    if log == True:
        for axs in ax:
            axs.set_xscale("log")
            axs.set_yscale("log")
    
    for axs in ax:
        axs.grid(True)
        axs.legend(loc='best')