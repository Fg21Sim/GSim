# Copyright (c) 2020-2022 Chenxi SHAN <cxshan@hey.com>
# Galactic Simulation Class Dev Mode

__version__ = "1.5"
__date__    = "2022-08-13"

# Last updated by Chenxi 08-13@13:14
"""
v1.3 add use_G02: G02 spectral index map; This is added to loadmap();
"""

# Basic modules
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import astropy.units as au

import time
from datetime import datetime
import timeit

import pickle, joblib

# MPI module
from threading import active_count

# PKGs related to healpix & SHT
import healpy as hp
import ducc0 as dc
import ducc0.healpix as dhp
import ducc0.sht as dsh

# Custom modules
from Log import Log

# cl2alm pkg
from cl2alm import cl2alm
from alm2cl import alm2cl

# Coord
from astropy.wcs import WCS
import scipy
from reproject import reproject_from_healpix, reproject_to_healpix
from astropy.coordinates import SkyCoord

# Fitting
from MCMC import MCMC
from MCMC_Model import Fitfunc

# Util
from Ut.sky import get_sky, SkyPatch, SkyHealpix
from Ut.io import read_fits_healpix, write_fits_healpix
from Ut.freqz import genfz

class GalacticDev:
    r"""
        *** The Galactic object to simulate Galactic Component with small scales *** # Basic function
        !!! Need to check the map we need to use float32 !!! # Important note
        +++ Update to take fg21sim config files; +++ # Improvements
        +++ Add unit check for sigma! +++
        +++ Add healpix info function +++
        +++ Add unit support, right now unit='K' does nothing +++
    """
    
    def __init__(self, 
                 nside = None, sigma_tem = None,                         # Resolution
                 freqs = None, unit = 'K',                               # Freq & unit
                 alpha = None, beta = None, gamma = None,                # Small scale & cl params
                 alm = None, cl = None,                                  # Input cl & alm
                 inmap = None, mmap = None, idmap = None,                # Input map, masked map, spectral index map
                 grfmap = None, grfpatch = True,                         # Input grfmap
                 fitlmin = None, fitlmax = None,                         # Gamma param fitting
                 psize = None, npatch = None, pcenter = None,            # Alpha & beta param fitting
                 fov = ( 10, 10 ), center = ( 0, -27 ), sim_pixel = 20,  # Sky patch simulation
                 fullsky_ = False, fit_patch_ = False,                   # Simulation settings | Not working rightnow
                 coord = 'C', frame = 'ICRS', proj = 'TAN',              # Projection info
                 comp = 'Gsync', order = 'RING',                         # Type & order
                 server = 'sgi', use_ns512 = True, use_G02 = True,       # Server info, which map version to use
                 extra_large_mode = True, default = False,               # Mode setting
                 level = 'debug', stdout = False ):                      # Log contral
        
        """
        Please use extra_large_mode = True when simulating large nside or high resolution simulations!
        You can use high res grfmap by setting grfmap!
        """
        
        # Initialize the logger
        self.level_ = level
        self.stdout_ = stdout
        self.type_ = comp
        self.default_ = default
        self.extra_large_mode_ = extra_large_mode
        self._set_logger()
        
        # Initialize the Simulation Params
        self.logger.info( ' ⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇ ' )
        self.logger.info( ' Galactic %s simulation is initializing! ' % self.type_ )
        self.logger.warning( " sigma_tem default unit is [arcmin]! " )
        
        self._set_comp() # Set the compID & name
        self._params_set_up = None # Set the status
        self._set_default_io()
        
        if self.default_: # Using default params if you can
            self._set_default_params()                  
        else:
            self.logger.info( ' Custom settings are used! ' )
            self._alpha = alpha
            self._beta = beta
            self._gamma = gamma
            self.sigma_tem = sigma_tem * au.arcmin # unit[arcmin]
        
        self._set_up_params()
            
        self.logger.info( ' %s alpha is initialized as %s ' % ( self.type_, self.alpha ) )
        self.logger.info( ' %s beta is initialized as %s ' % ( self.type_, self.beta ) )
        self.logger.info( ' %s gamma is initialized as %s ' % ( self.type_, self.gamma ) )
        self.logger.info( ' %s sigma_tem is initialized as %s [arcmin] ' % ( self.type_, self.sigma_tem ) )
        
        # Skypatch Grid >> Aimed simulation spec of the skypatch grid
        self.fov = fov                  # (tuple,deg) The size of the skypatch e.g. (10, 10);
        self.center = center            # (tuple,deg) Ra & dec in ICRS skycoordinate e.g. (0, -27);
        self.show_g_coord()             # Show the center in [G]
        self.sim_pixel = sim_pixel      # [arcsec] Simulated pixel size;
        self.sim_size, _, _ = self._gen_pix_grid( center, fov, sim_pixel )
        self._set_skypatch_grid()
        
        # Healpix Grid >> Aimed simulation spec of the healpix grid
        self.nside = nside
        self.order = order
        self.logger.info( ' Nside is set to %s ' % self.nside )
        self._set_healpix_grid()
        
        # Fitting Grid >> Gen the fitting grid for MCMC
        self._set_fitting_grid()
        self._fitmap = None
        
        # Maps
        self.inmap = inmap        # input map
        self.idmap = idmap        # input spectral index map
        self.mmap = mmap          # masked map
        self.grfmap = grfmap      # Gaussian Random Field map (from cl & alm)
        self.grfpatch = grfpatch  # Add patch mode
        self._set_stage()
        self._set_grf()
        self.inheader = None
        self.idheader = None
        
        if self.inmap is None:
            self.loadmap( server = server, use_ns512 = use_ns512, use_G02 = use_G02 )
            self.logger.info( 'No input maps are given, auto loading input map.' )
        
        # Alm Cl, they can be set by the user;
        self._alm = alm
        self._cl = cl
        self.logger.info( ' alm & cl are set to %s & %s ' % ( self._alm, self._cl ) )
        
        ## Fitting cl
        self.fitlmin = fitlmin
        self.fitlmax = fitlmax
        self._fitcl = None
        self.logger.info( ' Cl fitting using lmin=%s, lmax=%s. ' % ( fitlmin, fitlmax ) )
        
        # Frequency range
        self.freqs = freqs
        if freqs is None:
            self.freqs = self._gen_default_freqs()
            self.logger.info( 'Freqs list is not offered, generated default freq list.' )
        self.unit = unit # Unit of the intensity maps, should set as 'K' Currentlt not working   
        
        self.logger.info( ' Galactic %s simulation is initialized! ' % self.type_ )
        self.logger.info( ' ⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆ ' )
    
    def _set_logger( self ):
        self.logname = 'GSim' + '_' + self.type_ + '_' + "Default_" + str(self.default_)
        loging = Log( self.logname, self.level_, self.stdout_ )
        self.logger = loging.logger
    
    def _set_comp( self ):
        # Component name
        if self.type_ is 'Gsync':
            self.compID = "galactic/synchrotron"
            self.name = "Galactic synchrotron (unpolarized)"    
        elif self.type_ is 'Gfree':
            self.compID = "galactic/freefree"
            self.name = "Galactic free-free"
        self.logger.info( "compID is %s, name is %s" % ( self.compID, self.name ) )
        
    def _set_healpix_grid( self ):
        self.sky = self._set_sky( 'healpix' )
        self.sky.add_header( "NSIDE", self.nside, "Healpix resolution parameter" )
        self.sky.add_header( "CompID", self.compID, "Emission component ID" )
        self.sky.add_header( "CompName", self.name, "Emission component" )
        self.sky.add_header( "BUNIT", "K", "[Kelvin] Data unit" )
        self._hpinfo = {} # Store healpix info
        self.logger.info( "Healpix grid is all set!" )
    
    def _set_fitting_grid( self ):
        self.fittingpatch = self._set_sky( 'patch', fitting=True )
        self.fittingpatch.add_header( "CompID", self.compID, "Emission component ID" )
        self.fittingpatch.add_header( "CompName", self.name, "Emission component" )
        self.fittingpatch.add_header( "Aim", "Fitting", "MCMC Usage" )
        self.fittingpatch.add_header( "BUNIT", "K", "[Kelvin] Data unit" )
        self.logger.info( "Fitting grid is all set!" )
        
    def _set_skypatch_grid( self ):
        self.skypatch = self._set_sky( 'patch', fitting=False )
        self.skypatch.add_header( "CompID", self.compID, "Emission component ID" )
        self.skypatch.add_header( "CompName", self.name, "Emission component" )
        self.skypatch.add_header( "BUNIT", "K", "[Kelvin] Data unit" )
        self.logger.info( "Skypatch grid is all set!" )
    
    def _set_stage( self ):
        self._stage = {'inmap':{'0':'input map', '1':[]},
                       'idmap':{'0':'input spectral index map','1':[]},
                       'mmap':{'0':'masked map', '1':[]}, 
                       'mfmap': {'0': 'mask-filled map', '1':[]}, 
                       'upmap':{'0':'upgraded map', '1':[]}, 
                       'upidmap':{'0':'upgraded spectral index map', '1':[]},
                       'grfmap':{'0':'Gaussian Random Field map', '1':[]}, 
                       'ssmap':{'0':'small scale map', '1':[]}, 
                       'hpmap':{'0':'upgraded map w/ small scale', '1':[]}, 
                       'simmap':{'0':'Final simulated map', '1':[]} } # map stage
        self.mfmap = None # mask-filled map
        self.upmap = None # upgraded map
        self.upidmap = None # upgraded spectral index map
        self.ssmap = None # small scale map
        self.hpmap = None # upgraded map w/ small scale (intermediate)
        self.simmap = None # Final simulated map
    
    def _set_grf( self ):
        if self.grfmap is not None:
            if self.grfpatch:
                self.input_gss_fitmap = self.grfmap['fit']
                self.logger.info( "Input custom Gaussian Random Field patch for map fitting." )
                self.input_gss_simmap = self.grfmap['sim']
                self.logger.info( "Input custom Gaussian Random Field patch for map simulation." )
                self.grftag = True # Grf tag
            else:
                self.logger.info( 'Input custom Gaussian Random Field.' )
                nside_grf = self.ncheck( self.grfmap )
                if nside_grf != self.nside:
                    raise ValueError( "Nside of the input grfmap is not competible w/ the aimed nside!" )
                    self.logger.error( "Nside of the input grfmap is not competible w/ the aimed nside!" )
                self._stage['grfmap']['1'].append( "grfmap is inputed by the user w/ nside %s w/ default RING order." % nside_grf )
                self.grftag = True # Grf tag
        else:
            self.grftag = False # Grf tag
    
    def _gen_default_freqs( self ):
        bands = {}
        bands['hig'] = genfz( 'f', 188, 0.16, 196, print_=False )
        bands['mid'] = genfz( 'f', 154, 0.16, 162, print_=False )
        bands['low'] = genfz( 'f', 120, 0.16, 128, print_=False )
        return bands
    
    def _set_default_io( self ):
        if self.type_=='Gsync':
            self.prefix = "gsync"
        elif self.type_ == "Gfree":
            self.prefix = "gfree"
        self.output_dir = "/home/cxshan/data/fg21sim+/GalacticSimData/"
        self.filename_pattern = "{prefix}_{frequency:06.2f}.fits"
        self.clobber = True       # Overwrite exiting
        self.logger.info( "Default IO is set. Output dir is %s, clobber is %s." % ( self.output_dir, self.clobber ) )

    def _gen_pix_grid( self, center, FoV, Psize ):
        """
        *** Gen a pix grid using  ***
        """
        ra_FoV, dec_FoV = FoV
        ra_c, dec_c = center
        PsizeDeg = Psize / 3600
        ra_size = int( ra_FoV / PsizeDeg )
        dec_size = int( dec_FoV / PsizeDeg )
        size = ( ra_size, dec_size )             # Int
        pixelsize = Psize                        # Arcsec
        return size, pixelsize, center
    
    def _set_sky( self, skytype, fitting=True ):
        """
        *** Set the skybase for patch & fullsky based on skytype. ***
        """
        kwargs = {
            "float32": True,
            "clobber": False,
            "checksum": False,
        }
        if skytype=='patch':
            if fitting:
                size, pixelsize, center = self._gen_pix_grid(self.center, self.fov, self.resolution.value)
            else:
                size, pixelsize, center = self._gen_pix_grid(self.center, self.fov, self.sim_pixel)
            return SkyPatch(size=size, pixelsize=pixelsize,
                        center=center, **kwargs)
        elif skytype=='healpix':
            return SkyHealpix(nside=self.nside, **kwargs)
    
    def _get_sky_patch( self, temp_map, temp_header, skyobject, merge=False ):
        """
        *** Get sky patch from healpix map ***
        +++ Add fits history +++
        """
        nested = temp_header["ORDERING"].upper() == "NESTED"
        try:
            coordsys = temp_header["COORDSYS"]
        except KeyError:
            logger.warning( "No 'COORDSYS' keyword for map %s" % temp_map )
            logger.warning( "Assume to use the 'Galactic' coordinate" )
            coordsys = "Galactic"
        
        patch_image, __ = reproject_from_healpix( input_data = ( temp_map, coordsys ),
                                       output_projection = skyobject.wcs,
                                       shape_out = skyobject.shape, nested = nested ) 
        if merge:
            skyobject.merge_header( temp_header.copy( strip=True ) )
            skyobject.data = patch_image
            #skypatch.add_history(" ".join(sys.argv))
            #outname = self.outfile_pattern.format(prefix = outfile_prefix, z=z, f=f)
            #sky.write(outfile)
            #logger.info("Written extracted sky patch to file: %s" % outfile)
        return patch_image
    
    def show_g_coord( self ):
        ra_c = self.center[0]
        dec_c = self.center[1]
        c_icrs = SkyCoord(ra=ra_c*au.degree, dec=dec_c*au.degree, frame='icrs')
        g = c_icrs.galactic
        self.logger.info( "The center of the simulation in Galactic Coordinate is: %s." % g )
    
    def smooth_patch( self, sigma_npix, target='skypatch' ):
        """
        sigma_npix, target='skypatch'
        """
        self.logger.info( "Smoothing the %s with a Gaussian filter ..." % target )
        if target is 'skypatch':
            patch_image = self.skypatch.data
            pixelsize = self.sim_size
        elif target is 'fittingpatch':
            patch_image = self.fittingpatch.data
            pixelsize = self.resolution.value
        sigma = ( sigma_npix * self.sigma_tem * 60.0 / pixelsize )
        smoothed = scipy.ndimage.gaussian_filter( patch_image, sigma=sigma )
        self.logger.info("Smoothed sky patch using Gaussian filter of " +
                    "sigma = %.2f [pixel]" % sigma)
        return smoothed
    
    def smooth_sky( self, sigma_npix, skyobject ):
        """
        *** General sky smooth method ***
        sigma_npix, skyobject
        """
        self.logger.info( "Smoothing the %s with a Gaussian filter ..." % skyobject )
        patch_image = skyobject.data
        pixelsize = skyobject.pixelsize
        sigma = ( sigma_npix * self.sigma_tem * 60.0 / pixelsize )
        smoothed = scipy.ndimage.gaussian_filter( patch_image, sigma=sigma )
        self.logger.info("Smoothed sky patch using Gaussian filter of " +
                    "sigma = %.2f [pixel]" % sigma)
        return smoothed
    
    # cl fittings
    def gen_fitcl( self ):
        """
            *** Generate cls for gamma fitting. ***
        """
        if self._fitcl is None:
            self._fitcl = {}
            if self.inmap_nan:
                clmap = self.mfmap
                self.logger.info( ' The cls is generated from the mask filled input map. ' )
            else:
                clmap = self.inmap
                self.logger.info( ' The cls is generated from the input map. ' )
            self._fitcl['full'] = self.map2cl( clmap )
            self.plotcl( self._fitcl['full'] )
            self._fitcl['cl'], self._fitcl['l'] = self.cutcl( self._fitcl['full'], self.fitlmin, self.fitlmax )
            self.logger.info( ' Cut cls in range of ( %s, %s ). ' % ( self.fitlmin, self.fitlmax ) )
        else:
            self.logger.info( " Fitting maps were generated prior, if you want to generate new sets, using re_gen_fitcl instead. " )
            
    def re_gen_fitcl( self ):
        self._fitcl = None
        self.gen_fitcl()
        self.logger.info( " Fitting cls are regenerated. " )
    
    # Map_cl
    def map2cl_dc( self, healpixmap, nthreads, spin=0 ):
        """
        It is broken right now, please fix this!
        """
        nside = self.ncheck( healpixmap )
        lmax = 3 * nside - 1
        mmax = lmax
        
        hpmap64 = np.float64(healpixmap)
        rsmap = hpmap64.reshape((1,-1))
        inmap = rsmap
        base = dhp.Healpix_Base( nside, self.order )
        geom = base.sht_info()
        alm_out = dsh.experimental.adjoint_synthesis(lmax=lmax, spin=spin, map=inmap, nthreads=nthreads, **geom)
        cl_out = alm2cl( alm_out, lmax=lmax, mmax=mmax )
        return cl_out, alm_out
    
    # Map_cl
    def map2cl( self, healpixmap ):
        nside = self.ncheck( healpixmap )
        lmax = 3 * nside - 1
        mmax = lmax
        cl_out = hp.anafast( healpixmap, lmax=lmax, mmax=mmax )
        return cl_out
    
    # Plot_cl
    def plotcl( self, cl, cl_=True, log=True ):
        r"""
        *** plot cls w/ log option ***

        :Params cl: input cl;
        :Params cl_: plot only cl;
        :Params log: True for setting matplotlib.pyplot using log scale;
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
    
    # Get cl for fitting
    def cutcl( self, cl, lmin, lmax ):
        data = {}
        ell = np.arange(len(cl))
        data['cl'] = cl[lmin:lmax]
        data['l'] = ell[lmin:lmax]
        return data['cl'], data['l']
    
    def _set_up_cl_fitting( self ):
        if self.fitlmin is None or self.fitlmax is None:
            self.fitlmin = 30
            self.fitlmax = 90
            self.logger.info( 'The lmin or lmax is not set, using default value: (%s,%s)' % ( self.fitlmin, self.fitlmax ) )
        self.gen_fitcl()
        self.logger.info( "cl fitting are ready" )
    
    def _run_MCMC_cl( self, nsteps=20000, mult=False, nc=10, regen=False ):
        
        self._set_up_cl_fitting()
        if regen:
            self.re_gen_fitcl()
        
        IP = True
        nwalkers, ndim = 20, 2
        param_len, param_num = 1, 2
        np.random.seed(42)
        initial_params_cl = np.array([-2.2203067, 3.403917092]) + 1e-3 * np.random.randn(nwalkers, ndim)
        param_lim_cl = ( (-20, 20), (-20, 20) )

        namecl = self.type_ + '_' + 'cl_fitting_' + str( self.nside ) + '_lmin' + str(self.fitlmin) + '_lmax' + str(self.fitlmax) + \
                 '_nstep' + str( nsteps ) + '_IP_' + str(IP) + '_lim' + str(param_lim_cl[0][0]) + '_' + str(param_lim_cl[0][1])

        fitting_cl = Fitfunc()
        lh = fitting_cl.cl_likelihood
        fitting_cl.set_cl( cl = self._fitcl['cl'], l = self._fitcl['l'])

        Clsampler = MCMC(nwalkers, nsteps, ndim, param_num, param_len, param_lim_cl, initial_params_cl, lim_=True, multi_=mult, thread_num=nc, param_names=['gamma','Constant'], vars_list=[lh])
        Clsampler.run_MCMC()
        Clsampler.param_chain( 10, 1 )
        Clcoeff = Clsampler.param_fit_result()
        gamma_result = Clcoeff[0]
        gamma = Clcoeff[0][1]
        self.logger.info( "Gamma best fit is %s." % gamma )

        if self._params_set_up is None:
            self._set_up_params
        else:
            self._mod_params( gamma_result, p='gamma' )
        Clsampler.param_result_plot( legend="Result" )
        self._clsampler = Clsampler

    def mod_MCMC_cl( self, discard, thin, mod=False ):
        self.clsampler.param_chain( discard, thin )
        Clcoeff = self.clsampler.param_fit_result()
        gamma_result = Clcoeff[0]
        gamma = Clcoeff[0][1]
        self.logger.info( "Gamma best fit is %s." % gamma )
        self.clsampler.param_result_plot( legend="Result" )
        if mod:
            self._mod_params( gamma_result, p='gamma' )
    
    @property
    def clsampler( self ):
        return self._clsampler
    
    # map fittings
    def _set_up_map_fitting( self, nc=20 ):
        """
        +++ Add memory estimation for GRFmap generation. +++
        +++ Move to a consistent regen logic for all func: genalmcl, gaussianmap, gen_fitmap (Also, fix the names!) +++
        """
        if self.grftag and self.grfpatch:
            # Grf is input by user
            self.gen_fitmap()
        else:
            # Grf could be input by the user, but input is not all set.
            if self.grftag:
                self.gen_fitmap()
            else:
                if self.extra_large_mode_:
                    self.logger.warning( " Extra_large_mode is on! You are simulating an extrmely large nside, do you really want to generate the gaussian map? " )
                if self.grfmap is None:
                    self.logger.ino( "No input grfmap, generating new one!" )
                    self.genalmcl()
                    self.logger.info( "Using Gamma=%s to genrate cls. Is this your prefered param? If not, reset gamma using set_gamma()." % self.gamma )
                    self.gaussianmap( nthreads=nc )
                else:
                    self.logger.info( " Exiting map is ready to go, no need to regenerate the it! " )
                self.gen_fitmap()

    def viewslice( self, patch ):
        imageview = plt.imshow( patch, origin='lower',cmap=plt.cm.RdYlBu, aspect='auto')
        plt.colorbar(imageview)
        plt.xlabel('Right ascension (ICRS)')
        plt.ylabel('Declination (ICRS)')
        plt.show()

    def gen_fitmap( self ):
        """
        self.grftag and self.grfpatch controls the input of gssmap. gen_fitmap() will no longer compute grf data.
        """
        if self._fitmap is None:    
            self._fitmap = {}
            
            if self.grftag and self.grfpatch:
                self.grf_patch = self.input_gss_fitmap
            else:
                tic = timeit.default_timer()
                self.grf_patch = self._get_sky_patch( temp_map=self.grfmap, temp_header=self.inheader, skyobject=self.fittingpatch, merge=False )
                toc = timeit.default_timer()
                self.logger.info(' Finish generating the GRF patch image for map fitting, using %s s.' % (toc - tic))
            
            self._fitmap['gss'] = self.whiten( self.grf_patch )
            self.viewslice( self._fitmap['gss'] )
            
            if self.upmap is None:
                if self.inmap_nan:
                    inmap = self.mfmap
                    self.logger.info( 'Using mask filled map to generate map fitting image.' )
                else:
                    inmap = self.inmap
            else:
                inmap = self.upmap
                
            tic = timeit.default_timer()
            self.map_patch = self._get_sky_patch( temp_map=inmap, temp_header=self.inheader, skyobject=self.fittingpatch, merge=True )
            toc = timeit.default_timer()
            self.logger.info(' Finish generating the orignal patch image for map fitting, using %s s.' % (toc - tic))
            self._fitmap['ori'] = self.map_patch
            self.viewslice( self._fitmap['ori'] )
        else:
            self.logger.info( " Fitting maps were generated prior, if you want to generate new sets, using re_gen_fitmap instead. " )
            
    def re_gen_fitmap( self ):
        self._fimap = None
        self.gen_fitmap()
        self.logger.info( " Fitting maps are regenerated. " )

    def _run_MCMC_map( self, nsteps=500, mult=True, nc=20, regen=False ):

        self._set_up_map_fitting( nc=nc )
        if regen:
            self.re_gen_fitmap()

        IP = True
        nwalkers, ndim = 60, 2
        param_len, param_num = 1, 2
        np.random.seed(42)
        initial_params_map = np.array([0.0599, 0.782]) + 1e-3 * np.random.randn(nwalkers, ndim)
        param_lim_map = ( (0, 2), (0, 2) )

        namemap = self.type_ + '_map_fitting_' + str( self.nside ) + '_partial_' + str(self.fov[0]) + '_center_' + str(self.center[0]) + \
                str(self.center[1]) + '_nstep' + str( nsteps ) + '_IP_' + str(IP) + '_lim' + str(param_lim_map[0][0]) + '_' + str(param_lim_map[0][1])

        fitting_map = Fitfunc( size=self.fittingpatch.shape[0] )
        combine = fitting_map.combine

        fitting_map.set_map( gss = self._fitmap['gss'].flatten(), ori = self._fitmap['ori'].flatten() )

        Mapsampler = MCMC(nwalkers, nsteps, ndim, param_num, param_len, param_lim_map, initial_params_map, lim_=True, multi_=mult, thread_num=nc, param_names=['gamma','Constant'], vars_list=[combine])
        Mapsampler.run_MCMC()
        Mapsampler.param_chain( 10, 1 )
        Mapcoeff = Mapsampler.param_fit_result()
        alpha_result = Mapcoeff[0]
        alpha = Mapcoeff[0][1]
        self.logger.info( "Alpha best fit is %s." % alpha )
        beta_result = Mapcoeff[1]
        beta = Mapcoeff[1][1]
        self.logger.info( "Beta best fit is %s." % beta )

        if self._params_set_up is None:
            self._set_up_params
        else:
            self._mod_params( alpha_result, p='alpha' )
            self._mod_params( beta_result, p='beta' )
        Mapsampler.param_result_plot( legend="Result" )
        self._mapsampler = Mapsampler

    def mod_MCMC_map( self, discard, thin, smooth=False, mod=False ):
        Mapsampler.param_chain( discard, thin )
        Mapcoeff = Mapsampler.param_fit_result()
        alpha_result = Mapcoeff[0]
        alpha = Mapcoeff[0][1]
        self.logger.info( "Alpha best fit is %s." % alpha )
        beta_result = Mapcoeff[1]
        beta = Mapcoeff[1][1]
        self.logger.info( "Beta best fit is %s." % beta )
        Mapsampler.param_result_plot( smooth=smooth, legend="Result" )

        if mod:
            self._mod_params( alpha_result, p='alpha' )
            self._mod_params( beta_result, p='beta' )
            
    @property
    def mapsampler( self ):
        return self._mapsampler
    
    # Frequency simulation option
    
    def _set_up_freq( self ):
        """
        +++ Add patch & fullsky flag. Rightnow, we only have patch stuff. +++
        """
        self.gen_simmap()
        self._set_physical_params()
    
    def gen_simmap( self ):
        """
        Gen the patch images for patch simulation.
        """
        if self.inmap_nan:
            inmap = self.mfmap
            self.logger.info( 'Using mask filled map to generate simulation image.' )
        else:
            inmap = self.inmap
        tic = timeit.default_timer()
        self.patch_map = self._get_sky_patch( temp_map=inmap, temp_header=self.inheader, skyobject=self.skypatch, merge=True )
        toc = timeit.default_timer()
        self.logger.info(' Finish generating the original image for simulation, using %s s.' % (toc - tic))
        
        if self.grftag and self.grfpatch:
            self.patch_gss = self.input_gss_simmap # gss maps are set by _set_grf()
            self.logger.info( 'Input patch flag detected. Using user input GRF patch for simulation.' )
        else:
            tic = timeit.default_timer()
            self.patch_gss = self._get_sky_patch( temp_map=self.grfmap, temp_header=self.inheader, skyobject=self.skypatch, merge=False )
            toc = timeit.default_timer()
            self.logger.info(' Finish generating the GRF patch image for simulation, using %s s.' % (toc - tic))
        self.patch_gss = self.whiten( self.patch_gss )
        if self.type_ is "Gsync":
            tic = timeit.default_timer()
            self.patch_id = self._get_sky_patch( temp_map=self.idmap, temp_header=self.inheader, skyobject=self.skypatch, merge=False )
            toc = timeit.default_timer()
            self.logger.info(' Finish generating the spectral index patch image for simulation, using %s s.' % (toc - tic))
        elif self.type_ is "Gfree":
            self.patchid = None
    
    def add_ss( self, inmap ):
        self.logger.info( "The small scale map is generated using alpha=%s, beta=%s." % ( self.alpha, self.beta ) )
        whiten_gss = self.whiten(self.patch_gss)
        patch_ss = self.alpha * whiten_gss * inmap ** self.beta
        add_ss_patch_map = inmap + patch_ss
        return add_ss_patch_map, patch_ss
    
    def _outfilepath(self, frequency, **kwargs):
        """
        Generate the path/filename to the output file for writing
        the simulate sky images.
        Parameters
        ----------
        frequency : float
            The frequency of the output sky image.
            Unit: [MHz]
        Returns
        -------
        filepath : str
            The generated filepath for the output sky file.
        """
        filename = self.filename_pattern.format(
            prefix=self.prefix, frequency=frequency, **kwargs)
        filepath = os.path.join(self.output_dir, filename)
        return filepath
    
    def simulate_frequency( self, frequency, skytype, smooth=False, sigma_npix=0.3 ):
        """
        Simulate the free-free map at the specified frequency.
        Parameters
        ----------
        sigma_npix: default 0.3 for self.use_ns512=True;
        frequency : float
            The frequency where to simulate the emission map.
            Unit: [MHz]
        Returns
        -------
        sky : `~SkyBase`
            The simulated sky image as a new sky instance.
        +++ Add smooth for healpix grid +++
        """
        
        self.logger.info("Simulating {name} map at {freq:.2f} [MHz] ...".format(
            name=self.name, freq=frequency))
        
        if self.type_ is "Gfree":
            self.temp_ratio_K_R = self._calc_halpha_to_freefree( self.gfree_temp_freq )
            ratio_K_R = self._calc_halpha_to_freefree(frequency)
            if skytype=='patch':
                sky = self.skypatch.copy()
                sky.data = self.skypatch.data * ratio_K_R / self.temp_ratio_K_R
                if smooth:
                    smoothed = self.smooth_sky( sigma_npix, sky )
                    added, ss = self.add_ss( smoothed ) # Add small scales
                    sky.data = added
                else:
                    added, ss = self.add_ss( sky.data ) # Add small scales
                    sky.data = added
                sky.frequency = frequency
            elif skytype=='healpix':
                sky = self.sky.copy()
                sky.data = self.upmap * ratio_K_R / self.temp_ratio_K_R
                ssmap = self.gen_smallscales()
                sky.data = sky.data + ssmap # Add small scales
                sky.frequency = frequency
            else:
                print("Wrong skytype!")
            self.logger.info("Done simulate map at %.2f [MHz]." % frequency)
            
        elif self.type_ == "Gsync":
            ff = frequency / self.gsync_temp_freq
            if skytype == 'patch':
                sky = self.skypatch.copy()
                sky.data = self.skypatch.data * ff ** (-np.abs(self.patch_id))
                if smooth:
                    smoothed = self.smooth_sky( sigma_npix, sky )
                    added, ss = self.add_ss( smoothed ) # Add small scales
                    sky.data = added
                else:
                    added, ss = self.add_ss( sky.data ) # Add small scales
                    sky.data = added
                sky.frequency = frequency
            elif skytype == 'healpix':
                sky = self.sky.copy()
                sky.data = self.upmap * ff ** (-np.abs(self.upidmap))
                ssmap = self.gen_smallscales()
                sky.data = sky.data + ssmap # Add small scales
                sky.frequency = frequency
            else:
                print("Wrong skytype!")
            self.logger.info("Done simulate map at %.2f [MHz]." % frequency)
        return sky
    
    def simulate(self, frequencies=None, skytype='patch', smooth=False):
        """
        Simulate the emission maps.
        Parameters
        ----------
        frequencies : float, or list[float]
            The frequencies where to simulate the emission map.
            Unit: [MHz]
            Default: None (i.e., use ``self.frequencies``)
        Returns
        -------
        skyfiles : list[str]
            List of the filepath to the written sky files
        """
        if frequencies is None:
            frequencies = self.frequencies
        else:
            frequencies = np.array(frequencies, ndmin=1)

        self.logger.info("Simulating {name} ...".format(name=self.name))
        skyfiles = []
        for freq in frequencies:
            sky = self.simulate_frequency(freq, skytype, smooth)
            outfile = self._outfilepath(frequency=freq)
            sky.write(outfile)
            skyfiles.append(outfile)
        self.logger.info("Done simulate {name}!".format(name=self.name))
        return skyfiles
    
    # Params
    
    def _set_default_params( self ):
        self.logger.info( ' Default settings are used! ' )
        if self.type_=='Gsync':
            alpha = 0.03035716
            self.set_alpha( alpha )
            beta = 0.23375195
            self.set_beta( beta )
            gamma = -2.06863485792474
            self.set_gamma( gamma )
            self.sigma_tem = 56 # unit[arcmin]
        elif self.type_=='Gfree':
            alpha = 0.01019949
            self.set_alpha( alpha )
            beta = 0.97270131
            self.set_beta( beta )
            gamma = -2.42875121
            self.set_gamma( gamma )
            self.sigma_tem = 6 # unit[arcmin]
            
    def _set_up_params( self ):
        """
        *** steup the storage of the best fit params ***
        """
        params = {}
        if self.default_:
            params['alpha'] = { '0':[ self.alpha ], '1':[ 'Alpha is set by default.' ], 'user':[], 'info':[] }
            params['beta'] = { '0':[ self.beta ], '1':[ 'Beta is set by default.' ], 'user':[], 'info':[] }
            params['gamma'] = { '0':[ self.gamma ], '1':[ 'Gamma is set by default.' ], 'user':[], 'info':[] }
            
        else:
            params['alpha'] = { '0':[], '1':[], 'user':[], 'info':[] }
            params['beta'] = { '0':[], '1':[], 'user':[], 'info':[] }
            params['gamma'] = { '0':[], '1':[], 'user':[], 'info':[] }
        self._params = params
        self._params_set_up = True
        self.logger.info( " The params storage is setup! " )

    def _mod_params( self, x, p='gamma'):
        if p is 'gamma':
            d = datetime.now()
            dd = d.strftime('%m_%d_%Y_%H%M%S')
            self._params['gamma']['0'].append( x )
            self._params['gamma']['1'].append( '%s = %s is updated @ %s.' % ( p, x, dd ) )
            self.logger.info( '%s = %s is updated @ %s.' % ( p, x, dd ) )
        elif p is 'alpha':
            d = datetime.now()
            dd = d.strftime('%m_%d_%Y_%H%M%S')
            self._params['alpha']['0'].append( x )
            self._params['alpha']['1'].append( '%s = %s is updated @ %s.' % ( p, x, dd ) )
            self.logger.info( '%s = %s is updated @ %s.' % ( p, x, dd ) )
        elif p is 'beta':
            d = datetime.now()
            dd = d.strftime('%m_%d_%Y_%H%M%S')
            self._params['beta']['0'].append( x )
            self._params['beta']['1'].append( '%s = %s is updated @ %s.' % ( p, x, dd ) )
            self.logger.info( '%s = %s is updated @ %s.' % ( p, x, dd ) )

    def user_change_params( self, x, p='gamma'):
        if p is 'gamma':
            d = datetime.now()
            dd = d.strftime('%m_%d_%Y_%H%M%S')
            self._params['gamma']['user'].append( x )
            self._params['gamma']['info'].append( '%s = %s is updated @ %s.' % ( p, x, dd ) )
            self.logger.info( '%s = %s is changed by the user @ %s.' % ( p, x, dd ) )
        elif p is 'alpha':
            d = datetime.now()
            dd = d.strftime('%m_%d_%Y_%H%M%S')
            self._params['alpha']['user'].append( x )
            self._params['alpha']['info'].append( '%s = %s is updated @ %s.' % ( p, x, dd ) )
            self.logger.info( '%s = %s is changed by the user @ %s.' % ( p, x, dd ) )
        elif p is 'beta':
            d = datetime.now()
            dd = d.strftime('%m_%d_%Y_%H%M%S')
            self._params['beta']['user'].append( x )
            self._params['beta']['info'].append( '%s = %s is updated @ %s.' % ( p, x, dd ) )
            self.logger.info( '%s = %s is changed by the user @ %s.' % ( p, x, dd ) )

    def set_gamma( self, gamma ):
        self._gamma = gamma
        self.logger.info( " gamma is set as %s " % gamma )

    def set_alpha( self, alpha ):
        self._alpha = alpha
        self.logger.info( " alpha is set as %s " % alpha )

    def set_beta( self, beta ):
        self._beta = beta
        self.logger.info( " beta is set as %s " % beta )

    @property
    def params( self ):
        return self._params
    
    @property
    def gamma( self ):
        return self._gamma
    
    @property
    def alpha( self ):
        return self._alpha
    
    @property
    def beta( self ):
        return self._beta
    
    # SHT & Method
    
    def genalmcl( self, lmin=1 ):
        r"""
            *** Generate the angular power spectrum of the Gaussian random field with default settings ***
            !!! If you want custom_ lmax please call genGuass_cl from Gtools !!!
            !!! The Guassian Random Field should be in Radian !!!
            Check the reference: https://ui.adsabs.harvard.edu/abs/2013A%26A...553A..96D/abstract
            :Params lmin: multipole min; # Should default as 1!
            :Params from self: type_, lmax, gamma, sigma_tem, & sigma_sim [if type_='Gff'];
            :Output almcl: dict contains alm & cl; 
                           almcl['alm']: return alm of a Gaussian random field;
                           almcl['cl']: return the generated angular power spectrum;
        """
        if self.nside is None:
            self.logger.error( ' Nside currently is %s, must be set prior!' % self.nside )
            raise ValueError

        # Set up alm
        almcl = {}
        # Set the lmax & mmax
        lmax = self.lmax
        mmax = lmax
        nalm = ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)
        
        # Prepare gamma & sigma
        gamma = self.gamma  # index of the power spectrum between l [30, 90]
        _sigma_tem = self.sigma_tem  # original beam resolution of the template [ arcmin ]
        sigma_tem = self.arcsec2rad( _sigma_tem * 60 ) # [ au.rad ]
        
        # angular power spectrum of the Gaussian random field
        ell = np.arange(lmax+1, dtype=int)
        cl = np.zeros(ell.shape)
        ell_idx = ell >= lmin
        if self.type_=='Gsync':            
            cl[ell_idx] = ell[ell_idx] ** gamma * ( 1.0 - np.exp(-ell[ell_idx]**2 * sigma_tem.value**2) )
        elif self.type_=='Gfree':
            sigma_sim = self.sigma_sim  # simulated beam resolution [ au.rad ]
            cl[ell_idx] = ell[ell_idx] ** gamma * ( np.exp(-ell[ell_idx]**2 * sigma_sim.value**2) - np.exp(-ell[ell_idx]**2 * sigma_tem.value**2) )
        cl[ell < lmin] = cl[lmin]
        
        # convert cl to alm
        alm = cl2alm(cls=cl,lmax=lmax, mmax=mmax)
        
        self.logger.info( ' Generate alm & cl for Galactic %s simulation. ' % self.type_ )
        self.logger.info( ' %s alpha is %s ' % ( self.type_, self.alpha ) )
        self.logger.info( ' %s beta is %s ' % ( self.type_, self.beta ) )
        self.logger.info( ' %s gamma is %s ' % ( self.type_, self.gamma ) )
        self.logger.info( ' %s sigma_tem is %s ' % ( self.type_, self.sigma_tem ) )
        self.logger.info( ' %s sigma_sim is %s ' % ( self.type_, self.sigma_sim ) )
        self.logger.info( ' lmax, mmax, nalm is %s, %s, & %s ' % ( lmax, mmax, nalm ) )
        
        self._hpinfo['lmax'] = lmax
        self._hpinfo['mmax'] = mmax
        self._hpinfo['nalm'] = nalm
        
        # Modify the Class attribute
        self._alm = alm
        self._cl = cl
        
        # Gen output for the function
        almcl['alm'] = alm
        almcl['cl'] = cl
        return almcl
    
    def gaussianmap( self, nthreads, spin=0 ):
        r"""
            *** gaussianmap (a special copy of alm2map) using ducc0 in a healpix grid ***
            :Params alm: spherical harmonics coefficients; 
            :Params nside: nside of the extended map;
            :Params nthreads: number of threads;
            :Params spin: spin of the alm, default spin = 0;
            :Params test: test type;
            ::::::::::::: test == 0 using a RNG distribution alm;
            ::::::::::::: test == 1 using a custom alm;
            :Output hpmap: healpix map;
        """
        # Multi-threading
        allthreads = active_count()
        self.logger.info("alm2map supports multi-threading, use %s out of total %s threads." % (nthreads, allthreads))
        # Set the lmax & mmax
        lmax = self.lmax
        mmax = lmax
        alm = self.alm.copy()
        inalm = alm.reshape( (1,-1) ) # reshape the alm to dhp standard
        base = dhp.Healpix_Base( self.nside, self.order )
        geom = base.sht_info()

        # test adjointness between synthesis and adjoint_synthesis
        _grfmap = dsh.experimental.synthesis(alm=inalm, lmax=lmax, spin=spin, nthreads=nthreads, **geom)
        shape = _grfmap.shape
        grfmap = _grfmap.reshape(shape[1],)
        
        self._hpinfo['nside'] = self.nside
        self._hpinfo['order'] = self.order
        self._hpinfo['spin'] = spin
        self.logger.info( "grfmap is generated w/ nside=%s, lmax=%s, spin=%s, order=%s" % ( self.nside, self.lmax, spin, self.order ) )
        self._stage['grfmap']['1'].append( "grfmap is generated w/ nside=%s, lmax=%s, spin=%s, order=%s" % ( self.nside, self.lmax, spin, self.order ) )
        # Modify the Class attribute
        self.grfmap = grfmap
        return grfmap
    
    # ====== Maps ======
    
    def loadmap( self, server='gravity', use_ns512=True, use_G02=True ):
        r"""
            extra_large_mode_: if on, do not auto update maps >> control large Nside simulation, the update is not nessary for patch simulation;
            use_ns512: if on, use ns512 map. 2022-08-11 add use_ns512 flag;
            *** Load the default files on the computers ***
            +++ Add other machines such as SGI +++
            
            >> All healpy routines assume RING ordering, 
            in fact as soon as you read a map with read_map, 
            even if it was stored as NESTED, 
            it is transformed to RING. 
            However, you can work in NESTED ordering passing 
            the nest=True argument to most healpy routines.

            > https://healpy.readthedocs.io/en/latest/tutorial.html#:~:text=was%20stored%20as-,NESTED,-%2C%20it%20is%20transformed
        """
        self.server = server
        self.use_ns512 = use_ns512
        
        if server == 'gravity':
            if self.type_ == 'Gsync':
                datadir = "/home/cxshan/radiodata/"
                if use_ns512:
                    fn = "haslam408_dsds_Remazeilles2014_ns512.fits"
                else:
                    fn = "haslam408_dsds_Remazeilles2014_ns2048.fits"
                if use_G02:
                    idn = "GsyncSpectralIndex_Giardino2002_ns2048.fits"
                else:
                    idn = "synchrotron_specind2_ns512.fits"
            elif self.type_ == 'Gfree':
                datadir = "/home/cxshan/fg21sim+/GalacticData/gfree/"
                fn = "gfree_120.0.fits"
        elif server == 'sgi':
            if self.type_ == 'Gsync':
                datadir = "/mnt/ddnfs/data_users/cxshan/radiodata/"
                if use_ns512:
                    fn = "haslam408_dsds_Remazeilles2014_ns512.fits"
                else:
                    fn = "haslam408_dsds_Remazeilles2014_ns2048.fits"
                if use_G02:
                    idn = "GsyncSpectralIndex_Giardino2002_ns2048.fits"
                else:
                    idn = "synchrotron_specind2_ns512.fits"
            elif self.type_ == 'Gfree':
                datadir = "/mnt/ddnfs/data_users/cxshan/radiodata/GalacticData/gfree/"
                fn = "gfree_120.0.fits"
        else:
            print('Map location unknow! You should use gopen()')
        
        # Load the initial map
        fname = datadir + fn
        inmap, inheader = read_fits_healpix( fname )
        self.inmap = inmap
        self.inheader = inheader
        inmap_nside = self.ncheck( inmap )
        self.logger.info( "Input map is loaded from %s on %s server w/ nside=%s in RING order" % ( fname, server, inmap_nside ) )
        self._stage['inmap']['1'].append( "Input map is loaded from %s on %s server w/ nside=%s in RING order" % ( fname, server, inmap_nside ) )
        
        # Check nan
        inmap_nan = self.checknan( inmap )
        self.inmap_nan = inmap_nan
        if inmap_nan:
            self.logger.info( 'Input map contains NaN pixels, filtering the input map. ' )
            self.mmap = self.gopen_mask( datadir, fn )
            self.logger.info( "Input map is nan masked" )
            self._stage['mmap']['1'].append( "Input map is nan masked" )
            self.fillmap() # Fill the nan-masked map with 0
        
        # Load idmap
        if self.type_ == 'Gsync':
            idname = datadir + idn
            idmap, idheader = read_fits_healpix( idname ) # Default order is always RING;
            self.idmap = idmap
            self.idheader = idheader
            idmap_nside = self.ncheck( idmap )
            self.logger.info( "Input spectral index map is loaded from %s on %s server w/ nside=%s in RING order" % ( idname, server, idmap_nside ) )
            self._stage['idmap']['1'].append( "Input spectral index map is loaded from %s on %s server w/ nside=%s in RING order" % ( idname, server, idmap_nside ) )
            idmap_nan = self.checknan( idmap )
            if idmap_nan:
                self.logger.info( ' Input spectral index map contains NaN pixels, filtering the spectral index map. ' )
                idmap = self.gopen_mask( datadir, fn )
                self.logger.info( " Input spectral index map is nan masked. " )
                idmap = idmap.filled( fill_value = 0 )
                self.logger.info( " Input spectral index map's nan pixels are filled with 0. " )
        else:
            idmap = None
            self.logger.info( " Input spectral index map is not supported for type = %s! " % self.type_ )
            self._stage['idmap']['1'].append( " Input spectral index map is not supported for type = %s! " % self.type_ )
        
        if self.extra_large_mode_:
            print( " Please update the healpix maps when you need to. " )
        else:
            if inmap_nside != self.nside:
                self.logger.info( " Extra_large_mode is off, auto update is on" )
                self.update()
        
        return inmap, idmap
     
    def gopen( self, fdir, fn ):
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

    def gopen_mask( self, fdir, fn ):
        r"""
        *** Open the file map w/ hp.READ_MAP & mask all the nan ***
        !!! Check out the hp.READ_MAP instructuion !!!
        :Params fdir: path to the file;
        :Params fn: file name;
        :Output mask: masked map in healpix `RING` scheme;
        """
        file = fdir + fn
        x = hp.read_map(file)
        masked = np.ma.masked_array(x, np.isnan(x))
        return masked
    
    def maskmap( self, mask_='nan' ):
        r"""
            *** Mask nan from map *** # Utility
            :Output masked: maskedmap
        """
        x = self.inmap.copy()
        if maks_ == 'nan':
            masked = np.ma.masked_array(x, np.isnan(x))
        else:
            print('Mask your own mask or wait for the update!')
        self.mmap = masked
        
        self.logger.info( "Input map is nan masked" )
        self._stage['mmap']['1'].append( "Input map is nan masked" )
        
        return masked
    
    def fillmap( self, value=0 ):
        r"""
            *** Fill the nan-Masked map with value *** # Utility
            :Output mfmap: mask filled map;
        """
        mfmap = self.mmap.filled( fill_value=value )
        self.mfmap = mfmap
        
        self.logger.info( "Nan-masked map is filled w/ %s" % value )
        self._stage['mfmap']['1'].append( "Nan-masked map is filled w/ %s" % value )
        return mfmap
    
    def checknan( self, inputmap ):
        r"""
            *** Check if a map has any NaN pixels. *** # Utility
        """
        if( np.isnan( inputmap ).any() ):
            nanstatus = True
        else:
            nanstatus = False
        return nanstatus
    
    def ncheck( self, inmap ):
        r"""
        *** Check the nside of the map ***
        """
        ncheck = hp.get_nside(inmap)
        return ncheck
    
    def nchecks( self, map1, map2 ):
        r"""
        *** Check if the nside of two maps matches ***
        """
        n1 = self.ncheck( map1 )
        n2 = self.ncheck( map2 )
        if n1 == n2:
            status = True
        else:
            status = False
        return status
    
    def whiten( self, gss ):
        r"""
        *** Whiten a guass map ***
        """
        gss = (gss - gss.mean()) / gss.std()
        return gss
    
    def update( self, aim_='inmap' ):
        r"""
            *** Update the filled / original map to aimed nside *** # Utility
            :Output upmap: update original or mask filled map;
        """
        if aim_ == 'mmap': # masked map
            _map = self.mmap
        elif aim_ == 'mfmap': # mask-filled map
            _map = self.mfmap
        elif aim_ == 'inmap': # input map
            _map = self.inmap
        else:
            aim_ = 'inmap'
            _map = self.inmap
            self.logger.error( "aim handler %s is not supported!" % aim_ )
            raise ValueError
        
        if self.inmap_nan:
            _map = self.mfmap
            self.logger.info( 'Detected input map NaN status, using mask_filled map instead. ' )
        self.logger.info( "Starting the upgrade of %s." % aim_ )
        self._stage['upmap']['1'].append( "Detected input map NaN status, using mask_filled map instead." )
        
        n = self.ncheck( _map )
        upmap = self.up_method( _map )
        self.upmap = upmap
        self.logger.info( "%s is upgraded from %s to %s." % ( aim_, n, self.nside))
        self._stage['upmap']['1'].append( "%s is upgraded from %s to %s." % ( aim_, n, self.nside) )
        
        self.logger.info( "Starting the upgrade of spectral index map." )
        if self.idmap is None:
            self.logger.info( "Upgrade of spectral index map is not supported for type = %s!" % self.type_ )
            self._stage['idmap']['1'].append( "Upgrade of spectral index map is not supported for type = %s!" % self.type_ )
            upidmap = None
        else:
            upidmap = self.up_method( self.idmap )
            nid = self.ncheck( self.idmap )
            self.idmap = upidmap
            self.logger.info( " Spectral index map is upgraded from %s to %s." % ( nid, self.nside))
            self._stage['upmap']['1'].append( "Spectral index map is upgraded from %s to %s." % ( nid, self.nside) )
        return upmap, upidmap
    
    def up_method( self, lowresmap ):
        n = self.ncheck(lowresmap)
        if n < self.nside:
            highresmap = hp.ud_grade(lowresmap, nside_out=self.nside)
            self.logger.info(' Lowres map (Nside = %s) is upgrading to nside %s. ' % ( n, self.nside ))
        elif n == self.nside:
            highresmap = lowresmap
            self.logger.info(' Nside of lowres map %s is the same as aimed nside %s. ' % ( n, self.nside ))
        else:
            highresmap = lowresmap
            self.logger.error(' Nside of lowres map %s is larger than the aimed nside %s! Please reset your nside! ' % ( n, self.nside ))
            raise ValueError
        return highresmap
    
    def gen_smallscales( self, alpha=0.0599, beta=0.782, fitting_=False ):
        r"""
        *** Check the gss & hpmap and generate smallscales temp ***
        :Params alpha: ratio alpha param;
        :Params beta: ratio beta param;
        :Output ssmap: small scale map;
        """
        
        if fitting_ is True:
            _alpha = alpha
            _beta = beta
        else:
            _alpha = self.alpha
            _beta = self.beta
        
        # Whiten the GRFmap
        gss = self.grfmap.copy()
        gss = self.whiten( gss )
        self.logger.info( "A whiten copy of grfmap is generated." )
        
        
        match = self.nchecks( self.grfmap, self.upmap )
        if match == True:
            
            ssmap = _alpha * gss * self.upmap**_beta
            self.logger.info( "ssmap is generated w/ alpha=%s beta=%s" % ( _alpha, _beta ) )
            self._stage['ssmap']['1'].append( "ssmap is generated w/ alpha=%s beta=%s" % ( _alpha, _beta ) )
        else:
            self.logger.error( "!!! Nside Error, please check the Nside." )
            self.logger.error( "grfmap:", self.ncheck( self.grfmap ) )
            self.logger.error( "upmap:", self.ncheck( self.upmap ) )
            ssmap = gss * 0
            self.logger.info( "ssmap = 0 is generated due to nside error." )
            self._stage['ssmap']['1'].append( "ssmap = 0 is generated due to nside error." )
        self.ssmap = ssmap
        return ssmap
    
    def add_smallscales( self ):
        r"""
        *** Check the smallscales & hpmap and add smallscales to hpmap ***
        :Params smallscales: small scale map from a whitened GRF map;
        :Params hpmap: original healpix map;
        :Output addedmap: small scale added hpmap;
        """
        match = self.nchecks( self.ssmap, self.upmap )
        if match == True:
            hpmap = self.ssmap + self.upmap
            self.logger.info( "Small scales are added to hpmap." )
            self._stage['hpmap']['1'].append( "Small scales are added to hpmap." )
        else:
            self.logger.error( "!!! Nside Error, please check the Nside." )
            self.logger.error( "ssmap:", self.ncheck( self.ssmap ) )
            self.logger.error( "upmap:", self.ncheck( self.upmap ) )
            hpmap = self.upmap
            self._stage['hpmap']['1'].append( "hpmap = upmap due to nside error." )
        self.hpmap = hpmap
        return hpmap
    
    def pixsize( self, nside ):
        r"""
            *** nside >> pixel size in rad *** # Basic function
            !!! Inverse of rad2nside() !!! # Important note
            !!! For healpix usage !!! # Important note
            +++  +++ # Improvements
            :Params nside: nside of the map
            :Output pix_rad: pixel size in rad [Astropy Units Quantity]
        """
        pix_rad = ( math.sqrt(math.pi / 3 / nside / nside) ) * au.rad
        return pix_rad
    
    def rad2arcsec( self, rad ):
        r"""
            *** convert rad to arcsec *** # Basic function
            !!! Inverse of arcsec2rad() !!! # Important note
            !!! Verified by REF !!! # Important note
            # REF: https://www.advancedconverter.com/unit-conversions/angle-conversion/radians-to-arcseconds
            +++ Maybe use astropy? +++ # Improvements
            :Params rad: pixel size in rad
            :Output arcsec: pixel size in arcsec [Astropy Units Quantity]
        """
        arcsec = rad / math.pi * 180 * 3600 * au.arcsec
        return arcsec
    
    def arcsec2rad( self, arcsec ):
        r"""
            *** convert arcsec to rad *** # Basic function
            !!! Inverse of rad2arcsec() !!! # Important note
            !!! Verified by REF !!! # Important note
            # REF: https://www.advancedconverter.com/unit-conversions/angle-conversion/arcseconds-to-radians
            +++ Maybe use astropy? +++ # Improvements
            :Params arcsec: pixel size in arcsec
            :Output rad: pixel size in rad [Astropy Units Quantity]
        """
        rad = ( arcsec / 3600 / 180 * math.pi ) * au.rad
        return rad
    
    @property
    def resolution( self ):
        r"""
            *** nside >> pixel size in arcsec *** # Basic function
            !!! Inverse of arcsec2nside() !!! # Important note
            +++  +++ # Improvements
            :Params nside: nside of the map
            :Output arcsec: pixel size in arcsec
        """
        rad = self.pixsize( self.nside )
        arcsec = self.rad2arcsec( rad.value ) # arcsec
        self.logger.info( " Nside %s have a pixe size of %s " % ( self.nside, arcsec) )
        return arcsec
    
    @property
    def resolution_deg( self ):
        deg = self.resolution.value / 3600 * au.deg
    
    @property
    def lmax( self ):
        r"""
            *** Default est nside >> largest multipole of a Nside grid *** # Basic function
            !!! Inverse of defaultnside() !!! # Important note
            +++ Add reference +++ # Improvements
            :Params nside: nside of the map;
            :Output lmax: Largest multipole;
        """
        lmax = int ( 3 * self.nside - 1 )
        self.logger.info( "Nside %s have a max l of %s" % ( self.nside, lmax) )
        return lmax
    
    @property
    def sigma_sim( self ):
        sigma_sim = self.pixsize( self.nside ) # rad
        self.logger.info( " Nside %s have a simulated beam size of %s " % ( self.nside, sigma_sim) )
        return sigma_sim
    
    @property
    def alm( self ):
        return self._alm
    
    @property
    def cl( self ):
        return self._cl
    
    @property
    def stage( self ):
        return self._stage
    
    @property
    def hpinfo( self ):
        return self._hpinfo
    
    # Gfree methods
    
    def _set_physical_params( self ):
        self.gfree_temp_freq = 120.      # MHz Gfree
        self.f_dust = 0.33         # Effective dust fraction in the LoS
        self.halpha_abs_th = 1     # [mag]
        self.Te = 7000.0           # [K]
        self.gsync_temp_freq = 408   # MHz Gsync
        self.logger.info(' Physical params are set. ')
        
    def _correct_dust_absorption(self):
        """
        !!! Not needed at the moment !!!
        """
        """
        Correct the Hα map for dust absorption using the
        100-μm dust map.
        References: Ref.[dickinson2003],Eq.(1,3),Sec.(2.5)
        """
        if hasattr(self, "_dust_corrected") and self._dust_corrected:
            return

        logger.info("Correct H[alpha] map for dust absorption")
        logger.info("Effective dust fraction: {0}".format(self.f_dust))
        # Mask the regions where the true Halpha absorption is uncertain.
        # When the dust absorption goes rather large, the true Halpha
        # absorption can not well determined.
        # Corresponding dust absorption threshold, unit: [ MJy / sr ]
        dust_abs_th = self.halpha_abs_th / 0.0462 / self.f_dust
        logger.info("Dust absorption mask threshold: " +
                    "{0:.1f} MJy/sr ".format(dust_abs_th) +
                    "<-> H[alpha] absorption threshold: " +
                    "{0:.1f} mag".format(self.halpha_abs_th))
        mask = (self.dustmap.data > dust_abs_th)
        self.dustmap.data[mask] = np.nan
        fp_mask = 100 * mask.sum() / self.dustmap.data.size
        logger.warning("Dust map masked fraction: {0:.1f}%".format(fp_mask))
        #
        halphamap_corr = (self.halphamap.data *
                          10**(self.dustmap.data * 0.0185 * self.f_dust))
        self.halphamap.data = halphamap_corr
        self._dust_corrected = True
        logger.info("Done dust absorption correction")

    def _calc_factor_a(self, nu):
        """
        Calculate the ratio factor a(Te, ν), which will be used to
        convert the Halpha emission [Rayleigh] to free-free emission
        brightness temperature [K].
        Parameters
        ----------
        nu : float
            The frequency where to calculate the factor a(nu).
            Unit: [MHz]
        Returns
        -------
        a : float
            The factor for Hα to free-free conversion.
        References: [dickinson2003],Eq.(8)
        """
        term1 = 0.183 * nu**0.1 * self.Te**(-0.15)
        term2 = 3.91 - np.log(nu) + 1.5*np.log(self.Te)
        a = term1 * term2
        return a

    def _calc_halpha_to_freefree(self, nu):
        """
        Calculate the conversion factor between Hα emission [Rayleigh]
        to radio free-free emission [K] at frequency ν [MHz].
        Parameters
        ----------
        nu : float
            The frequency where to calculate the conversion factor.
            Unit: [MHz]
        Returns
        -------
        h2f : float
            The conversion factor between Hα emission and free-free emission.
        References: [dickinson2003],Eq.(11)
        NOTE: The above referred formula has a superfluous "10^3" term!
        """
        a = self._calc_factor_a(nu)
        h2f = 38.86 * a * nu**(-2.1) * 10**(290/self.Te) * self.Te**0.667
        return h2f
    
    # Fit maps >>> Funcs below will be dispatched very soooooon!
    def ___old_gen_fitmap( self ):
        """
            *** Generate cls for gamma fitting. ***
        """
        self._fitmap = {}
        self.logger.info( "Getting the GRF map & Upgraded map." )
        
        if self.fullsky_:
            self._fitmap['gss'], self._fitmap['ori'] = self.cut_fitmap( self.gssmap, self.upmap, partial_=False )
        else:
            if self.npatch == 1:
                if self.coord_aim == 'C':
                    rotation = True
                else:
                    rotation = False
                self._fitmap['gss'], self._fitmap['ori'] = self.cut_fitmap( self.grfmap, self.upmap, center=self.pcenter, size=self.psize, rot_=rotation )
            elif self.npatch > 1:
                self._fitmap['gss'], self._fitmap['ori'] = self.cut_fitmapsets( self.grfmap, self.upmap, size=self.psize, npatch=self.npatch, rot_=False )
    
    # One patch or fullsky
    def cut_fitmap( self, gssmap, orimap, partial_=True, center=[0, 0.3] ,size=1000, rot_=True ):
        wgssmap = self.whiten(gssmap)
        orimap = orimap
        # Rotation
        if rot_:
            coords=['G','C']
        else:
            coords=['G']
        if partial_: # Get patial sky for fitting
            fit_gssmap = hp.gnomview(wgssmap, rot=center, reso=self.resolution.value/60, xsize=size, coord=coords, return_projected_map=True)
            fit_orimap = hp.gnomview(orimap, rot=center, reso=self.resolution.value/60, xsize=size, coord=coords, return_projected_map=True)
            self.logger.info( "Using partial sky centered at %s w/ image size of %s w/ coord in %s. The image has a FoV of %s deg." % ( center, size, coords, self.resolution.value*size/3600 ))
        else: # Get full sky for fitting
            fit_gssmap = gssmap
            fit_orimap = orimap
            self.logger.info( " Using the full sky map. ")
        return fit_gssmap, fit_orimap
    
    # Multiple patches
    def cut_fitmapsets( self, gssmap, orimap, size=1000, npatch=5, rot_=True ):
        wgssmap = self.whiten(gssmap)
        orimap = orimap
        
        # Random center list
        centers = [np.array([0, 0])]
        crandom = np.random.randint(-360,360, size= (npatch-1,2))
        centers.extend(crandom)
        
        #+++ Add adjacent minimum requirements
        size_deg = size * self.resolution.value / 3600
        
        # Rotation
        if rot_:
            coords=['G','C']
        else:
            coords=['G']
        
        fit_gssmap = {}
        fit_orimap = {}
        
        #>> Get maps
        for i in range(0,npatch):
            fit_gssmap[i] = hp.gnomview(wgssmap, rot=centers[i], reso=self.resolution.value/60, coord=coords, xsize=size, return_projected_map=True)
            fit_orimap[i] = hp.gnomview(orimap, rot=centers[i], reso=self.resolution.value/60, coord=coords, xsize=size, return_projected_map=True)
            self.logger.info( "Generating the %s/%s partial sky map centered at %s w/ image size of %s w/ coord in %s. The image has a FoV of %s deg." % ( i, npatch, centers[i], size, coords, self.resolution.value*size/3600 ) )
        return fit_gssmap, fit_orimap
    
    # Visualization
    def compareview( self ):
        hp.gnomview(self.upmap,rot=[0, 0.3], reso=0.1, xsize= 5000, title='Upgrade Map',coord=['G','C'])
        hp.gnomview(self.ssmap,rot=[0, 0.3], reso=0.1, xsize= 5000, title='SmallScale Map',coord=['G','C'])
        hp.gnomview(self.hpmap,rot=[0, 0.3], reso=0.1, xsize= 5000, title='Upgrade + SmallScale Map',coord=['G','C'])


class Galactic:
    r"""
        *** The Galactic object to simulate Galactic Component with small scales *** # Basic function
        !!! Need to check the map float32 or float64 issue !!! # Important note
        +++ Update to take fg21sim config files; +++ # Improvements
        +++ Add unit check for sigma! +++
        :Params type_: type handle, "Gsync" & "Gff"
    """
    def __init__(self, 
                 nside = None, sigma_tem = None,                        # Resolution
                 freqs = None, unit = None,                             # Freq & unit
                 alpha = None, beta = None, gamma = None,               # Small scale & cl params
                 alm = None, cl = None,                                 # Input cl & alm
                 inmap = None, mmap = None, idmap = None,               # Input map, masked map, spectral index map
                 fitlmin = None, fitlmax = None,                        # Gamma param fitting
                 psize = None, npatch = None, pcenter = None,           # Alpha & beta param fitting
                 fov = None, center = None, sim_pixel = None,           # Sky patch simulation
                 fullsky_ = False, fit_patch_ = False,                  # Simulation settings
                 coord_ = 'C', frame_ = 'ICRS', proj_ = 'TAN',          # Projection info
                 type_ = 'Gsync', order_ = 'RING',                      # Type & order
                 server = 'sgi',                                        # Server info
                 fitting_mode_ = True, default_ = True,                 # Mode setting
                 level_ = 'debug', stdout_ = False ):                   # Log contral
        
        # Initialize the logger
        self.level_ = level_
        self.stdout_ = stdout_
        self.type_ = type_
        self.default_ = default_
        self.logname = 'GSim' + '_' + self.type_ + '_' + "Default_" + str(self.default_)
        loging = Log( self.logname, self.level_, self.stdout_ )
        self.logger = loging.logger
        
        # Initialize the Simulation Params
        self.logger.info( ' ⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇ ' )
        self.logger.info( ' Galactic %s simulation is initializing! ' % self.type_ )
        self.logger.warning( " sigma_tem default unit is [arcmin]! " )
        
        self.fullsky_ = fullsky_
        if fullsky_:
            self.logger.info( ' Simulation the full sky. ' )
        else:
            self.logger.info( ' Simulation a patch of the sky. ' )
        
        if default_: # Using default params if you can
            self.logger.info( ' Default settings are used! ' )
            
            if type_=='Gsync':
                self.alpha = alpha   # Add default version
                self.beta = beta     # Add default version
                self.gamma = -2.703  # R15 paper
                self.sigma_tem = 56  # unit[arcmin]
                
            elif type_=='Gff':
                self.alpha = alpha   # Add default version
                self.beta = beta     # Add default version
                self.gamma = -3.136  # Fitted by Chenxi
                self.sigma_tem = 6   # unit[arcmin]
                
        else:
            if fitting_mode:
                self.logger.info( ' Prepare fitting settings! ' )
                self.alpha = alpha
                self.beta = beta
                self.gamma = gamma
                if type_=='Gsync':
                    self.sigma_tem = 56 # unit[arcmin]
                elif type_=='Gff':
                    self.sigma_tem = 6 # unit[arcmin]
            else:
                self.logger.info( ' Custom settings are used! ' )
                self.alpha = alpha
                self.beta = beta
                self.gamma = gamma
                self.sigma_tem = sigma_tem * au.arcmin # unit[arcmin]
            
        self.logger.info( ' %s alpha is initialized as %s ' % ( self.type_, self.alpha ) )
        self.logger.info( ' %s beta is initialized as %s ' % ( self.type_, self.beta ) )
        self.logger.info( ' %s gamma is initialized as %s ' % ( self.type_, self.gamma ) )
        self.logger.info( ' %s sigma_tem is initialized as %s [arcmin] ' % ( self.type_, self.sigma_tem ) )
        
        self.nside = nside
        self.logger.info( ' Nside is set to %s ' % self.nside )
        
        self.unit = unit # Unit of the intensity maps, should set as 'mK'

        # Alm Cl, they can be set by the user;
        self._alm = alm
        self._cl = cl
        self.logger.info( ' alm & cl are set to %s & %s ' % ( self._alm, self._cl ) )
        
        # Maps
        self.order = order_ # Default map order
        self.logger.info( ' Default healpix map order is %s ' % order_ )
        
        self.inmap = inmap # input map
        self.idmap = idmap # input spectral index map
        self.mmap = mmap # masked map
        self.mfmap = None # mask-filled map
        self.upmap = None # upgraded map
        self.grfmap = None # Gaussian Random Field map (from cl & alm)
        self.ssmap = None # small scale map
        
        self.hpmap = None # upgraded map w/ small scale (intermediate)
        self.simmap = None # Final simulated map
        self._stage = { 'inmap':{'0':'input map', '1':[]}, 'idmap':{'0':'input spectral index map','1':[]}, 'mmap':{'0':'masked map', '1':[]}, 'mfmap': {'0': 'mask-filled map', '1':[]}, 'upmap':{'0':'upgraded map', '1':[]}, 'grfmap':{'0':'Gaussian Random Field map', '1':[]}, 'ssmap':{'0':'small scale map', '1':[]}, 'hpmap':{'0':'upgraded map w/ small scale', '1':[]}, 'simmap':{'0':'Final simulated map', '1':[]} } # map stage
        
        # Info store
        self._hpinfo = {} # Store healpix info
        
        # Simulation settings
        # Coordinates
        self.coordname = {"G": "Galactic", "E": "Ecliptic", "C": "Equatorial"}
        
        # Frequency range
        self.freqs = freqs
        
        # Sky pathc simulation config
        self.fov = fov # (tuple,deg) The size of the skypatch e.g. (10, 10);
        self.center = center # (tuple,deg) Ra & dec in ICRS skycoordinate e.g. (0, -27);
        self.sim_pixel = sim_pixel # [arcsec] Simulated pixel size;
        self._xsize = None # Image size of RA
        self._ysize = None # Image size of Dec
        
        # Coordinate system & projection
        self.coord_ori = 'G'     # Default is Galactic;
        self.coord_aim = coord_  # Aim is Equatorial;
        self.frame = frame_      # Coord name
        self.proj = proj_        # Projection name
        
        self.framename = {'ICRS', 'GALACTIC', 'EQUATORIAL', 'ECLIPTIC','G', 'C', 'E'}
        self.projname = {'TAN', 'CAR', 'HPX', 'MOL', 'SIN'}
        # https://docs.astropy.org/en/stable/wcs/supported_projections.html
        
        # Fitting Procedure
        ## Fitting cl
        self.fitlmin = fitlmin
        self.fitlmax = fitlmax
        self._fitcl = None
        self.logger.info( ' Cl fitting using lmin=%s, lmax=%s. ' % ( fitlmin, fitlmax ) )
        ## Fitting map
        self.fit_patch_ = fit_patch_
        if fit_patch_:           # If you are simulate one sky patch, make sure to turn on fit_patch_
            self.pcenter = center
            if fov is None:
                raise ValueError(' Input FoV can not be NoneType! ')
            _psize = 1.5 * max(fov[0],fov[1]) / self.resolution_deg.value
            _psize = int(_psize)
            self.psize = _psize
            self.npatch = 1
            self.logger.info( ' Map fitting using the aimed simulated sky w/ size=%s with a center of %s. ' % ( self.psize, center ) )
        else:
            self.pcenter = pcenter
            self.psize = psize
            self.npatch = npatch
            self.logger.info( ' Map fitting using %s batches w/ size=%s with one of centers fixed at the Galactic Center. ' % ( npatch, psize ) )
        self._fitmap = None
        
        if self.inmap is None:
            self.loadmap( server = server )
            self.logger.info( 'No input maps are given, auto loading input map.' )
        # Properties
        """
        - resolution
        - lmax
        - sigma_sim
        - stage
        - hpinfo
        - alm
        - cl
        """
        self.logger.info( ' Galactic %s simulation is initialized! ' % type_ )
        self.logger.info( ' ⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆ ' )
    
    
    def pixsize( self, nside ):
        r"""
            *** nside >> pixel size in rad *** # Basic function
            !!! Inverse of rad2nside() !!! # Important note
            !!! For healpix usage !!! # Important note
            +++  +++ # Improvements
            :Params nside: nside of the map
            :Output pix_rad: pixel size in rad [Astropy Units Quantity]
        """
        pix_rad = ( math.sqrt(math.pi / 3 / nside / nside) ) * au.rad
        return pix_rad
    
    
    def rad2arcsec( self, rad ):
        r"""
            *** convert rad to arcsec *** # Basic function
            !!! Inverse of arcsec2rad() !!! # Important note
            !!! Verified by REF !!! # Important note
            # REF: https://www.advancedconverter.com/unit-conversions/angle-conversion/radians-to-arcseconds
            +++ Maybe use astropy? +++ # Improvements
            :Params rad: pixel size in rad
            :Output arcsec: pixel size in arcsec [Astropy Units Quantity]
        """
        arcsec = rad / math.pi * 180 * 3600 * au.arcsec
        return arcsec
    
    
    def arcsec2rad( self, arcsec ):
        r"""
            *** convert arcsec to rad *** # Basic function
            !!! Inverse of rad2arcsec() !!! # Important note
            !!! Verified by REF !!! # Important note
            # REF: https://www.advancedconverter.com/unit-conversions/angle-conversion/arcseconds-to-radians
            +++ Maybe use astropy? +++ # Improvements
            :Params arcsec: pixel size in arcsec
            :Output rad: pixel size in rad [Astropy Units Quantity]
        """
        rad = ( arcsec / 3600 / 180 * math.pi ) * au.rad
        return rad
    
    
    @property
    def resolution( self ):
        r"""
            *** nside >> pixel size in arcsec *** # Basic function
            !!! Inverse of arcsec2nside() !!! # Important note
            +++  +++ # Improvements
            :Params nside: nside of the map
            :Output arcsec: pixel size in arcsec
        """
        rad = self.pixsize( self.nside )
        arcsec = self.rad2arcsec( rad.value ) # arcsec
        self.logger.info( " Nside %s have a pixe size of %s " % ( self.nside, arcsec) )
        return arcsec
    
    @property
    def resolution_deg( self ):
        deg = self.resolution.value / 3600 * au.deg
    
    @property
    def lmax( self ):
        r"""
            *** Default est nside >> largest multipole of a Nside grid *** # Basic function
            !!! Inverse of defaultnside() !!! # Important note
            +++ Add reference +++ # Improvements
            :Params nside: nside of the map;
            :Output lmax: Largest multipole;
        """
        lmax = int ( 3 * self.nside - 1 )
        self.logger.info( "Nside %s have a max l of %s" % ( self.nside, lmax) )
        return lmax
    
    @property
    def sigma_sim( self ):
        sigma_sim = self.pixsize( self.nside ) # rad
        self.logger.info( " Nside %s have a simulated beam size of %s " % ( self.nside, sigma_sim) )
        return sigma_sim
    
    @property
    def alm( self ):
        return self._alm
    
    @property
    def cl( self ):
        return self._cl
    
    @property
    def fitcl( self ):
        return self._fitcl
    
    @property
    def fitmap( self ):
        return self._fitmap
    
    @property
    def stage( self ):
        return self._stage
    
    
    @property
    def hpinfo( self ):
        return self._hpinfo
    
    
    def genalmcl( self, lmin=1 ):
        r"""
            *** Generate the angular power spectrum of the Gaussian random field with default settings ***
            !!! If you want custom_ lmax please call genGuass_cl from Gtools !!!
            !!! The Guassian Random Field should be in Radian !!!
            Check the reference: https://ui.adsabs.harvard.edu/abs/2013A%26A...553A..96D/abstract
            :Params lmin: multipole min; # Should default as 1!
            :Params from self: type_, lmax, gamma, sigma_tem, & sigma_sim [if type_='Gff'];
            :Output almcl: dict contains alm & cl; 
                           almcl['alm']: return alm of a Gaussian random field;
                           almcl['cl']: return the generated angular power spectrum;
        """
        
        if self.nside is None:
            self.logger.error( ' Nside currently is %s, must be set prior!' % self.nside )
            raise ValueError
        
        # Set up alm
        almcl = {}
        # Set the lmax & mmax
        lmax = self.lmax
        mmax = lmax
        nalm = ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)
        
        # Prepare gamma & sigma
        gamma = self.gamma  # index of the power spectrum between l [30, 90]
        _sigma_tem = self.sigma_tem  # original beam resolution of the template [ arcmin ]
        sigma_tem = self.arcsec2rad( _sigma_tem * 60 ) # [ au.rad ]
        
        # angular power spectrum of the Gaussian random field
        ell = np.arange(lmax+1, dtype=int)
        cl = np.zeros(ell.shape)
        ell_idx = ell >= lmin
        if self.type_=='Gsync':            
            cl[ell_idx] = ell[ell_idx] ** gamma * ( 1.0 - np.exp(-ell[ell_idx]**2 * sigma_tem.value**2) )
        elif self.type_=='Gff':
            sigma_sim = self.sigma_sim  # simulated beam resolution [ au.rad ]
            cl[ell_idx] = ell[ell_idx] ** gamma * ( np.exp(-ell[ell_idx]**2 * sigma_sim.value**2) - np.exp(-ell[ell_idx]**2 * sigma_tem.value**2) )
        cl[ell < lmin] = cl[lmin]
        
        # convert cl to alm
        alm = cl2alm(cls=cl,lmax=lmax, mmax=mmax)
        
        self.logger.info( ' Generate alm & cl for Galactic %s simulation. ' % self.type_ )
        self.logger.info( ' %s alpha is %s ' % ( self.type_, self.alpha ) )
        self.logger.info( ' %s beta is %s ' % ( self.type_, self.beta ) )
        self.logger.info( ' %s gamma is %s ' % ( self.type_, self.gamma ) )
        self.logger.info( ' %s sigma_tem is %s ' % ( self.type_, self.sigma_tem ) )
        self.logger.info( ' %s sigma_sim is %s ' % ( self.type_, self.sigma_sim ) )
        self.logger.info( ' lmax, mmax, nalm is %s, %s, & %s ' % ( lmax, mmax, nalm ) )
        
        self._hpinfo['lmax'] = lmax
        self._hpinfo['mmax'] = mmax
        self._hpinfo['nalm'] = nalm
        
        # Modify the Class attribute
        self._alm = alm
        self._cl = cl
        
        # Gen output for the function
        almcl['alm'] = alm
        almcl['cl'] = cl
        return almcl
    
    
    def gaussianmap( self, nthreads, spin=0 ):
        r"""
            *** gaussianmap (a special copy of alm2map) using ducc0 in a healpix grid ***
            :Params alm: spherical harmonics coefficients; 
            :Params nside: nside of the extended map;
            :Params nthreads: number of threads;
            :Params spin: spin of the alm, default spin = 0;
            :Params test: test type;
            ::::::::::::: test == 0 using a RNG distribution alm;
            ::::::::::::: test == 1 using a custom alm;
            :Output hpmap: healpix map;
        """
        # Multi-threading
        allthreads = active_count()
        self.logger.info("alm2map supports multi-threading, use %s out of total %s threads." % (nthreads, allthreads))
        # Set the lmax & mmax
        lmax = self.lmax
        mmax = lmax
        alm = self.alm.copy()
        inalm = alm.reshape( (1,-1) ) # reshape the alm to dhp standard
        base = dhp.Healpix_Base( self.nside, self.order )
        geom = base.sht_info()

        # test adjointness between synthesis and adjoint_synthesis
        _grfmap = dsh.experimental.synthesis(alm=inalm, lmax=lmax, spin=spin, nthreads=nthreads, **geom)
        shape = _grfmap.shape
        grfmap = _grfmap.reshape(shape[1],)
        
        self._hpinfo['nside'] = self.nside
        self._hpinfo['order'] = self.order
        self._hpinfo['spin'] = spin
        self.logger.info( "grfmap is generated w/ nside=%s, lmax=%s, spin=%s, order=%s" % ( self.nside, self.lmax, spin, self.order ) )
        self._stage['grfmap']['1'].append( "grfmap is generated w/ nside=%s, lmax=%s, spin=%s, order=%s" % ( self.nside, self.lmax, spin, self.order ) )
        # Modify the Class attribute
        self.grfmap = grfmap
        return grfmap
    
    
    # ====== Maps ======
    
    
    def loadmap( self, server='gravity' ):
        r"""
            *** Load the default files on the computers ***
            +++ Add other machines such as SGI +++
            
            >> All healpy routines assume RING ordering, 
            in fact as soon as you read a map with read_map, 
            even if it was stored as NESTED, 
            it is transformed to RING. 
            However, you can work in NESTED ordering passing 
            the nest=True argument to most healpy routines.

            > https://healpy.readthedocs.io/en/latest/tutorial.html#:~:text=was%20stored%20as-,NESTED,-%2C%20it%20is%20transformed
        """
        if server == 'gravity':
            if self.type_ == 'Gsync':
                datadir = "/home/cxshan/radiodata/"
                fn = "haslam408_dsds_Remazeilles2014_ns512.fits"
                idn = "synchrotron_specind2_ns512.fits"
            elif self.type_ == 'Gff':
                datadir = "/home/cxshan/fg21sim+/GalacticData"
                fn = "gfree_120.0.fits"
        elif server == 'sgi':
            if self.type_ == 'Gsync':
                datadir = "/mnt/ddnfs/data_users/cxshan/radiodata/"
                fn = "haslam408_dsds_Remazeilles2014_ns512.fits"
                idn = "synchrotron_specind2_ns512.fits"
            elif self.type_ == 'Gff':
                datadir = "/mnt/ddnfs/data_users/cxshan/radiodata/GalacticData/gfree/"
                fn = "gfree_120.0.fits"
        else:
            print('Map location unknow! You should use gopen()')
        fname = datadir + fn
        inmap = hp.read_map( fname ) # Default order is always RING;
        self.inmap = inmap
        inmap_nside = self.ncheck( inmap )
        self.logger.info( "Input map is loaded from %s on %s server w/ nside=%s in RING order" % ( fname, server, inmap_nside ) )
        self._stage['inmap']['1'].append( "Input map is loaded from %s on %s server w/ nside=%s in RING order" % ( fname, server, inmap_nside ) )
        
        inmap_nan = self.checknan( inmap )
        self.inmap_nan = inmap_nan
        if inmap_nan:
            self.logger.info( 'Input map contains NaN pixels, filtering the input map. ' )
            self.mmap = self.gopen_mask( datadir, fn )
            self.logger.info( "Input map is nan masked" )
            self._stage['mmap']['1'].append( "Input map is nan masked" )
            self.fillmap()
            
        if self.type_ == 'Gsync':
            idname = datadir + idn
            idmap = hp.read_map( idname ) # Default order is always RING;
            self.idmap = idmap
            idmap_nside = self.ncheck( idmap )
            self.logger.info( "Input spectral index map is loaded from %s on %s server w/ nside=%s in RING order" % ( idname, server, idmap_nside ) )
            self._stage['idmap']['1'].append( "Input spectral index map is loaded from %s on %s server w/ nside=%s in RING order" % ( idname, server, idmap_nside ) )
            idmap_nan = self.checknan( idmap )
            if idmap_nan:
                self.logger.info( ' Input spectral index map contains NaN pixels, filtering the spectral index map. ' )
                idmap = self.gopen_mask( datadir, fn )
                self.logger.info( " Input spectral index map is nan masked. " )
                idmap = idmap.filled( fill_value = 0 )
                self.logger.info( " Input spectral index map's nan pixels are filled with 0. " )
        else:
            idmap = None
            self.logger.info( " Input spectral index map is not supported for type = %s! " % self.type_ )
            self._stage['idmap']['1'].append( " Input spectral index map is not supported for type = %s! " % self.type_ )
        return inmap, idmap
    
    
    def gopen( self, fdir, fn ):
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

    
    def gopen_mask( self, fdir, fn ):
        r"""
        *** Open the file map w/ hp.READ_MAP & mask all the nan ***
        !!! Check out the hp.READ_MAP instructuion !!!
        :Params fdir: path to the file;
        :Params fn: file name;
        :Output mask: masked map in healpix `RING` scheme;
        """
        file = fdir + fn
        x = hp.read_map(file)
        masked = np.ma.masked_array(x, np.isnan(x))
        return masked
    
    
    def checknan( self, inputmap ):
        r"""
            *** Check if a map has any NaN pixels. *** # Utility
        """
        if( np.isnan( inputmap ).any() ):
            nanstatus = True
        else:
            nanstatus = False
        return nanstatus
    
    
    def maskmap( self, inputmap, mask_='nan', store_=False ):
        r"""
            *** Mask nan from map *** # Utility
            !!! Not working rightnow, use gopen_mask() instead !!!
            :Output masked: maskedmap
        """
        x = inputmap.copy()
        if mask_ == 'nan':
            masked = np.ma.masked_array(x, np.isnan(x))
        else:
            print(' Create your own mask or wait for the update! ')
        if store_:
            self.mmap = masked
            self.logger.info( "Input map is nan masked" )
            self._stage['mmap']['1'].append( "Input map is nan masked" )
        return masked
    
    
    def fillmap( self, value=0 ):
        r"""
            *** Fill the nan-Masked map with value *** # Utility
            :Output mfmap: mask filled map;
        """
        mfmap = self.mmap.filled( fill_value=value )
        self.mfmap = mfmap
        
        self.logger.info( "Nan-masked map is filled w/ %s" % value )
        self._stage['mfmap']['1'].append( "Nan-masked map is filled w/ %s" % value )
        return mfmap
    
    
    def ncheck( self, inmap ):
        r"""
        *** Check the nside of the map ***
        """
        ncheck = hp.get_nside(inmap)
        return ncheck
    
    
    def nchecks( self, map1, map2 ):
        r"""
        *** Check if the nside of two maps matches ***
        """
        n1 = self.ncheck( map1 )
        n2 = self.ncheck( map2 )
        if n1 == n2:
            status = True
        else:
            status = False
        return status
    
    
    def whiten( self, gss ):
        r"""
        *** Whiten a guass map ***
        """
        gss = (gss - gss.mean()) / gss.std()
        return gss
    
    
    def update( self, aim_='inmap' ):
        r"""
            *** Update the filled / original map to aimed nside *** # Utility
            :Output upmap: update original or mask filled map;
        """
        if aim_ == 'mmap': # masked map
            _map = self.mmap
        elif aim_ == 'mfmap': # mask-filled map
            _map = self.mfmap
        elif aim_ == 'inmap': # input map
            _map = self.inmap
        else:
            aim_ = 'inmap'
            _map = self.inmap
            self.logger.error( "aim handler %s is not supported!" % aim_ )
            raise ValueError
        
        if self.inmap_nan:
            _map = self.mfmap
            self.logger.info( 'Detected input map NaN status, using mask_filled map instead. ' )
        self.logger.info( "Starting the upgrade of %s." % aim_ )
        self._stage['upmap']['1'].append( "Detected input map NaN status, using mask_filled map instead." )
        
        n = self.ncheck( _map )
        upmap = self.up_method( _map )
        self.upmap = upmap
        self.logger.info( "%s is upgraded from %s to %s." % ( aim_, n, self.nside))
        self._stage['upmap']['1'].append( "%s is upgraded from %s to %s." % ( aim_, n, self.nside) )
        
        self.logger.info( "Starting the upgrade of spectral index map." )
        if self.idmap is None:
            self.logger.info( "Upgrade of spectral index map is not supported for type = %s!" % self.type_ )
            self._stage['idmap']['1'].append( "Upgrade of spectral index map is not supported for type = %s!" % self.type_ )
            upidmap = None
        else:
            upidmap = self.up_method( self.idmap )
            nid = self.ncheck( self.idmap )
            self.idmap = upidmap
            self.logger.info( " Spectral index map is upgraded from %s to %s." % ( nid, self.nside))
            self._stage['upmap']['1'].append( "Spectral index map is upgraded from %s to %s." % ( nid, self.nside) )
        return upmap, upidmap
    
    
    def up_method( self, lowresmap ):
        n = self.ncheck(lowresmap)
        if n < self.nside:
            highresmap = hp.ud_grade(lowresmap, nside_out=self.nside)
            self.logger.info(' Lowres map (Nside = %s) is upgrading to nside %s. ' % ( n, self.nside ))
        elif n == self.nside:
            highresmap = lowresmap
            self.logger.info(' Nside of lowres map %s is the same as aimed nside %s. ' % ( n, self.nside ))
        else:
            highresmap = lowresmap
            self.logger.error(' Nside of lowres map %s is larger than the aimed nside %s! Please reset your nside! ' % ( n, self.nside ))
            raise ValueError
        return highresmap
    
    
    def gen_smallscales( self, alpha=0.0599, beta=0.782, fitting_=False ):
        r"""
        *** Check the gss & hpmap and generate smallscales temp ***
        :Params alpha: ratio alpha param;
        :Params beta: ratio beta param;
        :Output ssmap: small scale map;
        """
        
        if fitting_ is True:
            _alpha = alpha
            _beta = beta
        else:
            _alpha = self.alpha
            _beta = self.beta
            self.final_ = True
        
        # Whiten the GRFmap
        gss = self.grfmap.copy()
        gss = self.whiten( gss )
        self.logger.info( "A whiten copy of grfmap is generated." )
        
        
        match = self.nchecks( self.grfmap, self.upmap )
        if match == True:
            if self.inmap_nan:
                self.logger.info( 'Detected input map NaN status, using upgraded mask_filled map instead. ' )
                self._stage['ssmap']['1'].append( "Detected input map NaN status, using upgraded mask_filled map instead." )
            ssmap = _alpha * gss * self.upmap ** _beta
            self.logger.info( "ssmap is generated w/ alpha=%s beta=%s" % ( _alpha, _beta ) )
            self._stage['ssmap']['1'].append( "ssmap is generated w/ alpha=%s beta=%s" % ( _alpha, _beta ) )
        else:
            self.logger.error( "!!! Nside Error, please check the Nside." )
            self.logger.error( "grfmap:", self.ncheck( self.grfmap ) )
            self.logger.error( "upmap:", self.ncheck( self.upmap ) )
            ssmap = gss * 0
            self.logger.info( "ssmap = 0 is generated due to nside error." )
            self._stage['ssmap']['1'].append( "ssmap = 0 is generated due to nside error." )
        self.ssmap = ssmap
        return ssmap
    
    
    def add_smallscales( self ):
        r"""
        *** Check the smallscales & hpmap and add smallscales to hpmap ***
        :Params smallscales: small scale map from a whitened GRF map;
        :Params hpmap: original healpix map;
        :Output addedmap: small scale added hpmap;
        """
        match = self.nchecks( self.ssmap, self.upmap )
        if match == True:
            if self.inmap_nan:
                self.logger.info( 'Detected input map NaN status, using upgraded mask_filled map & related ssmap instead. ' )
                self._stage['hpmap']['1'].append( "Detected input map NaN status, using upgraded mask_filled map & related ssmap instead." )
            hpmap = self.ssmap + self.upmap
            self.logger.info( "Small scales are added to hpmap." )
            self._stage['hpmap']['1'].append( "Small scales are added to hpmap." )
        else:
            self.logger.error( "!!! Nside Error, please check the Nside." )
            self.logger.error( "ssmap:", self.ncheck( self.ssmap ) )
            self.logger.error( "upmap:", self.ncheck( self.upmap ) )
            hpmap = self.upmap
            self._stage['hpmap']['1'].append( "hpmap = upmap due to nside error." )
        self.hpmap = hpmap
        if self.final_:
            self.simmap = hpmap
            self.logger.info( " Final stage is here, simmap is generated. " )
            self._stage['simmap']['1'].append( " Simmap is generated. " )
        return hpmap
    
    
    # Fit maps
    def gen_fitmap( self ):
        """
            *** Generate cls for gamma fitting. ***
        """
        self._fitmap = {}
        self.logger.info( "Getting the GRF map & Upgraded map." )
        
        if self.fullsky_:
            self._fitmap['gss'], self._fitmap['ori'] = self.cut_fitmap( self.gssmap, self.upmap, partial_=False )
        else:
            if self.npatch == 1:
                if self.coord_aim == 'C':
                    rotation = True
                else:
                    rotation = False
                self._fitmap['gss'], self._fitmap['ori'] = self.cut_fitmap( self.grfmap, self.upmap, center=self.pcenter, size=self.psize, rot_=rotation )
            elif self.npatch > 1:
                self._fitmap['gss'], self._fitmap['ori'] = self.cut_fitmapsets( self.grfmap, self.upmap, size=self.psize, npatch=self.npatch, rot_=False )
    
    
    # One patch or fullsky
    def cut_fitmap( self, gssmap, orimap, partial_=True, center=[0, 0.3] ,size=1000, rot_=True ):
        wgssmap = self.whiten(gssmap)
        orimap = orimap
        # Rotation
        if rot_:
            coords=['G','C']
        else:
            coords=['G']
        if partial_: # Get patial sky for fitting
            fit_gssmap = hp.gnomview(wgssmap, rot=center, reso=self.resolution.value/60, xsize=size, coord=coords, return_projected_map=True)
            fit_orimap = hp.gnomview(orimap, rot=center, reso=self.resolution.value/60, xsize=size, coord=coords, return_projected_map=True)
            self.logger.info( "Using partial sky centered at %s w/ image size of %s w/ coord in %s. The image has a FoV of %s deg." % ( center, size, coords, self.resolution.value*size/3600 ))
        else: # Get full sky for fitting
            fit_gssmap = gssmap
            fit_orimap = orimap
            self.logger.info( " Using the full sky map. ")
        return fit_gssmap, fit_orimap
    
    
    # Multiple patches
    def cut_fitmapsets( self, gssmap, orimap, size=1000, npatch=5, rot_=True ):
        wgssmap = self.whiten(gssmap)
        orimap = orimap
        
        # Random center list
        centers = [np.array([0, 0])]
        crandom = np.random.randint(-360,360, size= (npatch-1,2))
        centers.extend(crandom)
        
        #+++ Add adjacent minimum requirements
        size_deg = size * self.resolution.value / 3600
        
        # Rotation
        if rot_:
            coords=['G','C']
        else:
            coords=['G']
        
        fit_gssmap = {}
        fit_orimap = {}
        
        #>> Get maps
        for i in range(0,npatch):
            fit_gssmap[i] = hp.gnomview(wgssmap, rot=centers[i], reso=self.resolution.value/60, coord=coords, xsize=size, return_projected_map=True)
            fit_orimap[i] = hp.gnomview(orimap, rot=centers[i], reso=self.resolution.value/60, coord=coords, xsize=size, return_projected_map=True)
            self.logger.info( "Generating the %s/%s partial sky map centered at %s w/ image size of %s w/ coord in %s. The image has a FoV of %s deg." % ( i, npatch, centers[i], size, coords, self.resolution.value*size/3600 ) )
        return fit_gssmap, fit_orimap
    
    
    def gen_fitcl( self ):
        """
            *** Generate cls for gamma fitting. ***
        """
        self._fitcl = {}
        if self.inmap_nan:
            clmap = self.mfmap
            self.logger.info( ' The cls is generated from the mask filled input map. ' )
        else:
            clmap = self.inmap
            self.logger.info( ' The cls is generated from the input map. ' )
        self._fitcl['full'] = self.map2cl( clmap )
        self.plotcl( self._fitcl['full'] )
        self._fitcl['cl'], self._fitcl['l'] = self.cutcl( self._fitcl['full'], self.fitlmin, self.fitlmax )
        self.logger.info( ' Cut cls in range of ( %s, %s ). ' % ( self.fitlmin, self.fitlmax ) )
    
    
    # Map_cl
    def map2cl_dc( self, healpixmap, nthreads, spin=0 ):
        """
        It is broken right now, please fix this!
        """
        nside = self.ncheck( healpixmap )
        lmax = 3 * nside - 1
        mmax = lmax
        
        hpmap64 = np.float64(healpixmap)
        rsmap = hpmap64.reshape((1,-1))
        inmap = rsmap
        base = dhp.Healpix_Base( nside, self.order )
        geom = base.sht_info()
        alm_out = dsh.experimental.adjoint_synthesis(lmax=lmax, spin=spin, map=inmap, nthreads=nthreads, **geom)
        cl_out = alm2cl( alm_out, lmax=lmax, mmax=mmax )
        return cl_out, alm_out
    
    
    # Map_cl
    def map2cl( self, healpixmap ):
        nside = self.ncheck( healpixmap )
        lmax = 3 * nside - 1
        mmax = lmax
        cl_out = hp.anafast( healpixmap, lmax=lmax, mmax=mmax )
        return cl_out
    
    
    # Plot_cl
    def plotcl( self, cl, cl_=True, log=True ):
        r"""
        *** plot cls w/ log option ***

        :Params cl: input cl;
        :Params cl_: plot only cl;
        :Params log: True for setting matplotlib.pyplot using log scale;
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
    
    
    # Get cl for fitting
    def cutcl( self, cl, lmin, lmax ):
        data = {}
        ell = np.arange(len(cl))
        data['cl'] = cl[lmin:lmax]
        data['l'] = ell[lmin:lmax]
        return data['cl'], data['l']
    
    
    # Visualization
    def compareview( self ):
        hp.gnomview(self.upmap,rot=[0, 0.3], reso=0.1, xsize= 5000, title='Upgrade Map',coord=['G','C'])
        hp.gnomview(self.ssmap,rot=[0, 0.3], reso=0.1, xsize= 5000, title='SmallScale Map',coord=['G','C'])
        hp.gnomview(self.hpmap,rot=[0, 0.3], reso=0.1, xsize= 5000, title='Upgrade + SmallScale Map',coord=['G','C'])