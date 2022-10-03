# Copyright (c) 2020-2022 Chenxi SHAN <cxshan@hey.com>
# Modified based on Hekun Lee's mcmc_sampler
# A emcee realization of MCMC Paramization method w/ Parameter-limit, Logging, & Multi-threading support.

__version__ = "0.4"
__date__    = "2022-08-05"

# Basic modules
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import emcee

# MPI module
from multiprocessing import Pool
from getdist import plots, MCSamples
from schwimmbad import MultiPool
from pathos.multiprocessing import ProcessPool

# Custom modules
from Log import Log


class MCMC:
    r"""
        A emcee realization of MCMC Paramization method
        w/ Parameter-limit, Logging, & Multi-threading support.
        This is an updated version of Hekun Lee's mcmc_sampler.
    """
    def __init__( self, nwalkers=None, nsteps=None, 
                 ndim=None, param_num=None, param_len=None,
                 param_lim=None, initial=None, vars_list=None, param_names=None,
                 serial_time=None, thread_num=None,
                 lim_=False, multi_=True,
                 level_='debug', stdout_=False):
        
        # Initialize the logger
        self.level_ = level_
        self.stdout_ = stdout_
        self.logname = 'MCMC' + '_ndim_' + str(ndim) + '_nwalkers_' + str(nwalkers) + '_nsteps_' + str(nsteps) + '_lim_' + str(lim_) + '_multi_' + str(multi_) + '_thread_num_' + str(thread_num) + '_'
        
        loging = Log( self.logname, self.level_, self.stdout_ )
        self.logger = loging.logger
        
        self.nwalkers = nwalkers            # Num of walkers
        self.nsteps = nsteps                # Num of steps
        self._ndim = ndim                   # Num of dim in the Param space
        if self._ndim == 1:
            self.logger.warning( 'Fiting one len=1 param only, the param_lim should in [(lower, upper)] format!' )
        self.logger.info( ' nwalkers is initialized as %s. ' % self.nwalkers )
        self.logger.info( ' nsteps is initialized as %s. ' % self.nsteps )
        self.logger.info( ' ndim of the Param space is initialized as %s. ' % self._ndim )
        
        self.param_num = param_num          # Num of params
        self.param_len = param_len          # Shape of the param
        self._param_lim = param_lim         # Param limit
        self.param_names = param_names      # Parameter names in list with right order
        self.logger.info( ' %s parmas of %s dimension w/ value limit of %s is initialized. ' % ( self.param_num, self.param_len, self._param_lim ) )
        self.vars_list = [vars_list]
        
        self.thread_num = thread_num
        self._serial_time = serial_time
        self.lim_ = lim_                    # Param limit handler
        self.multi_ = multi_                # Multi-threading handler
        
        self._initial = initial
        self.logger.info( 'Input initial params are:\n %s' % self._initial)
        
        self._sampler = None
        self._chain = None
        self._results = None
        self._discard_step = None
        self._thin_step = None

    
    @property
    def ndim( self ):
        if self._ndim is None:
            self._ndim = self.param_len * self.param_num
            #self.logger.info('Ndim = %s is calcaulated using Param_len & Param_num ' % self._ndim)
        else:
            if self._ndim != self.param_len * self.param_num:
                self.logger.error('Ndim %s, Param_len %s, and Param_num %s are not compatible! \n Please reset those prams to get Ndim = Param_len * Param_num.' % ( self.ndim, self.param_len, self.param_num ))
                raise ValueError
            else:
                self._ndim = self._ndim
                #self.logger.info(' Ndim = %s is verified using Param_len & Param_num ' % self._ndim)
        return self._ndim
    
    
    # Initial_params method
    def initial( self ):
        r"""
            *** Initialize the parameters *** # Utility
            !!! The outputs are passing to self.initial_params @property !!! # Note
            +++  +++ # Todo
        """
        self.logger.info(' Initialize the parameters.')
        if self._initial is None:
            self.logger.info(' No user input parameters, generate the initial params on the fly.')
            if self.lim_:
                self.logger.info(' Param_lim flag is on, initialize the parameters w/ limits.')
                initial_params = np.zeros((self.nwalkers, self.ndim))
                for i in range(self.ndim):
                    a, b = self.param_lim[i]
                    initial_params[:, i] = np.random.uniform(a, b, self.nwalkers)
            else:
                self.logger.info(' Param_lim flag is off, initialize the parameters randomly. ')
                np.random.seed(42)
                initial_params = np.random.randn(self.nwalkers, self.ndim)
        else:
            self.logger.info(' Using user input parameters, passing the initial params.')
            initial_params = self._initial
        self.logger.info( ' Initialization is done, initial params are:\n %s' % initial_params )
        return initial_params
    
    
    @property
    def initial_params( self ):
        initial_params = self.initial()
        return initial_params
    
    
    @property
    def param_lim( self ):
        if len(self._param_lim) != self.ndim:
            self.logger.error("Ndim %s & input Param_lim %s are not compatible! Param_lim length %s should equals Ndim." % (self.ndim, self._param_lim, len(self._param_lim) ))
            raise ValueError
        else:
            #self.logger.info(' Using user input parameter limit:\n %s.' % (self._param_lim,) )
            return self._param_lim
    
    
    def log_prob_demo( self, theta ):
        r"""
            *** Demo log_prob for debug or testing. *** # Utility
            !!! This is a do nothing function w/ no real meaning params. !!! # Note
            +++ Add other demos & print demo to show log prob examples! +++ # Todo
        """
        self.logger.info('Calls the demo log_prob for debug or testing.')
        t = time.time() + np.random.uniform(0.005, 0.008)
        while True:
            if time.time() >= t:
                break
        return -0.5 * np.sum(theta**2)
    
    
    def log_prob(self, params, vars_list):
        r"""
            *** Add parameter limits & filter NaN output of your model function *** # Utility
            !!!  ⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇ !!! # Note
            - Vars_list:
                - The chisq_function and the additional function_variables are in 'vars_list';
                - The chisq_funtion should be defined as "def chisq_fun(params, func_vars);
                - func(param, var) w/ (x, y, yerr = var) should be passing as [func, xi, yi, yierr];
            - Params:
                - params are the fitting parameters, which will be sampled by MCMC;
            !!!  ⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆ !!! # Note
            +++  +++ # Todo
        """
        if self.lim_:
            #self.logger.info( ' Param_lim flag is on using the parameter limits! ' )
            lp = 0
            for i in range(self.ndim):
                low_bound, upper_bound = self.param_lim[i]
                if low_bound < params[i] < upper_bound:
                    lp += 0
                else:
                    lp = -np.inf
            if np.isfinite(lp):
                if len(vars_list) == 1:
                    chi_sq_func = vars_list[0]
                    chisq = chi_sq_func(params)
                    if np.isnan(chisq):
                        self.logger.info( ' Hit NaN at w/ params %s. ' % params )
                        chisq = -np.inf
                        self.logger.info( 'NaN filtering in action: %s. ' % chisq )
                else:
                    chi_sq_func, func_vars = vars_list[0], vars_list[1:]
                    chisq = chi_sq_func(params, func_vars)
                    if np.isnan(chisq):
                        self.logger.info( ' Hit NaN at w/ params %s. ' % params )
                        chisq = -np.inf
                        self.logger.info( 'NaN filtering in action: %s. ' % chisq )
                return lp + chisq
            else:
                return -np.inf
        else:
            #self.logger.info( ' Param_lim flag is off, sample the parameters freely! ' )
            if len(vars_list) == 1:
                chi_sq_func = vars_list[0]
                chisq = chi_sq_func(params)
                if np.isnan(chisq):
                    self.logger.info( ' Hit NaN at w/ params %s. ' % params )
                    chisq = -np.inf
                    self.logger.info( 'NaN filtering in action: %s. ' % chisq )
            else:
                chi_sq_func, func_vars = vars_list[0], vars_list[1:]
                chisq = chi_sq_func(params, func_vars)
                if np.isnan(chisq):
                    self.logger.info( ' Hit NaN at w/ params %s. ' % params )
                    chisq = -np.inf
                    self.logger.info( 'NaN filtering in action: %s. ' % chisq )
            return chisq
    
    
    def run_MCMC( self ):
        r"""
            *** Calls & runs the respective MCMC sampler *** # Utility
            !!!  !!! # Note
            +++ Add pool options +++ # Todo
        """
        if self.multi_:
            self.logger.info(' Multi_threading flag is on, calls the MCMC sampler w/ Pool Parallelization.')
            self.Pool_MCMC( self.thread_num )
        else:
            self.logger.info(' Multi_threading flag is off, calls the Serial MCMC sampler.')
            self.Serial_MCMC()
    
    
    def Pool_MCMC( self, thread_num ):
        r"""
            *** emcee MCMC sampler w/ Pool Parallelization. *** # Utility
            !!!  !!! # Note
            +++ Add multi-time property +++ # Todo
        """
        status = self.verify_cpu( thread_num )
        if status:
            self.logger.info( "Request %s threads, which confines within total (%s) cpu resources." % (thread_num, self.cpu) )
            thread_num = thread_num
        else:
            # Use half of the cores
            self.logger.info( "Request %s threads, which exceeds total (%s) cpu resources." % (thread_num, self.cpu) )
            thread_num = int( self.cpu / 2 )
            self.logger.info( "Using %s threads, which is half of the total (%s) cpu resources." % (thread_num, self.cpu) )
        with Pool(thread_num) as pool:
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_prob, args=self.vars_list, pool=pool)
            start = time.time()
            self.logger.info( 'Starting the MCMC sampling progress.' )
            sampler.run_mcmc(self.initial_params, self.nsteps, progress=True)
            end = time.time()
            self.logger.info( 'Ending the MCMC sampling progress.' )
            multi_time = end - start
            self.logger.info("Multiprocessing took {0:.1f} seconds".format(multi_time))
        self._sampler = sampler
        d = datetime.now()
        dd = d.strftime('%m_%d_%Y_%H%M%S')
        self.logger.info( 'The sampler is updated by Pool_MCMC @ %s.' % dd )
    
    
    def Serial_MCMC( self ):
        r"""
            *** emcee MCMC sampler without Pool Parallelization. *** # Utility
            !!!  !!! # Note
            +++ Add serial-time property +++ # Todo
        """
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_prob, args=self.vars_list)
        self.logger.info( 'Starting the MCMC sampling progress.' )
        start = time.time()
        sampler.run_mcmc(self.initial_params, self.nsteps, progress=True)
        end = time.time()
        self.logger.info( 'Ending the MCMC sampling progress.' )
        serial_time = end - start
        self.logger.info("Serial took {0:.1f} seconds".format(serial_time))
        self._sampler = sampler
        d = datetime.now()
        dd = d.strftime('%m_%d_%Y_%H%M%S')
        self.logger.info( 'The sampler is updated by Serial_MCMC @ %s.' % dd )
    
    
    @property
    def cpu( self ):
        cpuCount = os.cpu_count()
        self.logger.info("Number of CPUs in the system: %s" % cpuCount)
        return cpuCount
    
    
    def verify_cpu( self, thread_num ):
        r"""
            *** Verify if the request threads exiting the system config. *** # Utility
        """
        if thread_num <= self.cpu:
            status = True
        else:
            status = False
        return status
    
    
    @property
    def serial_time( self ):
        return self._serial_time
    
    
    @property
    def sampler( self ):
        return self._sampler
    
    
    @property
    def discard_step( self ):
        return self._discard_step
    
    
    @property
    def thin_step( self ):
        return self._thin_step
    
    
    # Steps modifier
    def set_steps( self, discard_step, thin_step ):
        self._discard_step = discard_step
        self._thin_step = thin_step
    
    
    @property
    def chain( self ):
        """
        Flat modifed chain, if you want the original chain please use chain_raw
        """
        return self._chain
    
    
    @property
    def chain_raw( self ):
        if self.sampler is None:
            chain_raw = None
        else:
            chain_raw = self.sampler.get_chain()
        return chain_raw
    
    
    # Custom chain modifier
    def param_chain(self, discard_step, thin_step):
        r"""
         The parameter thin allows the user to specify if and how much the MCMC chains should be 
         thinned out before storing them. By default thin = 1 is used, which corresponds to 
         keeping all values. A value thin = 10 would result in keeping every 10th value and 
         discarding all other values.
        """
        self._discard_step = discard_step
        self._thin_step = thin_step
        self._chain = self.sampler.get_chain(discard=self.discard_step, thin=self.thin_step, flat=True)
        self.logger.info("Get the Markov chain customly by discarding %s intial steps (out of %s steps) & keeping very nth value(n=%s)" %  (self.discard_step, self.nsteps, self.thin_step) )
        return self._chain
    
    
    # Auto chain modifier
    def param_chain_auto( self ):
        tau = self.sampler.get_autocorr_time()
        self._discard_step = int(tau.mean() * 2)
        self._thin_step = int(tau.mean() / 2)
        self._chain = self.sampler.get_chain(discard=self.discard_step, thin=self.thin_step, flat=True)
        self.logger.info("Get the Markov chain automatically by discarding %s intial steps (out of %s steps) & keeping very nth value(n=%s)" %  (self.discard_step, self.nsteps, self.thin_step) )
        return self._chain
    
    
    @property
    def results( self ):
        return self._results
    
    
    # Results modifier
    def get_results(self, quantiles=None, interpolation="midpoint"):
        if not quantiles:
            quantiles = np.array([0.16, 0.5, 0.84])*100
        results = np.zeros((self.ndim, len(quantiles)))
        for i in range(self.ndim):
            results[i] = np.percentile(self.chain[:,i], quantiles,interpolation=interpolation)
            self.logger.info('Fitting results of %s: %s' % ( self.param_names[i], results[i] ))
        self._results = results
        d = datetime.now()
        dd = d.strftime('%m_%d_%Y_%H%M%S')
        self.logger.info( 'The result is calculated using chain(%s, %s) @ %s.' % ( self.discard_step, self.thin_step, dd) )
        return results
    
    
    def plot_results(self, legend, width_inch=6, legend_fontsize=18, label_size=18, axs_fontsize=18, smooth_order=0, smooth_scale=0.3, smooth=False, show_setting=False, fig_path=None):
        r"""
        + Smooth > https://getdist.readthedocs.io/en/latest/plot_gallery.html
        """
        if smooth:
            samples = MCSamples(samples=self.chain,names=self.param_names, labels=self.param_names, settings={'mult_bias_correction_order':smooth_order,'smooth_scale_2D':smooth_scale, 'smooth_scale_1D':smooth_scale})
            self.logger.info('Smooth is on, plot w/ (%s, %s, %s).' % (smooth_order, smooth_scale, smooth_scale ))
        else:
            samples = MCSamples(samples=self.chain,names=self.param_names, labels=self.param_names)
            self.logger.info('Default settings attempt to minimize sampling noise and bias.')
        
        g = plots.get_subplot_plotter(width_inch=width_inch)
        g.settings.legend_fontsize = legend_fontsize
        g.settings.axes_labelsize = label_size
        g.settings.linewidth = 1.5
        g.settings.axes_fontsize = axs_fontsize
        g.settings.alpha_filled_add = 0.7
        g.settings.figure_legend_frame = False
        g.figure_legend_ncol = 1
        if show_setting:
            g.show_all_settings()
        g.triangle_plot(samples, filled=True, legend_labels=[legend])
        if fig_path:
            g.export(fig_path)
    
    
    def plot_density( self, raw=True ):
        r"""
            Default raw=True, plot raw chain density
        """
        fig, axes = plt.subplots( self.ndim, figsize=(10, 7), sharex=True )
        labels = self.param_names
        if raw:
            chain = self.chain_raw
            self.logger.info('Plot the density distribution of the raw chain')
        else:
            chain = self.chain
            self.logger.info('Plot the density distribution of the custom chain (%s,%s)' % (self.discard_steps, self.thin_steps) )
        for i in range(self.ndim):
            ax = axes[i]
            ax.hist(chain[:, :, i].flatten(), 100)
            #ax.set_xlim(0, len(chain))
            ax.set_xlabel(labels[i])
            ax.set_ylabel('Prob(%s)' % labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        
    
    def plot_steps( self ):
        r"""
            Plot raw chain steps
        """
        fig, axes = plt.subplots( self.ndim, figsize=(10, 7), sharex=True )
        samples = self.chain_raw
        labels = self.param_names
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number");

    
    # Alias
    param_result_plot = plot_results
    param_fit_result = get_results
    
    #======================= Auto correlation plot =======================
    
    def plot_autocorr( self ):
        chain = self.chain_raw[:, :, 0].T
        N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), 10)).astype(int)
        gw2010 = np.empty(len(N))
        new = np.empty(len(N))
        for i, n in enumerate(N):
            gw2010[i] = self.autocorr_gw2010(chain[:, :n])
            new[i] = self.autocorr_new(chain[:, :n])

        # Plot the comparisons
        plt.loglog(N, gw2010, "o-", label="G&W 2010")
        plt.loglog(N, new, "o-", label="new")
        ylim = plt.gca().get_ylim()
        plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
        plt.ylim(ylim)
        plt.xlabel("number of samples, $N$")
        plt.ylabel(r"$\tau$ estimates")
        plt.legend(fontsize=14);
    
    
    # Automated windowing procedure following Sokal (1989)
    def auto_window( self, taus, c ):
        m = np.arange(len(taus)) < c * taus
        if np.any(m):
            return np.argmin(m)
        return len(taus) - 1
    

    # Following the suggestion from Goodman & Weare (2010)
    def autocorr_gw2010( self, y, c=5.0 ):
        f = self.autocorr_func_1d(np.mean(y, axis=0))
        taus = 2.0 * np.cumsum(f) - 1.0
        window = self.auto_window(taus, c)
        return taus[window]
    

    def autocorr_func_1d( self, x, norm=True ):
        x = np.atleast_1d(x)
        if len(x.shape) != 1:
            raise ValueError("invalid dimensions for 1D autocorrelation function")
            self.loger.Error("invalid dimensions for 1D autocorrelation function")
        n = self.next_pow_two(len(x))

        # Compute the FFT and then (from that) the auto-correlation function
        f = np.fft.fft(x - np.mean(x), n=2 * n)
        acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
        acf /= 4 * n
        # Optionally normalize
        if norm:
            acf /= acf[0]
        return acf
    
    
    def autocorr_new( self, y, c=5.0 ):
        f = np.zeros(y.shape[1])
        for yy in y:
            f += self.autocorr_func_1d(yy)
        f /= len(y)
        taus = 2.0 * np.cumsum(f) - 1.0
        window = self.auto_window(taus, c)
        return taus[window]
    
    
    def next_pow_two( self, n ):
        i = 1
        while i < n:
            i = i << 1
        return i


# ======================= MCMC_dev ======================= #
class MCMC_bk:
    r"""
        Backup version. A emcee realization of MCMC Paramization method
        w/ Parameter-limit, Logging, & Multi-threading support.
        This is an updated version of Hekun Lee's mcmc_sampler.
    """
    def __init__( self, nwalkers=None, nsteps=None, 
                 ndim=None, param_num=None, param_len=None,
                 param_lim=None, initial=None, vars_list=None,
                 serial_time=None, thread_num=None,
                 lim_=False, multi_=True,
                 level_='debug', stdout_=False):
        
        # Initialize the logger
        self.level_ = level_
        self.stdout_ = stdout_
        self.logname = 'MCMC' + '_ndim_' + str(ndim) + '_nwalkers_' + str(nwalkers) + '_nsteps_' + str(nsteps) + '_lim_' + str(lim_) + '_multi_' + str(multi_) + '_thread_num_' + str(thread_num) + '_'
        
        loging = Log( self.logname, self.level_, self.stdout_ )
        self.logger = loging.logger
        
        self.nwalkers = nwalkers            # Num of walkers
        self.nsteps = nsteps                # Num of steps
        self._ndim = ndim                   # Num of dim in the Param space
        
        self.logger.info( ' nwalkers is initialized as %s. ' % self.nwalkers )
        self.logger.info( ' nsteps is initialized as %s. ' % self.nsteps )
        self.logger.info( ' ndim of the Param space is initialized as %s. ' % self._ndim )
        
        self.param_num = param_num          # Num of params
        self.param_len = param_len          # Shape of the param
        self._param_lim = param_lim         # Param limit
        self.logger.info( ' %s parmas of %s dimension w/ value limit of %s is initialized. ' % ( self.param_num, self.param_len, self._param_lim ) )
        self.vars_list = [vars_list]
        
        self.thread_num = thread_num
        self._serial_time = serial_time
        self.lim_ = lim_                    # Param limit handler
        self.multi_ = multi_                # Multi-threading handler
        
        self._initial = initial
        self.logger.info( 'Input initial params are:\n %s' % self._initial)
        
        self._sampler = None
        self.chain = None

    
    @property
    def ndim( self ):
        if self._ndim is None:
            self._ndim = self.param_len * self.param_num
            #self.logger.info('Ndim = %s is calcaulated using Param_len & Param_num ' % self._ndim)
        else:
            if self._ndim != self.param_len * self.param_num:
                self.logger.error('Ndim %s, Param_len %s, and Param_num %s are not compatible! \n Please reset those prams to get Ndim = Param_len * Param_num.' % ( self.ndim, self.param_len, self.param_num ))
                raise ValueError
            else:
                self._ndim = self._ndim
                #self.logger.info(' Ndim = %s is verified using Param_len & Param_num ' % self._ndim)
        return self._ndim
    
    
    # Initial_params method
    def initial( self ):
        r"""
            *** Initialize the parameters *** # Utility
            !!! The outputs are passing to self.initial_params @property !!! # Note
            +++  +++ # Todo
        """
        self.logger.info(' Initialize the parameters.')
        if self._initial is None:
            self.logger.info(' No user input parameters, generate the initial params on the fly.')
            if self.lim_:
                self.logger.info(' Param_lim flag is on, initialize the parameters w/ limits.')
                initial_params = np.zeros((self.nwalkers, self.ndim))
                for i in range(self.ndim):
                    a, b = self.param_lim[i]
                    initial_params[:, i] = np.random.uniform(a, b, self.nwalkers)
            else:
                self.logger.info(' Param_lim flag is off, initialize the parameters randomly. ')
                np.random.seed(42)
                initial_params = np.random.randn(self.nwalkers, self.ndim)
        else:
            self.logger.info(' Using user input parameters, passing the initial params.')
            initial_params = self._initial
        self.logger.info( ' Initialization is done, initial params are:\n %s' % initial_params )
        return initial_params
    
    
    @property
    def initial_params( self ):
        initial_params = self.initial()
        return initial_params
    
    
    @property
    def param_lim( self ):
        if len(self._param_lim) != self.ndim:
            self.logger.error("Ndim %s & input Param_lim %s are not compatible! Param_lim length %s should equals Ndim." % (self.ndim, self._param_lim, len(self._param_lim) ))
            raise ValueError
        else:
            #self.logger.info(' Using user input parameter limit:\n %s.' % (self._param_lim,) )
            return self._param_lim
    
    
    def log_prob_demo( self, theta ):
        r"""
            *** Demo log_prob for debug or testing. *** # Utility
            !!! This is a do nothing function w/ no real meaning params. !!! # Note
            +++ Add other demos & print demo to show log prob examples! +++ # Todo
        """
        self.logger.info('Calls the demo log_prob for debug or testing.')
        t = time.time() + np.random.uniform(0.005, 0.008)
        while True:
            if time.time() >= t:
                break
        return -0.5 * np.sum(theta**2)
    
    
    def log_prob(self, params, vars_list):
        r"""
            *** Add parameter limits & filter NaN output of your model function *** # Utility
            !!!  ⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇ !!! # Note
            - Vars_list:
                - The chisq_function and the additional function_variables are in 'vars_list';
                - The chisq_funtion should be defined as "def chisq_fun(params, func_vars);
                - func(param, var) w/ (x, y, yerr = var) should be passing as [func, xi, yi, yierr];
            - Params:
                - params are the fitting parameters, which will be sampled by MCMC;
            !!!  ⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆ !!! # Note
            +++  +++ # Todo
        """
        if self.lim_:
            #self.logger.info( ' Param_lim flag is on using the parameter limits! ' )
            lp = 0
            for i in range(self.ndim):
                low_bound, upper_bound = self.param_lim[i]
                if low_bound < params[i] < upper_bound:
                    lp += 0
                else:
                    lp = -np.inf
            if np.isfinite(lp):
                if len(vars_list) == 1:
                    chi_sq_func = vars_list[0]
                    chisq = chi_sq_func(params)
                    if np.isnan(chisq):
                        self.logger.info( ' Hit NaN at w/ params %s. ' % params )
                        chisq = -np.inf
                        self.logger.info( 'NaN filtering in action: %s. ' % chisq )
                else:
                    chi_sq_func, func_vars = vars_list[0], vars_list[1:]
                    chisq = chi_sq_func(params, func_vars)
                    if np.isnan(chisq):
                        self.logger.info( ' Hit NaN at w/ params %s. ' % params )
                        chisq = -np.inf
                        self.logger.info( 'NaN filtering in action: %s. ' % chisq )
                return lp + chisq
            else:
                return -np.inf
        else:
            #self.logger.info( ' Param_lim flag is off, sample the parameters freely! ' )
            if len(vars_list) == 1:
                chi_sq_func = vars_list[0]
                chisq = chi_sq_func(params)
                if np.isnan(chisq):
                    self.logger.info( ' Hit NaN at w/ params %s. ' % params )
                    chisq = -np.inf
                    self.logger.info( 'NaN filtering in action: %s. ' % chisq )
            else:
                chi_sq_func, func_vars = vars_list[0], vars_list[1:]
                chisq = chi_sq_func(params, func_vars)
                if np.isnan(chisq):
                    self.logger.info( ' Hit NaN at w/ params %s. ' % params )
                    chisq = -np.inf
                    self.logger.info( 'NaN filtering in action: %s. ' % chisq )
            return chisq
    
    
    def run_MCMC( self ):
        r"""
            *** Calls & runs the respective MCMC sampler *** # Utility
            !!!  !!! # Note
            +++ Add pool options +++ # Todo
        """
        if self.multi_:
            self.logger.info(' Multi_threading flag is on, calls the MCMC sampler w/ Pool Parallelization.')
            self.Pool_MCMC( self.thread_num )
        else:
            self.logger.info(' Multi_threading flag is off, calls the Serial MCMC sampler.')
            self.Serial_MCMC()
    
    
    def Pool_MCMC( self, thread_num ):
        r"""
            *** emcee MCMC sampler w/ Pool Parallelization. *** # Utility
            !!!  !!! # Note
            +++ Add multi-time property +++ # Todo
        """
        status = self.verify_cpu( thread_num )
        if status:
            self.logger.info( "Request %s threads, which confines within total (%s) cpu resources." % (thread_num, self.cpu) )
            thread_num = thread_num
        else:
            # Use half of the cores
            self.logger.info( "Request %s threads, which exceeds total (%s) cpu resources." % (thread_num, self.cpu) )
            thread_num = int( self.cpu / 2 )
            self.logger.info( "Using %s threads, which is half of the total (%s) cpu resources." % (thread_num, self.cpu) )
        with Pool(thread_num) as pool:
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_prob, args=self.vars_list, pool=pool)
            start = time.time()
            self.logger.info( 'Starting the MCMC sampling progress.' )
            sampler.run_mcmc(self.initial_params, self.nsteps, progress=True)
            end = time.time()
            self.logger.info( 'Ending the MCMC sampling progress.' )
            multi_time = end - start
            self.logger.info("Multiprocessing took {0:.1f} seconds".format(multi_time))
        self._sampler = sampler
        d = datetime.now()
        dd = d.strftime('%m_%d_%Y_%H%M%S')
        self.logger.info( 'The sampler is updated by Pool_MCMC @ %s.' % dd )
    
    
    def Serial_MCMC( self ):
        r"""
            *** emcee MCMC sampler without Pool Parallelization. *** # Utility
            !!!  !!! # Note
            +++ Add serial-time property +++ # Todo
        """
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_prob, args=self.vars_list)
        self.logger.info( 'Starting the MCMC sampling progress.' )
        start= time.time()
        sampler.run_mcmc(self.initial_params, self.nsteps, progress=True)
        end = time.time()
        self.logger.info( 'Ending the MCMC sampling progress.' )
        serial_time = end - start
        self.logger.info("Serial took {0:.1f} seconds".format(serial_time))
        self._sampler = sampler
        d = datetime.now()
        dd = d.strftime('%m_%d_%Y_%H%M%S')
        self.logger.info( 'The sampler is updated by Serial_MCMC @ %s.' % dd )
    
    
    @property
    def cpu( self ):
        cpuCount = os.cpu_count()
        self.logger.info("Number of CPUs in the system: %s" % cpuCount)
        return cpuCount
    
    
    def verify_cpu( self, thread_num ):
        r"""
            *** Verify if the request threads exiting the system config. *** # Utility
        """
        if thread_num <= self.cpu:
            status = True
        else:
            status = False
        return status
    
    
    @property
    def serial_time( self ):
        return self._serial_time
    
    
    @property
    def sampler( self ):
        return self._sampler
    
    
    def Single_test( self ):
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_prob, args=self.vars_list)
        start = time.time()
        sampler.run_mcmc(self.initial_params, self.nsteps, progress=True)
        end = time.time()
        serial_time = end - start
        print("Serial took {0:.1f} seconds".format(serial_time))
        self._serial_time = serial_time
        self._sampler = sampler
    
    
    def Pool_test( self, thread_num ):
        status = self.verify_cpu( thread_num )
        if status:
            thread_num = thread_num
        else:
            # Use half of the cores
            thread_num = int( self.cpu / 2 )
        with Pool(thread_num) as pool:
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_prob, args=self.vars_list, pool=pool)
            start = time.time()
            sampler.run_mcmc(self.initial_params, self.nsteps, progress=True)
            end = time.time()
            multi_time = end - start
            print("Multiprocessing took {0:.1f} seconds".format(multi_time))
            print("{0:.1f} times of serial".format(self.serial_time / multi_time))
        self._sampler = sampler
    
    
    def MultiPool_test( self, thread_num ):
        status = self.verify_cpu( thread_num )
        if status:
            thread_num = thread_num
        else:
            # Use half of the cores
            thread_num = int( self.cpu / 2 )
        with MultiPool(thread_num) as pool:
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_prob, args=self.vars_list, pool=pool)
            start = time.time()
            sampler.run_mcmc(self.initial_params, self.nsteps, progress=True)
            end = time.time()
            multi_time = end - start
            print("Multiprocessing took {0:.1f} seconds".format(multi_time))
            print("{0:.1f} times of serial".format(self.serial_time / multi_time))
        self._sampler = sampler
    
    
    def ProcessPool_test( self, thread_num ):
        status = self.verify_cpu( thread_num )
        if status:
            thread_num = thread_num
        else:
            # Use half of the cores
            thread_num = int( self.cpu / 2 )
        with ProcessPool( thread_num ) as pool:
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_prob, args=self.vars_list, pool=pool)
            start = time.time()
            self.sampler.run_mcmc(self.initial_params, self.nsteps, progress=True)
            end = time.time()
            multi_time = end - start
            print("Multiprocessing took {0:.1f} seconds".format(multi_time))
            print("{0:.1f} times of serial".format(self.serial_time / multi_time))
        self._sampler = sampler
    
    
    def param_chain(self, discard_step, thin_step):
        self.chain = self.sampler.get_chain(discard=discard_step, thin=thin_step, flat=True)
        return self.chain
    
    
    def param_chain_auto(self, print_info=False):
        tau = self.sampler.get_autocorr_time()
        discard_step = int(tau.mean() * 2)
        thin_step = int(tau.mean() / 2)
        if print_info:
            strs = "Corr step: "
            for i in range(self.ndim):
                strs += "%.2f "%tau[i]
            strs += ". Discard: %.2f steps. Thin: %.2f"%(discard_step, thin_step)
            print(strs)
        self.chain = self.sampler.get_chain(discard=discard_step, thin=thin_step, flat=True)
        return self.chain
    
    
    def param_fit_result(self, quantiles=None, interpolation="midpoint"):
        if not quantiles:
            quantiles = np.array([0.16, 0.5, 0.84])*100
        results = np.zeros((self.ndim, len(quantiles)))
        for i in range(self.ndim):
            results[i] = np.percentile(self.chain[:,i], quantiles,interpolation=interpolation)
        return results
    
    
    def param_result_plot(self, param_names, legend, width_inch=6, legend_fontsize=18, label_size=18, axs_fontsize=18, show_setting=False, fig_path=None):
        samples = MCSamples(samples=self.chain,names=param_names, labels=param_names)

        g = plots.get_subplot_plotter(width_inch=width_inch)
        g.settings.legend_fontsize = legend_fontsize
        g.settings.axes_labelsize = label_size
        g.settings.linewidth = 1.5
        g.settings.axes_fontsize = axs_fontsize
        g.settings.alpha_filled_add = 0.7
        g.settings.figure_legend_frame = False
        g.figure_legend_ncol = 1
        if show_setting:
            g.show_all_settings()
        g.triangle_plot(samples, filled=True, legend_labels=[legend])
        if fig_path:
            g.export(fig_path)