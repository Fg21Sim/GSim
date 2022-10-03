# Copyright (c) 2020-2022 Chenxi SHAN <cxshan@hey.com>
# MCMC models

__version__ = "0.3"
__date__    = "2022-08-05"

import numpy as np
import scipy.stats as st
from sklearn import metrics

import numpy as np
import scipy.stats as st
from sklearn import metrics

class Fitfunc:
    r"""
        Offers a selection of fitting function for GalacticSim
    """
    
    def __init__( self, tol=0.0001 , multi_=False, size=1000 ):
        self.multi_ = multi_        # Using multiple maps
        self.tol = tol              # Tolerance of accuracy
        self.size = size            # Size of the input image
    
    
    def set_map( self, gss=None, ori=None ):
        global g_gssmap, g_orimap
        g_gssmap = gss
        g_orimap = ori
        
        
    def set_cl( self, cl=None, l=None ):
        global g_cl, g_l
        g_cl = cl
        g_l = l
    
    
    def diff( self, x ):
        r"""
        *** Print the stat difference ***
        """
        if self.multi_:
            for i in g_gssmap:
                ssmap = x[0] * g_gssmap[i] * g_orimap[i] ** x[1]
                newmap = ssmap + g_orimap[i]
                orimean = np.mean( g_orimap[i] )
                orivar = np.var( g_orimap[i] )
                oriskew = st.skew( g_orimap[i] )
                orikurt = st.kurtosis( g_orimap[i] )
                
                print( 'Map patch: [%s]' % i )
                print( '========Mean========' )
                print( 'ssmap:', np.mean(ssmap) )
                print( 'newmap', np.mean(newmap) )
                print( 'orimap', orimean )
                print( 'Mean_diff', np.mean(newmap)-orimean )
                print( 'Mean_Tol', self.tol, orimean*self.tol )

                print( '========Var========' )
                print( 'ssmap:', np.var(ssmap) )
                print( 'newmap', np.var(newmap) )
                print( 'orimap', orivar )
                print( 'Var_diff', np.var(newmap)-orivar ) 
                print( 'Var_Tol', self.tol, orivar*self.tol )

                print( '========Skew========' )
                print( 'ssmap:', st.skew(ssmap) )
                print( 'newmap', st.skew(newmap) )
                print( 'orimap', oriskew )
                print( 'Skew_diff', st.skew(newmap)-oriskew )
                print( 'Skew_Tol', self.tol, oriskew*self.tol )

                print( '========Kurt========' )
                print( 'ssmap:', st.kurtosis(ssmap) )
                print( 'newmap', st.kurtosis(newmap) )
                print( 'orimap', orikurt )
                print( 'Kurt_diff', st.kurtosis(newmap)-orikurt )
                print( 'Kurt_Tol', self.tol, oriskew*self.tol )
                plt.hist( orimap, alpha=0.3, color='c', edgecolor='k' )
                plt.hist( newmap, alpha=0.3 )
            
        else:
            ssmap = x[0] * g_gssmap * g_orimap ** x[1]
            newmap = ssmap + g_orimap
            orimean = np.mean( g_orimap )
            orivar = np.var( g_orimap )
            oriskew = st.skew( g_orimap )
            orikurt = st.kurtosis( g_orimap )
            
            print( '========Mean========' )
            print( 'ssmap:', np.mean(ssmap) )
            print( 'newmap', np.mean(newmap) )
            print( 'orimap', orimean)
            print( 'Mean_diff', np.mean(newmap)-orimean )
            print( 'Mean_Tol', self.tol, orimean*self.tol )
            
            print( '========Var========' )
            print( 'ssmap:', np.var(ssmap) )
            print( 'newmap', np.var(newmap) )
            print( 'orimap', orivar )
            print( 'Var_diff', np.var(newmap)-orivar ) 
            print( 'Var_Tol', self.tol, orivar*self.tol )
            
            print( '========Skew========' )
            print( 'ssmap:', st.skew(ssmap) )
            print( 'newmap', st.skew(newmap) )
            print( 'orimap', oriskew )
            print( 'Skew_diff', st.skew(newmap)-oriskew )
            print( 'Skew_Tol', self.tol, oriskew*self.tol )
            
            print( '========Kurt========' )
            print( 'ssmap:', st.kurtosis(ssmap) )
            print( 'newmap', st.kurtosis(newmap) )
            print( 'orimap', orikurt )
            print( 'Kurt_diff', st.kurtosis(newmap)-orikurt )
            print( 'Kurt_Tol', self.tol, oriskew*self.tol )
            plt.hist( orimap, alpha=0.3, color='c', edgecolor='k' )
            plt.hist( newmap, alpha=0.3 )

    # ================= Cl Model =================
    
    def cl_likelihood( self, x ):
        r"""
        *** Linear relation of the Angular power spectrum in the log-log space ***
        """
        model = x[0] * np.log10( g_l ) + x[1]
        return -0.5 * np.sum( ( np.log10( g_cl ) - model ) ** 2 )
    

    def plot_cl( self, x, lmin, lmax ):
        r"""
        *** Plot the Cl-l function function in the log-log space ***
        """
        lower = np.log10( lmin ) - 0.2
        upper = np.log10( lmax ) + 0.2
        xfid = np.linspace( lower, upper )
        plt.plot( np.log10( g_l ), np.log10( g_cl ), c='coral' )
        plt.plot( xfid, x[0]*xfid + x[1], c='c', linestyle=':' )
        plt.xlabel( 'Log(l)' )
        plt.ylabel( 'Log(Cl)' )
        plt.show()
    
    
    def fitcl( self, lmin, lmax ):
        r"""
        *** Linear regression of Cl-l function in the log-log space w/ plot overview ***
        """
        lower = np.log10( lmin ) - 0.2
        upper = np.log10( lmax ) + 0.2
        xfid = np.linspace( lower, upper )
        slope, intercept, r_value, p_value, std_err = linregress(np.log10( g_l ), np.log10( g_cl ))
        plt.plot( np.log10( g_l ), np.log10( g_cl ), c='coral' )
        plt.plot( xfid, xfid*slope+intercept, c='darkolivegreen', linestyle='--' )
        plt.xlabel( 'Log(l)' )
        plt.ylabel( 'Log(Cl)' )
        plt.show()
        return slope, intercept, r_value, p_value, std_err
    
    # ================= Model =================
    
    def full( self, x ):
        r"""
        *** Using the full stat difference [ mean, variance, skewness, kurtosis ] ***
        """
        if self.multi_:
            stat = []
            for i in g_gssmap:
                newmap = x[0] * g_gssmap[i] * g_orimap[i] ** x[1] + g_orimap[i]
                s = ( np.mean( newmap ) - np.mean( g_orimap[i] ) ) **2 + \
                     np.sqrt( ( np.var( newmap ) - np.var( g_orimap[i] ) ) **2 ) + \
                     ( st.skew( newmap ) - st.skew( g_orimap[i] ) ) **2 + \
                     ( st.kurtosis( newmap ) - st.kurtosis( g_orimap[i] ) )**2
                stat.append( s )
            stat = sum( stat )
        else:
            newmap = x[0] * g_gssmap * g_orimap ** x[1] + g_orimap
            stat = ( np.mean( newmap ) - np.mean( g_orimap ) ) **2 + \
                    np.sqrt( ( np.var( newmap ) - np.var( g_orimap ) ) **2 ) + \
                    ( st.skew( newmap ) - st.skew( g_orimap ) ) **2  + \
                    (st.kurtosis( newmap ) - st.kurtosis( g_orimap ))**2
        return - 0.25 * stat
    
    
    def single( self, x, type_='m' ):
        r"""
        *** Using the one of the stat difference [ mean, variance, skewness, kurtosis ] ***
        type_ = 'mean', 'var', 'skew', 'kurt' or 'm', 'v', 's', 'k'
        """
        if type_ in { 'mean', 'var', 'skew', 'kurt', 'm', 'v', 's', 'k' }:
            if type_ in { 'mean', 'm' }:
                if self.multi_:
                    stat = []
                    for i in g_gssmap:
                        newmap = x[0] * g_gssmap[i] * g_orimap[i] ** x[1] + g_orimap[i]
                        s = ( np.mean( newmap ) - np.mean( g_orimap[i] ) ) **2
                        stat.append( s )
                    stat = sum( stat )
                else:
                    newmap = x[0] * g_gssmap * g_orimap ** x[1] + g_orimap
                    stat = ( np.mean( newmap ) - np.mean( g_orimap ) ) **2
            elif type_ in { 'var', 'v' }:
                if self.multi_:
                    stat = []
                    for i in g_gssmap:
                        newmap = x[0] * g_gssmap[i] * g_orimap[i] ** x[1] + g_orimap[i]
                        s =  ( np.var( newmap ) - np.var( g_orimap[i] ) ) **2 
                        stat.append( s )
                    stat = sum( stat )
                else:
                    newmap = x[0] * g_gssmap * g_orimap ** x[1] + g_orimap
                    stat = ( np.var( newmap ) - np.var( g_orimap ) ) **2
            elif type_ in { 'skew', 's' }:
                if self.multi_:
                    stat = []
                    for i in g_gssmap:
                        newmap = x[0] * g_gssmap[i] * g_orimap[i] ** x[1] + g_orimap[i]
                        s = ( st.skew( newmap ) - st.skew( g_orimap[i] ) ) **2
                        stat.append( s )
                    stat = sum( stat )
                else:
                    newmap = x[0] * g_gssmap * g_orimap ** x[1] + g_orimap
                    stat = ( st.skew( newmap ) - st.skew( g_orimap ) ) **2
            elif type_ in { 'kurt', 'k' }:
                if self.multi_:
                    stat = []
                    for i in g_gssmap:
                        newmap = x[0] * g_gssmap[i] * g_orimap[i] ** x[1] + g_orimap[i]
                        s = ( st.kurtosis( newmap ) - st.kurtosis( g_orimap[i] ) )**2
                        stat.append( s )
                    stat = sum( stat )
                else:
                    newmap = x[0] * g_gssmap * g_orimap ** x[1] + g_orimap
                    stat = (st.kurtosis( newmap ) - st.kurtosis( g_orimap ))**2
            return - stat
    
    
    def com_full_tost( self, x ):
        r"""
        A combined model using full stat difference [ mean, variance, skewness, kurtosis ], MMD, & TOST;
        TOST is using for set the tolerance;
        """
        if self.multi_:
            stat = []
            
            for i in g_gssmap:
                ssmap = x[0] * g_orimap[i] * g_orimap[i] ** x[1]
                newmap = ssmap + g_orimap[i]
                # The TOST Test
                mean = np.mean(g_orimap[i])
                low = -mean * self.tol
                upp = mean * self.tol
                t1, pv1 = st.ttest_1samp( ssmap, low, alternative='greater' )
                t2, pv2 = st.ttest_1samp( ssmap, upp, alternative='less' )
                p = max(pv1/2, pv2/2)
                tost = 1 - p

                if p > 0.005:
                    diff_p = 1000000000
                else:
                    diff_p = 0

                # Full stats
                diff_mean = ( np.mean( newmap ) - np.mean( g_orimap[i] ) ) ** 2
                diff_var = ( np.var( newmap ) - np.var( g_orimap[i] ) ) ** 2
                diff_skew = ( st.skew( newmap ) - st.skew( g_orimap[i] ) ) ** 2
                diff_kurt = (st.kurtosis( newmap ) - st.kurtosis( g_orimap[i] ) ) ** 2
            
                s = diff_mean + diff_var + diff_skew + diff_kurt + diff_p
                stat.append( s )
                
            stat = sum( stat )
        else:
            ssmap = x[0] * g_gssmap * g_orimap ** x[1]
            newmap = ssmap + g_orimap
            # The TOST Test
            mean = np.mean(g_orimap)
            low = -mean * self.tol
            upp = mean * self.tol
            t1, pv1 = st.ttest_1samp( ssmap, low, alternative='greater' )
            t2, pv2 = st.ttest_1samp( ssmap, upp, alternative='less' )
            p = max(pv1/2, pv2/2)
            tost = 1 - p
            
            if p > 0.005:
                diff_p = 1000000000
            else:
                diff_p = 0
            
            # Full stats
            diff_mean = ( np.mean( newmap ) - np.mean( g_orimap ) ) ** 2
            diff_var = ( np.var( newmap ) - np.var( g_orimap ) ) ** 2
            diff_skew = ( st.skew( newmap ) - st.skew( g_orimap ) ) ** 2
            diff_kurt = (st.kurtosis( newmap ) - st.kurtosis( g_orimap ) ) ** 2
            
            stat = diff_mean + diff_var + diff_skew + diff_kurt + diff_p
            
        return - 0.25 * stat
    
    
    def combine( self, x ):
        r"""
        A combined model using full stat difference [ mean, variance, skewness, kurtosis ], MMD, & TOST;
        TOST is using for set the tolerance;
        """
        if self.multi_:
            stat = []
            
            for i in g_gssmap:
                ssmap = x[0] * g_gssmap[i] * g_orimap[i] ** x[1]
                newmap = ssmap + g_orimap[i]
                # The TOST Test
                mean = np.mean(g_orimap[i])
                low = -mean * self.tol
                upp = mean * self.tol
                t1, pv1 = st.ttest_1samp( ssmap, low, alternative='greater' )
                t2, pv2 = st.ttest_1samp( ssmap, upp, alternative='less' )
                p = max(pv1/2, pv2/2)
                tost = 1 - p

                if p > 0.005:
                    diff_p = 1000000000
                else:
                    diff_p = 0

                # MMD
                ori_vector = g_orimap[i].copy()
                ori_vector = ori_vector.reshape( self.size, self.size )
                new_vector = newmap.copy()
                new_vector = new_vector.reshape( self.size, self.size )
                mml = self.mmd_linear( new_vector, ori_vector )

                # Full stats
                diff_mean = ( np.mean( newmap ) - np.mean( g_orimap[i] ) ) ** 2
                diff_var = np.sqrt( ( np.var( newmap ) - np.var( g_orimap[i] ) ) ** 2 )
                diff_skew = np.cbrt( ( st.skew( newmap ) - st.skew( g_orimap[i] ) ) ** 2 )
                diff_kurt = np.sqrt( np.sqrt( (st.kurtosis( newmap ) - st.kurtosis( g_orimap[i] ) ) ** 2 ) )
            
                s = ( diff_mean + mml ) / 2 + diff_var + diff_skew + diff_kurt + diff_p
                stat.append( s )
                
            stat = sum( stat )
        else:
            ssmap = x[0] * g_gssmap * g_orimap ** x[1]
            newmap = ssmap + g_orimap
            # The TOST Test
            mean = np.mean(g_orimap)
            low = -mean * self.tol
            upp = mean * self.tol
            t1, pv1 = st.ttest_1samp( ssmap, low, alternative='greater' )
            t2, pv2 = st.ttest_1samp( ssmap, upp, alternative='less' )
            p = max(pv1/2, pv2/2)
            tost = 1 - p
            
            if p > 0.005:
                diff_p = 1000000000
            else:
                diff_p = 0
            
            # MMD
            ori_vector = g_orimap.copy()
            ori_vector = ori_vector.reshape( self.size, self.size )
            new_vector = newmap.copy()
            new_vector = new_vector.reshape( self.size, self.size )
            mml = self.mmd_linear( new_vector, ori_vector )
            
            # Full stats
            diff_mean = ( np.mean( newmap ) - np.mean( g_orimap ) ) ** 2
            diff_var = np.sqrt( ( np.var( newmap ) - np.var( g_orimap ) ) ** 2 )
            diff_skew = np.cbrt( ( st.skew( newmap ) - st.skew( g_orimap ) ) ** 2 )
            diff_kurt = np.sqrt( np.sqrt( ( st.kurtosis( newmap ) - st.kurtosis( g_orimap ) ) ** 2 ) )
            
            stat = ( diff_mean + mml ) / 2 + diff_var + diff_skew + diff_kurt + diff_p
            
        return - 0.25 * stat
    
    
    # ================= MMD =================
    
    def MMD( self, x, shape=500 ):
        r"""
        *** Calculate the maximum mean discrepancy ***
        """
        ssmap = x[0] * g_gssmap * g_orimap ** x[1]
        newmap = ssmap + g_orimap
        ori_vector = g_orimap.copy()
        ori_vector = ori_vector.reshape( shape[0],shape[1] )
        new_vector = newmap.copy()
        new_vector = new_vector.reshape( shape[0],shape[1] )

        mml = self.mmd_linear( new_vector, ori_vector )
        mmg = self.mmd_rbf( new_vector, ori_vector )

        return mml, mmg
    
    
    def mmd_linear( self, X, Y ):
        r"""MMD using linear kernel (i.e., k(x,y) = <x,y>)
        Note that this is not the original linear MMD, only the reformulated and faster version.
        The original version is:
            def mmd_linear(X, Y):
                XX = np.dot(X, X.T)
                YY = np.dot(Y, Y.T)
                XY = np.dot(X, Y.T)
                return XX.mean() + YY.mean() - 2 * XY.mean()
        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]
        Returns:
            [scalar] -- [MMD value]
        """
        delta = X.mean(0) - Y.mean(0)
        return delta.dot(delta.T)


    def mmd_rbf( self, X, Y, gamma=1.0):
        r"""MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]
        Keyword Arguments:
            gamma {float} -- [kernel parameter] (default: {1.0})
        Returns:
            [scalar] -- [MMD value]
        """
        XX = metrics.pairwise.rbf_kernel(X, X, gamma)
        YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
        XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
        return XX.mean() + YY.mean() - 2 * XY.mean()


    def mmd_poly( self, X, Y, degree=2, gamma=1, coef0=0 ):
        r"""MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]
        Keyword Arguments:
            degree {int} -- [degree] (default: {2})
            gamma {int} -- [gamma] (default: {1})
            coef0 {int} -- [constant item] (default: {0})
        Returns:
            [scalar] -- [MMD value]
        """
        XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
        YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
        XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
        return XX.mean() + YY.mean() - 2 * XY.mean()
    
    # ================= Stat Tests =================
    
    def Ttest( self, x ):
        r"""
        a.k.a Wilcoxon signed-rank Test >> Mean Test for ND
        Uses the non-param wilcoxon test from Scipy
        """
        if self.multi_:
            tt = []
            for i in g_gssmap:
                newmap = x[0] * g_gssmap[i] * g_orimap[i] ** x[1] + g_orimap[i]
                t = st.wilcoxon(newmap, g_orimap[i], mode = 'approx')
                tt.append(t)
        else:
            newmap = x[0] * g_gssmap * g_orimap ** x[1] + g_orimap
            tt = st.wilcoxon(newmap, g_orimap, mode = 'approx')
        return tt
    
    
    def TOSTest( self, x ):
        r"""
        *** Equivalence test, test if two mean difference are within the tolerance range! ***
        """
        if self.multi_:
            tost = []
            for i in g_gssmap:
                ssmap = x[0] * g_gssmap * g_orimap ** x[1]
                mean = np.mean(g_orimap)

                # The TOST Test
                low = -mean * self.tol
                upp = mean * self.tol
                t1, pv1 = st.ttest_1samp( ssmap, low, alternative='greater' )
                t2, pv2 = st.ttest_1samp( ssmap, upp, alternative='less' )
                p = max(pv1/2, pv2/2)
                t = 1 - p
            tost.append(t)
        else:
            ssmap = x[0] * g_gssmap * g_orimap ** x[1]
            mean = np.mean(g_orimap)

            # The TOST Test
            low = -mean * self.tol
            upp = mean * self.tol
            t1, pv1 = st.ttest_1samp( ssmap, low, alternative='greater' )
            t2, pv2 = st.ttest_1samp( ssmap, upp, alternative='less' )
            p = max(pv1/2, pv2/2)
            tost = 1 - p
        return tost
    
    
    def Ftest( self, x ):
        r"""
        a.k.a Variance Test
        Uses the non-param ansari test from Scipy
        The high-res map & low-res map are considered as independent obs from
        a high-res telescope & a low-res telescope
        """    
        if self.multi_:
            aa = []
            for i in g_gssmap:
                newmap = x[0] * g_gssmap[i] * g_orimap[i] ** x[1] + g_orimap[i]
                a = st.ansari(newmap,  g_orimap[i])
                aa.append( a )
        else:
            newmap = x[0] * g_gssmap * g_orimap ** x[1] + g_orimap
            aa = st.ansari( newmap, g_orimap )
        return aa
    

    def EStest( self, x ):
        r"""
        Uses the non-param epps_singleton_2samp test from Scipy
        The high-res map & low-res map are considered as independent obs from
        a high-res telescope & a low-res telescope
        """
        if self.multi_:
            ee = []
            for i in g_gssmap:
                newmap = x[0] * g_gssmap[i] * g_orimap[i] ** x[1] + g_orimap[i]
                e = st.epps_singleton_2samp( newmap, g_orimap[i] )
                ee.append( e )
        else:
            newmap = x[0] * g_gssmap * g_orimap ** x[1] + g_orimap
            ee = st.epps_singleton_2samp( newmap, g_orimap )
        return ee

    
    def KStest( self, x ):
        r"""
        Uses the non-param Kolmogorov-Smirnov test from Scipy
        The high-res map & low-res map are considered as independent obs from
        a high-res telescope & a low-res telescope
        """
        if self.multi_:
            kk = []
            for i in g_gssmap:
                newmap = x[0] * g_gssmap[i] * g_orimap[i] ** x[1] + g_orimap[i]
                k = st.ks_2samp( newmap, g_orimap[i] )
                kk.append( k )
        else:
            newmap = x[0] * g_gssmap * g_orimap ** x[1] + g_orimap
            kk = st.ks_2samp( newmap, g_orimap )
        return kk