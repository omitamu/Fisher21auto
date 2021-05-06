import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline1d
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from scipy.interpolate import interp1d
import os, copy
import scipy
import pylab

import HI_experiments
# import cross_bispec
from scipy.stats import chi2


# KAVI
import settings
import binning
import cosmological_parameters
import power_spec_functions
import sys
sys.path.append('cosmolopy')
import cosmolopy.distance as cd
import cosmolopy.density as density
cosmo = {'omega_M_0':cosmological_parameters.om_m, 'omega_lambda_0':cosmological_parameters.om_L, 'omega_k_0':0.0, 'h':cosmological_parameters.h}

output_dir = 'Output/'

if not os.path.exists(output_dir):
		os.makedirs(output_dir)

SAVEFIG = settings.SAVEFIG
SHOWFIG = settings.SHOWFIG

### SET UP LENSING KERNEL GLOBALLY
# k_lensing_arr = np.linspace(0.0,3.0,100000)
# CMB_lensing_kernel_arr =  lensing_kernel.get_lensing_kernel_fourier_sp(k_lensing_arr) # only good up to k=1.17 # smooth function

### SET UP EXPT GLOBALLY 		# >>>>> REMOVE EVENTUALLY
expt=HI_experiments.getHIExptObject(settings.expt_name, settings.expt_mode)
fsky = expt.getfsky()
Sarea_sr = 4.*np.pi*fsky



#############################################################################################################################################
######### Kappa-Kappa AUTO POWER SPECTRUM, 21-21 AUTO POWER SPECTRUM, 21-Kappa CROSS POWER SPECTRUM, 21-21-Kappa CROSS BISPECTRUM,  #########

def plot_ellipse(semimaj=1,semimin=1,phi=0,x_cent=0,y_cent=0,theta_num=1e3,ax=None,plot_kwargs=None,\
                    fill=False,fill_kwargs=None,data_out=False,cov=None,mass_level=0.68):
    '''
        An easy to use function for plotting ellipses in Python 2.7!

        The function creates a 2D ellipse in polar coordinates then transforms to cartesian coordinates.
        It can take a covariance matrix and plot contours from it.

        semimaj : float
            length of semimajor axis (always taken to be some phi (-90<phi<90 deg) from positive x-axis!)

        semimin : float
            length of semiminor axis

        phi : float
            angle in radians of semimajor axis above positive x axis

        x_cent : float
            X coordinate center

        y_cent : float
            Y coordinate center

        theta_num : int
            Number of points to sample along ellipse from 0-2pi

        ax : matplotlib axis property
            A pre-created matplotlib axis

        plot_kwargs : dictionary
            matplotlib.plot() keyword arguments

        fill : bool
            A flag to fill the inside of the ellipse

        fill_kwargs : dictionary
            Keyword arguments for matplotlib.fill()

        data_out : bool
            A flag to return the ellipse samples without plotting

        cov : ndarray of shape (2,2)
            A 2x2 covariance matrix, if given this will overwrite semimaj, semimin and phi

        mass_level : float
            if supplied cov, mass_level is the contour defining fractional probability mass enclosed
            for example: mass_level = 0.68 is the standard 68% mass

    '''

    # Get Ellipse Properties from cov matrix
    if cov is not None:
        eig_vec,eig_val,u = np.linalg.svd(cov)
        # Make sure 0th eigenvector has positive x-coordinate
        if eig_vec[0][0] < 0:
            eig_vec[0] *= -1
        semimaj = np.sqrt(eig_val[0])
        semimin = np.sqrt(eig_val[1])
        if mass_level is None:
            multiplier = np.sqrt(2.279)
        else:
            distances = np.linspace(0,20,20001)
            chi2_cdf = chi2.cdf(distances,df=2)
            multiplier = np.sqrt(distances[np.where(np.abs(chi2_cdf-mass_level)==np.abs(chi2_cdf-mass_level).min())[0][0]])
        semimaj *= multiplier
        semimin *= multiplier
        phi = np.arccos(np.dot(eig_vec[0],np.array([1,0])))
        if eig_vec[0][1] < 0 and phi > 0:
            phi *= -1

    # Generate data for ellipse structure
    theta = np.linspace(0,2*np.pi,int(theta_num))
    r = 1 / np.sqrt((np.cos(theta))**2 + (np.sin(theta))**2)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    data = np.array([x,y])
    S = np.array([[semimaj,0],[0,semimin]])
    R = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
    T = np.dot(R,S)
    data = np.dot(T,data)
    data[0] += x_cent
    data[1] += y_cent

    # Output data?
    if data_out == True:
        return data

    # Plot!
    return_fig = False
    if ax is None:
        return_fig = True
        fig,ax = plt.subplots(figsize=(100,100))
        ax.locator_params(axis='x',tight=True,nbins=1)
        ax.tick_params(axis='both', which='minor', labelsize=13)
        #ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useOffset=False)

    if plot_kwargs is None:
        ax.plot(data[0],data[1],color='b',linestyle='-')
        ax.locator_params(axis='x',tight=True,nbins=1)
        ax.tick_params(axis='both', which='minor', labelsize=13)
        #ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useOffset=False)
    else:
        ax.plot(data[0],data[1],**plot_kwargs)
        ax.locator_params(axis='x',tight=True,nbins=1)
        ax.tick_params(axis='both', which='minor', labelsize=13)
        #ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useOffset=False)

    if fill == True:
        ax.fill(data[0],data[1],**fill_kwargs)
        ax.locator_params(axis='x',tight=True,nbins=1)
        ax.tick_params(axis='both', which='minor', labelsize=13)
        #ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useOffset=False)

    if return_fig == True:
        return fig
    ax.locator_params(axis='x',tight=True,nbins=1)
    ax.tick_params(axis='both', which='minor', labelsize=13)
    #ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useOffset=False)


def plot_2d_contour(twod_mean, twod_cov_matrix):

	nsig1,nsig2 = np.sqrt(np.array([6.17, 2.3]))

	x0,y0 = twod_mean
	sigxsq, sigysq, sigxy= twod_cov_matrix[0,0], twod_cov_matrix[1,1], twod_cov_matrix[0,1]
	asq, bsq = 0.5*(sigxsq + sigysq) + 0.25 * np.sqrt((sigxsq - sigysq)**2 + sigxy**2), 0.5*(sigxsq + sigysq) - 0.25 * np.sqrt((sigxsq - sigysq)**2 + sigxy**2)

	sin2ang = 2 * sigxy ; cos2ang = sigxsq - sigysq
	ang = np.arctan(cos2ang/sin2ang) / 2.0

	print 	(sin2ang, cos2ang)
	#raise KeyboardInterrupt

	Ellipse = matplotlib.patches.Ellipse
	patch = Ellipse(xy=(x0,y0), width=2*nsig1*np.sqrt(asq), height=2*nsig1*np.sqrt(bsq), angle=ang*180/np.pi, fill=True)
	print (2*nsig1*np.sqrt(asq), 2*nsig1*np.sqrt(bsq), ang*180/np.pi)
	#patch.set_fc('b')
	#patch.set_ec('b')
	#patch.set_alpha(1)
	#patch.set_zorder(1)
	#patch.set_lw(1)
	fig = plt.figure(0)
	ax = fig.add_subplot(111)
	patch.set_lw(2)
	patch.set_clip_box(ax.bbox)
	patch.set_alpha(1)
	patch.set_facecolor([0.1,0.2,0.3])
	ax.add_artist(patch)


	plt.show()

	#gca().add_patch(patch)

def logpk_derivative(pk, kgrid):
    """
    Calculate the first derivative of the (log) power spectrum,
    d log P(k) / d k. Sets the derivative to zero wherever P(k) is not defined.

    Parameters
    ----------

    pk : function
        Callable function (usually an interpolation fn.) for P(k)

    kgrid : array_like
        Array of k values on which the integral will be computed.
    """
    # Calculate dlog(P(k))/dk using central difference technique
    # (Sets lowest-k values to zero since P(k) not defined there)
    # (Suppresses div/0 error temporarily)
    dk = 1e-7
    np.seterr(invalid='ignore')
    dP = pk(kgrid + 0.5*dk) / pk(kgrid - 0.5*dk)
    np.seterr(invalid=None)
    dP[np.where(np.isnan(dP))] = 1. # Set NaN values to 1 (sets deriv. to zero)
    dlogpk_dk = np.log(dP) / dk
    return dlogpk_dk

def HI_angular_noise_ell_y(ells, zbin_prop,T_obs = 2., experiment_name='hirax', mode='interferometer', show_plots=False):   # independent of y
	#be careful of ells, if ell goes too high then n(u) is zero and 1/n(u) makes no sense. Noise will be dodgy
   z_A = zbin_prop.z_A
   nu=1420e6/(z_A+1.)
   expt=HI_experiments.getHIExptObject(experiment_name, mode, tobsyears = T_obs)
   noise_HI_ell_y=expt.getNoiseAngularPower(ells, nu,0.4e6)  #400e6 is bandwidth in Hz but not used if we use Bull expression. It will become important if you change the getNoiseAngularPower function to use

   return noise_HI_ell_y#*(9.)

def HI_angular_noise_ell_y_allsky(HI_angular_noise_ell_y):
    return HI_angular_noise_ell_y/fsky

def Cl_21_auto_ell_y_zA(ell_arr, y_arr, zbin_prop, bias_var=1., gamma_var=1.): # 21cm angular PS at z vs ell,y

   n_ell = ell_arr.size ; n_y = y_arr.size
   z_A = zbin_prop.z_A
   chi_A = zbin_prop.chi_A ; nutilde_A = zbin_prop.nutilde_A ; rnu_A = zbin_prop.rnu_A
   kperp_arr = ell_arr/chi_A; kpar_arr = y_arr/rnu_A

   ktot_2d_arr = np.sqrt(np.outer(ell_arr,np.ones(n_y))**2/chi_A**2 + np.outer(np.ones(n_ell),y_arr)**2/rnu_A**2)
   kpar_2d_arr = np.outer(np.ones(n_ell),kpar_arr)
   kperp_2d_arr = np.outer(kperp_arr,np.ones(n_y))
   mu_k_2d_arr = kpar_2d_arr/ktot_2d_arr

   F_bias_rsd_sq_2d_arr = (power_spec_functions.get_HI_bias(z_A, bias_var)+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2)**2
   P_21_tot_2d_arr = F_bias_rsd_sq_2d_arr *power_spec_functions.get_P_m_0(ktot_2d_arr) * \
		power_spec_functions.get_mean_temp(z_A)**2 * power_spec_functions.get_growth_function_D(z_A)**2  # mK^2

   Cl_21_auto_ell_y_2d_arr = P_21_tot_2d_arr / (chi_A**2 * rnu_A)

   return Cl_21_auto_ell_y_2d_arr

def EOS_parameters_derivs():

    C = 3e5*3.24e-23
    w0 = cosmological_parameters.w0; wa = cosmological_parameters.wa
    om = cosmological_parameters.om_m; ol = cosmological_parameters.om_L
    ok = 1. - om - ol

    # Omega_DE(a) and E(a) functions
    omegaDE = lambda a: ol * np.exp(3.*wa*(a - 1.)) / a**(3.*(1. + w0 + wa))
    E = lambda a: np.sqrt( om * a**(-3.) + ok * a**(-2.) + omegaDE(a) )

    # Derivatives of E(z) w.r.t. parameters
    #dE_omegaM = lambda a: 0.5 * a**(-3.) / E(a)
    if np.abs(ok) < 1e-7: # Effectively zero
        dE_omegak = lambda a: 0.5 * a**(-2.) / E(a)
    else:
        dE_omegak = lambda a: 0.5 * a**(-2.) / E(a) * (1. - 1./a)
    dE_omegaM = lambda a: 0.5 * a**(-3.) / E(a)
    dE_omegaDE = lambda a: 0.5 / E(a) * (1. - 1./a**3.)
    dE_w0 = lambda a: -1.5 * omegaDE(a) * np.log(a) / E(a)
    dE_wa = lambda a: -1.5 * omegaDE(a) * (np.log(a) + 1. - a) / E(a)

    # Bundle functions into list (for performing repetitive operations with them)
    fns = [dE_omegak, dE_omegaDE, dE_w0, dE_wa]
    #HH, rr, DD, ff = cosmo_fns

    aa = np.linspace(1., 1e-4, 500)
    zz = 1./aa - 1.
    EE = E(aa); fz = power_spec_functions.get_growth_factor_f(aa)#ff(aa)
    gamma = cosmological_parameters.gamma; H0 = 100. * cosmological_parameters.h; h = cosmological_parameters.h

    # Derivatives of apar w.r.t. parameters
    derivs_apar = [f(aa)/EE for f in fns]

    # Derivatives of f(z) w.r.t. parameters
    f_fac = -gamma * fz / EE
    df_domegak  = f_fac * (EE/om + dE_omegak(aa))
    df_domegaDE = f_fac * (EE/om + dE_omegaDE(aa))
    df_w0 = f_fac * dE_w0(aa)
    df_wa = f_fac * dE_wa(aa)
    df_dh = np.zeros(aa.shape)
    df_dgamma = fz * np.log(density.omega_M_z(zz, **cosmo))
    derivs_f = [df_domegak, df_domegaDE, df_w0, df_wa, df_dh, df_dgamma]

    # Calculate comoving distance (including curvature)
    r_c = scipy.integrate.cumtrapz(1./(aa**2. * EE), aa)
    r_c = np.concatenate(([0.], r_c))
    if ok > 0.:
        r = C/(H0*np.sqrt(ok)) * np.sinh(r_c * np.sqrt(ok))
    elif ok < 0.:
        r = C/(H0*np.sqrt(-ok)) * np.sin(r_c * np.sqrt(-ok))
    else:
        r = C/H0 * r_c

    # Perform integrals needed to calculate derivs. of aperp
    derivs_aperp = [(C/H0)/r[1:] * scipy.integrate.cumtrapz(f(aa)/(aa * EE)**2., aa)
                        for f in fns]

    # Add additional term to curvature integral (idx 1)
    # N.B. I think Pedro's result is wrong (for fiducial Omega_k=0 at least),
    # so I'm commenting it out
    #derivs_aperp[1] -= (H0 * r[1:] / C)**2. / 6.

    # Add initial values (to deal with 1/(r=0) at origin)
    inivals = [0.5, 0.0, 0., 0.]
    derivs_aperp = [ np.concatenate(([inivals[i]], derivs_aperp[i]))
                     for i in range(len(derivs_aperp)) ]

    # Add (h, gamma) derivs to aperp,apar
    derivs_aperp += [np.ones(aa.shape)/h, np.zeros(aa.shape)]
    derivs_apar  += [np.ones(aa.shape)/h, np.zeros(aa.shape)]

    # Construct interpolation functions
    interp_f     = [scipy.interpolate.interp1d(aa[::-1], d[::-1],
                    kind='linear', bounds_error=False) for d in derivs_f]
    interp_apar  = [scipy.interpolate.interp1d(aa[::-1], d[::-1],
                    kind='linear', bounds_error=False) for d in derivs_apar]
    interp_aperp = [scipy.interpolate.interp1d(aa[::-1], d[::-1],
                    kind='linear', bounds_error=False) for d in derivs_aperp]

    return [interp_f, interp_aperp, interp_apar]


def Cl21_Distance_derivs(zbin_prop,ell_arr, y_arr,fid_params, bias_var=1., gamma_var=1.):

   n_ell = ell_arr.size ; n_y = y_arr.size
   z_A = zbin_prop.z_A
   chi_A = zbin_prop.chi_A ; nutilde_A = zbin_prop.nutilde_A ; rnu_A = zbin_prop.rnu_A
   kperp_arr = ell_arr/chi_A; kpar_arr = y_arr/rnu_A
   apar = cosmological_parameters.apar ; aperp = cosmological_parameters.aperp

   ktot_2d_arr = np.sqrt(np.outer(ell_arr,np.ones(n_y))**2/chi_A**2 + np.outer(np.ones(n_ell),y_arr)**2/rnu_A**2)
   kpar_2d_arr = np.outer(np.ones(n_ell),kpar_arr)
   kperp_2d_arr = np.outer(kperp_arr,np.ones(n_y))
   mu_k_2d_arr = kpar_2d_arr/ktot_2d_arr
   fbao_2d_arr = power_spec_functions.get_fbao(ktot_2d_arr)
   alpha_fnl_2d_arr = 3e5**2.*2.*ktot_2d_arr**2.* power_spec_functions.get_transfer_function(ktot_2d_arr)*power_spec_functions.get_growth_function_D(z_A)\
    				  /(3. * cosmological_parameters.om_m * (cosmological_parameters.H0)**2.)
   beta_fnl = 2.*cosmological_parameters.delta_c_fnl* ( power_spec_functions.get_HI_bias(z_A,bias_var) - 1.)


   F_bias_rsd_2d_arr = power_spec_functions.get_HI_bias(z_A, bias_var)+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2
   F_bias_rsd_sq_2d_arr = F_bias_rsd_2d_arr**2
   F_bias_rsd_fnl_2d_arr = power_spec_functions.get_HI_bias(z_A, bias_var)+power_spec_functions.get_growth_factor_f(z_A, gamma_var)*mu_k_2d_arr**2 + \
   cosmological_parameters.f_nl * beta_fnl / alpha_fnl_2d_arr

   P_21_tot_2d_arr = F_bias_rsd_sq_2d_arr *power_spec_functions.get_P_m_0(ktot_2d_arr) * \
	power_spec_functions.get_mean_temp(z_A)**2 * power_spec_functions.get_growth_function_D(z_A)**2  # mK^2
   P_21_tot_2d_arr_fnl =  F_bias_rsd_fnl_2d_arr*power_spec_functions.get_P_m_0(ktot_2d_arr) * \
 	power_spec_functions.get_mean_temp(z_A)**2 * power_spec_functions.get_growth_function_D(z_A)**2

   Cl_21_auto_ell_y_2d_arr = P_21_tot_2d_arr / (chi_A**2 * rnu_A)
   Cl_21_auto_ell_y_2d_arr_fnl = P_21_tot_2d_arr_fnl / (chi_A**2 * rnu_A)

   f_zA = power_spec_functions.get_growth_factor_f(z_A, bias_var)
   drsd_du2 = 2.*f_zA/ F_bias_rsd_2d_arr

   h=cosmological_parameters.h
   kP=np.loadtxt('Datafiles/P_k_nonlin_camb_z0.dat')  # goes up to k=18/Mpc
   P_spl_k=interp1d(kP[:,0]/h, kP[:,1]*h**3,bounds_error=False,fill_value=0.0)


   dlogpk_dk = logpk_derivative(P_spl_k, ktot_2d_arr) # Numerical deriv.
   daperp_u2 = -2. * (kperp_2d_arr/kpar_2d_arr * aperp/apar * mu_k_2d_arr**2.)**2. / aperp
   dapar_u2 =   2. * (kperp_2d_arr/kpar_2d_arr* aperp/apar * mu_k_2d_arr**2.)**2. / apar
   daperp_k = (aperp*kperp_2d_arr)**2. / (ktot_2d_arr*aperp)
   dapar_k  = (apar*kpar_2d_arr)**2. / (ktot_2d_arr*apar)

   ### For nonlinear cases sigma_nl must add extra terms dlogpk_dk + drsd_dk + dbias_k below as in Bull et al.
   deriv_aperp = ( (2./aperp) + drsd_du2 * daperp_u2 \
                       + (dlogpk_dk)*daperp_k ) * Cl_21_auto_ell_y_2d_arr
   deriv_apar =  ( (1./apar)  + drsd_du2 * dapar_u2 \
                       + (dlogpk_dk)*dapar_k  ) * Cl_21_auto_ell_y_2d_arr

   deriv_f = 2.*mu_k_2d_arr**2./ F_bias_rsd_2d_arr *  Cl_21_auto_ell_y_2d_arr


   '''
   plt.plot(ell_arr, deriv_apar[:,100]**2.,color='C4', linestyle='-',label='Total Derivative Signal')
   plt.plot(ell_arr, deriv_apar[:,500]**2.,color='C4', linestyle='--',label='Total Derivative Signal')
   plt.plot(ell_arr, deriv_apar[:,1000]**2.,color='C4', linestyle='-.',label='Total Derivative Signal')
   plt.plot(ell_arr, deriv_apar[:,2000]**2.,color='C4', linestyle=':',label='Total Derivative Signal')

   plt.plot(ell_arr, Cl_21_auto_ell_y_2d_arr[:,100]**2.,color='C5', linestyle='-',label='21cm signal')
   plt.plot(ell_arr, Cl_21_auto_ell_y_2d_arr[:,500]**2.,color='C5', linestyle='--',label='21cm signal')
   plt.plot(ell_arr, Cl_21_auto_ell_y_2d_arr[:,1000]**2.,color='C5', linestyle='-.',label='21cm signal')
   plt.plot(ell_arr, Cl_21_auto_ell_y_2d_arr[:,2000]**2.,color='C5', linestyle=':',label='21cm signal')
   #plt.legend()
   from matplotlib.lines import Line2D
   leg0 = Line2D([0], [0], color='C0', lw=4, label='T1')
   leg1 = Line2D([0], [0], color='C1', lw=4, label='T2')
   leg2 = Line2D([0], [0], color='C2', lw=4, label='T3')
   leg3 = Line2D([0], [0], color='C3', lw=4, label='T4')
   leg4 = Line2D([0], [0], color='C4', lw=4, label='Total Derivative')
   leg5 = Line2D([0], [0], color='C5', lw=4, label='21cm Signal')
   #plt.legend(handles=[leg0, leg1, leg2, leg3, leg4])
   plt.legend(handles=[leg4, leg5])
   plt.xlabel('$\ell$')
   plt.yscale('log')
   plt.ylabel(r'21cm $\alpha_\parallel$ derivative term squared')
   plt.show()
   raise KeyboardInterrupt
   '''
   #Abao_zA,sig8, b1_zA, b2_zA,  f_zA, aperp, apar= fid_params
   f_zA, aperp, apar = fid_params
   #Abao_zA,sig8, f_zA, b1_zA, b2_zA = fid_params 
   #sig8_zA = sig8*power_spec_functions.get_growth_function_D(z_A)
   #f_zA, aperp, apar= fid_params

   #dC_ell_y_dAbao = Cl_21_auto_ell_y_2d_arr*fbao_2d_arr/(1.+fbao_2d_arr)

   #sig8_deriv_kernel = 2.0/(sig8_zA*power_spec_functions.get_growth_function_D(z_A))
   #dC_ell_y_dsig8_zA = Cl_21_auto_ell_y_2d_arr * sig8_deriv_kernel

   b2_deriv_kernel = 0.0
   dC_ell_y_db2_zA = Cl_21_auto_ell_y_2d_arr * b2_deriv_kernel

   b1_deriv_kernel = 2.0/F_bias_rsd_2d_arr
   dC_ell_y_db1_zA = Cl_21_auto_ell_y_2d_arr*b1_deriv_kernel

   fnl_deriv_kernel = 2.0*beta_fnl/(alpha_fnl_2d_arr*F_bias_rsd_fnl_2d_arr)
   dC_ell_y_dfnl_zA = Cl_21_auto_ell_y_2d_arr*fnl_deriv_kernel

   #return dC_ell_y_dAbao, dC_ell_y_dsig8_zA, dC_ell_y_db1_zA, dC_ell_y_db2_zA, deriv_f, deriv_aperp, deriv_apar
   return deriv_f, deriv_aperp, deriv_apar
   
   
def get_FisherMatrix_Cl_21_21_auto(zbin_prop, num_k_bins_ell = 15, num_k_bins_y=14):

	## SET Z BIN PROPERTIES

    Deltanutilde = zbin_prop.Deltanutilde    # Bin width
    deltanutilde = zbin_prop.deltanutilde	 # Channel width

    z_A = zbin_prop.z_A

    nutilde_A = zbin_prop.nutilde_A
    chi_A=zbin_prop.chi_A
    rnu_A=zbin_prop.rnu_A
    lambda_A = zbin_prop.lambda_A
    FOV_A_sr = zbin_prop.FOV_A_sr
    theta_min_sr = zbin_prop.theta_min_sr
    print('r_z_A, z_A', chi_A, z_A)
    print('Rnu_zA,zA', rnu_A, z_A)
    #raise KeyboardInterrupt

	### SET RADIAL MODES

    y_min = zbin_prop.y_min_expt ; y_max = zbin_prop.y_max_expt
    result = zbin_prop.get_y_arr_expt() ; n_y = result[0]  ; y_arr = result[1]

	### SET TRANSVERSE MODES

    ell_min = zbin_prop.ell_min_expt  ; ell_max = zbin_prop.ell_max_expt
    result = zbin_prop.get_ell_arr_expt() ; n_ell = result[0]  ; ell_arr = result[1]   # CHANGE THESE TO _expt
    print('n_y',n_y)
    N_patches = zbin_prop.N_patches
    Mode_Volume_Factor = zbin_prop.Mode_Volume_Factor
    print('Vol',0.5 * Mode_Volume_Factor/(2*np.pi**2))
	### COMPUTE AUTO-21-21 Fisher matrix for Tb, f, b1,b2 - MERGE ALL FISHER MATRIX CALCS INTO ONE METHOD

    sig8=cosmological_parameters.sig8
    Abao=cosmological_parameters.Abao
    aperp=cosmological_parameters.aperp
    apar=cosmological_parameters.apar

    ell_min_ltd = ell_min ; y_min_ltd = y_min
    ell_max_ltd = 0.01*num_k_bins_ell*chi_A + ell_min_ltd
    y_max_ltd = 0.01*num_k_bins_y*rnu_A + y_min_ltd #y_max#
    ell_arr_ltd = np.linspace(ell_min_ltd,ell_max_ltd,n_ell); y_arr_ltd = np.linspace(y_min_ltd,y_max_ltd,n_y)

    Omega_HI_zA=power_spec_functions.get_Omega_HI(z_A) ; f_zA = power_spec_functions.get_growth_factor_f(z_A, gamma_var=1.0)
    b1_zA=power_spec_functions.get_HI_bias(z_A, bias_var=1.0); b2_zA = power_spec_functions.get_HI_bias_2nd_order(z_A)

    #fid_fisher_params = np.array([Abao,sig8, b1_zA, b2_zA, f_zA, aperp, apar])	#, Omega_HI_zA # T_b(z_i), sig8, f, b1, b2   // T_b(z_i) in place of Omega_HI(z_i)
    fid_fisher_params = np.array([f_zA, aperp, apar])
    n_fisher_params = fid_fisher_params.size

    #Cl_Abao,Cl_sig8,Cl_f,Cl_b1,Cl_b2,Cl_aperp,Cl_apar = Cl21_Distance_derivs(zbin_prop, ell_arr_ltd, y_arr_ltd, fid_fisher_params, bias_var=1., gamma_var=1.)
    Cl_f,Cl_aperp,Cl_apar = Cl21_Distance_derivs(zbin_prop, ell_arr_ltd, y_arr_ltd, fid_fisher_params, bias_var=1., gamma_var=1.)
    #, Cl_Omega_HI
    # returns fractional derivatives
    Cl_signal = Cl_21_auto_ell_y_zA(ell_arr_ltd, y_arr_ltd, zbin_prop, bias_var=1., gamma_var=1.)

    #parameter_deriv_C_ell_y_arr = np.array([Cl_Abao,Cl_sig8,Cl_b1,Cl_b2,Cl_f,Cl_aperp,Cl_apar]) #,Cl_Omega_HI
    parameter_deriv_C_ell_y_arr = np.array([Cl_f,Cl_aperp,Cl_apar])
    
    ell_2d_arr_ltd=np.outer(ell_arr_ltd,np.ones(n_y))

    Cl_21_noise_ell_arr=HI_angular_noise_ell_y(ell_arr_ltd, zbin_prop, experiment_name='hirax', mode='interferometer', show_plots=False)
    Cl_21_noise_ell_y_2d_arr=  np.outer(Cl_21_noise_ell_arr,np.ones(n_y))
    Cl_21_noise_ell_y_2d_arr_allsky = HI_angular_noise_ell_y_allsky(Cl_21_noise_ell_y_2d_arr)
    #plt.loglog(ell_arr_ltd,Cl_signal[:,94],label='Our Cl21')
    #plt.loglog(ell_arr_ltd, Cl_21_noise_ell_arr,label='Our Noise')
    #Dpk = np.loadtxt('Devin_pk_y100.dat')
    #Dnk = np.loadtxt('Devin_nk_y100.dat')
    #np.savez('Cn21_kavi',ell_arr_ltd, y_arr_ltd,Cl_signal)
    #np.savez('Cl21_kavi',ell_arr_ltd, y_arr_ltd,Cl_21_noise_ell_y_2d_arr_allsky )
    #plt.loglog(Dnk[0],Dnk[1],label='Devin Noise Bull code')
    #plt.loglog(Dpk[0],Dpk[1],label='Devin Cl21 Bull code')
    #plt.xlim(ell_arr_ltd[0],ell_arr_ltd[-1])
    #plt.xlabel(r'$\ell$')
    #plt.ylabel(r"$C_\ell^{21}$")
    #plt.ylim(1e-12,1e-7)
    #plt.legend()
    #plt.show()
    #print('y is ', y_arr_ltd[94]) ; raise KeyboardInterrupt
    Cl_21_auto_ell_y_2d_arr=Cl_21_auto_ell_y_zA(ell_arr_ltd, y_arr_ltd, zbin_prop, bias_var=1., gamma_var=1.) # 21cm angular PS at z vs ell,y

    variance_arr = (Cl_21_noise_ell_y_2d_arr_allsky + Cl_21_auto_ell_y_2d_arr)**2.

    fisher_arr_zA_21_21 = np.zeros((n_fisher_params,n_fisher_params)) ; corr_arr_zA_21_21 = np.zeros((n_fisher_params,n_fisher_params))
    for ii, pp_i in enumerate(fid_fisher_params):
        deriv_arr_ii = parameter_deriv_C_ell_y_arr[ii,]
        for jj, pp_j in enumerate(fid_fisher_params):
            deriv_arr_jj = parameter_deriv_C_ell_y_arr[jj,]
            signal_arr_sq = deriv_arr_ii * deriv_arr_jj
            fisher_ratio_arr = signal_arr_sq/variance_arr
            #Vsur = 467407923.347056
            fisher_arr_zA_21_21[ii,jj] = 0.5 * Mode_Volume_Factor/(2*np.pi**2) * np.trapz(np.trapz(fisher_ratio_arr*ell_2d_arr_ltd, ell_arr_ltd, axis=0), y_arr_ltd, axis=0)
            #for ii, pp_i in enumerate(fid_fisher_params[:n_fisher_params-1]):
		#for jj, pp_j in enumerate(fid_fisher_params[:n_fisher_params-1]):
			#corr_arr_zA_21_21[ii,jj]=fisher_arr_zA_21_21[ii,jj]/np.sqrt(np.abs(fisher_arr_zA_21_21[ii,ii]*fisher_arr_zA_21_21[jj,jj]))

    #fid_param_label = ['Abao','sig8', 'f', 'b1', 'b2', 'aperp', 'apar'] #,'Omega_HI
    fid_param_label = ['f', 'aperp', 'apar']
    #fid_param_label = ['BHI_Omega_HI', 'f_Omega_HI']
    print ('z=', z_A)
    #print ('Fid values Abao,sig8, f, b1, b2, fid_fisher_params #, Omega_HI')
    print ('Fisher matrix', fisher_arr_zA_21_21)
	#print corr_arr_zA_21_21

    print ('Amp SN', np.sqrt(0.5 * Mode_Volume_Factor/(2*np.pi**2) * \
        np.trapz(np.trapz(Cl_signal**2/variance_arr*ell_2d_arr_ltd, ell_arr_ltd, axis=0), y_arr_ltd, axis=0)))

	#print 'z=', z_A
	#print 'Fid values sig8, f, b1, b2', fid_fisher_params
	#np.savetxt('z_index_' + str(zbin_index) + '_Fisher_21_21_auto.txt',str(zbin_prop.z_A))
    np.savetxt('z_index_' + str(zbin_index) + '_fiducial_params_s8_f_b1_b2.txt',fid_fisher_params)

    np.savetxt('z_index_' + str(zbin_index) + '_Fisher_21_21_auto.txt',fisher_arr_zA_21_21)
    np.savetxt('z_index_' + str(zbin_index) + '_Fisher_21_21_auto_2params.txt',fisher_arr_zA_21_21)

    return fid_param_label, fid_fisher_params, fisher_arr_zA_21_21

	###### raise KeyboardInterrupt

	#variance_auto_arr = HIRAX_noise**2 + Cly_21**2
	#plt.loglog(ell_arr,Cly_21) ; plt.loglog(ell_arr,HIRAX_noise/(fsky * (delta_ell*ell_arr) ));  plt.loglog(ell_arr,np.sqrt(variance_auto_arr/(fsky * (delta_ell*ell_arr) ))) ; plt.show()
	#SN_sq_auto_ell_y = fsky * delta_y_target * (delta_ell*ell_arr)*(Cly_21/np.sqrt(variance_arr))**2
	#cumulative_SN_auto_ell_y = np.sqrt(np.cumsum(SN_sq_auto_ell_y,axis=0)) # total SN over all ell for one y
	
def expand_fisher_matrix(zbin_prop, derivs, F, names, exclude=[]):
    """
    Transform Fisher matrix to with (f, aperp, apar) parameters into one with
    dark energy EOS parameters (Omega_k, Omega_DE, w0, wa, h, gamma) instead.

    Parameters
    ----------

    z : float
        Central redshift of the survey.

    derivs : 2D list of interp. fns.
        Array of interpolation functions used to evaluate derivatives needed to
        transform to new parameter set. Precompute this using the
         function.

    F : array_like
        Fisher matrix for the old parameters.

    names : list
        List of names of the parameters in the current Fisher matrix.

    exclude : list, optional
        Prevent a subset of the functions [f, aperp, apar] from being converted
        to EOS parameters. e.g. exclude = [1,] will prevent aperp from
        contributing to the EOS parameter constraints.

    Returns
    -------

    Fnew : array_like
        Fisher matrix for the new parameters.

    paramnames : list, optional
        Names parameters in the expanded Fisher matrix.
    """
    z_A = zbin_prop.z_A
    a = 1. / (1. + z_A)

    # Define mapping between old and new Fisher matrices (including expanded P(k) terms)
    old = copy.deepcopy(names)
    Nold = len(old)
    #oldidxs = [old.index(p) for p in ['f', 'aperp', 'apar']]
    oldidxs = [old.index(p) for p in ['f', 'aperp', 'apar']]

    # Insert new parameters immediately after 'apar'
    new_params = [r'$\Omega_k$', r'$\Omega_\Lambda$', 'w0', 'wa', 'h']#, r'$\gamma$']
    new = old[:old.index('apar')+1]
    #new += new_params
    new = new_params
    new += old[old.index('apar')+1:]
    newidxs = [new.index(p) for p in new_params]
    Nnew = len(new)

    # Construct extension operator, d(f,aperp,par)/d(beta)
    S = np.zeros((Nold, Nnew))
    for i in range(Nold):
      for j in range(Nnew):
        # Check if this is one of the indices that is being replaced
        if i in oldidxs and j in newidxs:
            # Old parameter is being replaced
            ii = oldidxs.index(i) # newidxs
            jj = newidxs.index(j)
            if ii not in exclude:
                S[i,j] = derivs[ii][jj](a)
        else:
            if old[i] == new[j]: S[i,j] = 1.

    # Multiply old Fisher matrix by extension operator to get new Fisher matrix
    Fnew = np.dot(S.T, np.dot(F, S))
    Omega_DE = cosmological_parameters.om_L
    Omega_k = cosmological_parameters.om_k
    w0 = cosmological_parameters.w0
    wa = cosmological_parameters.wa
    h = cosmological_parameters.h
    gamma = cosmological_parameters.gamma

    Fid = np.array([ Omega_k,Omega_DE, w0, wa, h])#, gamma])
    return  new, Fid, Fnew

def plot_ellipse_sub(twod_mean, twod_cov_matrix, twod_label, figsub):
    figsub.set_xlabel(twod_label[0],fontsize=10)
    figsub.set_ylabel(twod_label[1],fontsize=10)
    plot_kwargs_b = {'color':'b','linestyle':'-','linewidth':3,'alpha':0.8} ;	fill_kwargs_b = {'color':'b','alpha':0.5}
    plot_kwargs_r = {'color':'r','linestyle':'-','linewidth':3,'alpha':0.8} ;	fill_kwargs_r = {'color':'r','alpha':0.5}
    plot_ellipse(x_cent=twod_mean[0], y_cent=twod_mean[1], ax = figsub, cov=twod_cov_matrix, mass_level=0.67, plot_kwargs=plot_kwargs_b,fill=True,fill_kwargs=fill_kwargs_b)
    plot_ellipse(x_cent=twod_mean[0], y_cent=twod_mean[1], ax = figsub, cov=twod_cov_matrix, mass_level=0.95, plot_kwargs=plot_kwargs_r,fill=True,fill_kwargs=fill_kwargs_r)

def plot_cov_ellipses(lbl, zA, param_mean, param_Fisher_matrix, param_label):

    param_cov = scipy.linalg.inv(param_Fisher_matrix)

    fig = plt.figure()
    for i in range(param_mean.size):
       for j in range(param_mean.size):
           if i < j:
               figsub=plt.subplot2grid((param_mean.size,param_mean.size), (j,i))
               sig = np.sqrt(np.abs(param_cov[i,i]))
               twod_mean=np.array([param_mean[i],param_mean[j]])
               twod_label=np.array([param_label[i],param_label[j]])
               twod_cov_matrix = np.array([[param_cov[i,i],param_cov[i,j]], [param_cov[j,i],param_cov[j,j]]])
               plot_ellipse_sub(twod_mean, twod_cov_matrix, twod_label, figsub)
               if j != param_mean.size-1 :
                  figsub.set_xticklabels([])
                  figsub.tick_params(axis='both', which='major', labelsize=16)
               if i !=0  :
                  figsub.set_yticklabels([])
                  figsub.set_yticks([])
                  figsub.set_ylabel(str( ))
                  figsub.tick_params(axis='both', which='major', labelsize=16)
               else:
                  figsub.set_ylabel(param_label[j],fontsize=15)
               #figsub.set_xlim(param_mean[i]- 0.05,param_mean[i]+0.05)
               #if j == param_mean.size-1:
                   #figsub.set_ylim(param_mean[j]-1,param_mean[j]+1)
               #else:
                   #figsub.set_ylim(param_mean[j]-0.05,param_mean[j]+0.05)

           elif i==j:
               figsub=plt.subplot2grid((param_mean.size,param_mean.size), (j,i))
               sig = np.sqrt(np.abs(param_cov[i,i]))
               xx = np.linspace(param_mean[i]-2.*sig, param_mean[i]+2.*sig, 4000)
               yy = 1./np.sqrt(2.*np.pi*sig**2.) * np.exp(-0.5 * ((xx-param_mean[i])/sig)**2.)
               yy /= np.max(yy)
               figsub.plot(xx, yy, ls='solid', color='red', lw=1.5) #colours[k][0]
               figsub.set_xlabel(param_label[i],fontsize=15)
               if i is 0:
                  fig.legend()
                  figsub.tick_params(axis='both', which='major', labelsize=16)
               if i != param_mean.size-1:
                  figsub.set_xticklabels([])
                  figsub.tick_params(axis='both', which='major', labelsize=16)
               figsub.set_yticks([])
           #figsub.set_xlabel(param_label[i],fontsize=25)
           figsub.tick_params(axis='both', which='major', labelsize=16)
    fig.subplots_adjust(wspace=0, hspace=0)
    #fig.tight_layout()
    if SAVEFIG: plt.savefig(output_dir + lbl + '_z_'+np.str(zA)+'_Fisher_matrix.png')
    plt.show()
	
     
zbinning_struct = binning.setup_ZBinningStruct(expt)
zbin_index=3
'''
if len(sys.argv)==2:
    zbin_index= int(sys.argv[1])
else:
    zbin_index =2# int(sys.argv[1])
'''
'''
for i in range(1,4):
        zbin_index = i
        zbin_prop = binning.setup_ZBinProp(zbin_index, zbinning_struct, expt)
        fid_param_label, fid_fisher_params, fisher_arr_zA_21_21 = get_FisherMatrix_Cl_21_21_auto(zbin_prop)
        derivs = EOS_parameters_derivs()
        params, fid, Fnew = expand_fisher_matrix(zbin_prop,derivs,fisher_arr_zA_21_21,fid_param_label)
        np.savetxt('fisher21_'+str(zbin_index)+'.dat',Fnew)
        plot_cov_ellipses('21 auto',zbin_prop, fid, Fnew, params)
        
'''
zbin_prop = binning.setup_ZBinProp(zbin_index, zbinning_struct, expt)
print('z bin')
print(zbin_prop)
fid_param_label, fid_fisher_params, fisher_arr_zA_21_21 = get_FisherMatrix_Cl_21_21_auto(zbin_prop)
derivs = EOS_parameters_derivs()
params, fid, Fnew = expand_fisher_matrix(zbin_prop,derivs,fisher_arr_zA_21_21,fid_param_label)
np.savetxt('fisher21_zbin_index'+str(zbin_index)+'.dat',Fnew)
print('Parameters',params)
print('Fiducial values',fid)
plot_cov_ellipses('21auto',zbin_index, fid, Fnew, params)

        
