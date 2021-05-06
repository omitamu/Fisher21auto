import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline

import cosmological_parameters

import sys
sys.path.append('cosmolopy')
import cosmolopy.distance as cd
cosmo = {'omega_M_0':cosmological_parameters.om_m, 'omega_lambda_0':cosmological_parameters.om_L, 'omega_k_0':0.0, 'h':cosmological_parameters.h}

show_plots=False


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# power spec functions - simplify so less repetition, one function and tell what you want?
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def get_transfer_function(k):
    Tk = np.loadtxt('Datafiles/Transfer_function.dat')
    if show_plots:
        plt.plot(Tk[0],Tk[1])
        plt.xlabel('k')
        plt.ylabel('$T_k(k)$')
        plt.title('Transfer function')
    Tk_spl = interp1d(Tk[0],Tk[1],bounds_error=False,fill_value=0.0)
    return Tk_spl(k)

def get_2D_transfer_function(kperp,kpar):
    Tk = np.loadtxt('Datafiles/Transfer_function.dat')
    Kperp_arr, Kpar_arr = np.meshgrid(kperp,kpar)
    Tk_spl = interp1d(Tk[0],Tk[1],bounds_error=False,fill_value=0.0)
    K_vec = np.sqrt(Kperp_arr**2. + Kpar_arr**2.)
    Tk_spl_2d = RectBivariateSpline(kperp,kpar,Tk_spl(K_vec))
    return Tk_spl_2d(kperp,kpar)

def get_growth_function_D(z):
    zD=np.loadtxt('Datafiles/growth_function_D.dat')

    if show_plots:
        plt.plot(zD[:,0], zD[:,1])
        plt.xlim(0, 3)
        plt.xlabel('z')
        plt.ylabel('D')
        plt.title('Growth Function')
        #   plt.savefig('TestingPlots/growth_function_D')
        # plt.show() # Kavi

    z_in=zD[:,0]
    D=zD[:,1]
    D_spl_z=interp1d(z_in, D,bounds_error=False,fill_value=0.0)
    return D_spl_z(z)

def get_growth_function_D_chi(chi):
    zD=np.loadtxt('Datafiles/growth_function_D.dat')
    chis=cd.comoving_distance(zD[:,0], **cosmo)

#    if show_plots:
#        plt.plot(chis, zD[:,1])
#        plt.xlabel(r'$\chi$')
#        plt.ylabel('D')
#        plt.title('Growth Function')
#        plt.savefig('TestingPlots/growth_function_D_chi')
#        plt.show()  # Kavi

    D_spl_chi=interp1d(chis, zD[:,1],bounds_error=False,fill_value=0.0)
    return D_spl_chi(chi)

def get_growth_function_D_2D(zs):
    zD=np.loadtxt('Datafiles/growth_function_D.dat')
    z_in=zD[:,0]
    D=zD[:,1]
    D_spl_z=interp1d(z_in, D,bounds_error=False,fill_value=0.0)
    D=np.zeros(zs.shape)
    for i in range(zs.shape[1]):
        D[:,i]=D_spl_z(zs[:,i])
    return D

def get_growth_function_D_chi_2D(chi):
    zD=np.loadtxt('Datafiles/growth_function_D.dat')
    chis=cd.comoving_distance(zD[:,0], **cosmo)
    D_spl_chi=interp1d(chis, zD[:,1],bounds_error=False,fill_value=0.0)
    D=np.zeros(chi.shape)
    for i in range(chi.shape[1]):
        D[:,i]=D_spl_chi(chi[:,i])
    return D


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def get_HI_bias(z, bias_var=1.):
    zb=np.loadtxt('Datafiles/bias_HI.dat')
    #print('fit coeff', np.polyfit(zb[:,0],zb[:,1],1))
    if show_plots:
        plt.plot(zb[:,0], zb[:,1])
        plt.xlim(0, 3)
        plt.ylim(0,2)
        plt.xlabel('z')
        plt.ylabel('b')
        plt.title('HI bias')
        #    plt.savefig('TestingPlots/bias')
        plt.show()


    b_spl_z=interp1d(zb[:,0], zb[:,1],bounds_error=False)
    b=b_spl_z(z)

    b*=bias_var
    return b

def get_HI_bias_chi(chi, bias_var=1.):
    zb=np.loadtxt('Datafiles/bias_HI.dat')
    chis=cd.comoving_distance(zb[:,0], **cosmo)

    if show_plots:
        plt.plot(chis, zb[:,1])
        #plt.xlim(0, 7000)
        plt.ylim(0,2)
        plt.xlabel(r'$\chi$')
        plt.ylabel('b')
        plt.title('HI bias')
        plt.savefig('TestingPlots/bias_chi')
        plt.show()

    b_spl_chi=interp1d(chis, zb[:,1],bounds_error=False)
    b=b_spl_chi(chi)

    b*=bias_var
    return b

def get_HI_bias_2D(zs, bias_var=1.):
    zb=np.loadtxt('Datafiles/bias_HI.dat')
    b_spl_z=interp1d(zb[:,0], zb[:,1],bounds_error=False)
    b=np.zeros(zs.shape)
    for i in range(zs.shape[1]):
        b[:,i]=b_spl_z(zs[:,i])

    b*=bias_var
    return b

def get_HI_bias_chi_2D(chi, bias_var=1.):
    zb=np.loadtxt('Datafiles/bias_HI.dat')
    chis=cd.comoving_distance(zb[:,0], **cosmo)
    b_spl_chi=interp1d(chis, zb[:,1],bounds_error=False)
    b=np.zeros(chi.shape)
    for i in range(chi.shape[1]):
        b[:,i]=b_spl_chi(chi[:,i])



    b*=bias_var
    return b



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def get_HI_bias_2nd_order(z):
    zb=np.loadtxt('Datafiles/bias2_HI.dat')
    b_spl=interp1d(zb[:,0], zb[:,1],bounds_error=False)
    print(np.polyfit(zb[:,0],zb[:,1],3))
    #b2=get_HI_bias_chi(chi)**2

    if show_plots:
        plt.plot(zb[:,0], zb[:,1])
        plt.xlim(0, 3)
        plt.ylim(-1,1)
        plt.xlabel('z')
        plt.ylabel('b')
        plt.title('HI 2nd order bias')
        #    plt.savefig('TestingPlots/bias')
        plt.show()


    return b_spl(z)

def get_HI_bias_2nd_order_chi(chi):
    zb=np.loadtxt('Datafiles/bias2_HI.dat')
    chis=cd.comoving_distance(zb[:,0], **cosmo)
    b_spl_chi=interp1d(chis, zb[:,1],bounds_error=False)

    if show_plots:
        plt.plot(chi, b_spl_chi(chi))
        #plt.xlim(0, 3)
        plt.ylim(-1,0)
        plt.xlabel(r'$\chi$')
        plt.ylabel('b')
        plt.title('HI 2nd order bias')
        #    plt.savefig('TestingPlots/bias')
        plt.show()


    return b_spl_chi(chi)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def get_growth_factor_f(z, gamma_var=1.):
    Omega_m_z=cosmological_parameters.om_m*(1.0+z)**3*(cd.hubble_z(0, **cosmo)/cd.hubble_z(z, **cosmo))**2

    gamma = 0.55*gamma_var
    growth_factor = Omega_m_z ** gamma

    if show_plots:
        # zf=np.loadtxt('Datafiles/growth_factor_f.dat')

        #plt.plot(zf[:,0], zf[:,1])
        plt.plot(z, growth_factor)
        plt.xlim(0, 3)
        plt.ylim(0,1)
        plt.xlabel('z')
        plt.ylabel('f')
        plt.title('Growth factor f')
        #plt.savefig('TestingPlots/growth_factor_f')
        plt.show()


#    f_spl_z=interp1d(zf[:,0], zf[:,1])
    return growth_factor

def get_growth_factor_f_chi(chi, gamma_var=1.):
    z=np.linspace(0,3,1001)
    f=get_growth_factor_f(z, gamma_var)
    zf=np.zeros((f.size,2))
    zf[:,0]=z
    zf[:,1]=f
    #zf=np.loadtxt('Datafiles/growth_factor_f.dat')
    chis=cd.comoving_distance(zf[:,0], **cosmo)

    f_spl_chi=interp1d(chis, zf[:,1],bounds_error=False)


    if show_plots:

        plt.plot(chi, f_spl_chi(chi))
        #plt.xlim(0, 7000)
        plt.ylim(0,1)
        plt.xlabel(r'$\chi$')
        plt.ylabel('f')
        plt.title('Growth factor f')
        plt.savefig('TestingPlots/growth_factor_f_chi')
        plt.show()



    return f_spl_chi(chi)

def get_growth_factor_f_2D(zs, gamma_var=1.):
    z=np.linspace(0,3,1001)
    f=get_growth_factor_f(z, gamma_var)
    zf=np.zeros((f.size,2))
    zf[:,0]=z
    zf[:,1]=f
    #    zf=np.loadtxt('Datafiles/growth_factor_f.dat')
    f_spl_z=interp1d(zf[:,0], zf[:,1],bounds_error=False)
    f=np.zeros(zs.shape)
    for i in range(zs.shape[1]):
        f[:,i]=f_spl_z(zs[:,i])
    return f

def get_growth_factor_f_chi_2D(chi, gamma_var=1.):
    z=np.linspace(0,3,1001)
    f=get_growth_factor_f(z, gamma_var)
    zf=np.zeros((f.size,2))
    zf[:,0]=z
    zf[:,1]=f
    #    zf=np.loadtxt('Datafiles/growth_factor_f.dat')
    chis=cd.comoving_distance(zf[:,0], **cosmo)
    f_spl_chi=interp1d(chis, zf[:,1],bounds_error=False)
    f=np.zeros(chi.shape)
    for i in range(chi.shape[1]):
        f[:,i]=f_spl_chi(chi[:,i])
    return f



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def get_mean_temp(z):
    zT=np.loadtxt('Datafiles/mean_temp.dat')    #mK

    if show_plots:
        plt.plot(zT[:,0], zT[:,1])
        plt.xlim(0, 3)
        #plt.ylim(0,2)
        plt.xlabel('z')
        plt.ylabel('T [mK]')
        plt.title('Mean Temperature')
        #plt.savefig('TestingPlots/mean_temp')
        plt.show()


    T_spl_z=interp1d(zT[:,0], zT[:,1],bounds_error=False)
    return T_spl_z(z)

def get_mean_temp_chi(chi):
    zT=np.loadtxt('Datafiles/mean_temp.dat')    #mK
    chis=cd.comoving_distance(zT[:,0], **cosmo)

    if show_plots:
        plt.plot(chis, zT[:,1])
    #    plt.xlim(0, 7000)
        #plt.ylim(0,2)
        plt.xlabel(r'$\chi$')
        plt.ylabel('T [mK]')
        plt.title('Mean Temperature')
        plt.savefig('TestingPlots/mean_temp_chi')
        plt.show()


    T_spl_chi=interp1d(chis, zT[:,1],bounds_error=False)
    return T_spl_chi(chi)

def A_b(zs):
    A_b = 566*cosmological_parameters.h/0.003 * (1+zs)**2. *(cd.hubble_z(0, **cosmo)/cd.hubble_z(zs, **cosmo))*1e-3
    return A_b

def get_mean_temp_2D(zs):
    zT=np.loadtxt('Datafiles/mean_temp.dat')    #mK
    #    plt.plot(zT[:,0], zT[:,1])
    #    plt.show()
    T_spl_z=interp1d(zT[:,0], zT[:,1],bounds_error=False)
    T=np.zeros(zs.shape)
    for i in range(zs.shape[1]):
        T[:,i]=T_spl_z(zs[:,i])
    return T


def get_mean_temp_chi_2D(chi):
    zT=np.loadtxt('Datafiles/mean_temp.dat')    #mK
    chis=cd.comoving_distance(zT[:,0], **cosmo)
    T_spl_chi=interp1d(chis, zT[:,1],bounds_error=False)
    T=np.zeros(chi.shape)
    for i in range(chi.shape[1]):
        T[:,i]=T_spl_chi(chi[:,i])
    return T


#def get_P_m_0(k):    #nonlinear matter power at z=0, questionable for k>1??
#    h=0.67
#    kP=np.loadtxt('Datafiles/P_k_nonlin_camb_z0.dat')
#    P_spl_k=interp1d(kP[:,0]/h, kP[:,1]*h**3)
#    return P_spl_k(k)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def get_P_m_0(k):    #nonlinear (coz we go to high k) matter power at z=0 - datafile from camb
    h=cosmological_parameters.h
    kP=np.loadtxt('Datafiles/P_k_nonlin_camb_z0.dat')  # goes up to k=18/Mpc
    P_spl_k=interp1d(kP[:,0]/h, kP[:,1]*h**3,bounds_error=False,fill_value=0.0)
    return P_spl_k(k)

def get_P_m_0_2D(ks):
    h=cosmological_parameters.h
    kP=np.loadtxt('Datafiles/P_k_nonlin_camb_z0.dat')
    P_spl_k=interp1d(kP[:,0]/h, kP[:,1]*h**3,bounds_error=False,fill_value=0.0)
    P=np.zeros(ks.shape)
    for i in range(ks.shape[1]):
        P[:,i]=P_spl_k(ks[:,i])

    return P

def get_fbao(k):
    fbao = np.loadtxt('Datafiles/fbao.dat')
    fbao_spl = interp1d(fbao[0], fbao[1],bounds_error=False,fill_value=0.0)
    return fbao_spl(k)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# plot power spectra
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def plot_P():
    ks=np.logspace(-4, 1.5, 1001)
    k_par=0.

    zs=[0.]#0.8,2.5]


    for z in zs:
        P21=get_mean_temp(z)**2*(get_HI_bias(z)+get_growth_factor_f(z)*(k_par/ks)**2)**2*get_P_m_0(ks)*get_growth_function_D(z)**2
        print (P21)
        plt.loglog(ks, P21, label='z='+str(z))
    plt.legend()
    plt.xlabel(r'$k$')
    plt.ylabel(r'$P_{21}(k) [K^2 Mpc^3]$')
    plt.title('21cm power spectrum')
    plt.show()

    for z in zs:
        P=get_P_m_0(ks)*get_growth_function_D(z)**2
        plt.loglog(ks, ks**3*P, label='z='+str(z))
    #plt.legend()
    plt.xlabel(r'$k$')
    plt.ylabel(r'$k^3 P_{m}(k) $')
    plt.title('Matter power spectrum')
    plt.show()


def get_Omega_HI(zs):
    z, OmHI = np.loadtxt('Datafiles/Omega_HI.dat')
    return interp1d(z,OmHI,bounds_error=False,fill_value=0.0)(zs)

def b1(zs, model = None):
    b0 = cosmological_parameters.b1_0
    b1 = cosmological_parameters.b1_1
    b2 = cosmological_parameters.b1_2
    if model == 'constant' :
        return b0*np.ones(len(zs))
    elif model == 'linear' :
        return b0 + b1*zs
    else:
        return b0 + b1*zs + b2*zs**2.

def b2(zs, model = None):
    b0 = cosmological_parameters.b2_0
    b1 = cosmological_parameters.b2_1
    b2 = cosmological_parameters.b2_2
    if model == 'constant' :
        return b0*np.ones(len(zs))
    elif model == 'linear' :
        return b0 + b1*zs
    else:
        return b0 + b1*zs + b2*zs**2.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# main
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if __name__=='__main__':
    #global show_plots
    show_plots=True
    zs=np.linspace(0, 3, 1001)
    zi=np.linspace(0, 3, 4)
    plt.plot(zs,get_Omega_HI(zs)*1e3);plt.show()
    #get_growth_function_D(zs)
    #get_growth_factor_f(zs)
    #get_HI_bias(zs)
    #get_HI_bias_2nd_order(zs)
    #get_mean_temp(zs)
    #b1 = lambda z: 0.04612641*z**2. + 0.11576764*z + 0.70015328
    #b2 = lambda z : 0.01967713*z**3. -0.02374002*z**2. -0.07189535*z -0.30780636

    #plt.plot(zi, b2(zi),'k--')
    Fisher00 = np.loadtxt('Datafiles/Fisher_matrix_sum0.dat')
    Fisher0 = np.linalg.inv(Fisher00)
    Fisher11 = np.loadtxt('Datafiles/Fisher_matrix_sum1.dat')
    Fisher1 = np.linalg.inv(Fisher11)
    Fisher22 = np.loadtxt('Datafiles/Fisher_matrix_sum2.dat')
    Fisher2 = np.linalg.inv(Fisher22)
    Fisher33 = np.loadtxt('Datafiles/Fisher_matrix_sum3.dat')
    Fisher3 = np.linalg.inv(Fisher33)
    sigma_errors = [Fisher3[0,0],Fisher2[0,0],Fisher1[0,0],Fisher0[0,0]]
    f_errors = [Fisher3[1,1],Fisher2[1,1],Fisher1[1,1],Fisher0[1,1]]
    b1_errors = np.sqrt([Fisher3[2,2],Fisher2[2,2],Fisher1[2,2],Fisher0[2,2]])
    b2_errors = np.sqrt([Fisher3[3,3],Fisher2[3,3],Fisher1[3,3],Fisher0[3,3]])
    x_arr = np.array([0.81,0.95,1.27,1.95])

    plt.errorbar(x_arr,get_HI_bias(x_arr),yerr=b1_errors,ecolor='k',capsize=3)
    #plt.errorbar(x_arr,get_HI_bias_2nd_order(x_arr),yerr=b2_errors,ecolor='k',capsize=3)
    #plt.errorbar(x_arr,get_growth_factor_f(x_arr),yerr=np.sqrt(f_errors),ecolor='k',capsize=3)
    #plt.errorbar(x_arr,cosmological_parameters.sig8*np.ones(len(x_arr)),yerr=np.sqrt(sigma_errors),ecolor='k',capsize=3)
    #plt.plot(zs,get_HI_bias(zs),'k',label='b')
    #plt.plot(zs,b1(zs, model='linear'),'b--',label='linear')
    #plt.plot(zs,b1(zs, model='constant'),'b-.',label='constant')

    #plt.errorbar(xerr,get_HI_bias_2nd_order(xerr),yerr=y2err,fmt='k--')
    #plt.plot(zs,get_HI_bias_2nd_order(zs),'r',label='b2')
    #plt.plot(zs,b2(zs, model='linear'),'r--',label='b2 linear')
    #plt.plot(zs,b2(zs, model='constant'),'r-.',label='b2 constant')

    #chis=np.arange(0, 6, 10)
    plt.xlim([0.8,2.0])
    #plt.ylim([0.68,1.15])
    plt.xlabel('z')
    plt.ylabel('b1(z)')
    plt.legend(loc=2)
    #get_growth_function_D_chi(chis)
    #get_growth_factor_f_chi(chis)
    #get_HI_bias_chi(chis)
    #get_HI_bias_2nd_order_chi(chis)
    #get_mean_temp_chi(chis)
