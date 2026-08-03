import numpy as np
from scipy.integrate import simps
import camb
from camb import model,initialpower
from time import time
import pylab as pl


start = time()


#=======================================================================================================================
# CUSTOM FUNCTIONS
#=======================================================================================================================


def Dz(z,gamma_0,gamma_a):
    gamma = gamma_0 + gamma_a*z/(1.+z)
    z_integral = np.linspace(0.,z,500)
    integrand_dz = (results.get_Omega('cdm',z=z_integral)+results.get_Omega('baryon',z=z_integral)+results.get_Omega('neutrino',z=z_integral))**gamma/(1.+z_integral)
    return np.exp(-simps(integrand_dz,z_integral))


def lens_ker(chis,zs,H_z):  
    W_k = (3.*Om0*(H0**2)*chis*(1.+zs)*(chistar-chis))/(2.*chistar*c*H_z)	
    return W_k


def gal_ker(dist,zs,alpha,bias,chis,H_z):
    
    W_g = np.zeros(shape=(2,len(zs)))
    
    dist = dist/simps(dist,zs)
    #W_g[0,:] = np.poly1d(bias)(zs) * dist
    W_g[0,:] = bias * dist
    
    for i in range(len(zs)):
	    zs_intgrl = zs[i:]
	    ch_intgrl = chis[i:]
	    H_z_intgrl = H_z[i:]
	    dist_intgrl = dist[i:]
    
	    integrand = (1.-ch_intgrl[0]/ch_intgrl)*(alpha-1.)*dist_intgrl
	    val_intgrl = simps(y=integrand,x=zs_intgrl)

	    W_g[1,i] = (3.*Om0*(H0**2)*chis[i]*(1.+zs[i]))*val_intgrl/(2.*c*H_z[i])

    return (W_g[0,:] + W_g[1,:])

    

#=======================================================================================================================
# FILE PATH NAME DECLARATIONS
#=======================================================================================================================


infile_dist = '/home/dekak/Desktop/SOxLSST/LSSTY1/outputs/gal_dist_truez_gold_lsst.txt'
outdir_theor = '/home/dekak/Desktop/SOxLSST/LSSTY1/camb_cls_ztrue/'
outdir_eff_bias = '/home/dekak/Desktop/SOxLSST/LSSTY1/camb_cls_ztrue/'


#=======================================================================================================================
# VARIABLE DECLARATIONS
#=======================================================================================================================


nside = 2048

lmin = 0
lmax = 7000

kmin = 1e-4
kmax = 20.0
c = 3.e5

zbins_start = 1
zbins_end = 8

#bias = [0.2, 0.6, 1.3]
#bias = [0., 0., 1.]
bias = np.ones(zbins_end - zbins_start + 1)  # a bias array for redshift bins
#bias = [1.05, 1.10, 1.18, 1.26, 1.35, 1.45, 1.70, 2.0]
#bias = np.median(np.load('../outputs/specsA/postsample_chains_b0_A.npy'), axis=1)[:,0]

alpha = 1.
bias_evol = False

gamma_0 = 6./11.
gamma_a = 0.

zmin_kk = 0
zmax_kk = 100
spacing_kk = 1000


theor_gg = True
theor_gg_cross = True
theor_kk = True
theor_kg = True


#=======================================================================================================================
# INITIALIZATION
#=======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# Making matter power spectra at redshift z = 0
# ----------------------------------------------------------------------------------------------------------------------
pars = camb.set_params(WantTransfer=True,NonLinear='NonLinear_both',num_nu_massless = 2.046,
					    nu_mass_degeneracies=[0.0],share_delta_neff=True,nu_mass_fractions=[1.0],
					    nu_mass_numbers=[1],MassiveNuMethod='Nu_trunc',H0=67.32117,ombh2=0.0223828,
					    omch2=0.1201075, mnu=0.06451439,omk=0,tau=0.05430842)
pars.Transfer.high_precision=True
pars.InitPower.set_params(parameterization=1,ns=0.9660499,As=2.100549e-09)
pars.set_for_lmax(lmax=lmax,lens_potential_accuracy=4)
pars.NonLinearModel.set_params(halofit_version='original')
pars.NonLinearModel.Min_kh_nonlinear = 0.15


pars.InitPower.At=0.0
pars.Reion.helium_delta_redshift = 0.5
pars.Reion.helium_redshiftstart = 6.0

pars.SourceTerms.limber_phi_lmin = 100
pars.SourceTerms.limber_windows = True
pars.set_matter_power(redshifts=[0.], kmax=kmax, nonlinear=True, accurate_massive_neutrino_transfers=False)
pars.InitPower.k_min = kmin
pars.DoLensing = True

results = camb.get_results(pars)
chistar = results.conformal_time(0)-results.tau_maxvis

H0 = results.hubble_parameter(z=0.0)
Om0 = results.get_Omega('cdm',z=0)+results.get_Omega('baryon',z=0)+results.get_Omega('neutrino',z=0)
sigma8_0 = results.get_sigma8_0()

print ('Value of Hubble cosntant:',H0)
print ('Value of Density Parameter:',Om0)
print (r'Value of sigma_8:', sigma8_0)


ls = np.arange(lmin,lmax+1,dtype=np.float32)
gal_dist = np.loadtxt(infile_dist)

dz = (gal_dist[2:,0]-gal_dist[:-2,0])/2.
chis = results.comoving_radial_distance(gal_dist[:,0])
H_z = results.hubble_parameter(gal_dist[:,0])

D_z = np.zeros_like(gal_dist[:,0])
for x in range(len(gal_dist[:,0])):
    D_z[x] = Dz(gal_dist[x,0],gamma_0,gamma_a)


PK = camb.get_matter_power_interpolator(pars,nonlinear=True,hubble_units=False,k_hunit=False,kmax=kmax, \
                                 var1=model.Transfer_tot,var2=model.Transfer_tot,zmax=110.)


#=======================================================================================================================
# COMPUTING KERNELS
#=======================================================================================================================

gal_bias = np.zeros(shape=zbins_end-zbins_start+1)

meanz = np.zeros(shape=zbins_end-zbins_start+1)
mean_chi = np.zeros(shape=zbins_end-zbins_start+1)
mean_Dz = np.zeros(shape=zbins_end-zbins_start+1)
for ii in range(zbins_start,zbins_end+1):
    print(ii)
    meanz[ii-zbins_start] = simps(gal_dist[:,0]*(gal_dist[:,ii-zbins_start+1]*chis)**2/H_z,gal_dist[:,0])/simps((gal_dist[:,ii-zbins_start+1]*chis)**2/H_z,gal_dist[:,0])
    mean_chi[ii-zbins_start] = results.comoving_radial_distance(meanz[ii-zbins_start])
    mean_Dz[ii-zbins_start] = Dz(meanz[ii-zbins_start],gamma_0,gamma_a)
    gal_bias[ii-zbins_start] = bias[ii-1]/Dz(meanz[ii-zbins_start],gamma_0,gamma_a)

np.savetxt(outdir_eff_bias+'effective_bias.txt',np.array([meanz,mean_chi, mean_Dz, gal_bias]).T)

del meanz, mean_Dz, mean_chi

#=======================================================================================================================
# COMPUTING KERNELS
#=======================================================================================================================

for ii in range(zbins_start,zbins_end+1):
    bias_ii = bias[ii-1]
    if theor_gg or theor_kg or theor_gg_cross is True:
	    Wg = gal_ker(gal_dist[1:-1,ii-zbins_start+1],gal_dist[1:-1,0],alpha, \
	                bias_ii,chis[1:-1],H_z[1:-1])
        
    if theor_kg is True:
	    Wk = lens_ker(chis[1:-1],gal_dist[1:-1,0],H_z[1:-1])
	    
    if theor_gg_cross is True:	
        if ii < zbins_end - 1:
            Wg2 = gal_ker(gal_dist[1:-1,ii-zbins_start+2],gal_dist[1:-1,0],alpha, \
                    bias_ii,chis[1:-1],H_z[1:-1])
            Wg3 = gal_ker(gal_dist[1:-1,ii-zbins_start+3],gal_dist[1:-1,0],alpha, \
                    bias_ii,chis[1:-1],H_z[1:-1])
        elif ii == zbins_end-1 :
            Wg2 = gal_ker(gal_dist[1:-1,ii-zbins_start+2],gal_dist[1:-1,0],alpha, \
                    bias_ii,chis[1:-1],H_z[1:-1])

#=======================================================================================================================
# COMPUTING CLS
#=======================================================================================================================


#---------------------------------------------------------------------------------------------------------------
# COMPUTING GG SPECTRA
#---------------------------------------------------------------------------------------------------------------


    if theor_gg is True:

	    outfile_cl = outdir_theor+'cl_gg_lmax_'+str(lmax)+'_bin_'+str(ii)+'.dat'
	    cl = np.zeros(shape=(lmax-lmin+1))
	    
	    for l in range(lmin,lmax+1):
		    k = (l+0.5)/chis[1:-1]
		    w = np.ones(k.shape)
		    w[k<kmin] = 0.
		    w[k>kmax] = 0.
	    
		    if bias_evol is True:
			    cl[l] = np.dot(dz,w*H_z[1:-1]*Wg**2*PK.P(0,k,grid=False)/chis[1:-1]**2/c) / sigma8_0**2
		    if bias_evol is False:
			    cl[l] = np.dot(dz,w*H_z[1:-1]*Wg**2*PK.P(0,k,grid=False)*(D_z[1:-1])**2/chis[1:-1]**2/c) / sigma8_0**2
		    
		    del k,w
		    
	    np.savetxt(outfile_cl,cl)
		    
	    del outfile_cl,cl,l

#---------------------------------------------------------------------------------------------------------------
# COMPUTING GG cross SPECTRA
#---------------------------------------------------------------------------------------------------------------

    if theor_gg_cross is True:
        ## cross clgg between i and i+1 or i and i+2
        if ii < zbins_end - 1 : 
            
            ## cross clgg between i and i+1
            outfile_cl = outdir_theor+'cl_gg_lmax_'+str(lmax)+'_bin_'+str(ii)+str(ii+1)+'.dat'
            cl = np.zeros(shape=(lmax-lmin+1))
	        
            for l in range(lmin,lmax+1):
                k = (l+0.5)/chis[1:-1]
                w = np.ones(k.shape)
                w[k<kmin] = 0.
                w[k>kmax] = 0.

                if bias_evol is True:
                    cl[l] = np.dot(dz,w*H_z[1:-1]*Wg*Wg2*PK.P(0,k,grid=False)/chis[1:-1]**2/c) / sigma8_0**2
                if bias_evol is False:
                    cl[l] = np.dot(dz,w*H_z[1:-1]*Wg*Wg2*PK.P(0,k,grid=False)*(D_z[1:-1])**2/ chis[1:-1]**2/c) / sigma8_0**2

                del k,w
		        
            np.savetxt(outfile_cl,cl)
		        
            del outfile_cl,cl,l,Wg2
            
            ## cross clgg between i and i+2
            outfile_cl = outdir_theor+'cl_gg_lmax_'+str(lmax)+'_bin_'+str(ii)+str(ii+2)+'.dat'
            cl = np.zeros(shape=(lmax-lmin+1))

            for l in range(lmin,lmax+1):
                k = (l+0.5)/chis[1:-1]
                w = np.ones(k.shape)
                w[k<kmin] = 0.
                w[k>kmax] = 0.

                if bias_evol is True:
                    cl[l] = np.dot(dz,w*H_z[1:-1]*Wg*Wg3*PK.P(0,k,grid=False)/chis[1:-1]**2/c) / sigma8_0**2
                if bias_evol is False:
                    cl[l] = np.dot(dz,w*H_z[1:-1]*Wg*Wg3*PK.P(0,k,grid=False)*(D_z[1:-1])**2 / chis[1:-1]**2/c) / sigma8_0**2

                del k,w

            np.savetxt(outfile_cl,cl)

            del outfile_cl,cl,l,Wg3
	        
        elif ii == zbins_end-1 :
            outfile_cl = outdir_theor+'cl_gg_lmax_'+str(lmax)+'_bin_'+str(ii)+str(ii+1)+'.dat'
            cl = np.zeros(shape=(lmax-lmin+1))

            for l in range(lmin,lmax+1):
                k = (l+0.5)/chis[1:-1]
                w = np.ones(k.shape)
                w[k<kmin] = 0.
                w[k>kmax] = 0.

                if bias_evol is True:
                    cl[l] = np.dot(dz,w*H_z[1:-1]*Wg*Wg2*PK.P(0,k,grid=False)/chis[1:-1]**2/c) / sigma8_0**2
                if bias_evol is False:
                    cl[l] = np.dot(dz,w*H_z[1:-1]*Wg*Wg2*PK.P(0,k,grid=False)*(D_z[1:-1])**2 / chis[1:-1]**2/c) / sigma8_0**2

                del k,w
	            
            np.savetxt(outfile_cl,cl)
                
            del outfile_cl,cl,l,Wg2
	        
	        

#---------------------------------------------------------------------------------------------------------------
# COMPUTING KG SPECTRA
#---------------------------------------------------------------------------------------------------------------

    
    if theor_kg is True:

	    outfile_cl = outdir_theor+'cl_kg_lmax_'+str(lmax)+'_bin_'+str(ii)+'.dat'
	    cl = np.zeros(shape=(lmax-lmin+1))
	    
	    for l in range(lmin,lmax+1):
		    k = (l+0.5)/chis[1:-1]
		    w = np.ones(k.shape)
		    w[k<kmin] = 0.
		    w[k>kmax] = 0.
	    
		    if bias_evol is True:
			    cl[l] = np.dot(dz,w*H_z[1:-1]*Wg*Wk*PK.P(0,k,grid=False)*(D_z[1:-1])/chis[1:-1]**2/c) / sigma8_0**2
		    if bias_evol is False:
			    cl[l] = np.dot(dz,w*H_z[1:-1]*Wg*Wk*PK.P(0,k,grid=False)*(D_z[1:-1])**2/chis[1:-1]**2/c) / sigma8_0**2
		    
		    del k,w
		    
	    np.savetxt(outfile_cl,cl)
		    
	    del outfile_cl,cl,l
    
    
    if theor_gg or theor_kg is True:del Wg
    if theor_kg is True: del Wk

del bias

#---------------------------------------------------------------------------------------------------------------
# COMPUTING KK SPECTRA
#---------------------------------------------------------------------------------------------------------------


if theor_kk is True:
    
    zs = np.linspace(zmin_kk,zmax_kk,np.int64((zmax_kk-zmin_kk)*spacing_kk))
    dz = (zs[2:]-zs[:-2])/2.
    zs = zs[1:-1]
    chis = results.comoving_radial_distance(zs)
    H_z = results.hubble_parameter(zs)
    D_z = np.zeros_like(zs)
    
    
    for x in range(len(zs)):
	    D_z[x] = Dz(zs[x],gamma_0,gamma_a)
    
    Wk = lens_ker(chis,zs,H_z)
    np.savetxt(outdir_theor+'cmb_kk_window_function.dat',np.array([zs,Wk]).T)
    cl = np.zeros(shape=(len(ls)))
    
    for i,l in enumerate(ls):
	    k = (l+0.5)/chis
	    w = np.ones(k.shape)
	    w[k<kmin] = 0.
	    w[k>kmax] = 0.
	    cl[i] = np.dot(dz,w*H_z*Wk**2*PK.P(0,k,grid=False)*D_z**2/chis**2/c)
	    
	    del k,w


    outfile_cl = outdir_theor+'cl_kk_lmax_'+str(lmax)+'.dat'
    np.savetxt(outfile_cl,cl)
    
    del zs,dz,chis,H_z,D_z,Wk,cl,outfile_cl,l


#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------		
#---------------------------------------------------------------------------------------------------------------


end = time()
print('Time elasped for theoretical calculations:',(end-start),'seconds')


#=====================================================================================================================
# DEALLOCATION
#=====================================================================================================================


del infile_dist,outdir_theor
del nside,lmin,lmax,kmin,kmax,c,alpha,bias_evol,gamma_0,gamma_a,zbins_start,zbins_end,zmin_kk,zmax_kk,spacing_kk
del theor_gg,theor_kk,theor_kg,pars,results,chistar,H0,Om0,ls,gal_dist,PK, gal_bias
del start,end
