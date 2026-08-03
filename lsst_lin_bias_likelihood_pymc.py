import numpy as np
import healpy as hp
import emcee
import os, sys
import corner
import pylab as pl
import pymaster as nmt
from multiprocessing import Pool
import pymc as pm
import arviz as az
import pytensor
import pytensor.tensor as pt

set_string = sys.argv[1]

os.environ["OMP_NUM_THREADS"] = "32"

def load_cls_theory(file, lmax, path = 'theory_cls/'):
    # Load the file
    data = np.loadtxt(path + file)
    spectrum_row = data[0] if data.ndim > 1 else data
    # Limit to lmax
    cls = spectrum_row[:lmax + 1]
    return cls
        
def bias_estimate_pymc(cl_data, cl_fid, cl_noise, cov_mat, nbins, output_samples):
	
    cov_mat = pt.as_tensor_variable(cov_mat)		
    L = pt.linalg.cholesky(cov_mat)
    dof_per_bin = (2 * bin_l) - 2   # datapoints - parameters
    
    cl_data_gg = pt.as_tensor_variable(cl_data[0])
    cl_data_kg = pt.as_tensor_variable(cl_data[1])
    cl_fid_gg = pt.as_tensor_variable(cl_fid[0])
    cl_fid_kg = pt.as_tensor_variable(cl_fid[1])
    cl_noise_gg = pt.as_tensor_variable(cl_noise)
    
    with pm.Model() as model:
        # priors: one bias per bin, one shared amplitude
        b = pm.Uniform("b_z", lower=0.0, upper=10.0, shape=nbins)
        s8 = pm.Uniform("s8_0", lower=0.0, upper=5.0)
 
        # per-bin models
        model_gg = (b[:, None] ** 2) * (s8 ** 2) * cl_fid_gg + cl_noise_gg   # (nbins, bin_l)
        model_kg = b[:, None] * (s8 ** 2) * cl_fid_kg                    # (nbins, bin_l)
 
        # concatenate gg and kg per bin into one 2*bin_l vector
        res = pt.concatenate((cl_data_gg - model_gg, cl_data_kg - model_kg), axis=1)
 
        factor = pt.linalg.solve_triangular(L, res, lower=True, b_ndim=1)
        quad = pt.sum(factor ** 2, axis=1)        
 
        # multivariate-t log-likelihood per bin, summed over bins
        loglike_per_bin = -0.5 * (2*bin_l + dof_per_bin) * pt.log1p(quad / dof_per_bin)
        pm.Potential("loglike", pt.sum(loglike_per_bin))
 
        idata = pm.sample(draws=draws,tune=tune,chains=chains,target_accept=target_accept,
        			cores=cores,progressbar=True)

    idata.to_netcdf(outfile_samples)
        
    
    
## bias estimate
Lmax = 2048
lowLL = [0, 30, 60, 90,140, 250, 400, 600, 900, 1200, 1650]
highLL = [29, 59, 89, 139, 249, 399, 599, 899, 1199, 1599, 2049]
b = nmt.NmtBin.from_edges(lowLL, highLL)
ell_bin = b.get_effective_ells()

lmin = 2
lmax = len(ell_bin)
bin_l = lmax - lmin

## parameters of the chain
chains = 10
draws = 5000
tune = 1000
cores = 14
target_accept = 0.95

zbin = 8
nside = 2048
pix_area = hp.nside2pixarea(nside)
pix_win = hp.pixwin(nside=2048, lmax=Lmax)

specs_path = '../outputs/specs%s/' %set_string
n_bar = np.load(specs_path + 'mean_galaxy_count_set%s.npy' %set_string)  # mean galaxy count
cl_gg_master = np.load(specs_path + 'clgg_auto_bin_mask_apo.npy')
cl_kg_master = np.load(specs_path + 'clkg_cross_bin_mask_apo.npy')

mask_lsst = hp.read_map('../../../simons_sims/hitmaps/lsst_mask_apo_c2_8deg_nside_2048.fits')
mask_lsst_bin = mask_lsst > 0.
mask_so = hp.read_map('../../../simons_sims/hitmaps/simons_lat_mask_gal70_apo10deg_nside_2048.fits')
mask_so_bin = mask_so > 0

f1 = nmt.NmtField(mask_lsst, None, spin=0, lmax=Lmax, lmax_mask=Lmax, n_iter=3)
f2 = nmt.NmtField(mask_so, None, spin=0, lmax=Lmax, lmax_mask=Lmax, n_iter=3)
wsp11 = nmt.NmtWorkspace.from_fields(f1, f1, b)
wsp12 = nmt.NmtWorkspace.from_fields(f1, f2, b)
del f1, f2

cl_fid = np.zeros((2, zbin, bin_l))
cl_noise = np.zeros((zbin, bin_l))
cl_data = np.zeros((2, zbin, bin_l)) 
for i in range(zbin):
    nlgg = np.full(Lmax+1, np.mean(mask_lsst)*pix_area/n_bar[i], dtype=np.float32)  # shot noise
    nlgg_bin = wsp11.decouple_cell(wsp11.couple_cell([nlgg]))[0]
    ## load gg spectra
    clgg = load_cls_theory('cl_gg_lmax_7000_bin_%s.dat' %(i+1), Lmax, '../camb_cls_ztrue/')
    clgg_bin = wsp11.decouple_cell(wsp11.couple_cell([clgg*pix_win**2]))[0]
    cl_fid[0,i] = clgg_bin[lmin:lmax]
    cl_noise[i] = nlgg_bin[lmin:lmax]
    cl_data[0,i] = cl_gg_master[i, lmin:lmax]
    ## load kg spectra
    clkg = load_cls_theory('cl_kg_lmax_7000_bin_%s.dat' %(i+1), Lmax, '../camb_cls_ztrue/')
    clkg_bin = wsp12.decouple_cell(wsp12.couple_cell([clkg*pix_win**2]))[0]
    cl_fid[1,i] = clkg_bin[lmin:lmax]
    cl_data[1,i] = cl_kg_master[i, lmin:lmax]
    del nlgg, nlgg_bin, clgg, clgg_bin
del cl_gg_master,cl_kg_master, n_bar
del mask_so, mask_lsst, wsp11, wsp12

f_sky = 0.35
cov_mat_temp = np.load('../outputs/cov_mat_gg_kg_jackknife_lmax_2048.npy')
N_jk, N_data = 1000, 16
hartlap_fac = (N_jk - N_data - 2)/(N_jk - 1)
cov_mat = np.zeros((zbin, 2*bin_l, 2*bin_l))
cov_mat[:,0:bin_l, 0:bin_l] = cov_mat_temp[:,0,:,:]
cov_mat[:,bin_l:2*bin_l, bin_l:2*bin_l] = cov_mat_temp[:,1,:,:]
cov_mat = cov_mat / hartlap_fac

outfile_samples = specs_path + 'postsample_chains_b_s8_pymc_ztrue.nc'
    
bias_estimate_pymc(cl_data, cl_fid, cl_noise, cov_mat, zbin, outfile_samples)
