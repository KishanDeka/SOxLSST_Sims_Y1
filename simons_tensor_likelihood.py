import numpy as np
import healpy as hp
import emcee
import os
import corner
import pylab as pl
import pymaster as nmt
from multiprocessing import Pool
from plancklens import utils

set_string = 'B'
data_path = '/home/dekak/Desktop/simons_sims/camb_cls/'
cl_len = utils.camb_clfile(data_path + 'camb_lensed_cls.dat')
cl_unl = utils.camb_clfile(data_path + 'camb_unlensed_cls.dat')

#os.environ["OMP_NUM_THREADS"] = "32"
        
def gaussian_likelihood(theta, cl_data, cl_lens, cl_tens, cl_noise, inv_cov_mat):
    log10_r, Alens = theta
    r = 10**log10_r                      # transform back to linear r
    cl_model = r*cl_tens + Alens*cl_lens + cl_noise
    diff = cl_data - cl_model
    chi2 = np.dot(np.dot(diff, inv_cov_mat), diff.T)
    return -0.5 * chi2

def log_prior(theta):
    log10_r, Alens = theta
    # e.g. r between 1e-5 and 1  ->  log10(r) between -5 and 0
    if -4. < log10_r < 0. and 0.< Alens < 1.:
        return 0.0
    return -np.inf

def logposterior(theta, data, lens, tens, noise, inv_cov_mat):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + gaussian_likelihood(theta, data, lens, tens, noise, inv_cov_mat)

def bias_estimate(cl_data, cl_lens, cl_tens, cl_noise, inv_cov_mat):
    Nens = 32
    # initialize walkers inside the prior box
    pos = np.empty((Nens, 2))
    pos[:, 0] = np.random.uniform(-4., 0., Nens)   # log10(r)
    pos[:, 1] = np.random.uniform(0., 1., Nens)   # Alens
    nwalkers, ndims = pos.shape

    Nburnin  = 1000
    Nsamples = 4000
    argslist = (cl_data, cl_lens, cl_tens, cl_noise, inv_cov_mat)

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(Nens, ndims, logposterior,
                                        args=argslist, pool=pool)
        sampler.run_mcmc(pos, Nburnin + Nsamples, progress=True)
        postsamples = sampler.chain[:, Nburnin:, :].reshape((-1, ndims))

    # column 0 is log10(r); convert if you want samples of r itself
    postsamples_r = postsamples.copy()
    postsamples_r[:, 0] = 10**postsamples_r[:, 0]
    return postsamples_r    
    
    
## tensor-to-scalar ratio estimate
NSIDE = 512
Lmax = 600
lmin, lmax = 2, 10
bin_l = lmax - lmin
b = nmt.NmtBin.from_lmax_linear(Lmax, 20)

#mask_so_sat = hp.read_map('../hitmaps/simons_sat_mask_gal70_apo10deg_nside_512.fits')
#mask_lsst = hp.ud_grade(hp.read_map('../hitmaps/lsst_mask_apo_c2_8deg_nside_2048.fits'), nside_out = NSIDE, power=0)
#mask_lsst_so_bin = (mask_so_sat * mask_lsst) > 0.
#del mask_so_sat, mask_lsst

sat_hits = hits = hp.read_map("../../simons_sims/hitmaps/sat_hitmap_nside_512.fits")
hilc_mask = np.where(hits > 0, hits/hits.max(), 0.)  # normalised hitmap mask
hilc_mask_bin = hilc_mask > 0.

beam = 30.
transf = hp.gauss_beam(np.radians(beam/60.), lmax=Lmax)

f0 = nmt.NmtField(hilc_mask, None, spin=0, lmax=Lmax, lmax_mask=Lmax, n_iter=3)
wsp = nmt.NmtWorkspace.from_fields(f0, f0, b)

for sid in range(1,2):
    
    cov_mat = np.load('../outputs/specs%s/cov_mat_clbb_del_jackkniffe_hilc_set%s.npy' %(set_string, set_string))[lmin:lmax, lmin:lmax]
    #cov_mat = np.load('../mocks/covariance_matrix_sims_100.npy')[lmin:lmax, lmin:lmax]
    
    N_jk, N_data = 91, 8
    hartlap_fac = (N_jk - N_data - 2) / (N_jk - 1)
    inv_cov_mat = np.linalg.inv(cov_mat) * hartlap_fac
    del cov_mat
    
    #### internal delensing estimate
    Blm_temp_opt = hp.read_alm('../lensQE/template_blm_internal_lmax_%d_sim_%03d_set%s.fits' %(Lmax, sid, set_string), hdu=1)
    tempB_opt = hp.alm2map(Blm_temp_opt, nside=NSIDE)
    Blm_sat = hp.read_alm('../hilc_sat/hilc_alms/alm_eb_apomask_sat_hilc_%03d_set%s.fits' %(sid, set_string), hdu=2)
    Bmap_obs = hp.alm2map(Blm_sat, nside=NSIDE)

    #Elm, Blm = hp.map2alm_spin(hp.read_map('../sims/lcmb_TQU_nsim_1.fits', field=(1,2)), lmax=Lmax, spin=2)
    #Bmap = hp.alm2map(Blm, nside=NSIDE) * mask_lsst_so_bin

    f1 =  nmt.NmtField(hilc_mask, [Bmap_obs - tempB_opt * hilc_mask_bin], spin=0, lmax=Lmax, lmax_mask=Lmax, n_iter=3)
    cl_data = wsp.decouple_cell(nmt.compute_coupled_cell(f1,f1))[0, lmin:lmax]
    del f1

    cl_lens = b.bin_cell(cl_len['bb'][:Lmax+1] * transf**2)[lmin:lmax]
    cl_tens = b.bin_cell(cl_unl['bb'][:Lmax+1] * transf**2)[lmin:lmax]  
    #cl_noise = b.bin_cell(np.load('../outputs/clbb_res_intp_sat_lmax_600.npy'))[lmin:lmax]
    cl_noise = np.mean(np.load('../outputs/clbb_res_master.npy'), axis=0)[lmin:lmax]


    #print(chi2)
    postsample = bias_estimate(cl_data, cl_lens, cl_tens, cl_noise, inv_cov_mat)
    print(np.median(postsample, axis=0), np.std(postsample, axis=0))

    np.save('../postsamples/postsample_chains_r_Alens_sim_%03d_int_set%s.npy' %(sid, set_string), postsample)
    del Blm_temp_opt, tempB_opt, cl_data, postsample
    
    #### optimal delensing estimate
    
    Blm_temp_opt = hp.read_alm('../lensQE/template_blm_optimal_lmax_%d_sim_%03d_set%s.fits' %(Lmax, sid, set_string), hdu=1)
    tempB_opt = hp.alm2map(Blm_temp_opt, nside=NSIDE)

    f1 =  nmt.NmtField(hilc_mask, [Bmap_obs - tempB_opt * hilc_mask_bin], spin=0, lmax=Lmax, lmax_mask=Lmax, n_iter=3)
    cl_data = wsp.decouple_cell(nmt.compute_coupled_cell(f1,f1))[0, lmin:lmax]
    del f1

    #print(chi2)
    postsample = bias_estimate(cl_data, cl_lens, cl_tens, cl_noise, inv_cov_mat)
    print(np.median(postsample, axis=0), np.std(postsample, axis=0))

    np.save('../postsamples/postsample_chains_r_Alens_sim_%03d_opt_set%s_true.npy' %(sid, set_string), postsample)
    del Blm_temp_opt, tempB_opt, cl_data, postsample 
    
    
    #### optimal delensing estimate

    Blm_temp_opt = hp.read_alm('../lensQE/template_blm_optimal_lmax_%d_sim_%03d_set%s_true.fits' %(Lmax, sid, set_string), hdu=1)
    tempB_opt = hp.alm2map(Blm_temp_opt, nside=NSIDE)

    f1 =  nmt.NmtField(hilc_mask, [Bmap_obs - tempB_opt * hilc_mask_bin], spin=0, lmax=Lmax, lmax_mask=Lmax, n_iter=3)
    cl_data = wsp.decouple_cell(nmt.compute_coupled_cell(f1,f1))[0, lmin:lmax]
    del f1

    #print(chi2)
    postsample = bias_estimate(cl_data, cl_lens, cl_tens, cl_noise, inv_cov_mat)
    print(np.median(postsample, axis=0), np.std(postsample, axis=0))

    np.save('../postsamples/postsample_chains_r_Alens_sim_%03d_opt_set%s_true.npy' %(sid, set_string), postsample)
    del Blm_temp_opt, tempB_opt, cl_data, postsample 
    
del hilc_mask, hilc_mask_bin    
