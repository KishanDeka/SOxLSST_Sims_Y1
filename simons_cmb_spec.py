import numpy as np
import camb
import os,argparse

parser = argparse.ArgumentParser(description='Simulation spectra')
parser.add_argument('-r', dest='rval', default=0, type=float, help='tensor-to-scalar ratio')
args = parser.parse_args()


Lmax = 7000
lmax_unl = 6800
clkk_dir = '../camb_cls/'  # path to clkk file directory
outdir = '../camb_cls/'   # path to store theory cls
cosmos = camb.model.CAMBparams(WantTensors=True,max_l=Lmax,max_l_tensor=Lmax)

cosmos.set_cosmology(H0=67.32117,ombh2=0.0223828,omch2=0.1201075, omk=0.0, tau=0.05430842)
cosmos.InitPower.set_params(As=2.100549e-09, ns=0.9660499,nt=0., ntrun=0., r=args.rval)
cosmos.Reion.helium_delta_redshift = 0.5
cosmos.Reion.helium_redshiftstart = 6.0
cosmos.set_for_lmax(Lmax, lens_potential_accuracy=4)


result = camb.get_results(cosmos)
result.calc_power_spectra(cosmos)

ell = np.arange(lmax_unl+1)

unl_cls = result.get_unlensed_total_cls(lmax=lmax_unl,CMB_unit='muK')
ps_unl_arr = np.vstack([ell, unl_cls.T])
print(ps_unl_arr.shape)

# store unlensed Cls
np.savetxt(outdir + 'camb_unlensed_cls.dat', ps_unl_arr.T,  fmt='%d  %1.5e  %1.5e  %1.5e  %1.5e')
del ps_unl_arr, unl_cls, result

#exit()

## get lensed theory Cls
data = np.loadtxt(clkk_dir + 'cl_kk_lmax_7000.dat')   # load your lensing convergence Cls
spectrum_row = data[0] if data.ndim > 1 else data  
# Limit to lmax
clkk = 2*spectrum_row[:Lmax+1] / np.pi

cosmos.set_for_lmax(lmax_unl, lens_potential_accuracy=4)
result = camb.get_results(cosmos)
result.calc_power_spectra(cosmos)

lmax_len = lmax_unl
tot_cls = result.get_lensed_cls_with_spectrum(clkk, lmax=lmax_len,CMB_unit='muK')

ell = np.arange(lmax_len+1)
ps_len_arr = np.vstack([ell, tot_cls.T])

print(ps_len_arr.shape)

# store lensed Cls
np.savetxt(outdir + 'camb_lensed_cls.dat' , ps_len_arr.T, fmt='%d  %1.5e  %1.5e  %1.5e  %1.5e')

del ps_len_arr, tot_cls, result, cosmos
