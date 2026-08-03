import numpy as np
import healpy as hp
import pysm3 as ps3
import pysm3.units as u
import os

os.environ["OMP_NUM_THREADS"] = "16"
    
outdir = '../outputs/fg_maps/'

## make sure the out path exists
try :
    os.makedirs(outdir)
except : print('exists')    
    
df = dict()
df['frequency'] = np.array([27, 39, 93, 145, 225, 280])
df['fwhm_lat'] = np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9])
df['fwhm_sat'] = np.array([91, 63, 30, 17, 11, 9])
nfreq = len(df['frequency'])

####################################################
## create foregrounds for LAT channels (beam-smoothed)
# LAT maps are stored as NSIDE=2048 and lmax=6144

nside_lat = 2048
sky = ps3.Sky(nside=nside_lat, preset_strings=["d12","s7"]) 

for fwhm_lat, freq in zip(df['fwhm_lat'], df['frequency']):
    
    emission = sky.get_emission(freq * u.GHz).to(u.uK_CMB,
                        equivalencies=u.cmb_equivalencies(freq * u.GHz))
    
    rot = hp.Rotator(coord=('G', 'C'))
    # smooth and produce LAT map                    
    emission_lat = ps3.apply_smoothing_and_coord_transform((emission.value), fwhm = fwhm_lat/60. * u.deg,
                    rot=rot, map2alm_lsq_maxiter=0, return_car = False, lmax=3*nside_lat-1)
    del emission, rot
     
    # sky coverage of SO  
    so_coverage = hp.read_map("../hitmaps/simons_lat_coverage_nside_%d.fits" %nside_lat) > 0                 
    for m in emission_lat:
        m *= so_coverage
    del so_coverage
                        
    hp.write_map(outdir+'lat_fg_d12s7_%dGHz_%0.1f_arcmin_nside_%d.fits' %(freq, fwhm_lat, nside_lat), 
                    emission_lat,  overwrite = True, dtype=np.float32)            
    
    del emission_lat
    
del nside_lat, sky

####################################################
## create foregrounds for SAT channels (beam-smoothed)
# SAT maps are stored as NSIDE=512 and lmax=1536
## SAT maps are stored only for Q and U polarisation 
## (only polarisation maps needed for r constrain)

nside_sat = 512
sky = ps3.Sky(nside=nside_sat, preset_strings=["d12","s7"])

for fwhm_sat, freq in zip(df['fwhm_sat'], df['frequency']):
    emission = sky.get_emission(freq * u.GHz).to(u.uK_CMB,
                        equivalencies=u.cmb_equivalencies(freq * u.GHz))
    
    rot = hp.Rotator(coord=('G', 'C'))    
    # smooth and produce SAT map                                    
    emission_sat = ps3.apply_smoothing_and_coord_transform((emission.value), fwhm = fwhm_sat/60. * u.deg,
                    rot=rot,  return_car = False, lmax=3*nside_sat-1)
    del emission, rot
    
    # sky coverage of SO
    so_coverage = hp.read_map("../hitmaps/simons_sat_coverage_nside_%d.fits" %nside_sat) > 0  
    for m in emission_sat:
        m *= so_coverage
    del so_coverage
    
    hp.write_map(outdir+'sat_fg_d12s7_%dGHz_%d_arcmin_nside_%d.fits' %(freq, fwhm_sat, nside_sat), 
                    emission_sat,  overwrite = True, dtype=np.float32)                
    
    del emission_sat
    
del sky, nside_sat
del df, outdir, nfreq
