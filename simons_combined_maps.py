import healpy as hp
import numpy as np
import os

os.environ["OMP_NUM_THREADS"] = "32"
    
Nsim_start = 1
Nsim_end = 50

parentdir = '/mnt/home/kdeka/simons_sims/'
outdir = parentdir + 'outputs/lat_maps/'
cmbdir = parentdir + 'outputs/cmb_maps/'
noisedir = parentdir + 'outputs/lat_noise/'
fgdir = parentdir + 'outputs/fg_maps/'

## make sure the out path exists
try :
    os.makedirs(outdir)
except : print('exists')    
    
df = dict()
df['frequency'] = np.array([27, 39, 93, 145, 225, 280])
df['fwhm_lat'] = np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9])
df['depth_lat'] = np.array([52, 27, 5.8, 6.3, 15, 37]) #np.array([52, 27, 5.8, 6.3, 15, 37])
df['fwhm_sat'] = np.array([91, 63, 30, 17, 11, 9])

####################################################
## create observations for LAT channels (beam-smoothed)
# LAT maps are stored as NSIDE=2048 and lmax=6144

N_splits = 1

nside_lat = 2048
lmin_lat = 50
lmax_lat = 3*nside_lat
lat_mask = hp.read_map(parentdir + 'hitmaps/simons_lat_coverage_nside_2048.fits', field=0)
for idx in range(Nsim_start, Nsim_end+1):
    for fwhm, freq in zip(df['fwhm_lat'], df['frequency']):
        print(freq)
        cmb_map = hp.read_map(cmbdir+'lcmb_lmax_4096_sid_%03d.fits' %idx, field=(0,1,2))
        Tlm, Elm, Blm = hp.map2alm(cmb_map, lmax=lmax_lat, pol=True, iter=3, use_pixel_weights=True)
        del cmb_map
        
        transf = hp.gauss_beam(np.radians(fwhm/60.), lmax=lmax_lat)
        for alm in [Tlm, Elm, Blm]:
            hp.almxfl(alm, transf, inplace=True)
        
        TQU = hp.alm2map([Tlm, Elm, Blm], nside=nside_lat, pol=True, pixwin=False)
        del Tlm, Elm, Blm, transf #, noise_cl, nTlm, nElm, nBlm
        
        fg_map = hp.read_map(fgdir+'lat_fg_d12s7_%dGHz_%0.1f_arcmin_nside_%d.fits' %(freq, fwhm, nside_lat), field=(0,1,2))
        TQU += fg_map
        del fg_map
        
        for n in range(1,N_splits+1):
            TQUnoise = hp.read_map(noisedir + 'lat_noise_%dGHz_split_%d_sid_%03d.fits' %(freq, n, idx), field=(0,1,2))
            TQUnoise += TQU
            
            Tlm, Elm, Blm = hp.map2alm(TQUnoise, lmax=lmax_lat, pol=True, iter=3, use_pixel_weights=True)
            del TQUnoise
            T, E, B = hp.alm2map([Tlm, Elm, Blm], nside=nside_lat, pol=False, pixwin=False)
            del Tlm, Elm, Blm
            
            for m in [T, E, B] :
                m *= lat_mask
                
            hp.write_map(outdir+'map_simons_lat_nside_2048_freq_%d_split_%d_nsim_%03d.fits' %(freq, n, idx), [T,E,B],
                           dtype=np.float32, overwrite=True)
            del T, E, B
        del TQU
del nside_lat, lmin_lat, lmax_lat

exit()
####################################################
## create observations for SAT channels (beam-smoothed)
# LAT maps are stored as NSIDE=512 and lmax=1536

nside_sat = 512
lmin_sat = 20
lmax_sat = 3*nside_sat
sat_mask = hp.read_map(parentdir + 'hitmaps/simons_sat_coverage_nside_512.fits', field=0)
for idx in range(Nsim_start, Nsim_end+1):
    for fwhm, freq in zip(df['fwhm_sat'], df['frequency']):
        print(freq)
        cmb_map = hp.read_map(cmbdir+'lcmb_lmax_4096_sid_%03d.fits' %idx, field=(1,2))
        Elm, Blm = hp.map2alm_spin(cmb_map, spin=2, lmax=lmax_sat, mmax=lmax_sat)
        del cmb_map
        
        transf = hp.gauss_beam(np.radians(fwhm/60.), lmax=lmax_sat)
        for alm in [Elm, Blm]:
            hp.almxfl(alm, transf, inplace=True)
        
        QU = hp.alm2map_spin([Elm, Blm], spin=2, nside=nside_sat, lmax=lmax_sat, mmax=lmax_sat)
        #TQU = hp.alm2map([Tlm, Elm, Blm], nside=nside_lat, pol=True, pixwin=True)
        del Elm, Blm, transf #, noise_cl, nTlm, nElm, nBlm
        
        fg_map = hp.read_map(fgdir+'sat_fg_d12s7_%dGHz_%0.0f_arcmin_nside_%d.fits' %(freq, fwhm, nside_sat), field=(1,2))
        QU += fg_map
        del fg_map
        
        

        QUnoise = hp.read_map(noisedir + 'sat_noise/sat_noise_%dGHz_sid_%03d.fits' %(freq, idx), field=(0,1))
        QU = QUnoise
        del QUnoise
        
        Elm, Blm = hp.map2alm_spin(QU, spin=2, lmax=lmax_sat, mmax=lmax_sat)
        del QU
        E, B = hp.alm2map([Elm, Blm], nside=nside_sat, pol=False, pixwin=False)
        
        for m in [E, B] :
            m *= sat_mask
            
        hp.write_map(outdir+'map_simons_sat_nside_2048_freq_%d_nsim_%03d.fits' %(freq, idx), [E,B],
                       dtype=np.float32, overwrite=True)
        del E, B
del nside_sat, lmin_sat, lmax_sat
del df, Nsim_start, Nsim_end, outdir, cmbdir, noisedir, fgdir
