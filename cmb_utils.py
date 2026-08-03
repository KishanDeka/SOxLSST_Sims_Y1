"""Planck 2018 lensing utils module
"""

import os
import healpy as hp
from healpy.projector import CartesianProj
import time
import numpy as np
import sys

def alm_copy(alm, lmax=None):
    """Copies the healpy alm array, with the option to reduce its lmax

    Args:
        alm (ndarray): healpy alm array.
        lmax (int, optional): new alm lmax.
    """
    alm_lmax = int(np.floor(np.sqrt(2 * len(alm)) - 1))
    assert lmax <= alm_lmax, (lmax, alm_lmax)
    if (alm_lmax == lmax) or (lmax is None):
        ret = np.copy(alm)
    else:
        ret = np.zeros((lmax + 1) * (lmax + 2) // 2, dtype=complex)
        for m in range(0, lmax + 1):
            ret[((m * (2 * lmax + 1 - m) // 2) + m):(m * (2 * lmax + 1 - m) // 2 + lmax + 1)] \
                = alm[(m * (2 * alm_lmax + 1 - m) // 2 + m):(m * (2 * alm_lmax + 1 - m) // 2 + lmax + 1)]
    return ret


def cli(cl):
    """Pseudo-inverse for positive cl-arrays.
    
    """
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
    return ret


def joincls(cls_list):
    lmaxp1 = np.min([len(cl) for cl in cls_list])
    return np.prod(np.array([cl[:lmaxp1] for cl in cls_list]), axis=0)


def camb_clfile(fname, lmax=None):
    """CAMB spectra (lenspotentialCls, lensedCls or tensCls types) returned as a dict of numpy arrays.

    Args:
        fname (str): path to CAMB output file
        lmax (int, optional): outputs cls truncated at this multipole.

    """
    cols = np.loadtxt(fname).transpose()
    ell = cols[0].astype(int)
    if lmax is None: lmax = ell[-1]
    assert ell[-1] >= lmax, (ell[-1], lmax)
    cls = {k : np.zeros(lmax + 1, dtype=float) for k in ['tt', 'ee', 'bb', 'te']}
    w = ell * (ell + 1) / (2. * np.pi)  # weights in output file
    idc = np.where(ell <= lmax) if lmax is not None else np.arange(len(ell), dtype=int)
    for i, k in enumerate(['tt', 'ee', 'bb', 'te']):
        cls[k][ell[idc]] = cols[i + 1][idc] / w[idc]
    if len(cols) > 5:
        wpp = lambda ell : ell ** 2 * (ell + 1) ** 2 / (2. * np.pi)
        wptpe = lambda ell : np.sqrt(ell.astype(float) ** 3 * (ell + 1.) ** 3) / (2. * np.pi)
        for i, k in enumerate(['pp', 'pt', 'pe']):
            cls[k] = np.zeros(lmax + 1, dtype=float)
        cls['pp'][ell[idc]] = cols[5][idc] / wpp(ell[idc])
        cls['pt'][ell[idc]] = cols[6][idc] / wptpe(ell[idc])
        cls['pe'][ell[idc]] = cols[7][idc] / wptpe(ell[idc])
    return cls


def _cldict2arr(cls_dict):
    lmaxp1 = np.max([len(cl) for cl in cls_dict.values()])
    ret = np.zeros((3, 3, lmaxp1), dtype=float)
    for i, x in enumerate(['t', 'e', 'b']):
        for j, y in enumerate(['t', 'e', 'b']):
            ret[i, j] =  extcl(lmaxp1 - 1, cls_dict.get(x + y, cls_dict.get(y + x, np.array([0.]))))
    return ret
    
class LAT_noise_info:
    
    def __init__(self, f_sky=1., splits=4, lmin=50):
        self.fsky = f_sky
        self.nsplits = splits
        self.lmin = lmin
        self.freqs = np.array([27,39,93,145,225,280])
        self.depth_t = np.array([44, 23, 3.8, 4.1, 10, 25])
        self.N_red_t = np.array([100, 39, 230, 1500, 17000, 31000])
        self.depth_p = np.array([44, 23, 3.8, 4.1, 10, 25]) * np.sqrt(2)
        self.fwhm = np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9])
        self.lKnee_t = np.array([1000, 1000, 1000, 1000, 1000, 1000])
        self.lKnee_p = np.array([700, 700, 700, 700, 700, 700])
        self.gamma_t = np.array([-3.5,-3.5,-3.5,-3.5,-3.5,-3.5])
        self.gamma_p = np.array([-1.4,-1.4,-1.4,-1.4,-1.4,-1.4])
        
    def get_N_ell(self, lmax=6144, frequency=30):
        '''
            lmax : maximum multipole of the Cls
            freq : frequency
            
            returns T, E and B noise Cls
        '''
        
        idx = np.where(self.freqs==frequency)
        temp_t = (np.arange(lmax+1)/self.lKnee_t[idx])**self.gamma_t[idx]
        temp_p = (np.arange(lmax+1)/self.lKnee_p[idx])**self.gamma_p[idx]
        temp_t[:self.lmin], temp_p[:self.lmin] = 0., 0.
        
        #transf = hp.gauss_beam(np.radians(self.fwhm[idx]/60.), lmax=lmax,pol=False)
        # convert unit from muK^2 sec to muK^2 sr
        N_red_conv = self.N_red_t[idx] * 1.6 * 1e-7
        Nell_T = (np.radians(self.depth_t[idx]/60.)**2 + N_red_conv * temp_t)  #* (self.fsky/0.4)**2
        Nell_E = Nell_B = (np.radians(self.depth_p[idx]/60.)**2 * (1+temp_p))  #* (self.fsky/0.4)**2
        
        return np.array([Nell_T, Nell_E, Nell_B]) * np.sqrt(self.nsplits)
    
    def get_noise_map(self, nside=2048, lmax=6144, frequency=30, seed=1234, hits = True):
        
        '''
            lmax : maximum multipole of the maps
            freq : frequency
            seed : seed for realisations
            hits : True or False (hit map weight)
            
            returns T, Q and U noise maps
        '''
        
        #idx = np.where(self.freqs==freq)
        Nl_T, Nl_E, Nl_B = self.get_N_ell(lmax, frequency)
        
        np.random.seed(seed)
        nlms = hp.synalm((Nl_T, Nl_E, Nl_B, None), lmax=lmax, new=True)
        nmap = hp.alm2map(nlms, nside=nside, lmax=lmax, mmax=lmax, pol=True)
        del nlms
        
        if hits == True:
            hitmap = hp.read_map('/home/dekak/Desktop/simons_sims/hitmaps/lat_hitmap_nside_2048.fits', field=0)
            #hitmap = hp.ud_grade(hitmap, nside_out=nside, power=0)
            
            norm_hits = np.where(hitmap > 0, hitmap/hitmap.max(), 0)
            del hitmap
            inv_hits = np.where(norm_hits > 0, 1/np.sqrt(norm_hits), 0)
            
            for m in nmap:
                m *= inv_hits
            del norm_hits, inv_hits
            
        return nmap
        

class SAT_noise_info:
    
    def __init__(self, f_sky=1., lmin=20):
        self.fsky = f_sky
        self.lmin = lmin
        self.freqs = np.array([27,39,93,145,225,280])
        self.depth_p = np.array([25, 17, 1.9, 2.1, 4.2, 10]) * np.sqrt(2)
        self.fwhm = np.array([91, 63, 30, 17, 11, 9])
        self.lKnee_p = np.array([15, 15, 25, 25, 35, 40])
        self.gamma_p = np.array([-2.4, -2.4, -2.5, -3.0, -3.0, -3.0])
        
    def get_N_ell(self, lmax=1536, frequency=30):
        '''
            lmax : maximum multipole of the Cls
            freq : frequency
            
            returns polarisation E and B noise Cls
        '''
        
        idx = np.where(self.freqs==frequency)
        temp_p = (np.arange(lmax+1)/self.lKnee_p[idx])**self.gamma_p[idx]
        temp_p[:self.lmin] = 0.
        
        #transf = hp.gauss_beam(np.radians(self.fwhm[idx]/60.), lmax=lmax, pol=False)
        
        Nell_E = Nell_B = (np.radians(self.depth_p[idx]/60.)**2 * (1+temp_p)) #* (self.fsky/0.1)**2
        
        return np.array([Nell_E, Nell_B])
    
    def get_noise_map(self, nside = 512, lmax=1536, frequency=30, seed=1234, hits = True):
        
        '''
            lmax : maximum multipole of the maps
            freq : frequency
            seed : seed for realisations
            hits : True or False (hit map weight)
            
            returns polarisation Q and U noise maps
        '''
        
        #idx = np.where(self.freqs==freq)
        Nl_E, Nl_B = self.get_N_ell(lmax, frequency)
        
        np.random.seed(seed)
        nlms = hp.synalm((Nl_E, Nl_B, None), lmax=lmax, new=True)
        nmap = hp.alm2map_spin(nlms, nside=nside, spin=2, lmax=lmax, mmax=lmax)
        del nlms
        
        if hits == True:
            hitmap = hp.read_map('/home/dekak/Desktop/simons_sims/hitmaps/sat_hitmap_nside_512.fits', field=0)
            #hitmap = hp.ud_grade(hitmap, nside_out=nside, power=0)
            
            norm_hits = np.where(hitmap > 0, hitmap/hitmap.max(), 0)
            del hitmap
            inv_hits = np.where(norm_hits > 0, 1/np.sqrt(norm_hits), 0)
            
            for m in nmap:
                m *= inv_hits
            del norm_hits, inv_hits
            
        return nmap    
