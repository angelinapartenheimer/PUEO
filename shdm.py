import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import sympy as sp
import scipy
import h5py
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from HDMSpectra import HDMSpectra
from common import *



#Make bins
def bins(minval, maxval, nbins = 100): 

    dx = (maxval - minval)/(nbins)
    bins = np.linspace(minval, maxval, nbins+1)[:-1] + dx/2
    
    return bins, dx



#Constants
rho0 = 0.4 #GeV/cm^3 Galactic DM density 
rc = 20 #kpc Galactic radius 
Rs = 8.25 #kpc distance to Galactic center

h = 0.674 #Hubble constant
H0 = h * 1.08e-28 #Hubble constant converted to cm^-1
OmegaL = 0.685
OmegaM = 0.315
OmegaDM = 0.265
rhoc = 4.79e-6 #critical energy density of the universe (GeV/cm^3)

x = np.logspace(-6.,0.,1000) # Energy fraction values, x = 2E/mDM



#Spectrum from the decay of a single DM particle
def DMSpectrum(x, mDM, delta = False): 

    nue = 12
    numu = 14
    nutau = 16
    
    #Decay into neutrinos of same type as primary
    dNdx11 = HDMSpectra.spec(nue, nue, x, mDM, 'HDMSpectra/data/HDMSpectra.hdf5', delta = True)
    dNdx22 = HDMSpectra.spec(numu, numu, x, mDM, 'HDMSpectra/data/HDMSpectra.hdf5', delta = True)
    dNdx33 = HDMSpectra.spec(nutau, nutau, x, mDM, 'HDMSpectra/data/HDMSpectra.hdf5', delta = True)
    
    delta_correction = (dNdx11[-1] + dNdx22[-1] + dNdx33[-1])/3
    
    if delta: 
        return delta_correction * 2 #for nu + nubar
    
    #Plus a contribution from decay into other flavors (through electroweak corrections)
    dNdx12 = HDMSpectra.spec(nue, numu, x, mDM, 'HDMSpectra/data/HDMSpectra.hdf5')
    dNdx13 = HDMSpectra.spec(nue, nutau, x, mDM, 'HDMSpectra/data/HDMSpectra.hdf5')
    dNdx21 = HDMSpectra.spec(numu, nue, x, mDM, 'HDMSpectra/data/HDMSpectra.hdf5')
    dNdx23 = HDMSpectra.spec(numu, nutau, x, mDM, 'HDMSpectra/data/HDMSpectra.hdf5')
    dNdx31 = HDMSpectra.spec(nutau, nue, x, mDM, 'HDMSpectra/data/HDMSpectra.hdf5')
    dNdx32 = HDMSpectra.spec(nutau, numu, x, mDM, 'HDMSpectra/data/HDMSpectra.hdf5')
    
    dNdx_primary = dNdx11[:-1] + dNdx22[:-1] + dNdx33[:-1]
    dNdx_secondary = dNdx12 + dNdx13 + dNdx21 + dNdx23 + dNdx31 + dNdx32
    
    dNdx = (dNdx_primary + dNdx_secondary)/3 #assume equal branching ratios
        
    dNdx *= 2 #also include nubar

    Ebins = mDM * x /2 #Energy bins
    dNdE = (2/mDM) * dNdx
    
    return Ebins, dNdE



#Galactocentric distance
def r(s, b, l):
    return np.sqrt(s**2 + Rs**2 - 2*s*Rs*np.cos(b)*np.cos(l))



#Argument for the integral over the Galactic DM density
def integrand(l, b, s):
    
    r_val = r(s, b, l)
    y = r_val / rc
    denominator = y * (1 + y)**2
    
    return np.sin(b) * rho0 / denominator



#Do the integral
def Galactic_integral(): 
    
    s_lower = 0
    s_upper = np.inf
    b_lower = 0
    b_upper = np.pi
    l_lower = 0
    l_upper = 2 * np.pi
    
    result, error = scipy.integrate.tplquad(integrand, s_lower, s_upper, b_lower, b_upper, l_lower, l_upper)
    result /= 4*np.pi #all-sky average
    result *= 3.086e21 #convert from kpc to cm
    
    return result


#Save this as a constant
Galactic_average = Galactic_integral()



#Observed spectrum from Galactic DM distribution
def Galactic_E2dNdE(mDM, tauDM = 1e29): 

    Ebins, dNdE = DMSpectrum(x, mDM)
    spec =  dNdE * Galactic_average / (4*np.pi * tauDM * mDM)
    
    #delta correction
    Cdelta = DMSpectrum(x, mDM, delta = True)
    spec[-2] += (2/mDM) * Cdelta * Galactic_average / (4*np.pi * tauDM * mDM)
    
    return Ebins, Ebins**2 * spec



#Hubble function
def H(z): 
    return H0 * np.sqrt(OmegaL + OmegaM*(1+z)**3)



#Observed spectrum from cosmological DM distribution
def Extragalactic_E2dNdE(mDM, tauDM = 1e29): 
    
    x = np.logspace(-6.,0.,1000)
    
    zmin = 0
    zmax = 1000
    nbins = 1000000
    zbins, dz = bins(zmin, zmax, nbins)
    
    Ebins, dNdE0 = DMSpectrum(x, mDM)
    
    spec = np.zeros_like(Ebins)
    
    dE = np.insert(Ebins[1:] - Ebins[:-1], 0, Ebins[0])
    Cdelta = DMSpectrum(x, mDM, delta = True)
    
    for z in zbins:
        
        dNdE = np.interp(Ebins*(1+z), Ebins, dNdE0) 
        spec += dNdE*dz/H(z)
    
    #delta correction
    #comes from replacing dN/dx with Cdelta * delta(x-1) in the integral
    z_corr = mDM/2/Ebins - 1 #Where E*(1+z) = mDM/2
    spec += Cdelta / (Ebins * H(z_corr))  
    
    spec *= OmegaDM*rhoc/(4*np.pi * tauDM * mDM)
        
    return Ebins, Ebins**2 * spec

    
    
#Figure 2 of 2512.20594
def SHDM_figure(): 
    
    mDM_vals = np.array([1e11])
    mDM_labels = ['1e11']
    mDM_linewidths = [2]
    colors = ['darkgreen']
    
    plt.figure(figsize = (15, 10))
    plt.semilogy()
    plt.xlim(2e8, 2e12)
    plt.ylim(1e-11, 1e-6)
    plt.xlabel(r'$E_{\nu}$ [GeV]')
    plt.ylabel(r'$E_{\nu}^2 dN_{\nu}/dE_{\nu}$ [GeV cm$^{-2}$ sr$^{-1}$ s$^{-1}$]')
    
    plt.plot(PUEO_GeV, PUEO_sens, color = 'dimgray', label = 'PUEO 30-day sensitivity', linewidth = 4)
    
    tauDM = 1e30
    for i, mDM in enumerate(mDM_vals): 
        
        Ebins_Egal, E2dNdE_Egal = Extragalactic_E2dNdE(mDM, tauDM)
        plt.loglog(Ebins_Egal, E2dNdE_Egal, linewidth = mDM_linewidths[i], color = colors[i], linestyle = 'dotted', label = 'Extragalactic') 
        
        Ebins_Gal, E2dNdE_Gal = Galactic_E2dNdE(mDM, tauDM)
        plt.loglog(Ebins_Gal, E2dNdE_Gal, color = colors[i], linestyle = 'dashed', label = 'Galactic')
        
        plt.loglog(Ebins_Egal, E2dNdE_Gal+E2dNdE_Egal, color = colors[i], linestyle = 'solid', label = 'Total')
    
    plt.legend()
    plt.show()



#Smallest possible tauDM inferred from a PUEO nondetection
def min_tauDM(mDM): 
    
    Ns = 2.3 #Poisson
      
    Ebins_old = mDM * x /2
    tauDM = 1e29
    
    #(Unnormalized) expected number of neutrinos
    _, E2dNdE_Egal = Extragalactic_E2dNdE(mDM, tauDM)
    spec_bins, E2dNdE_Gal = Galactic_E2dNdE(mDM, tauDM)
    spec = E2dNdE_Egal + E2dNdE_Gal
    N = Nexp(spec_bins, spec)
    norm = Ns/N
    
    return tauDM/norm


#Minimum tauDM for a range of mDM values
def tau_vs_mDM(): 
    
    mDM_vals = np.logspace(7, 11)
    tau_vals = [min_tauDM(mDM) for mDM in mDM_vals]
    
    return mDM_vals, tau_vals


    
#Figure 5 of 2512.20594
def TauvsMdm_figure(): 
    
    plt.figure(figsize = (15, 10))
    plt.semilogy
    plt.xlim(1.5e7, 1e11)
    plt.ylim(1e27, 1e30)
    plt.xlabel(r'$m_{DM}$ [GeV]')
    plt.ylabel(r'$\tau_{DM}$ [s]')
    
    #This step takes awhile
    mDM, tauDM = tau_vs_mDM()
    
    #Plot PUEO limit
    plt.loglog(mDM, tauDM, color = 'orange', linestyle = 'solid')
    plt.fill_between(mDM, tauDM, interpolate = True, hatch = '/', facecolor = 'none', edgecolor = 'orange', label = 'projected PUEO limit')
    
    #Existing limits
    mDM_lim_nu, tauDM_lim_nu = read_digitized_spec('existing_tauDM_lim_nuonly.txt')
    mDM_lim, tauDM_lim = read_digitized_spec('existing_tauDM_limit.txt')
    
    plt.loglog(mDM_lim, tauDM_lim, color = 'dimgray', linestyle = 'dotted', label = r'existing $\gamma$-ray limits')
    plt.loglog(mDM_lim_nu, tauDM_lim_nu, color = 'dimgray', linestyle = 'dashed', label = r'existing $\nu$ limits')
    plt.fill_between(mDM_lim_nu, tauDM_lim_nu, alpha = 0.1, color = 'dimgray')
    
    plt.legend()
    #plt.grid()
    plt.show()
