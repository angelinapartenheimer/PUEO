import crpropa
from crpropa import *
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import json
from common import *


### Setup ###

#Plot settings for figures
font = { 'family': 'DejaVu Sans', 'weight' : 'normal', 'size': 30}
pl.rc('font', **font)
legendfont = {'fontsize' : 24, 'frameon' : True}
pl.rc('legend', **legendfont)
pl.rc('lines', linewidth = 4)
plt.figure()


#Convert https://pos.sissa.it/301/486/pdf Figure 5 units to E2dNdE and GeV units (used for normalization)
def AugerE3dNdE_to_GeVE2dNdE(Auger_Ebin, Auger_spec): # eV^2/km^2/sr/yr   
    GeV_Ebin = Auger_Ebin/1e9
    E2_spec = Auger_spec/Auger_Ebin # eV/km^2/sr/yr
    E2_spec /= 1e9 # GeV/km^2/sr/yr
    E2_spec /= (1e5)**2 # GeV/cm^2/sr/yr
    E2_spec /= 365*24*60*60 # GeV/cm^2/sr/s
    return GeV_Ebin, E2_spec



#Data from https://pos.sissa.it/301/486/pdf Figure 5
def get_Auger():
    Auger_eV, Auger_E3dNdE, Auger_unc = read_digitized_spec('Auger_E3dNdE.txt', err = True)
    AugerGeV, Auger_E2dNdE = AugerE3dNdE_to_GeVE2dNdE(Auger_eV, Auger_E3dNdE)
    _, Auger_E2dNdE_unc = AugerE3dNdE_to_GeVE2dNdE(Auger_eV, Auger_unc)
    return AugerGeV, Auger_E2dNdE, Auger_E2dNdE_unc


#Data from Auger
AugerBins, AugerSpec, Auger_err = get_Auger()


#Read CRPropa simulation output
def read_crpropa(outfilename): 
    data = pl.genfromtxt(outfilename, names=True)
    cr_data = data[np.where([pid not in [12, 14, 16] for pid in abs(data['ID'])])[0]]
    nu_data = data[np.where([pid in [12, 14, 16] for pid in abs(data['ID'])])[0]]
    return cr_data, nu_data


### Main functions ###


#Read CRPropa data, rescale spectrum, rescale source evolution, and normalize
def get_E2dNdE(crdata, Rmax_EV, alpha, m):
       
    z = np.array([comovingDistance2Redshift(X0) for X0 in crdata["X0"]*Mpc])
    E0 = crdata["E0"] #CRPropa output in GeV
    E = crdata["E"] 
    
    #Get mass/charge number of injected parent
    ID0 = [int(id) for id in crdata["ID0"]]
    Z = np.array([int(str(pid)[-6:-4]) for pid in ID0])
    Emax_EeV = Rmax_EV*Z
    Emax = Emax_EeV * 1e9 #Convert EeV -> GeV
    
    #Make energy bins
    nbins = 50
    logE = np.log10(E)
    binedges = np.logspace(np.min(logE), np.ceil(np.max(logE)), nbins + 1)
    dE = binedges[1:] - binedges[:-1]
    Ebins = binedges[:-1] + dE/2
    
    #Rescale to correct spectral shape
    alpha_scale = 1 - alpha #Simulation injects as E0^-1
    scaling = E0**alpha_scale 
    scaling *= np.exp(-E0/Emax) #Convert injection E^-1 --> E^-alpha * e^(-E0/Emax)
    
    #Rescale to desired source evolution
    if m > 0:
        lowz = np.where(z < 1.5)[0]
        scaling[lowz] *= (1 + z[lowz])**m
        scaling[np.where(z >= 1.5)[0]] *= 2.5**m
    else: 
        scaling *= (1+z)**m
    
    Nevents, _ = np.histogram(E, binedges, weights = scaling)
    spectrum = Nevents/dE #dN/dE
    
    return Ebins, spectrum * Ebins**2



#Normalize spectrum to an observed N = 2.3 events observed by PUEO(expectation for Poisson background-free case)
def normalize_spec(nuBins, nuSpec): 
    
    Ns = 2.3 #Poisson
    N = Nexp(nuBins, nuSpec)
    norm = Ns/N
    
    return norm

    

#Neutrino spectrum corresponding to observed best-fit UHECR composition parameters
def best_fit_spectrum(crdata, nudata, alpha, Emax_EeV, m, crNormE = 10**10.55, crNormVal = 5e-9):
    
    crBins, crSpec = get_E2dNdE(crdata, Emax_EeV, alpha, m)
    nuBins, nuSpec = get_E2dNdE(nudata, Emax_EeV, alpha, m) 
    
    crSimVal = np.interp(crNormE, crBins, crSpec)
    norm = crNormVal/crSimVal #Normalize to best-fit proton spectrum at a chosen energy
    
    nuSpec *= norm
    crSpec *= norm
    
    return nuBins, nuSpec


#Note the typo in the TA ICRC proceedings axis label: they write km^-2 but mean m^-2
CR_best_fits = {'Auger best-fit model': {'alpha': 0.96, 'Emax_EeV': 4.8, 'm': 0, 'filename': 'Auger_best_fit', 'linestyle': 'solid', 'source': 'arxiv 1612.07155'}, 
                'TA best-fit model': {'alpha': 2.06, 'Emax_EeV': 182, 'm': 3, 'filename': 'TA_best_fit', 'linestyle': 'dashdot', 'source': 'https://pos.sissa.it/395/338/'}}



#Figure 1 of 2512.20594
def CosmogenicSpectra_figure(pdata, nudata):
    
    mvals = [3, 5, 7]
    alpha = 2
    Emax_EeV = 100
             
    plt.figure(figsize = (15, 10))
    plt.semilogy()
    plt.xlim(2e8, 2e12)
    plt.ylim(1e-11, 1e-6)
    plt.xlabel(r'$E$ [GeV]')
    plt.ylabel(r'$E^2 dN/dE$ [GeV cm$^{-2}$ sr$^{-1}$ s$^{-1}$]')
    
    plt.plot(PUEO_GeV, PUEO_sens, color = 'dimgray', label = 'PUEO 30-day sensitivity', linewidth = 4)
    plt.errorbar(AugerBins, AugerSpec, yerr = Auger_err, fmt = 'o', color = 'black', label = 'Auger observed spectrum')
    
    key = []
    pcolor = 'maroon'
    nucolor = 'turquoise'
    linestyles = ['dotted', 'dashed', 'solid']
    linewidths = [3, 4, 5]
    
    for i, mindex in enumerate(mvals): 
        
        pBins, pSpec = get_E2dNdE(pdata, Emax_EeV, alpha, mindex)
        nuBins, nuSpec = get_E2dNdE(nudata, Emax_EeV, alpha, mindex) 

        norm = normalize_spec(nuBins, nuSpec)
        nuSpec *= norm
        pSpec *= norm
    
        #Plot spectra 
        plt.loglog(nuBins, nuSpec, color = nucolor, linestyle = linestyles[i], linewidth = linewidths[i])
        plt.loglog(pBins, pSpec, color = pcolor, linestyle = linestyles[i], linewidth = linewidths[i])
        key.append(Line2D([0], [0], linestyle = linestyles[i], linewidth = linewidths[i], color = 'black', label = 'm = {}'.format(mindex)))
    
    #Best fits for TA, Auger
    normE = (10**19.55)/1e9 #GeV
    normVal = 5e-9 #From https://arxiv.org/pdf/1612.07155 Figure 3
    
    #Plot neutrino spectra for the best-fit models
    for fit in list(CR_best_fits.keys()):      
                
        model = CR_best_fits[fit]
        filename = model['filename']
        Emax_EeV = model['Emax_EeV']
        alpha = model['alpha']
        mindex = model['m']
        
        cr_data, nu_data = read_crpropa(filename+'.txt')
    
        #My simulated spectra
        nuBins, nuSpec = best_fit_spectrum(cr_data, nu_data, alpha, Emax_EeV, mindex, normE, normVal)
        plt.loglog(nuBins, nuSpec, color = 'lightgray', linestyle = model['linestyle'], label = fit)
    
    key.append(Patch(facecolor = pcolor, label = 'proton'))
    key.append(Patch(facecolor = nucolor, label = 'neutrino'))
    legend2 = plt.legend(handles = key, loc = 1, framealpha = 1, ncol = 2)
    legend = plt.legend(loc = 4, framealpha = 1)
    plt.gca().add_artist(legend2)

    

#Evaluate proton fraction fp for a given spectrum, source evolution, and sensitivity 
def proton_fraction(pdata, nudata, alpha, Emax_EeV, mindex):
    
    nuBins, nuSpec = get_E2dNdE(nudata, Emax_EeV, alpha, m = mindex) #simulated data
    pBins, pSpec = get_E2dNdE(pdata, Emax_EeV, alpha, m = mindex)
    
    norm = normalize_spec(nuBins, nuSpec)
    pEnergy = 10**19.55 / 1e9 #For fixed energy definition
    
    nuSpec *= norm
    pSpec *= norm
    
    pFlux = np.interp(pEnergy, pBins, pSpec) #Proton flux at fp
    AugerFlux = np.interp(pEnergy, AugerBins, AugerSpec) #Total CR flux at fp energy
    
    fp = (pFlux/AugerFlux)
    if fp > 1: 
        fp = 1
    
    return fp
        
        
        
#Get the minimum or maximum fp for a list of m values
def get_fractions(pdata, nudata, alpha, Emax): 
    
    mvals = np.linspace(0, 7, 20)
    fractions = []
    for m in mvals:
        fp = proton_fraction(pdata, nudata, float(alpha), float(Emax), mindex = float(m))
        fractions.append(fp)
        
    return fractions



#Figure 4 of 2512.20594
def Fp_figure(pdata, nudata): 
    
    plt.figure(figsize = (15, 10))
    plt.xlim(1, 7)
    plt.ylim(0, 1)
    plt.xlabel(r'$m$')
    plt.ylabel(r'proton fraction $(f_p)$')
    
    mvals = np.linspace(0, 7, 20)
    linestyles = ['solid', 'dashed', 'dashdot']
    key = []
    
    Emax1 = 40
    lines1 = []
    for i, gamma in enumerate([1, 2, 3]): 
        fp = get_fractions(pdata, nudata, gamma, Emax1)
        lines1.append(fp)
        plt.plot(mvals, fp, color = 'burlywood', linestyle = linestyles[i]) 
    plt.fill_between(mvals, lines1[0], lines1[2], alpha = 0.2, color = 'burlywood')
    
    Emax2 = 1000
    lines2 = []
    for i, gamma in enumerate([1, 2, 3]): 
        fp = get_fractions(pdata, nudata, gamma, Emax2)
        plt.plot(mvals, fp, color = 'cyan', linestyle = linestyles[i])  
        lines2.append(fp)
        key.append(Line2D([0], [0], linestyle = linestyles[i], color = 'black', label = r'$\gamma$ ='+str(gamma)))
    plt.fill_between(mvals, lines2[0], lines2[2], alpha = 0.2, color = 'cyan')
    
    IC_m, IC_lim = read_digitized_spec('IceCube_fp.txt')
    plt.plot(IC_m, IC_lim, linestyle = 'dashed', color = 'dimgray', label = 'IceCube limit')
    key.append(Line2D([0], [0], linestyle = 'dashed', color = 'dimgray', label = 'IceCube conservative limit'))
    
    key.append(Patch(facecolor = 'burlywood', label = r'$E_{max} = $'+str(Emax1)+' EeV'))
    key.append(Patch(facecolor = 'cyan', label = r'$E_{max} = $'+str(Emax2)+' EeV'))
    
    plt.axvline(5, color = 'black', linestyle = 'dotted', linewidth = '2')
    plt.text(5 + 0.1, 0.05, 'MHL-AGN', color = 'black')
    
    plt.axvline(3.4, color = 'black', linestyle = 'dotted', linewidth = '2')
    plt.text(3.4 + 0.1, 0.05, 'SFR', color = 'black')
    
    plt.axvline(2.1, color = 'black', linestyle = 'dotted', linewidth = '2')
    plt.text(2.1 + 0.1, 0.05, 'GRB', color = 'black')
    
    legend1 = plt.legend(handles = key, framealpha = 0.5, loc = 2, ncol = 2)
    plt.gca().add_artist(legend1)
    plt.show()
        
