import numpy as np
import pylab as pl
import matplotlib.pyplot as plt


#Plot settings for figures
font = { 'family': 'DejaVu Sans', 'weight' : 'normal', 'size': 30}
pl.rc('font', **font)
legendfont = {'fontsize' : 24, 'frameon' : True}
pl.rc('legend', **legendfont)
pl.rc('lines', linewidth = 4)
plt.figure()


#Read a digitized spectrum from a text file; used for comparing to published work
def read_digitized_spec(filename, err = False): 
    data = pl.genfromtxt('digitized/'+filename, delimiter = ',')
    bins, spec = data[:, 0], data[:, 1]
    if err: 
        errorbar = data[:, 2] - spec
        return bins, spec, errorbar
    return bins, spec



#PUEO effective area published in https://pos.sissa.it/444/1154/pdf
def get_PUEO_Aeff(): 
    E_eV, Aeff_km2sr = read_digitized_spec('PUEO_Aeff.txt') 
    E_GeV = E_eV/1e9
    Aeff = Aeff_km2sr * 1e10 #km^2 -> cm^2
    return E_GeV, Aeff


#PUEO sensitivity
PUEO_eV, PUEO_sens_eV = read_digitized_spec('PUEO_30d.txt')
PUEO_GeV = PUEO_eV/1e9
PUEO_sens = PUEO_sens_eV*PUEO_GeV #E^2 dN/dE


#Expected number of observed neutrinos given an input Ebins [GeV], E^2dNdE [GeV cm^-2 s^-1 sr^-1]
#Used to calculate values in Table 1 of 2512.20594
def Nexp(spec_bins, spec): 
    
    Tobs = 30 * 24 * 60 * 60   
    dig_Ebins, dig_Aeff = get_PUEO_Aeff() 
    
    #This populates the digitized effective area more densely
    E_edges = np.logspace(np.log10(dig_Ebins[0]), np.log10(dig_Ebins[-1]), 1000)
    Aeff_edges = np.interp(E_edges, dig_Ebins, dig_Aeff)
    
    Ebins = (E_edges[1:] + E_edges[:-1])/2
    Aeff = (Aeff_edges[1:] + Aeff_edges[:-1])/2
    dE = E_edges[1:] - E_edges[:-1]
    
    dNdE = np.interp(Ebins, spec_bins, spec/(spec_bins**2), left = 0, right = 0)
    Nexp = Tobs * np.sum(dNdE * Aeff * dE) 
    
    return Nexp
