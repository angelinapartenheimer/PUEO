import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from common import *


#All figures, equations from Berezinsky (2011) https://arxiv.org/abs/1108.2509 

#Minimum neutrino energy, GeV (Equation 35)
epsilon = 1


#Neutrino horizon (maximum of redshift integration) 
def z_nu(E_GeV):   
    if E_GeV < 2e11: 
        return 2.5e2 * (E_GeV/1e11)**(-2/5)
    else: 
        return 200



#Equation 55: owest redshift from which neutrinos from cosmic string radiation may be expected (minimum of redshift integration)
def zmin(E, alpha7, zstar_znu): 
    E0 = 2.7e13 * epsilon #GeV
    zmin = (E0/E)**(4/7) * alpha7**(8/7) * (zstar_znu)**(3/7) - 1 
    return zmin


#Equation 56
def E2J(alpha7, m5, p = 1, zstar_znu = 1): 
    E = np.logspace(6, 16)
    spec = np.zeros_like(E)
    znu = 200
    zm = zmin(E, alpha7, zstar_znu)
    spec = 2.5e-9 * (1/p) * alpha7**2 * m5**(-1/2) * np.sqrt(znu/200) * (1 - np.sqrt((1 + zm)/(1+znu)))
    spec[np.where(spec < 0)[0]] = 0  
    return E, spec



#Figure 3 of 2512.20594
def CosmicStrings_figure(): 
    
    plt.figure(figsize = (15, 10))
    plt.semilogy()
    plt.xlim(2e8, 2e12)
    plt.ylim(1e-11, 1e-6)
    plt.xlabel(r'$E_{\nu}$ [GeV]')
    plt.ylabel(r'$E_{\nu}^2 dN_{\nu}/dE_{\nu}$ [GeV cm$^{-2}$ sr$^{-1}$ s$^{-1}$]')
    
    plt.plot(PUEO_GeV, PUEO_sens, color = 'dimgray', label = 'PUEO 30-day sensitivity', linewidth = 4)
    
    #E = np.logspace(6, 16)
    E1, E2J1 = E2J(1, 1)
    plt.loglog(E1, E2J1,  color = 'firebrick', linestyle = 'dashed', linewidth = 4, label = r'$\alpha=10^7$, $m = 10^5$ GeV')
    
    E2, E2J2 = E2J(3, 4)
    plt.loglog(E2, E2J2,  color = 'teal', linestyle = 'dotted', linewidth = 4, label = r'$\alpha=3 \times 10^7$, $m = 4 \times 10^5$ GeV')
    
    E3, E2J3 = E2J(2, 0.1)
    plt.loglog(E3, E2J3,  color = 'darkkhaki', linestyle = 'dashdot', linewidth = 4, label = r'$\alpha=2 \times 10^7$, $m = 10^4$ GeV')
    
    plt.legend()
    plt.show()