#CRpropa simulation used to generate data in cosmogenic neutrino analysis
#Composition can be customized for the relevant model; here we generate pure protons

import crpropa
from crpropa import *
import numpy as np

cmb = CMB()
ebl = IRB_Gilmore12()
urb = URB_Protheroe96()
neutrinos = True
photons = False
electrons = False

def simulate(Nevents):

    # module setup
    m = ModuleList()
    m.add(SimplePropagation(10 * kpc, 10 * Mpc))
    m.add(Redshift())
    m.add(PhotoPionProduction(cmb, photons, neutrinos))
    m.add(PhotoPionProduction(ebl, photons, neutrinos))
    m.add(PhotoPionProduction(urb, photons, neutrinos))
    m.add(NuclearDecay(electrons, photons, neutrinos))
    m.add(PhotoDisintegration(cmb))
    m.add(PhotoDisintegration(ebl))
    m.add(PhotoDisintegration(urb))
    m.add(ElectronPairProduction(cmb))
    m.add(ElectronPairProduction(ebl))
    m.add(MinimumEnergy(1e14 * eV))

    obs = Observer()
    obs.add(Observer1D())
    output = TextOutput('data/simulation/output_protons.txt', Output.Event1D)
    output.setEnergyScale(GeV)
    output.enable(Output.CreatedEnergyColumn)
    output.enable(Output.SourcePositionColumn)
    obs.onDetection(output)
    m.add(obs)
    
    source = Source()
    source.add(SourceUniform1D(0., redshift2ComovingDistance(4)))
    source.add(SourceRedshift1D())
    source.add(SourcePowerLawSpectrum((10**17)*eV, (10**22)*eV, -1))
    
    composition = SourceMultipleParticleTypes()
    composition.add(1000010010, 1) #H
    composition.add(1000020040, 0.) #He
    composition.add(1000070140, 0.) #N
    composition.add(1000140280, 0.) #Si
    composition.add(1000260560, 0.) #Fe 
    source.add(composition)

    m.setShowProgress(True)
    m.run(source, Nevents, True)

    output.close()

simulate(100000)
