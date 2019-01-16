import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameters
from lmfit import Model

f, _, vrms, _, _, _, _, vrms2, pha = np.loadtxt("data/a2_c100h200_313k_torsional_data.csv", delimiter=',', unpack=True, skiprows=3)  # loading first interval
targetf= 313*1000

def lorentzian(f,a, fr, gamma, c):
    return a/(gamma**2+(f-fr)**2)+c

mod = Model(lorentzian)
parrs = Parameters()


parrs.add('a', value=100, min=0,
          max=10000)
parrs.add('fr', value=targetf, min=targetf-10*1000,
          max=targetf+10*1000)
parrs.add('gamma', value=500, min=10,
          max=10000)
parrs.add('c', value=0, min=-10,
          max=10)

result = mod.fit(vrms2, parrs, f=f) # fitting
best_a = result.best_values['a']
best_fr = result.best_values['fr']
best_gamma = result.best_values['gamma']
best_c = result.best_values['c']

print(result.fit_report())

qfactor = best_fr/2/best_gamma

plt.subplot(2,1,1)
plt.plot(f/1000, vrms2, "o")
plt.plot(f/1000, lorentzian(f, best_a, best_fr, best_gamma, best_c), "-r",lw=3, label="Q = {:0.2f}".format(qfactor))
plt.ylabel(r"Signal (A. U.)", fontsize=18)
plt.yticks([],[],fontsize=18)
plt.xticks([],[],fontsize=18)
plt.legend()

plt.subplot(2,1,2)
plt.plot(f/1000, pha, "o")
plt.ylabel(r"Phase (A. U.)", fontsize=18)
plt.xticks(fontsize=18)
plt.locator_params(nbins=5)
plt.xlabel(r"$f_m$ (kHz)", fontsize=18)
plt.yticks([],[],fontsize=18)

plt.tight_layout()

plt.show()

Q = best_fr/(2*best_gamma)
print(Q)