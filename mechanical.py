import numpy as np
from lmfit import Parameters
from lmfit import Model
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()


#f, _, vrms, _, _, _, _, vrms2, pha = np.loadtxt("data/fg_c100h100_600khz_h14m08.csv", delimiter=',', unpack=True, skiprows=3)  # loading first interval
f, _, vrms, _, _, _, _, vrms2, pha = np.loadtxt(file_path, delimiter=',' , unpack=True, skiprows=3)  # loading first interval


targetf= 1314*1000

def lorentzian(f,a, fr, gamma, c):
    return a/(gamma**2+(f-fr)**2)+c

idMax = np.argmax(vrms2)

fPeak = f[idMax]
vrmsPeak = vrms2[idMax]
mod = Model(lorentzian)
parrs = Parameters()
parrs.add('a', value=0.5, min=0,
          max=1)
parrs.add('fr', value=fPeak, min=fPeak-10*1000,
          max=fPeak+10*1000)
parrs.add('gamma', value=200, min=0,
          max=10000)

parrs.add('c', value=0, min=-0.001,
          max=0.001)

result = mod.fit(vrms2, parrs, f=f) # fitting
best_a = result.best_values['a']
best_fr = result.best_values['fr']
best_gamma = result.best_values['gamma']
best_c = result.best_values['c']

print(result.fit_report())

qfactor = best_fr/2/best_gamma

plt.subplot(2,1,1)
plt.plot(f/1000, vrms2*1e6, "o")
plt.plot(f/1000, lorentzian(f, best_a, best_fr, best_gamma, best_c)*1e6, "-r",lw=3, label="f = {:0.3f} kHz\n Q = {:0.2f}".format(best_fr/1000,qfactor))
plt.ylabel(r"Signal (uV)", fontsize=18)
# plt.yticks([],[],fontsize=18)
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

