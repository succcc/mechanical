import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameters
from lmfit import Model
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()

f, _, vrms, _, _, _, _, vrms2, pha = np.loadtxt(file_path,dtype='float', delimiter=',' , unpack=True, skiprows=3)  # loading first interval

def lorentzian(f,a, fr, gamma):
    return a/(gamma**2+(f-fr)**2)
idMax = np.argmax(vrms2)
fPeak = f[idMax]
vrmsPeak = vrms2[idMax]
mod = Model(lorentzian)
parrs = Parameters()
parrs.add('a', value=1, min=0,
          max=100)
parrs.add('fr', value=fPeak, min=fPeak-1*1000,
          max=fPeak+1*1000)
parrs.add('gamma', value=5, min=0,
          max=10000)


result = mod.fit(vrms2, parrs, f=f) # fitting
best_a = result.best_values['a']
best_fr = result.best_values['fr']
best_gamma = result.best_values['gamma']


print(result.fit_report())

plt.subplot(2,1,1)
plt.plot(f/1000, vrms2, "o")
plt.plot(f/1000, lorentzian(f, best_a, best_fr, best_gamma), "-r",lw=3)
plt.ylabel(r"Signal (A. U.)", fontsize=20)
plt.yticks([],[],fontsize=18)
plt.xticks(fontsize=18)


plt.subplot(2,1,2)
plt.plot(f/1000, pha, "o")
plt.ylabel(r"Phase (A. U.)", fontsize=18)
plt.xticks(fontsize=18)
plt.xlabel(r"$f_m$ (kHz)", fontsize=18)
plt.yticks([],[],fontsize=18)

plt.show()

Q = best_fr/(2*best_gamma)
print(Q)