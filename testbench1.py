# -*- coding: utf-8 -*-
## Inicialización del Notebook del TP4

import numpy as np
from pandas import DataFrame
from IPython.display import HTML
from scipy import signal as sig
from  matplotlib import patches
import matplotlib as mpl
import scipy.io as scipyio
import matplotlib.pyplot as plt
import scipy.fftpack as sc 

def zplane(b,a,filename=None):
    """Plot the complex z-plane given a transfer function.
    """

    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = b/float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a/float(kd)
    else:
        kd = 1
        
    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    # Plot the zeros and set marker properties    
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0,
              markeredgecolor='k', markerfacecolor='g')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0,
              markeredgecolor='r', markerfacecolor='r')

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set the ticks
    r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    

    return z, p, k


#ecg_lead: Registro de ECG muestreado a  𝑓𝑠=1  KHz durante una prueba de esfuerzo
#qrs_pattern1: Complejo de ondas QRS normal
#heartbeat_pattern1: Latido normal
#heartbeat_pattern2: Latido de origen ventricular
#qrs_detections: vector con las localizaciones (en # de muestras) donde ocurren los latidos
mat = scipyio.loadmat('ECG_TP4.mat')

ecg_lead = mat['ecg_lead']  
qrs_pattern1 = mat['qrs_pattern1']
heartbeat_pattern1 = mat['heartbeat_pattern1']
heartbeat_pattern2 = mat['heartbeat_pattern2']
qrs_detections = mat['qrs_detections']

fs = 1000 #1KHz
ts = 1/fs
ecg_N = 25000 #len(ecg_lead)
N = len(ecg_lead[:ecg_N])
tt = np.linspace(0, (N-1)*ts, N).flatten()

plt.figure(1)

plt.plot(tt, ecg_lead[:ecg_N])

plt.title('Registro ECG')
plt.xlabel('Tiempo en Segundos')
plt.ylabel('Amplitud')
plt.grid(which='both', axis='both')


ff = np.linspace(0, fs/2, N/2).flatten()

spectrum = (2/N)*np.abs(sc.fft(ecg_lead[:ecg_N]))
halfSpectrum = spectrum[:N//2] 

plt.figure(6)

plt.plot(ff, halfSpectrum)

plt.title('FFT: Registro ECG')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud Normalizada')
plt.grid(which='both', axis='both')

N = len(qrs_pattern1)
tt = np.linspace(0, (N-1)*ts, N).flatten()
