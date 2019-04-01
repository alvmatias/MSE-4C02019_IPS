import numpy as np
from scipy import signal as sgn
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.fftpack as sc 
import time

#from pandas import DataFrame
#from IPython.display import HTML

def generador_senoidal (fs, f0, N, a0=1, p0=0):
    """ 
    
    brief:  Generador de señales senoidal, con argumentos
    
    fs:     frecuencia de muestreo de la señal [Hz]
    N:      cantidad de muestras de la señal a generar
    f0:     frecuencia de la senoidal [Hz]
    a0:     amplitud pico de la señal [V]
    p0:     fase de la señal sinusoidal [rad]
    
    como resultado la señal devuelve:
    
    signal: senoidal evaluada en cada instante 
    tt:     base de tiempo de la señal
    """    

    # comienzo de la función

    ts = 1/fs # tiempo de muestreo
    
    # grilla de sampleo temporal
    tt = np.linspace(0, (N-1)*ts, N).flatten()

    # prestar atención que las tuplas dentro de los diccionarios también pueden direccionarse mediante "ii"
    signal = a0 * np.sin( 2*np.pi*f0*tt + p0 )          
    
    # fin de la función
    
    return tt, signal

def generador_ruido (fs, N, media, varianza):

    ts = 1/fs # tiempo de muestreo
    
    # grilla de sampleo temporal
    tt = np.linspace(0, (N-1)*ts, N).flatten()

    signal = np.sqrt(varianza) * np.random.randn(N,1) + media

    # fin de la función
    
    return tt, signal

def generador_cuadrada (A, d, N, fs):

    ts = 1/fs # tiempo de muestreo
    
    # grilla de sampleo temporal
    tt = np.linspace(0, (N-1)*ts, N).flatten()
    
    #t = np.arange(0, N-1, ts)
    signal = A * sgn.square(2 * np.pi *tt, d)

    # fin de la función
    
    return tt, signal


def dft(signal):

    N = len(signal)
    
    n = np.arange(N)
    k = n.reshape([N, 1])
    exp = np.exp(-((2j * np.pi * n * k) / N))
    spectrum = np.dot(signal, exp)

    return spectrum


def estimar_fundamental(spectrum, fs, N):
     
    f = 0 

    n = []

    L =  len(spectrum)
    
    m = max(spectrum)
    
    for i in range(L):
        if(spectrum[i] == m):
            n.append(i)

    for i in range(len(n)):
        f += n[i]

    f = (f / len(n)) * fs/N

    return f


N  = 1024     # muestras
fs = 1024     # Hz

fd = np.array([[0.00], [0.01], [0.25], [0.5]])

a0 = 1      # Volts
p0 = 0      # radianes
f0 = fs/4 + fd      # Hz

L = len(fd)

tt, signal = generador_senoidal(fs, f0, N, a0, p0)

padSignal = np.pad(signal, (N//10, 10*N), 'constant')

NN = N//10 + N + 10*N

spectrum = (2/NN)*np.abs(sc.fft(padSignal))     #arreglo bidimensional

halfSpectrum = spectrum[:,:NN//2]            #arreglo bidimensional

f = np.linspace(0, fs/2, NN/2).flatten()

for i in range(L):
    plt.figure(i+1)

    plt.stem(f, halfSpectrum[i])

    plt.title('FFT: $F_s/4$ + ' + str(fd[i]))
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud[V]')

    plt.show()