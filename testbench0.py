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


N  = [16, 32, 64, 128, 256, 512, 1024, 2048] # muestras
fs = [16, 32, 64, 128, 256, 512, 1024, 2048] # Hz

a0 = 1     # Volts
p0 = 0     # radianes
f0 = 5     # Hz

L = len(N)

#Vectores de tiempo de cada dft
fftTime = [0] * L

dftTime = [0] * L

#Por cada prueba
for i in range(L):
    #Obtenemos la señal
    tt, signal = generador_senoidal(fs[i], f0, N[i], a0, p0)

    #FFT
    timeStart = time.perf_counter()

    spectrumFFT = (2/N[i])*np.abs(sc.fft(signal))

    halfFFT = spectrumFFT[:N[i]//2] 

    fftTime[i] = time.perf_counter() - timeStart
    
    #DFT
    timeStart = time.perf_counter()

    spectrumDFT = (2/N[i])*np.abs(dft(signal))

    halfDFT = spectrumDFT[:N[i]//2]

    dftTime[i] = time.perf_counter() - timeStart

#Grafico
plt.plot(N, fftTime, '-*r', label='FFT')
plt.plot(N, dftTime, '-*g', label='DFT')

plt.title('' )
plt.xlabel('Cantidad de muestras')
plt.ylabel('Tiempo de ejecucion[S]')

plt.show()

#a = np.array([1]) #Filtro FIR
#b = (1/average)*(np.ones(average))
#y = signal.lfilter(b, a , xn)

#t-filter -> pagina para generar filtros