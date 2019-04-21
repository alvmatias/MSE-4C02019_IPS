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

a0 = 1      # Volts
p0 = 0      # radianes

#Arreglo de frecuencias de desintonia
fd = np.array([[0.00], [0.01], [0.25], [0.5]])

#Arreglo de frecuencias fundamentales
f0 = fs/4 + fd      # Hz

#Vector de "paddings"
zeros = [0, N//10, N, 10*N]

#Largo del vector de "paddings"
Lz = len(zeros)

#Largo del vector de frecuencias de desintonia
Lfd = len(fd)

#Matriz de porcentaje de error de estimacion de frecuencia fundamental
percentage = [[0 for x in range(Lfd)] for y in range(Lz)]

#Por cada frecuencia de desintonia
for i in range(Lfd): 
    #Por cada ejemplo de padding
    for j in range(Lz):
        #Calculamos señal
        tt, signal = generador_senoidal(fs, f0[i], N, a0, p0)
        #Realizamos ventaneo
        padSignal = np.pad(signal, (zeros[j], zeros[j]), 'constant')
        #Numero de muestras
        NN = zeros[j]*2 + N
        #Arreglo de frecuencias del espectro
        ff = np.linspace(0, fs/2, NN/2).flatten()
        #Calculamos espectro
        spectrum = (2/NN)*np.abs(sc.fft(padSignal))    
        #Nos quedamos con la primer mitad del espectro
        halfSpectrum = spectrum[:NN//2]  
        #Obtenemos el estimador de la fundamental
        f = estimar_fundamental(halfSpectrum, fs, NN)      
        #Calculamos el porcentaje de error
        percentage[i][j] = abs(f0[i][0]-f)*100/(f0[i][0])
        #Grafico
        plt.figure(i+1)

        plt.stem(ff, halfSpectrum)

        plt.title('FFT: $F_s/4$ + ' + str(fd[i][0]) + ' Hz ' + 'Padding simetrico: ' + str(zeros[j]))
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Amplitud normalizada')

        plt.show()


