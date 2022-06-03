# -*- coding: utf-8 -*-
"""
Created on Sun May 29 11:16:39 2016

@author: FredyGabriel
"""
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                     F U N C I O N E S   Ú T I L E S                         #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

def interpL(x1, y1, x2, y2, x):
    '''Interpolación lineal'''
    return (y2 - y1) / (x2 - x1) * (x - x1) + y1

def nudo_cercano(x, x_lista):
    '''
    Índice de posición del valor de x_lista más cercano a x.
    '''
    if x >= x_lista[-1]:
        indice = len(x_lista)-1
    elif x <= x_lista[0]:
        indice = 0
    else:
        b = np.where(x_lista<=x)[0][-1] #índice del elemento de x_lista más
                                    #cercano a x por la izquierda
        d1 = x_lista[b+1] - x #diferencia por la derecha
        d2 = x - x_lista[b] #diferencia por la izquierda
        if d1 < d2:
            indice = b+1
        else:
            indice = b
    return indice

def yLista(L1, L2, x):
    '''Devuelve el valor de la variable dependiente en una función biyectiva
    monotónica definida con numpy.array. Los valores de la lista L1, deben
    estar ordenados de menor a mayor.
    L1: dominio de la función
    L2: recorrido de la función
    x: valor realacionado con L1 para el cual se desea conocer el
        correspondiente valor de L2
    L1 y L2 tienen la misma longitud y se corresponden uno a uno'''
    
    if x in L1: #Si x es igual alguno de los elementos de L1
        i, = np.where(L1 == x)
        return L2[i][0]    
    else:   
        if x >= L1[-1]: #si es mayor o igual al último elemento de L1
            b = len(L1)-1 #El índice es igual al del último elem. de L1
            #Puede ocurrir extrapolación
            x1 = L1[b-1]
            x2 = L1[b]
            y1 = L2[b-1]
            y2 = L2[b]
        elif x <= L1[0]: #si es menor o igual al primer elemento de L1
            #Puede ocurrir extrapolación
            x1 = L1[0]
            x2 = L1[1]
            y1 = L2[0]
            y2 = L2[1]
        else:
            b = np.where(L1<=x)[0][-1] #índice del elemento de L1 más
                                        #cercano a x por la izquierda
            x1 = L1[b]
            x2 = L1[b+1]
            y1 = L2[b]
            y2 = L2[b+1]
        
        return interpL(x1, y1, x2, y2, x)

def M_unit(x, z, L, viga_tipo):
    '''Momentos debidos a cargas puntuales unitarias
    Parámetros:
    ----------
    x_izq: valores x a la izquierda de la carga unitaria
    x_der: valores x a la derecha de la carga unitaria
    z: posición de la carga unitaria desde el extremo izquierdo de la viga
    L: longitud de la viga
    gdr: condiciones de apoyo, tupla (1, 0, 1, 0), 1: restringido, 0: libre
    '''
    #VIGA SIMPLEMENTE APOYADA
    if viga_tipo == 1:
        if 0 <= x <= z:
            return (1-z/L)*x #Momento a la izquierda de la carga unitaria
        elif z < x <= L:
            return z*(1-x/L) #Momento a la derecha de la carga unitaria
        else:
            return 0
        
    #VIGA EMPOTRADA
    elif viga_tipo == 2:
        if 0 <= x <= z:
            return x - z #Momento a la izquierda de la carga unitaria
        else:
            return 0 #Momento en cualquier otro lugar
        
def dea5cm(numero):
    '''Redondea en el múltiplo de 0.05 más cercano hacia abajo.
    Ejemplo:
        dea5cm(0.24) = 0.20
        dea5cm(0.37) = 0.35'''
    if numero*100 % 5 == 0:
        return numero
    elif numero*100 % 10 < 5:
        return round(numero*10)/10
    else:
        return (numero*100 - (numero*100 % 5))/100
        

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                              E S P A C I O                                  #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

class Espacio:
    '''Espacio de parámetros generales para una viga cargada.
       L: longitud de la viga en m. Debe estar entre 1m y 10m (adoptado)
       NE: Número de elementos finitos, 50 por defecto.'''
       
    def __init__(self, L=5.0, NE=50):
        self.L = L #longitud
        self.NE = NE #Número de elementos finitos
        
    #Para evitar valores incorrectos de L
    @property
    def L(self):
        '''Longitud de la viga'''
        return self.__L
    @L.setter
    def L(self, L):
        assert 1.00 <= L <= 10.00, "Debe estar entre 1.00m y 10.00m"
        self.__L = L
        
    @property
    def NE(self):
        '''Número de elementos finitos'''
        return self.__NE
    @NE.setter
    def NE(self, NE):
        assert type(NE) == int, "Debe ser entero positivo"
        self.__NE = NE
        
    @property
    def NN(self):
        '''Número de nudos'''
        return self.NE + 1
    
    @property
    def gdl(self):
        '''Número de grados de libertad (se considera 2 por nudo)'''
        return 2 * self.NN
        
    @property
    def le(self):
        '''Longitud de cada elemento finito'''
        return self.L/self.NE
    
    @property
    def x_centros(self):
        '''Coordenadas de los centros de los elementos finitos'''
        x_centros = np.linspace(self.le/2, self.L-self.le/2, self.NE)
        return x_centros
    
    @property
    def x_nudos(self):
        '''Coordenadas de los nudos de los elementos finitos'''
        x_nudos = np.linspace(0, self.L, self.NN)
        return x_nudos
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                   R E F E R E N T E   A   C A R G A S                       #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

def P_nodales(P, esp):
    '''Devuelve las cargas nodales en todos los EF de una carga puntual.
        P: objeto Carga.puntual()
        esp: objeto Espacio()
    '''
    vector = np.zeros((esp.gdl, 1)) #inicialización del vector de cargas
    x = P.ubic*esp.L
    x_lista = esp.x_nudos
    indice = nudo_cercano(x, x_lista)
    vector[2*indice, 0] = -P.mayorada
    return vector

def q_nodales(q, esp):
    '''Devuelve las cargas nodales en todos los EF de una carga uniformemente
    distribuida en toda la longitud de una viga.
        q: objeto Carga.uniforme()
        esp: objeto Espacio()
    '''
    le = esp.le #longitud del EF
    carga = q.mayorada #carga mayorada
    
    #cargas nodales en coord. locales
    Fe = -carga*le/12 * np.array([[6], [le], [6], [-le]])
    vector = np.zeros((esp.gdl, 1)) #Inicialización del vector de Fuerzas
    #Ensamble
    for i in range(esp.NE):
        vector[2*i:2*i+4, :1] += Fe
    return vector
    
def carga_elemento(ql, esp):
    '''Carga distribuida media en cada elemento'''
    indice_inicio = nudo_cercano(ql.inicio, esp.x_centros)
    indice_fin = nudo_cercano(ql.fin, esp.x_centros)
    q0 = ql.mayorada[0] #Carga inicial
    qf = ql.mayorada[1] #Carga final
    x0 = ql.inicio #Coordenada inicial
    xf = ql.fin #Coordenada final
    x = esp.x_centros[indice_inicio:indice_fin+1] #x válidas
    q = q0 + (qf-q0)/(xf-x0)*(x-x0) #Interpolación de las cargas
    carga = np.zeros(esp.NE) #Inicialización de cargas en c/ EF       
    carga[indice_inicio:indice_fin+1] = q
    return carga

def ql_nodales(ql, esp):
    '''Devuelve las cargas nodales en todos los EF de una carga lineal.
        ql: objeto Carga.lineal()
        esp: objeto Espacio()'''
    le = esp.le
    #Cargas nodales equivalentes por cada elemento
    def Fe(q):
        Fe = -q*le/12 * np.array([
                [6],
                [le],
                [6],
                [-le]
                ])
        return Fe
    #Ensamble del vector de cargas nodales
    Ql = np.zeros((esp.gdl, 1)) #inicialización del vector de cargas
    cargas = carga_elemento(ql, esp)
    for i in range(esp.NE):
        Ql[2*i:2*i+4, :1] += Fe(cargas[i])
    return Ql

def cargas_nodales(cargas, esp):
    '''Genera el vector de cargas nodales a partir de una lista de cargas
    creadas con HoAo.Cargas()
    cargas: lista de cargas
    esp: espacio '''
    carga_ext = 0
    for i in range(len(cargas)):
        carga = cargas[i]
        clase = cargas[i].clase
        if clase == 0: #Carga puntual
            carga_ext += P_nodales(carga, esp)
        elif clase == 1: #Carga uniforme
            carga_ext += q_nodales(carga, esp)
        elif clase == 2: #Carga lineal
            carga_ext += ql_nodales(carga, esp)
        else:
            carga_ext += 0
            
    return carga_ext