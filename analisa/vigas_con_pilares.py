#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 15:20:02 2020

author: @fgrv
"""
import numpy as np
import csv

class vigaContinua:
    '''
    Cálculo de vigas continuas apoyadas sobre pilares, por el método matricial
    de rigidez.
    
    Sistema de coordenadas:
    ---------------------
    xx: en dirección del eje de la viga, positivo a la derecha;
    yy: en dirección de la altura de la viga, positivo hacia arriba;
    zz: en dirección de la base de la viga, positivo saliente de la pantalla;
    origen: en el extremo izquierdo de la viga continua.
    
    Se utilizan elementos de viga con 3 grados de libertad por nudo
    1. Desplazamiento horizontal;
    2. Desplazamiento vertical;
    3. Giro (positivo en sentido antihorario).

    Notas:
    -----
    1. Se admite la posibilidad de voladizos en ambos extremos de la viga.
    2. El apoyo en base de los pilares se considera del tipo empotramiento.
    3. Debe haber por lo menos un pilar.
    4. Se considera un módulo de elasticidad para cada tramo, que podría ser
        igual o diferente a los demás tramos.
    5. Los pilares pueden ser, cada uno, de un material diferente al de los
        demás, es decir, es posible considerar un módulo de elasticidad por
        pilar.
    6. No se considera el pandeo en los pilares. Los datos de los pilares son
        solo a los efectos de tomar en cuenta su rigidez en el cálculo de la
        viga continua.
    7.  Se admiten cargas solo sobre la viga continua, no así en los pilares
        (cargas laterales, por ejm.)

    Wiki:
    ----
    1. Viga continua: elemento estructural horizontal, apoyado sobre uno o varios
        pilares. Consta de uno o varios tramos, cada uno de los cuales puede
        tener secciones transversales diferentes pero constantes en el tramo.
    2. Tramo: elemento de viga comprendido entre ejes de pilares, o entre el eje
        de un pilar y el extremo de un voladizo.
    3. Pilar o columna: elemento estructural vertical que sirve de apoyo a la
        viga.
    4. Carga puntual:
    5. Momento concentrado:
    '''    
    def __init__(self, NPT=10, bb=[20], hb=[40], spanValues=[4], cantL=0,
                 cantR=0, eb=[30000], tc=[20, 20], wc=[30, 30], hc=[4, 4],
                 ec=[30000, 30000], cargas=[[(3, 1)]]):
        '''            
        Los valores por defecto consideran una viga de 20cm x 40cm, 4m de luz
        sobre pilares de 20cm x 30cm de 4m de altura en los extremos de la
        viga. El módulo de elasticidad por defecto es de 30000 MPa en todos
        los elementos.
        
        PARÁMETROS:
        ----------
        NPT: número de puntos por tramo, para el trazado de los diagramas.
            (en todos los tramos se adoptará el mismo valor).
        
        PARÁMETROS GEOMÉTRICOS Y DE MATERIALES:
        --------------------------------------
        
        Relativos a la viga continua:
        ----------------------------
        bb:
            type: list of floats
            Bases de la viga en cm, medidas en dirección zz.
            Ingresar valores tramo a tramo, observando la viga de izquierda a
            derecha. La cantidad de valores de esta lista debe ser igual al
            número de tramos de la viga continua.
            Si hay voladizo izquierdo, el primer valor de la lista debe ser
            la base de este.
            Si hay voladizo derecho, el último valor de la lista debe ser la
            base de este.
        hb:
            type: list of floats
            Alturas de la viga en cm, medidas en dirección yy
            Ingresar valores tramoa a tramo, observando a la viga de izquierda
            a derecha. La cantidad de valores de esta lista debe ser igual al
            número de tramos de la viga continua.
            Si hay voladizo izquierdo, el primer valor de la lista debe ser
            la altura de este.
            Si hay voladizo derecho, el último valor de la lista debe ser la
            altura de este.
        spanValues:
            type: list of floats
            Luces en m (no incluye voladizos)
        cantL:
            type: float
            Luz del voladizo izquierdo en m (0 si no existe)
        cantR:
            type: float
            Luz del voladizo derecho en m (0 si no existe)
        eb:
            type: list of floats
            Módulos de elasticidad, en MPa, de los tramos de la viga.
            La cantidad de valores de esta lista debe ser igual al número de 
            tramos de la viga continua.
            
        Relativos a los pilares:
        -----------------------
        Cada parámetro es una lista de floats y el número de elementos de cada
        una de estas listas debe ser igual al número de pilares.
        - tc: Espesores de los pilares [cm], medidos en dirección zz.
        - wc: Anchos de los pilares [cm], medidos en dirección xx.
        - hc: Alturas de los pilares [m], medidas en dirección yy.
        - ec: Módulos de elaticidad de los pilares [MPa].
            
        CARGAS:
        ------
        Se admiten cargas solo sobre la viga, no así en los pilares (cargas
        laterales, por ejm.)

        El parámetro 'cargas' es una lista de la forma:
        [cargas_tramo_1, ..., cargas_tramo_NT], en donde NT es el número total
        de tramos.
        IMPORTANTE: El orden de cargas por tramo debe respetarse. Si no hay
                    cargas en un tramo, debe introducirse una lista [] vacía.

        Cada cargas_tramo_i es una sublista de la forma:
        [datos_carga_1, ..., datos_carga_NQi], en donde NQi es el número de
        cargas del tramo i.
        
        Los datos_carga_i son a su vez tuplas, que dependen del tipo de carga,
        según lo siguiente:
        - Carga puntual vertical: (1, P, a)
        - Momento concentrado: (2, M, a)
        - Carga uniforme en todo el tramo: (3, q)
        - Carga linealmente variable: (4, q0, qf, a, b)
        
        En donde:
        --------
        El primer valor de cada paréntesis es un número entero que indica el
        tipo de carga según se especificó.

        Las demás variables son del tipo float:
        
        P: valor de la carga puntual [kN] positivo hacia abajo;
        a: posición de la carga puntual [m], desde el apoyo izq. inmediato.
        
        M: valor del momento concentrado [kNm], positivo antihorario;
        a: posición del momento concentrado [m], desde el apoyo izquierdo.
        
        q: valor de la carga uniformemente distribuida [kN/m], positivo hacia
            abajo;
        
        q0: valor a la izquierda (valor inicial) de la carga variable [kN/m],
            positivo hacia abajo;
        qf: valor a la derecha (valor final) de la carga variable [kN/m],
            positivo hacia abajo;
        a: pos. izq. (inicio) de la carga variable [m], desde el apoyo izq;
        b: pos. der. de la carga variable [m], desde el apoyo derecho.
            
        IMPORTANTE: Cada una de las posiciones 'a' debe ser medida desde el
                    apoyo izquierdo inmediato del tramo actual, la distancia
                    'b' se mide desde el apoyo derecho.
                    
                    El peso propio es una carga que debe ser agregada como
                    cualquier otra, con su correspondiente coef. de seguridad.
        
        Unidades de introducción de datos:
        ---------------------------------
        - Fuerza: kN
        - Momento flector: kNm
        - Carga distribuida: kN/m
        - Distancia (longitud): m
        - Dimensión de la sección transversal: cm
        - Módulo de elasticidad: MPa
        '''
        
        self.NPT = NPT  # Número de puntos por tramo para los diagramas
        
        #Viga
        #Cuando es necesario se agrega la letra 'b' (b: beam)
        self.bb = bb #Bases de la viga en cm (dirección zz)
        self.hb = hb #Alturas de la viga en cm (dirección yy)
        self.spanValues = spanValues #luces en m (no incluye voladizos)
        self.cantL = cantL #luz del voladizo izquierdo en m (0 sin voladizo)
        self.cantR = cantR #luz del voladizo derecho en m (0 sin voladizo)
        self.eb = eb #Módulo de elasticidad de la viga en MPa
        
        #Pilares
        #Cuando es necesario se agrega la letra 'c' (c: column)
        self.tc = tc #Espesor de los pilares en cm (zz)
        self.wc = wc #Ancho de los pilares en cm (xx)
        self.hc = hc #Altura de los pilares en m (yy)
        self.ec = ec #Módulos de elasticidad de los pilares en MPa
        
        #Cargas
        self.cargas = cargas
        
        #Homogeneización de unidades relativas a la viga
        self.Bb = np.array(bb)*1e-2 #bases de las vigas en m
        self.Hb = np.array(hb)*1e-2 #alturas de las vigas en m
        self.Lb = np.array(spanValues) #longitudes de las vigas en m
        self.Eb = np.array(eb)*1e6 #módulos de elasticidad de la viga en Pa
        
        #Homogeneización de unidades relativas a los pilares
        self.Tc = np.array(tc)*1e-2 #Espesores de los pilares en m
        self.Wc = np.array(wc)*1e-2 #Ancho de los pilares en m
        self.Hc = np.array(hc) #Alturas de los pilares en m
        self.Ec = np.array(ec)*1e6 #Elasticidad de los pilares en Pa
        
        self.Lmin = 0.05 # Longitud mínima de voladizos en m
        self.minV = 1e-4 # Valores inferiores a este se consideran cero.

    
    @property
    def VL(self):
        '''Devuelve una variable booleana, indicando la existencia o no del
        voladizo izquierdo. Se considera su existencia siempre que su longitud
        sea mayor o igual que la longitud mínima adoptada.'''
        if self.cantL >= self.Lmin:
            return True
        else:
            return False
    
    @property
    def CL(self):
        '''Devuelve la LONGITUD del VOLADIZO IZQUIERDO si es mayor a la
        longitud mínima (Lmin), 0 en caso contrario'''
        if self.VL:
            return self.cantL
        else:
            return 0
            
    @property
    def VR(self):
        '''Devuelve una variable booleana, indicando la EXISTENCIA o no del
        VOLADIZO DERECHO. Se considera su existencia siempre que su longitud
        sea mayor o igual que la longitud mínima adoptada.'''
        if self.cantR >= self.Lmin:
            vR = True
        else:
            vR = False
        return vR
    
    @property
    def CR(self):
        '''Devuelve la LONGITUD del VOLADIZO DERECHO si es mayor o igual a la
        longitud mínima Lmin, 0 en caso contrario'''
        if self.VR:
            return self.cantR
        else:
            return 0
    
    @property
    def NS(self):
        '''Número de luces (no incluye voladizos)'''
        return len(self.spanValues)
    
    @property
    def NT(self):
        '''Número de tramos de la viga (incluido voladizos)'''
        return len(self.bb)
    
    @property
    def NC(self):
        '''Número de columnas'''
        return len(self.tc)
    
    @property
    def luces(self):
        '''
        Luces [m] de los diferentes tramos de la viga (incluído voladizos).
        type: numpy.ndarray de NBx1
        '''
        if self.NS > 0: #si hay tramos entre pilares
            luces = np.append(np.append(self.CL, self.Lb), self.CR)
        else: #si no hay tramos entre pilares
            luces = np.append(self.CL, self.CR)
        
        if not self.VL: #Si no hay voladizo izquierdo
            luces = luces[1:]
        if not self.VR: #Si no hay voladizo derecho
            luces = luces[:-1]
        
        return luces
    
    @property
    def long_barras(self):
        '''
        Longitudes [m] de las barras.
        Devuelve un numpy.ndarray de 1xNB de las longitudes de los
        tramos de la viga seguidas de las alturas de los pilares
        '''
        return np.append(self.luces, self.Hc)
    
    @property
    def LT(self):
        '''Longitud total de la viga'''     
        return sum(self.luces)

    @property
    def abcisas_nudos(self):
        '''Devuelve las abcisas [m] de los nudos de la viga continua'''
        xx = np.append(0, self.luces)
        return np.cumsum(xx)

    @property
    def coordenadas(self):
        '''
        Devuelve las coordenadas de los nudos, como una lista de tuplas, en el
        siguiente orden:
        - Coordenadas de los nudos de la viga seguidas de las de las bases de
            los pilares.
        
        Formato devuelto: [(x1, y1), ..., (xn, yn)]
        '''
        Hc = self.Hc
        NC = self.NC
        NT = self.NT # Número de tramos
        abcisas = self.abcisas_nudos # Abcisas de los nudos

        coord = []
        #Viga
        for ii in range(NT + 1):
            coord.append((abcisas[ii], 0))

        # Pilares
        if self.VL: # Si hay voladizo izq.
            for jj in range(NC):
                coord.append((abcisas[jj+1], -Hc[jj]))
        else: # Si no hay voladizo izq.
            for jj in range(NC):
                coord.append((abcisas[jj], -Hc[jj]))
        
        #Devuelve la lista de coordenadas nodales                
        return coord
    
    @property
    def NN(self):
        '''Número de nudos'''
        return len(self.coordenadas)
    
    @property
    def areas(self):
        '''
        Devuelve un numpy.ndarray con las áreas de las secciones transversales
        de los tramos de la viga seguidas de las de los pilares [m2].
        '''
        Ab = self.Bb * self.Hb #Áreas de los tramos de la viga
        Ac = self.Tc * self.Wc #Áreas de pilares
        
        return np.append(Ab, Ac)
    
    @property
    def NB(self):
        '''Número de barras (tramos de viga + pilares)'''
        return len(self.areas)
            
    @property
    def inercias(self):
        '''
        Devuelve un numpy.ndarray con las inercias de las secciones [m4]
        transversales de los tramos de la viga seguidas de las de los pilares.
        '''
        Ib = self.Bb * self.Hb**3/12 #Inercias de los tramos de la viga
        Ic = self.Tc * self.Wc**3/12 #Inercias de pilares
        
        return np.append(Ib, Ic)
    
    @property
    def elasticidades(self):
        '''
        Devuelve un numpy.ndarray con las elasticidades de los tramos de la
        viga seguidas de las elasticidades de los pilares [Pa].
        '''
        return np.append(self.Eb, self.Ec)
    
    @property
    def apoyos(self):
        '''
        Devuelve una lista de tuplas con los siguientes valores por cada pilar
        [(nudo, restX, restY, restGiro, recalX, recalY, recalGiro)]
        En donde:
        --------
            nudo: número de nudo asignado a la base del pilar
            restX = 1, restricción en dirección xx
            restY = 1, restricción en dirección yy
            restGiro = 1, restricción en giro
            Estas restricciones son todas iguales a 1, debido a la suposición
            de empotramiento. Podría considerarse otros tipos de apoyo en
            futuras versiones del programa.
            
            recalX = 0, recalque en dirección xx
            recalY = 0, recalque en dirección yy
            recalGiro = 0, recalque de giro
            Estos recalques son todos cero pues no se consideran posibles
            posibles movimientos de la fundación, pero podría agregarse en
            futuras versiones del programa
        '''
        lista = []
        for ii in range(self.NC):
            lista.append((ii+1, 1, 1, 1, 0, 0, 0))
        
        return lista
    
    @property
    def conect(self):
        '''
        Devuelve la tabla de conectividad de la estructura, en una lista de 
        tuplas con el siguiente formato, en orden de enumeración de las barras
        [(inicio, fin, #area, #inercia, #elasticidad), (...)]
        
        En donde:
        --------
            - inicio: nudo inicial de la barra (viga o pilar)
            - fin: nudo final de la barra (viga o pilar)
            - #area: número de área, ubicación en la lista del método 'areas'
            - #inercia: número de inercia, ubicación en la lista del método
                                                                'inercias'
            - #elasticidad: número de elasticidad, ubicacón en la lista del
                            método 'elasticidades'
        '''
        con = []
        AA = self.areas
        II = self.inercias
        EE = self.elasticidades
        NT = self.NT
        NC = self.NC
            
        #Conectividad de los tramos de viga
        for ii in range(NT):
            con.append((ii+1, ii+2, AA[ii], II[ii], EE[ii]))
            
        #Conectividad de los pilares
        if self.VL: #Existencia del voladizo izquierdo.
            CL = 1
        else:
            CL = 0
            
        for jj in range(NC):
            con.append((NT+jj+2, CL+1+jj, AA[NT+jj], II[NT+jj], EE[NT+jj]))
            
        return con
    
    @property
    def rr(self):
        '''Número de reacciones (o restricciones de desplazamiento)'''
        return sum([par[1] + par[2] + par[3]  for par in self.apoyos])
    
    @property
    def gdl(self):
        '''Número de grados de libertad'''
        return 3*self.NN - self.rr
    
    #############################################################
    ### P R O C E S A M I E N T O   D E   L A S   C A R G A S ###
    #############################################################
    
    @property
    def NQ(self):
        '''Devuelve el número total de cargas que actúan sobre la viga'''
        NQ = sum([len(qq) for qq in self.cargas])
        return NQ

    @property
    def tipos_de_carga(self):
        '''
        Devuelve un diccionario con los tipos de carga, según la
        convención siguiente:
            1: Carga puntual;
            2: Momento concentrado;
            3: Carga uniformemente distribuida en todo el tramo;
            4: Carga linealmente variable.
        '''
        return {'Carga puntual': 1, 'Momento concentrado': 2,
                'Carga uniforme': 3, 'Carga lineal': 4}
    
    #Reacciones nodales equivalentes
    def RNE_qe(self, datos_de_carga, indice_tramo):
        '''
        Reacciones Nodales Equivalentes, por carga y por elemento.
        Devuelve las reacciones nodales equivalentes como vector columna
        numpy.ndarray de 6x1
        
        Devuelve
            - Fuerzas en N
            - Momentos en Nm
        
        Parámetros:
        ----------
        datos_de_carga: tupla que contiene los datos de la carga, según las
                        reglas de introducción de datos de carga;
        indice_tramo: indica en qué tramo está la carga, inicia en 0.
            
        Fórmulas de Aslam Kassimali. Matrix Analysis of Structures. 2nd. Ed.
        '''

        assert self.NQ > 0, "No existen cargas en esta viga"
        
        tipo_carga = datos_de_carga[0] #Obtención del tipo de carga
        assert tipo_carga in self.tipos_de_carga.values(),\
            "Tipo de carga incorrecto"

        L = self.luces[indice_tramo] #Luz del tramo
        
        #Carga puntual
        if tipo_carga == 1: #datos_de_carga = (1, P, xp)
            assert len(datos_de_carga) == 3, "Verificar los datos de carga"
            P = datos_de_carga[1]*1e3 # [N]
            a = datos_de_carga[2] # [m]
            b = L - a # [m]
            return P/L**2 * np.array([
                    [0],
                    [b**2 / L * (3*a + b)],
                    [a * b**2],
                    [0],
                    [a**2 / L * (a + 3*b)],
                    [-a**2 * b]
                ])
        
        #Momento concentrado
        elif tipo_carga == 2: #datos_de_carga = (2, M, xm)
            assert len(datos_de_carga) == 3, "Verificar los datos de carga"
            M = datos_de_carga[1]*1e3 # [Nm]
            a = datos_de_carga[2] # [m]
            b = L - a # [m]
            return M / L**2 * np.array([
                    [0],
                    [-6*a*b/L],
                    [b*(b - 2*a)],
                    [0],
                    [6*a*b/L],
                    [a*(a - 2*b)]
                ])
        
        #Carga uniformemente distribuida en todo el tramo
        elif tipo_carga == 3: #datos_de_carga = (3, q)
            assert len(datos_de_carga) == 2, "Verificar los datos de carga"
            q = datos_de_carga[1]*1e3 # [N]
            return q * L / 2 * np.array([
                    [0],
                    [1],
                    [L/6],
                    [0],
                    [1],
                    [-L/6]
                ])
        
        #Carga linealmente variable
        elif tipo_carga == 4: #datos_de_carga = (4, q0, qf, a, b)
            assert len(datos_de_carga) == 5, "Verificar los datos de carga"
            w1 = datos_de_carga[1]*1e3 # [N/m]
            w2 = datos_de_carga[2]*1e3 # [N/m]
            a = datos_de_carga[3] # [m]
            b = datos_de_carga[4] # [m]
            
            #Reacción vertical en el nudo izquierdo
            FSb = w1*(L-a)**3/(20*L**3)*((7*L+8*a) - b*(3*L+2*a)/(L-a)*(1 +\
                b/(L-a) + b**2/(L-a)**2) + 2*b**4/(L-a)**3) +\
                w2*(L-a)**3/(20*L**3)*((3*L+2*a)*(1 + b/(L-a) +\
                b**2/(L-a)**2) - b**3/(L-a)**2*(2 + (15*L-8*b)/(L-a)))
                    
            #Reacción momento en el nudo izquierdo
            FMb = w1*(L-a)**3/(60*L**2)*(3*(L+4*a) - b*(2*L+3*a)/(L-a)*(1 +\
                b/(L-a) + b**2/(L-a)**2) + 3*b**4/(L-a)**3) +\
                w2*(L-a)**3/(60*L**2)*((2*L+3*a)*(1 + b/(L-a) +\
                b**2/(L-a)**2) - 3*a**3/(L-a)**2*(1 + (5*L-4*b)/(L-a)))
                    
            #Reacción vertical en el nudo derecho
            FSe = (w1+w2)/2*(L-a-b) - FSb
            
            #Reacción momento en el nudo derecho
            FMe = (L-a-b)/6*(w1*(-2*L+2*a-b) - w2*(L-a+2*b)) + FSb*L - FMb
                    
            return np.array([
                [0],
                [FSb],
                [FMb],
                [0],
                [FSe],
                [FMe]
                ])
        else: # Para cualquier otra situación
            return np.zeros((6, 1)) # Devuelve vector de ceros
    
    #Fuerza cortante en una sección (viga sin apoyos)
    def V_sqe(self, xt, datos_de_carga, indice_tramo):
        '''
        Cortante en una sección, para una carga dada y en un tramo dado [N]
        
        Parámetros:
        ----------
        xt: posición [m] de la sección considerada respecto al extremo
            izquierdo del tramo.
        datos_de_carga: tupla que contiene los datos de la carga, en donde el
                        primer número es el tipo de carga;
        indice_tramo: indica en qué tramo está la carga, inicia en 0.            
        '''
        assert self.NQ > 0, "No existen cargas en esta viga"

        tipo_carga = datos_de_carga[0]  # Obtención del tipo de carga
        assert tipo_carga in self.tipos_de_carga.values(),\
            "Tipo de carga incorrecto"

        L = self.luces[indice_tramo]  # Luz del tramo

        #Carga puntual
        if tipo_carga == 1:  # datos_de_carga = (1, P, a)
            P = datos_de_carga[1]*1e3  # Valor de la carga puntual [N]
            a = datos_de_carga[2]  # Posición de la carga puntual [m]
            if 0 < xt < a:  # A la izquierda de P
                return 0
            elif a < xt <= L:  # A la derecha de P
                return -P
            else:  # En el apoyo izq. y fuera del tramo
                return 0

        # Momento concentrado
        elif tipo_carga == 2:  # datos_de_carga = (2, M, a)
            return 0  # En todo lugar

        # Carga uniformemente distribuida en todo el tramo
        elif tipo_carga == 3:  # datos_de_carga = (3, q)
            q = datos_de_carga[1]*1e3  # Valor carga unif. dist. [N/m]
            if 0 < xt <= L:  # En el interior del tramo
                return -q * xt
            else:  # En el apoyo izq. y fuera del tramo
                return 0

        # Carga lineal
        elif tipo_carga == 4:  # datos_de_carga = (4, q0, qf, a, b)
            q0 = datos_de_carga[1]*1e3  # Valor a la izq. de la carga [N/m]
            qf = datos_de_carga[2]*1e3  # Valor a la der. de la carga [N/m]
            a = datos_de_carga[3]  # Posición a la izquierda [m]
            b = datos_de_carga[4]  # Posición a la derecha [m]
            ll = L - a - b # Longitud de distribución de la carga [m]
            x = xt # Cambio de variabla para acortar
            if 0 < xt <= a:  # A la izquierda de la carga
                return 0
            elif a < x < L - b:  # Entre los límites de la carga
                R1 = (qf - q0)/(2*ll)*(x - a)**2 # Resultante triángulo
                R2 = q0*(x - a) # Resultante rectángulo
                return -R1 - R2
            elif L - b <= x <= L:  # A la derecha de la carga
                R1 = (qf - q0)*ll/2 # Triángulo total
                R2 = q0*ll # Rectángulo total
                return -R1 - R2
            else:  # En el apoyo izq. y fuera del tramo
                return 0
        # Para cualquier otra situación
        else:
            return 0

    #Momento flector en una sección (viga simplemente apoyada)
    def M_sqe(self, xt, datos_de_carga, indice_tramo):
        '''
        Aporte de la carga al Momento flector [Nm] en una sección.

        Las ecuaciones corresponden a una viga simplemente apoyada, la cual
        es la estructura isostática fundamental.
        
        Parámetros:
        ----------
        xt: posición [m] de la sección considerada respecto al extremo
            izquierdo del tramo.
        datos_de_carga: tupla que contiene los datos de la carga, en donde el
                        primer número es el tipo de carga;
        indice_tramo: indica en qué tramo está la carga, inicia en 0. 
        '''
        assert self.NQ > 0, "No existen cargas en esta viga"

        tipo_carga = datos_de_carga[0] # Obtención del tipo de carga
        assert tipo_carga in self.tipos_de_carga.values(),\
            "Tipo de carga incorrecto"

        L = self.luces[indice_tramo] # Luz del tramo
        
        #Carga puntual
        if tipo_carga == 1: #datos_de_carga = (1, P, a)
            P = datos_de_carga[1]*1e3 #Valor de la carga puntual [N]
            a = datos_de_carga[2] #Pos. de la carga puntual en el tramo [m]
            b = L - a #Distancia entre P y el fin del tramo [m]
            if 0 < xt <= a: #A la izquierda de P
                return b * P * xt/L
            elif a < xt < L: #A la derecha de P
                return a * P * (1 - xt/L)
            else: #En los extremos y fuera del tramo
                return 0
            
        #Momento concentrado
        elif tipo_carga == 2: #datos_de_carga = (2, M, a)
            M = datos_de_carga[1]*1e3 #Valor del momento concentrado [Nm]
            a = datos_de_carga[2] #Posición de M en el tramo [m]
            if 0 < xt < a: #A la izquierda del momento
                return M/L*xt
            elif a < xt < L: #A la derecha del momento
                return -M/L*(L-xt)
            else: #En los extremos y fuera del tramo
                return 0
            
        #Carga uniformemente distribuida en todo el tramo
        elif tipo_carga == 3: #datos_de_carga = (3, q)
            q = datos_de_carga[1]*1e3 #Valor de la carga uniforme [N/m]
            if 0 < xt < L: #En el interior del tramo
                return q*L/2*xt - q*xt**2/2
            else: # En los extremos y fuera del tramo
                return 0
            
        #Carga linealmente variable
        elif tipo_carga == 4: #datos_de_carga = (4, q0, qf, a, b)
            q0 = datos_de_carga[1]*1e3 #Valor a la izq. de la carga [N/m]
            qf = datos_de_carga[2]*1e3 #Valor a la derecha de la carga [N/m]
            a = datos_de_carga[3] #Posición a la izquierda [m]
            b = datos_de_carga[4] #Posición a la derecha [m]
            ll = L - a - b # Longitud de distribución de la carga [m]
            x = xt # Cambio de variable para acortar [m]

            R1 = (qf - q0)*ll/2 # Resultante de la parte triangular
            R2 = q0*ll # Resultante de la parte rectangular
            Ay = (ll/3 + b)/L*R1 + (ll/2 + b)/L*R2 #Reacción izquierda

            if 0 < x <= a: #A la izquierda de la carga
                M1 = Ay*x # Momento flector
                return M1
            elif a < x <= L - b: #Entre los límites de la carga
                r1 = (qf - q0)/(2*ll)*(x - a)**2 # Parte triangular
                r2 = q0*(x - a) # Parte rectangular
                M2 = Ay*x - r1*(x - a)/3 - r2*(x - a)/2 # Momento flector
                return M2
            elif L - b < x < L: #A la derecha de la carga
                #M3 = By*(L - x) # Momento flector
                M3 = -(L - x)*(-L + a + b)*(3*L*(q0 + qf) -\
                    3*q0*(L - a + b) + (q0 - qf)*(L - a + 2*b))/(6*L)
                return M3
            else: # En los extremos y fuera del tramo
                return 0
        
        #En cualquier otra situación
        else:
            return 0
        
    def D_sqe(self, xt, datos_de_carga, indice_tramo):
        '''
        Devuelve el desplazamiento de un punto (xt) del eje de la viga,
        correspondiente al aporte de la carga considerada.

        Las ecuaciones corresponden a una viga biempotrada.
        
        Parámetros:
        ----------
        xt: posición [m] de la sección considerada, medida desde el apoyo
            izquierdo del tramo actual.
        datos_de_carga: tupla que contiene los datos de la carga, en donde el
                        primer número es el tipo de carga;
        indice_tramo: indica en qué tramo está la carga, inicia en 0.
        '''
        assert self.NQ > 0, "No existen cargas en esta viga"

        tipo_carga = datos_de_carga[0]  # Obtención del tipo de carga
        assert tipo_carga in self.tipos_de_carga.values(),\
            "Tipo de carga incorrecto"

        L = self.luces[indice_tramo] #Luz del tramo [m]
        EI = self.Eb[indice_tramo] * self.inercias[indice_tramo] # Rigidez
        x = xt # Cambio de variable
        
        #Carga puntual
        if tipo_carga == 1: # datos_de_carga = (1, P, a)
            P = datos_de_carga[1]*1e3 # Valor de la carga puntual [N]
            a = datos_de_carga[2] # Posición de la carga [m]
            if 0 < x <= a: # A la izquierda de la carga
                d1 = x**2*(-L + a)**2*(-3*L*a + L*x + 2*a*x)/(6*L**3)
                return d1*P/EI
            elif a < x < L: # A la derecha de la carga
                d2 = a**2*(-L + x)**2*(L*a - 3*L*x + 2*a*x)/(6*L**3)
                return d2*P/EI
            else: # En los extremos y fuera del tramo
                return 0
                    
        #Momento concentrado
        elif tipo_carga == 2: #datos_de_carga = (2, M, a)
            M = datos_de_carga[1]*1e3 #Valor del momento [Nm]
            a = datos_de_carga[2] #Posición de M [m]
            if 0 < x <= a: # A la izq. del momento
                d1 = -x**2*(-L + a)*(L**2 - 3*L*a + 2*a*x)/(2*L**3)
                return d1*M/EI
            elif a < x <= L: # A la derecha del momento
                d2 = -a*(-L + x)**2*(L*a - 2*L*x + 2*a*x)/(2*L**3)
                return d2*M/EI
            else: # En los extremos y fuera del tramo
                return 0
            
        #Carga uniformemente distribuida
        elif tipo_carga == 3: #datos_de_carga = (3, q)
            q = datos_de_carga[1]*1e3 # Valor de la carga [N/m]
            if 0 < xt < L: # En el interior del tramo
                d = -x**2*(-L + x)**2/24
                return d*q/EI
            else: # En los extremos y fuera del tramo
                return 0
            
        #Carga linealmente variable
        elif tipo_carga == 4: #datos_de_carga = (4, q0, qf, a, b)
            q0 = datos_de_carga[1]*1e3 # Valor de carga a la izq. [N/m]
            qf = datos_de_carga[2]*1e3 # Valor de carga a la der. [N/m]
            a = datos_de_carga[3] # Pos. desde nudo izq. [m]
            b = datos_de_carga[4] # Pos. desde nudo der. [m]
            ll = L - a - b # Longitud de distribución de la carga [m]
            x = xt # Cambio de variable

            # PARTE TRIANGULAR 
            # A la izquierda de la carga
            vt1 = -x**2*(-L + a + b)*(-2*L**4 + L**3*a - 4*L**3*b +\
                3*L**3*x + 4*L**2*a**2 - 2*L**2*a*b - 4*L**2*a*x -\
                6*L**2*b**2 + 6*L**2*b*x - 3*L*a**3 + 6*L*a**2*b -\
                L*a**2*x - 9*L*a*b**2 - 2*L*a*b*x + 12*L*b**3 + 9*L*b**2*x +\
                2*a**3*x - 4*a**2*b*x + 6*a*b**2*x - 8*b**3*x)/(120*L**3)
            
            # Entre los límites de la carga
            vt2 = (2*L**6*x**2 - 5*L**5*a*x**2 - 3*L**5*x**3 + 10*L**4*a*x**3 -\
                L**3*a**5 + 5*L**3*a**4*x - 5*L**3*a*x**4 - 20*L**3*b**3*x**2 +\
                L**3*x**5 - 10*L**2*a**4*x**2 + 20*L**2*a*b**3*x**2 +\
                30*L**2*b**4*x**2 + 20*L**2*b**3*x**3 + 3*L*a**5*x**2 +\
                5*L*a**4*x**3 - 15*L*a*b**4*x**2 - 20*L*a*b**3*x**3 -\
                12*L*b**5*x**2 - 25*L*b**4*x**3 - 2*a**5*x**3 + 10*a*b**4*x**3 +\
                8*b**5*x**3)/(120*L**3*(-L + a + b))
            
            # A la derecha de la carga
            vt3 = -(-L + x)**2*(-L + a + b)*(4*L**4 + 3*L**3*a - 12*L**3*b -\
                7*L**3*x + 2*L**2*a**2 - 6*L**2*a*b - 4*L**2*a*x + 12*L**2*b**2 +\
                6*L**2*b*x + L*a**3 - 2*L*a**2*b - L*a**2*x + 3*L*a*b**2 -\
                2*L*a*b*x - 4*L*b**3 + 9*L*b**2*x + 2*a**3*x - 4*a**2*b*x +\
                6*a*b**2*x - 8*b**3*x)/(120*L**3)

            # PARTE RECTANGULAR
            # A la izquierda de la carga
            vr1 = x**2*(-L**4 + 6*L**2*a**2 - 8*L*a**3 + 4*L*b**3 +\
                3*a**4 - 3*b**4)/(24*L**2) + x**3*(L**4 - 2*L**3*a +\
                2*L*a**3 - 2*L*b**3 - a**4 + b**4)/(12*L**3)
            
            # Entre los límites de la carga
            vr2 = -a**4/24 + a**3*x/6 - x**4/24 + x**2*(-L**4 -\
                8*L*a**3 + 4*L*b**3 + 3*a**4 - 3*b**4)/(24*L**2) +\
                x**3*(L**4 + 2*L*a**3 - 2*L*b**3 - a**4 + b**4)/(12*L**3)

            # A la derecha de la carga
            vr3 = L**4/24 - L**3*b/6 + L**2*b**2/4 - L*b**3/6 - a**4/24 +\
                b**4/24 + x*(-L**3/6 + L**2*b/2 - L*b**2/2 + a**3/6 +\
                b**3/6) + x**2*(5*L**4 - 12*L**3*b + 6*L**2*b**2 -\
                8*L*a**3 + 4*L*b**3 + 3*a**4 - 3*b**4)/(24*L**2) + x**3*(
                -L**4 + 2*L**3*b + 2*L*a**3 - 2*L*b**3 - a**4 + b**4)/(12*L**3)

            if 0 < xt <= a: # A la izquierda de la carga
                d1 = (vr1*q0 + vt1*(qf - q0))/EI
                return d1
            elif a < xt <= a + ll: # Entre los límites de la carga
                d2 = (vr2*q0 + vt2*(qf - q0))/EI
                return d2
            elif a + ll < xt < L: # A la derecha de la carga
                d3 = (vr3*q0 + vt3*(qf - q0))/EI
                return d3
            else: # En los extremos y fuera del tramo
                return 0
        
        #En cualquier otra situación
        else:
            return 0
            
    #############################################################
    ### P R O C E S A M I E N T O   D E   L A S   B A R R A S ###
    #############################################################    
    
    @property
    def coord_gl_nudos(self):
        '''
        Enumeración de las coordenadas globales de cada nudo
        [horizontal, vertical, giro], 3 grados de libertad por nudo.
        Devuelve una lista de sublistas, con 3 elementos en cada sublista.
        '''
        NN = self.NN #Número de nudos
        enum = [[1 + 3*ii, 2 + 3*ii, 3 + 3*ii] for ii in range(NN)]
        return enum
    
    @property
    def coord_gl_barras(self):
        '''
        Enumeración de las coordenadas globales de cada barra según la tabla
        de conectividad.
        
        Devuelve una lista de sublistas en donde cada sublista contiene la
        enumeración asignada a las coordenadas globales de los extremos de
        cada barra.
        
        Los 3 primeros valores de cada sublista corresponden a la enumeración
        del nudo inicial de la barra y los 3 últimos a la enumeración del nudo
        final, en el orden: horizontal (h), vertical (v), giro (g)
        [h_i, v_i, g_i, h_f, v_f, g_f]

        Inicia en nudos de la viga y termina en nudos base de pilares.
        '''
        conect = self.conect #Tabla de conectividad
        NB = self.NB #Número de barras
        cgn= self.coord_gl_nudos #Coordenadas globales de los nudos
        
        enum2 = [] #Contenedor
        for ii in range(NB): #Para todas las barras
            lista = [0]*6 #Inicialización
            lista[:3] = cgn[conect[ii][0] - 1] #Números del nudo inicial
            lista[3:] = cgn[conect[ii][1] - 1] #Se agrega núm. del nudo final
            enum2.append(lista)
        
        return enum2
    
    @property
    def vectores_direccion(self):
        '''
        Componentes vectoriales de las barra (consideradas como vector).
        Devuelve un numpy.ndarray de NBx2.
        En orden de enumeración de las barras según la tabla self.conect
        '''
        coord = self.coordenadas #Coordenadas de los nudos
        vectores = []
        
        for barra in self.conect: #Recorre todas las barras
            Ni = barra[0] #Nudo inicial
            Nf = barra[1] #Nudo final
            
            Xi = coord[Ni - 1][0] #X inicial
            Xf = coord[Nf - 1][0] #X final
            Yi = coord[Ni - 1][1] #Y inicial
            Yf = coord[Nf - 1][1] #Y final
        
            X = Xf - Xi #componente X
            Y = Yf - Yi #componente Y
            
            vector = [X, Y] #Vector de dirección de la barra
            vectores.append(vector) #Se agrega a la lista de vectores
            
        return np.array(vectores)
    
    @property
    def vectores_unitarios(self):
        '''
        Vectores unitarios en dirección de cada barra.
        Devuelve un numpy.ndarray de NBx2
        '''
        VD = self.vectores_direccion
        LB = self.long_barras
        NB = self.NB # Número de barras

        return VD / LB.reshape(NB, 1)
    
    #Matriz de transformación de coordenadas.
    def TT(self, indice_barra):
        '''
        Matriz de transformación de coordenadas, de globales a locales.
        Devuelve un numpy.ndarray de 6x6
        
        Parámetros:
        ----------
        indice_barra: Índice para identificación de la barra.
                        Inicia en cero.
        '''
        ib = indice_barra # Índice de la barra
        v_unit = self.vectores_unitarios  # cosenos directores
        
        # Por cada barra
        c = v_unit[ib][0]  # coseno
        s = v_unit[ib][1]  # seno

        # Matriz de transformación de coordenadas
        T = np.array([
            [c, s, 0, 0, 0, 0],
            [-s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, s, 0],
            [0, 0, 0, -s, c, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        return T
    
    #Matriz de rigidez de una barra en coordenadas locales
    def ke(self, indice_barra):
        '''
        Matriz de rigidez de una barra en coordenadas locales.
        Devuelve un numpy.ndarray de 6x6
        
        Parámetro:
        ---------
        indice_barra: Identificación de la barra, inicia en 0
        '''
        ib = indice_barra  # Índice de barra
        conect = self.conect  # Tabla de conectividad
        longitudes = self.long_barras  # Longitudes de las barras

        # Para cada barra
        A = conect[ib][2]  # Área [m2]
        I = conect[ib][3]  # Inercia [m4]
        E = conect[ib][4]  # Elasticidad [Pa]
        L = longitudes[ib]  # Longitud [m]

        # Matriz de rigidez en coordenadas locales
        ke = E*I/L**3 * np.array([
            [A*L**2/I, 0, 0, -A*L**2/I, 0, 0],
            [0, 12, 6*L, 0, -12, 6*L],
            [0, 6*L, 4*L**2, 0, -6*L, 2*L**2],
            [-A*L**2/I, 0, 0, A*L**2/I, 0, 0],
            [0, -12, -6*L, 0, 12, -6*L],
            [0, 6*L, 2*L**2, 0, -6*L, 4*L**2]
        ])
        return ke
    
    #Matriz de rigidez de una barra en coordenadas globales
    def Ke(self, indice_barra):
        '''
        Matriz de rigidez de una barra en coordenadas globales.
        Parámetros:
        ----------
        indice_barra: Índice de para identificación de la barra,
                inicia en 0.
        '''
        ib = indice_barra # Índice de barra

        ke = self.ke(ib)  # Matriz de rigidez en coord. locales
        T = self.TT(ib)  # Matriz de transformación de coordenadas

        #Matriz de rigidez en coordenadas globales
        K = T.T @ ke @ T
        
        return K
    
    @property
    def KG(self):
        '''
        Matriz de rigidez global de la estructura (viga y pilares).
        Devuelve un numpy.ndarray de 3NN x 3NN
        '''
        
        NN = self.NN #Número de nudos
        NB = self.NB #Número de barras
        cgb = self.coord_gl_barras #Coordenadas globales de las barras
        
        #Ensamble de la matriz de rigidez global
        KG = np.zeros((3*NN, 3*NN)) #Inicialización
        for bb in range(NB): #Recorre todas las barras
            
            #Matriz de rigidez en coordenadas globales
            K = self.Ke(bb)
            
            #Ensamble de la matriz de rigidez
            serie = cgb[bb] #Se extraen las coord. globales de los extremos
            for r in range(6): #Recorre las 6 filas de la matriz K de la barra
                for s in range(6): #Recorre las 6 columnas de K
                    
                    #Toma el elemento de la posición (r,s) de la matriz K
                    #y lo coloca en KG, según las coordenadas globales
                    KG[serie[r] - 1, serie[s] - 1] += K[r, s]
        return KG
    
    @property
    def RNE(self):
        '''
        Reacciones nodales equivalentes de los tramos de la viga.
        Devuelve un numpy.ndarray de 6xNT donde, en cada columna, se
        encuentran las reacciones nodales equivalentes correspondientes a cada
        tramo.
        '''
        NT = self.NT #Número barras (tramos de viga + pilares)
        M_rne = np.zeros((6, NT)) #Inicialización matriz de reac. nod. equiv.
        
        for n in range(NT): #Para cada tramo de la viga
            
            for qq in self.cargas[n]: #P/ c/ carga del tramo
                rne = self.RNE_qe(qq, n) #Reacción nodal equivalente
                M_rne[:, n] = M_rne[:, n] + rne.T #Suma la reacción nod. eq.
        
        return M_rne
    
    @property
    def FG(self):
        '''
        Ensamble del vector de fuerzas nodales.
        Devuelve el vector de fuerzas nodales en coordenadas globales.
        '''
        NN = self.NN  # Número de nudos
        NT = self.NT # Número de tramos de la viga
        RNE = self.RNE # Reacciones nodales equivalentes
        
        PG = np.zeros(3*NN)  # Inicialización con ceros
        for ii in range(NT): # Para todos los tramos
            PG[3*ii:3*ii+6] = PG[3*ii:3*ii+6] - RNE[:, ii]
        
        return PG

    ###################################################
    ### R E S O L U C I Ó N   D E L   S I S T E M A ###
    ###################################################

    @property
    def DyR(self):
        '''
        Desplazamientods nodales de los nudos de la viga y reacciones en
        bases de los pilares.
        Devuelve una tupla con dos numpy.ndarray, el primero corresponde
        a los desplazamientos y el segundo a las reacciones.
        '''
        KG = self.KG # Matriz de rigidez de la estructura global
        FG = self.FG # Vector de cargas nodales
        gdl = self.gdl # Grados de libertad

        # Submatrices de rigidez
        K00 = KG[:gdl, :gdl]  # Considera solo los grados de libertad
        K10 = KG[gdl:, :gdl]  # 2a matriz de acoplam. e/ gdl y gdr

        #Subvector de fuerza
        F00 = FG[:gdl] # Cargas en los grados de liberdad
        
        # Desplazamientos en los nudos de la viga
        D0 = np.linalg.inv(K00) @ F00

        # Reacciones en las bases de los pilares
        R = K10 @ D0

        return D0, R

    ###########################
    ### R E S U L T A D O S ###
    ###########################

    # Desplazamientos por tramo de viga
    @property
    def despl_tramos(self):
        '''
        Devuelve los desplazamientos [m] de los nudos izq. y der. de los
        tramos de la viga, en un numpy.ndarray de 6xNT.
        En cada columna se encuentran los desplazamientos de las barras.
        '''
        NT = self.NT # Número de tramos
        D = self.DyR[0] # Desplazamientos

        # Matriz de desplazamientos de tramos
        matriz = np.zeros((6, NT)) # Una columna por tramo
        for ib in range(NT): # Recorre los tramos
            matriz[:, ib] = D[3*ib:3*ib + 6]
        
        return matriz

    # Fuerzas nodales por tramo de viga
    @property
    def fuerzas_tramos(self):
        '''
        Devuelve las fuerzas nodales [kN] de los tramos de la viga,
        en un numpy.ndarray de 6xNT.
        En cada columna se encuentran las fuerzas de cada barra.
        '''
        NT = self.NT # Número de tramos

        matriz = np.zeros((6, NT)) # Inicialización
        for ib in range(NT): # Recorre los tramos
            matriz[:, ib] = self.Ke(ib) @ self.despl_tramos[:, ib]\
                + self.RNE[:, ib]
        
        return matriz

    def f_despl(self, Xt, indice_tramo):
        '''
        Función de desplazamiento.
        Devuelve los desplazamientos verticales considerando las
        deformaciones nodales unicamente.
        Parámetros:
        ----------
        Xt: np.ndarray con el listado de coordenadas de puntos del tramo
        indice_tramo: Identificación del tramo, inicia en cero.
        '''
        L = self.luces[indice_tramo] # Lontitud del tramo
        u = self.despl_tramos[:, indice_tramo] # Despl. nodales del tramo
        X = Xt # Cambio de variable para acortar

        u1 = u[1] # Desplazamiento vertical nudo izquierdo
        u2 = u[2] # Giro nudo izquierdo
        u3 = u[4] # Desplazamiento vertical nudo derecho
        u4 = u[5] # Giro nudo derecho

        # Funciones de forma
        N1 = 1 - 3*(X/L)**2 + 2*(X/L)**3
        N2 = X*(1 - X/L)**2
        N3 = 3*(X/L)**2 - 2*(X/L)**3
        N4 = X**2/L*(-1 + X/L)

        # Desplazamientos verticales del tramo
        v = N1*u1 + N2*u2 + N3*u3 + N4*u4
        return v

    # Gráficos
    def diagramas(self):
        '''
        Devuelve:
        1. Abcisas de los diagramas [m]
        2. Fuerzas cortantes [kN]
        2. Momentos flectores [kNm]
        3. Desplazamientos verticales [mm]
        '''
        n = self.NPT # Número de puntos por tramo
        NT = self.NT # Número de tramos de la viga
        luces = self.luces # Luces de los tramos de la viga
        FNET = self.fuerzas_tramos # fuerzas nodales equiv. por tramo
        abcisas = self.abcisas_nudos # Abcisas de los nudos

        X = np.zeros(n*NT) # Contenedor de las abcisas del diagrama
        FQ = np.zeros(n*NT) # Contenedor de fuerza cortante
        MF = np.zeros(n*NT) # Contenedor de los momentos flectores
        D = np.zeros(n*NT) # Contenedor de desplazamientos
        for it in range(NT): # Recorre los tramos
            L = luces[it] # Longitud del tramo
            Xt = np.linspace(0, L, n) # Posiciones de sec.
            V_iso = np.zeros(n) # Contenedor de cortantes del tramo
            M_iso = np.zeros(n) # Contenedor de flectores del tramo
            D_iso = np.zeros(n) # Contenedor de desplazam. del tramo
            cargas = self.cargas[it] # Lista de cargas del tramo
            m = 0 # Contador de secciones
            for x in Xt: # Posición
                # Isostática fundamental
                for qq in cargas: # Cargas
                    # Cortante
                    V_iso[m] += self.V_sqe(x, qq, it)

                    # Flexión
                    M_iso[m] += self.M_sqe(x, qq, it)

                    # Deformada
                    D_iso[m] += self.D_sqe(x, qq, it)

                m += 1  # Incremento para la siguiente sección
            
            # Cortante de empotramiento
            V_emp = FNET[1, it] # A la izq. del tramo

            # Momentos de empotramiento
            M1 = FNET[2, it] # A la izq. del tramo
            M2 = FNET[5, it] # A la der. del tramo
            M_emp = (M1 + M2)/L*Xt - M1 # M de empotramiento

            # Deformada de empotramiento
            D_emp = self.f_despl(Xt, it)

            # Valores finales
            X[n*it:n*(1 + it)] = Xt + abcisas[it] # Abcisas
            FQ[n*it:n*(1 + it)] = V_iso + V_emp # Cortantes
            MF[n*it:n*(1 + it)] = M_iso + M_emp # Flectores
            D[n*it:n*(1 + it)] = D_iso + D_emp # Desplazamientos
        
        return X, FQ/1000, MF/1000, D*1000

    def listado_csv(self):
        '''
        Listado en formato csv, con Abcisas X [m];
        Fuerzas cortantes V [kN], Momentos flectores M [kNm] y
        Desplazamientos verticales D [mm].
        '''
        X, FQ, MF, D = self.diagramas() # Info
        minV = self.minV # Mínimo valor
        n = len(X) # Número de filas de datos
        contador = 0
        with open('listado.csv', 'w', newline='') as csvfile:
            cols = ['Num', 'X [m]', 'Cortante [kN]', 'Flector [kNm]',
            'Despl. vertical [mm]']
            escritor = csv.DictWriter(csvfile, fieldnames = cols)
            escritor.writeheader()

            for ii in range(n):
                contador += 1
                # Se eliminan valores pequeños
                if abs(FQ[ii]) < minV:
                    FQ[ii] = 0
                if abs(MF[ii]) < minV:
                    MF[ii] = 0
                if abs(D[ii]) < minV:
                    D[ii] = 0

                escritor.writerow({'Num':contador, 'X [m]':X[ii],
                    'Cortante [kN]':FQ[ii], 'Flector [kNm]':MF[ii],
                                   'Despl. vertical [mm]':D[ii]})
