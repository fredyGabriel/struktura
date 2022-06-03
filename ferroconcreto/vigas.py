# -*- coding: utf-8 -*-
"""
Propiedades de una sección transversal rectangular de hormigón
armado

-----------------------------------
Created on Sat Jul 23 09:04:36 2016
@author: Fredy Gabriel Ramírez Villanueva
ESTO FUNCIONA?: No sé todavía
-----------------------------------

Análisis de vigas de hormigón armado, de un solo tramo, de sección
rectangular constante en toda su longitud.
El armado longitudinal se supone constante de extremo a extremo de
la viga.

UNIDADES:
    Todos el SI sin múltiplos ni submúltiplos.

Ejemplos:
--------

"""

#Módulos de python
import math
import numpy as np
import Utiles as ut
import Materiales as mat

#Valores por defecto de los materiales de la viga
Ho = mat.Hnl() #Hormigón para análisis estructural
Ao = mat.Acero() #Acero
esp = ut.Espacio() #Parámetros generales

# - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - #
#                        V I G A S                               #
# - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - #
class VigaRecta:
    '''Viga de hormigón armado con sección rectangular constante
    en toda su longitud. El armado longitudinal se extiende en
    toda su longitud.'''
    
    def __init__(self, b=0.2, h=0.5, espacio=esp, hormigon=Ho,
                 acero=Ao, rec=0.025, viga_tipo=1, estado_limite=\
                 'ELU', Ast=2/10000, Asc=1/10000):
        ''' >>> V1 = HoAo.Vigas.VigaRecta(b, h, espacio, hormigon,
        acero, rec, viga_tipo, estado_limite, Ast, Asc)

        Parámetros:
        ----------
        b, h: base y altura de la viga
        espacio: objeto definido en Utiles.espacio, en el cual se
        consigna la longitud y el número de elementos finitos.
        hormigon: material definido en materiales.Hormigon()
        acero: material definido en Materiales.Acero()
        rec: recubrimiento en m
        viga_tipo: 1: simplemente apoyada, 2: empotrada,
                    3: empotrada-apoyada, 4: biempotrada
        estado_limite: 'ELU' (último), 'ELS' (servicio),
        'VMS' (valores medios)
        Ast: armadura de tracción
        Asc: armadura de compresión'''
        self.espacio = espacio #longitud y NE
        self.b = b #base
        self.h = h #altura
        self.hormigon = hormigon #def. en Materiales.Hormigon()
        self.acero = acero #definido en Materiales.Acero()
        self.rec = rec #recubrimiento
        self.viga_tipo = viga_tipo #según condiciones de apoyo
        self.estado_limite = estado_limite
        self.Ast = Ast #armadura de tracción [m2]
        self.Asc = Asc #armadura de compresión [m2]
    
        #Costos monetarios
        self.precio_madera = 0.70 #[USD/pulg]
        
    def __str__(self):
        datos_viga = "Viga recta de hormigón armado con sección\
        transversal\n" +  "y armado constantes.\n" +\
        "Estado limite = " + str(self.estado_limite) + "\nb = "+\
        str(self.b*100) + "cm\t\th = " + str(self.h*100) +\
        "cm\t\tL = " + str(self.L) + "m\n" + "fck = " +\
        str(self.hormigon.fck*1e-6) + "MPa\t\t" + "fyk = " + \
        str(self.acero.fyk*1e-6) + "MPa\t\t" + "pp = " +\
        str(self.rec*100)
        return datos_viga
    
    @property
    def L(self):
        '''Longitud de la viga [m]''' 
        return self.espacio.L
    
    #Rango válido de valores de b
    @property
    def b(self):
        '''Dimensión de base de sección transversal de la viga'''
        return self.__b
    @b.setter
    def b(self, b):
        assert 0.05 <= b <= 0.50, "debe estar entre 0.05m y 0.50m"
        self.__b = b
        
    #Rango válido de valores de h
    @property
    def h(self):
        '''dimensión de la altura de la sección transversal de la
        viga'''
        return self.__h
    @h.setter
    def h(self, h):
        assert self.l/20 <= h <= self.l/3, "entre l/20 y l/3"
        self.__h = h
    
    #Restricción de los valores para el recubrimiento
    @property
    def rec(self):
        '''Recubrimiento geométrico'''
        return self.__rec
    @rec.setter
    def rec(self, rec):
        assert 0.01 <= rec <= 0.04,\
        "debe estar entre 0.01m (EC2 4.4.1.2 (4.2)) y 0.04m\
        Calavera 'Proyecto y cálculo de estructuras de hormigón\
        para edificios' 2da. Ed. Pág 469"
        self.__rec = rec
    
    #Validación de los estados límite
    @property
    def estados_limite(self):
        '''Coeficientes para distinguir entre estados límite'''
        return {'ELU':0, 'ELS':1, 'VMS':2}
    @property
    def estado_limite(self):
        '''Definición del estado límite de análisis'''
        return self.__estado_limite
    @estado_limite.setter
    def estado_limite(self, estado_limite):
        assert estado_limite in self.estados_limite,\
        "Debe ser 'ELU', 'ELS' o 'VMS'"
        self.__estado_limite = estado_limite
    
    @property
    def d(self):
        '''Canto útil geométrico'''
        return self.h - self.rec
    
    @property    
    def Ac(self):
        '''
        Área de hormigón igual al área bruta de la sección [m2]
        '''
        return self.b * self.h
    
    #Sobre el armado
    @property 
    def As_min(self):
        '''Área mínima a tracción EC2 9.1.1.1 (9.1N)'''
        Amin1 = 0.26*self.hormigon.fctm/self.acero.fyk*self.b\
        *self.d
        Amin2 = 0.0013*self.b*self.d
        if Amin1 < Amin2:
            return Amin2 #[m2]
        else:
            return Amin1 #[m2]
    @property
    def As_max(self):
        '''Área máxima a tracción o compresión'''
        Amax = 0.04*self.Ac
        return Amax #[m2]
    
#    @property
#    def Ast(self):
#        '''Armadura de tracción'''
#        return self.__Ast
#    @Ast.setter
#    def Ast(self, Ast):
#        "No cumple cuantía"
#        assert self.As_min <= Ast <= self.As_max,
#        self.__Ast = Ast
        
    #Adoptamos el mismo criterio para las armaduras de comp.
#    @property
#    def Asc(self):
#        '''Armadura de compresión'''
#        return self.__Asc
#    @Asc.setter
#    def Asc(self, Asc):
#       "No cumple cuantía"
#        assert self.As_min <= Asc <= self.As_max,
#        self.__Asc = Asc

    @property
    def areas_varillas(self):
        '''
        numpy.array con las áreas de las secciones transversales
        de las varillas longitudinales de acero [m2]
        '''
        area = np.asarray([self.Ast, self.Asc])
        return area
    
    @property
    def fck(self):
        '''Resistencia característica del hormigón utilizado en
        la viga'''
        return self.hormigon.fk
    
    @property
    def fyk(self):
        '''Resistencia característica del acero utilizado en la
        viga'''
        return self.acero.fk
    
    @property
    def E_Ho(self):
        '''
        Devuelve el módulo de elasticidad del hormigón en la viga
        [Pa]
        '''
        if self.estado_limite == 'VMS':
            E = self.hormigon.Ecm
        else:
            E = self.hormigon.Ecd
        return E
    
    @property
    def E_Ao(self):
        '''
        Devuelve el módulo de elasticidad del acero utilizado en
        la viga [Pa]
        '''
        return self.acero.Es
    
    @property
    def inercia(self):
        '''Inercia de la sección bruta [m4]'''
        I = self.b * self.h**3/12
        return I
    
    @property
    def cantos_utiles(self):
        '''
        numpy.array de cantos útiles mecánicos para cada
        agrupación de varilla a compresión y tracción'
        '''
        return np.asarray([self.d, self.rec])

    @property
    def canto_util(self):
        '''Obtención del canto útil mecánico máximo [m]'''
        #Canto útil máximo
        d1 = np.max(self.cantos_utiles)
        return d1
    
    @property
    def vigas_tipo(self):
        '''Tipo de viga según condiciones de apoyo'''
        return {1:"Simplemente apoyada", 2:"Empotrada",
                3:"Empotrada-apoyada", 4:"biempotrada"}
        
    #Rango válido de valores de h
    @property
    def viga_tipo(self):
        '''Viga tipo según condiciones de apoyo'''
        return self.__viga_tipo
    @viga_tipo.setter
    def viga_tipo(self, viga_tipo):
        assert viga_tipo in self.vigas_tipo, "debe ser 1,2,3 o 4"
        self.__viga_tipo = viga_tipo
        
    @property
    def gdrs(self):
        '''Grados de restricción según condiciones de apoyo
        1: restringido
        0: libre
        devuelve: tupla de 4 elementos
        Según lo siguiente:
            Viga tipo 1: apoyada-apoyada: (1, 0, 1, 0)
            Viga tipo 2: empotrada: (1, 1, 0, 0)
            Viga tipo 3: empotrada-apoyada: (1, 1, 1, 0)
            Viga tipo 4: biempotrada: (1, 1, 1, 1)
        Los demás casos adaptar a estos'''
        return {1:(1,0,1,0),2:(1,1,0,0),3:(1,1,1,0),4:(1,1,1,1)}
    
    @property
    def Kse(self):
        '''Factor de sistema estructural'''
        vt = self.viga_tipo
        if vt == 1:
            return 1.0
        elif vt == 2:
            return 0.4
        elif vt == 3:
            return 1.3
        elif vt == 4:
            return 1.5
        
    @property
    def rho_0(self):
        '''Cuantía de referencia. EC2 7.4.2 (2)'''
        return math.sqrt(self.fck*1e-6)*1e-3
    
    @property
    def rho_t(self):
        '''Cuantía de la armadura de tracción'''
        return self.Ast / self.Ac
    
    @property
    def rho_c(self):
        '''Cuantía de la armadura de compresión'''
        return self.Asc / self.Ac
    
    @property
    def lim_luz_canto(self):
        '''Límite de la relación luz/canto con el cual puede
        omitirse el cálculo de flecha'''
        rho_0 = self.rho_0 #cuantía de referencia
        rho_t = self.rho_t #cuantía de tracción
        rho_c = self.rho_c #cuantía de compresión
        K = self.Kse #factor
        fck = self.fck*1e-6
        if rho_t <= rho_0: #EC2 (7.16.a)
            limite = K*(11 + 1.5*math.sqrt(fck)*rho_0/rho_t +\
                     3.2*math.sqrt(fck)*(rho_0/rho_t-1)**(3/2))
        else: #EC2 (7.16.b)
            limite = K*(11 + 1.5*math.sqrt(fck)*rho_0\
                        /(rho_t-rho_c) + 1/12*math.sqrt(fck)\
                        *math.sqrt(rho_c/rho_0))
        return limite
    
    #- - - - - - -  
    # C O S T O S
    #- - - - - - - 
    @property
    def costo_encofrado(self):
        '''
        Costo del encofrado para la viga.
        Se supone hecho con tabla en los laterales y en el fondo.
        Consideramos uso de puntales 3"x3" de 3m de altura
        disctibuidos a lo largo del eje de la viga, separados
        80cm entre sí
        '''
        tablas = (2*self.h + self.b)*50 #tablas de 1"
        return tablas * self.precio_madera
    
    @property
    def Vol_Ho(self):
        '''Volumen de hormigón en masa [m3]'''
        return self.b * self.h * self.L
    
    @property
    def costo_Ho(self):
        '''Costo total del volumen de hormigón en masa'''
        return self.Vol_Ho * self.hormigon.costo_unit
    
    @property
    def masa_Ao(self):
        '''Peso total del acero en N'''   
        #Consid. en la longitud total de las varillas el 20% de h
        vol = (self.Ast + self.Asc)*(self.L + .2*self.h)
        return vol * self.acero.densidad
    
    @property
    def costo_Ao(self):
        '''Costo total del acero'''
        return self.masa_Ao * self.acero.costo_unit
    
    @property
    def costo_viga(self):
        '''Costo total def la viga de hormigón armado'''
        return self.costo_Ho + self.costo_Ao+self.costo_encofrado
    
    # - - - - - - - 
    # C A R G A S
    # - - - - - - -
    @property
    def pp(self):
        '''Carga uniformemente distribuida debida al peso propio 
        N/m]. Sin considerar aún el coeficiente de seguridad'''
        Peso_Ho = self.Vol_Ho * self.hormigon.peso_esp
        Peso_Ao = self.masa_Ao * 9.80665
        return (Peso_Ho + Peso_Ao)/self.L
    
    @property
    def pp_nodales(self):
        '''Cargas debidas al peso propio, como cargas nodales.
        Vector columna de orden gdlx1. Ya incluye el coeficiente
        de seguridad.
        '''
        #Determinación del coeficiente de seguridad
        if self.estado_limite == 'ELU':
            gamma_G = 1.35
        else:
            gamma_G = 1.00
        
        le = self.espacio.le
        pp = self.pp
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
        Q = np.zeros((self.espacio.gdl, 1)) #ini. vector de cargas
        for i in range(self.espacio.NE):
            Q[2*i:2*i+4, :1] += Fe(pp)
        return Q * gamma_G
    
    def cargas_nodales_totales(self, cargas_nodales_ext=0):
        '''Carga total en cada nudo de la viga.
        Parámetro:
            cargas_nodales_totales: cargas nodales externas [N],
            ya debe incluir el coeficiente de seguridad
            correspondiente'''
        return  cargas_nodales_ext + self.pp_nodales # [N]
        
    # - - - - - - - - - - - - - - - - - - - - - - - -
    # Valores para el método de los ELEMENTOS FINITOS
    # - - - - - - - - - - - - - - - - - - - - - - - - 
    @property
    def rigidez_elemento(self):
        '''Matriz de rigidez del elemento finito considerando 2
        grados de libertad por nudo.
        E: módulo de elasticidad [puede ser definido en
        Materiales.hormigon()]
        VER 5.8.7.2 PARA CONSIDERAR EFECTOS DE SEGUNDO ORDEN
        '''
        le = self.espacio.le #longitud del elemento finito
        E = self.E_Ho #toma el módulo de elast. del Hº en N/m2
        I = self.inercia #inercia en m4
            
        rigidez = E * I / le**3 * np.array([
                [12, 6*le, -12, 6*le],
                [6*le, 4*le**2, -6*le, 2*le**2],
                [-12, -6*le, 12, -6*le],
                [6*le, 2*le**2, -6*le, 4*le**2]
            ])
        return rigidez
    
    @property
    def rigidez_global(self):
        '''Matriz de rigidez global de la viga considerando 2
        grados de libertad por nudo.
        E: módulo de elasticidad [puede ser definido en
        materiales.hormigon()]
        '''
        NE = self.espacio.NE
        gdl = self.espacio.gdl
        Kg = np.zeros((gdl, gdl))
        ke = self.rigidez_elemento
        for i in range(NE):
            Kg[2*i:2*i+4, 2*i:2*i+4] += ke
        return Kg
    
    @property
    def rigidez_modif(self):
        '''Matriz de rigidez global modificada según el envoque de
        penalización'''
        S = self.rigidez_global
        C = np.amax(np.abs(S))*1e4 #Rigidez del resorte del método
        NE = self.espacio.NE #número de elementos finitos
        gdr = self.gdrs[self.viga_tipo] #grados de restricción
        if gdr[0]:
            S[0, 0] += C #desplazamiento
        if gdr[1]:
            S[1, 1] += C #giro
        if gdr[2]:
            S[2*NE, 2*NE] += C #desplazamiento
        if gdr[3]:
            S[2*NE+1, 2*NE+1] += C #giro
        return S
    
     #########################
    ## R E S U L T A D O S ##
   #########################
    
    def desp_nodales(self, cargas_nodales_ext=0):
        '''Desplazamientos nodales, deflexiones verticales y giros
        [m, rad]
        Parámetro:
        ----------
        >>> viga.desp_nodales(cargas_nodales_ext)
        
        cargas_nodales_ext: numpy.array() de las cargas nodales
        externas [N, m] (sin peso propio)'''
        inversa = np.linalg.inv(self.rigidez_modif)
        cargas = self.cargas_nodales_totales(cargas_nodales_ext)
        return inversa @ cargas
    
    def elastica_fem(self, cargas_nodales_ext=0):
        '''Curva elástica obtenida por el método de los elementos
        finitos [m].
        Devuelve un numpy.array()
        Ejm:
        >>> viga.elastica_fem(cargas_nodales_ext)
        Parámetro:
        ---------
        cargas_nodales_ext: numpy.array() de las cargas nodales
        externas [N, m] (sin peso propio)
        '''
        NN = self.espacio.NN #Número de nudos
        v = np.zeros(NN) #inic. del vector de deflexiones vert.
        #Vector de desplazamientos nodales
        Q = self.desp_nodales(cargas_nodales_ext)
        #Extraemos solo los desplazamientos verticales
        for i in range(NN):
            v[i] += Q[2*i, 0]
        return v
    
    def flecha_fem(self, cargas_nodales_ext=0):
        '''
        Deflexión máxima [m] calculada con el Método de Elementos
        Finitos. Devuelve una tupla (flecha, posición).
        Flecha: es la deflexión máxima de la viga (en valor
        absoluto)
        Posición: es la ubicación de la flecha.
        Ejm:
        >>> viga.flecha_fem(cargas_nodales_ext)
        Parámetro:
        ---------
        cargas_nodales_ext: numpy.array() de las cargas nodales
        externas. [kN, m] (sin peso propio)
        '''
        #Curva elástica de la viga
        deflexiones = self.elastica_fem(cargas_nodales_ext)
        flecha = np.min(deflexiones)
        x_flecha = self.espacio.x_nudos[np.where(deflexiones ==\
                                                 flecha)]
        return abs(flecha), x_flecha[0]
    
    def matriz_desplazamientos(self,cargas_nodales_ext=0):
        '''
        Matriz cuyas filas son los desplazamientos de cada EF
        [m, rad]
        Ejm:
        >>> viga.matiz_desplazamientos(cargas_nodales_ext)
        Parámetro:
        ---------
        cargas_nodales_ext: numpy.array() de las cargas nodales
        externas
                        [N, m] (sin peso propio)'''
        NE = self.espacio.NE
        #Desplazamientos de cada elemento finito
        D = np.zeros((NE, 4)) #Matriz con filas de desp. en c/ EF
        #Vector de desplazamientos nodales
        Q = self.desp_nodales(cargas_nodales_ext)
        for i in range(NE):
            D[i, 0] += Q[2*i, 0]
            D[i, 1] += Q[2*i+1, 0]
            D[i, 2] += Q[2*i+2, 0]
            D[i, 3] += Q[2*i+3, 0]
        return D
    
    def DMF(self, cargas_nodales_ext=0):
        '''Diagrama de momento flector.
        Ejm:
            >>> viga.DMF(cargas_nodales_ext)
        Parámetro:
        ---------
        cargas_nodales_ext: numpy.array(NE) de las cargas nodales
        externas. [kN, m] (sin peso propio)
        '''
        E = self.E_Ho
        I = self.inercia
        le = self.espacio.le
        NE = self.espacio.NE
        
        #Cálculo de momentos en el centro de c/ EF (\xi=0)
        def Mf(q2, q4):
            '''Función para cálculo del momento flector dados los
            giros nodales'''
            M = E*I/le**2*(-le*q2 + le*q4)
            return M
        
        D = self.matriz_desplazamientos(cargas_nodales_ext)
        M = np.zeros(NE) #inicialización
        for i in range(NE):
            M[i] = Mf(D[i, 1], D[i, 3])

        return M #[Nm]
    
    def Md(self, cargas_nodales=0):
        '''Momento máximo de cálculo'''
        return np.max(self.DMF(cargas_nodales))
    
    # - - - - - - - - - - - - -
    # Propiedades de la sección
    # - - - - - - - - - - - - -
        
    def dmc(self):
        '''DIAGRAMA DE MOMENTO-CURVATURA
        Devuelve los valores de Momento [Nm] y Curvatura [m^(-1)]
        del diagrama Momento-Curvatura de una sección de Hormigón
        Armado.
            
        Devuelve dos numpy.array:
            1. Los valores de las curvaturas en mm^{-1}
            2. Los valores del momento flector correspondiente en
            kNm
        '''
        numx=250 #Número de iteraciones
        d1 = self.canto_util #canto mecánico útil
        areas_varillas = self.areas_varillas #np.array con las
                                            #áreas As [m2]
        b = self.b #base
        h = self.h #altura
        fc = self.hormigon.fc
        cantos_utiles = self.cantos_utiles
        
        #Funciones para calcular tensiones
        #Tensión ACERO en valores medios (Pa)     
        sigma_s = self.acero.sigma_s
        #Deformación del ACERO en el lím. elástico, en valor medio
        e_y = self.acero.epsilon_y
        #Deformación última del acero, en tanto por uno
        e_yu = self.acero.epsilon_ud
        #Área del diag. del HORMIGÓN entre a y b en valores medios
        area = self.hormigon.area_c
        #Momento estático del HORMIGÓN resp. al origen
        momento_estatico = self.hormigon.momento_estatico_c
    
        #Deformación última del hormigón, en tanto por uno, es (-)    
        e_cu = -self.hormigon.epsilon_cu1 #0.0035
        #Deformación última del acero, en tanto por uno, es (+)
        e_su = self.acero.epsilon_su #0.01
        
        #Límites de prof. de LN en Dominios    
        x2 = -e_cu*d1/(e_su - e_cu) #Límite Dominio 2
        x_lim = -e_cu*d1/(e_y - e_cu) #Límite Dominio 3
        
        #Tolerancia para considerar cero los valores de F
        tol = b * h * fc / numx/25000
        
        #Inicialización    
        e_sup = e_cu/numx #esta en tanto por uno, es (-)
        k = [0] #Para guardar las curvaturas
        M = [0] #Para guardar los momentos
        
        while e_sup >= e_cu:
            #Prof. mínima condicionada por la def. máxima del Aº
            x_min = -e_sup*d1/(-e_sup + e_yu) 
            x = x_min*1.1
            
            #Lista de fuerzas totales [kN]
            F_lista = [b * h * fc]
            #Lista de prof. de LN [m]
            x_lista = [x_min]
            while x <= h: #va hasta la fibra inferior
            
                #Deformaciones en el Acero en tanto por uno
                Epsilon_s = np.asarray([
                        -e_sup*(d-x)/x for d in cantos_utiles
                        ])
                
                #Tensiones en el ACERO MPa
                Ss = np.asarray([sigma_s(e) for e in Epsilon_s])
                #Fuerzas en cada camada kN            
                Ts = Ss * areas_varillas
                #Fuerza total en las armaduras en kN            
                T = np.sum(Ts)
                
                #Fuerza en el HORMIGÓN
                e_inf = -(h-x)/x*e_sup
                C = -b*x/e_sup*area(e_sup, e_inf) #kN
                    
                #FUERZA TOTAL EN LA SECCION
                F = T + C
                F_anterior = F_lista[-1]
                
                #Profundidad anterior de LN
                x_anterior = x_lista[-1]
    
                #El valor de F debe ir decreciendo
                if F < F_anterior:
                    #Si es menor a la tolerancia calculamos M y k
                    if abs(F) <= tol:              
                        #Cálc. de la excent. de las fuerzas del Hº
                        MC = -b*x**2/e_sup**2\
                        * momento_estatico(e_sup, e_inf)
                        ec = MC/C
                        
                        #Cálc. de la exc.de las fuerzas del ACERO
                        es = np.sum(Ts*cantos_utiles)/T - x
                        
                        #Momento flector
                        #Promediamos T y C buscando precisión
                        Mf = (T-C)/2*(ec+es)
                        
                        #Para que la curva sea siempre ascendente
                        if Mf > M[-1]:
                            M.append(Mf) #Momento en Nm
                            k.append(-e_sup/x) #Curvatura en 1/m
        
                        x += h #para terminar bucle while interno
                        
                    #No es menor a la tolerancia pero F pasa por
                    #un cero interpolamos
                    elif math.copysign(1, F) <\
                    math.copysign(1, F_anterior):
                        x = F_anterior/(F_anterior-F)\
                        *(x-x_anterior) + x_anterior
                    
                    else:
                        F_lista.append(F)
                        x_lista.append(x)
                        #Probamos con prof. de LN en límites de
                        #dominios
                        if e_sup == e_cu:
                            if 0 < x2 - x < d1/numx:
                                x = x2
                            elif 0 < x_lim - x < d1/numx:
                                x = x_lim
                            elif 0 < d1 - x < d1/numx:
                                x = d1
                            elif 0 < h - x < d1/numx:
                                x = h
                            else:
                                x += d1/numx #inc. en prof. de LN
                        else:
                            x += d1/numx #inc. en la prof. de LN
                else:
                    x += h #para terminar el bucle while interno
            
            #Incremento en la fibra superior (negativo)        
            if 0 < e_sup - e_cu < abs(e_cu)/numx:
                #Probar con valor máximo de e_sup
                e_sup = e_cu
            else:
                e_sup += e_cu/numx
        
        curvaturas = np.array(k)
        momentos = np.array(M)
        
        return curvaturas, momentos
    
    def diag_interaccion(self):
        '''
        Devuelve las abcisas (N: fuerza normal en N) y ordenadas
        (M: momento flector en Nm) correspondientes al diagrama
        de interacción de una sección rectangular de Hormigón
        Armado.
        '''
        numx=50 #es el núm. de iteraciones por c/ dominio de def.
        d1 = self.canto_util #Canto útil
        cantos_utiles = self.cantos_utiles
        areas_varillas = self.areas_varillas
        b = self.b
        h = self.h
        
        #Funciones para calcular tensiones
        #Tensión del ACERO
        sigma_s = self.acero.sigma_s
        #Tensión del HORMIGÓN en valores medios        
        sigma_c = self.hormigon.sigma_c
        #Área de la curva del hormigón entre a y b en val. medios        
        area = self.hormigon.area_c
        #Momento estático de respecto al origen del área de la
        #curva del hormigón entre a y b en valores medios
        momento_estatico = self.hormigon.momento_estatico_c
        #Def. del acero en el límite elástico (positivo)        
        e_y = self.acero.epsilon_y
        
        e_cu = -self.hormigon.epsilon_cu1 #Def. última del Hº (-)
        e_c1 = -self.hormigon.epsilon_c1 #Def. unit. pico del Hº
        e_su = self.acero.epsilon_su #Def. última del acero (+)
        traccion = self.hormigon.traccion
        
        #Listas para guardar los valores Nu y Mu calculados
        Nu, Mu = [], []
    
    #DOMINIO 1
        for e_sup in np.linspace(e_su, 0, numx): #Decrece
        #Tracción simple
            if e_sup == e_su:
            #ACERO
                Ss = sigma_s(e_su) #Tensión igual en todas las
                                    #camadas Pa
                Ts = Ss * areas_varillas #Fuerzas en c/ camada [N]
                T = np.sum(Ts) #Fuerza total en las armaduras [N]
            
            #HORMIGÓN
                C = b * h * sigma_c(e_su) #[N]
                
            #FUERZA TOTAL
                Nu.append(T + C)
            
            #MOMENTO TOTAL
                Mu.append(0) #Tracción simple
                
        #Tracción compuesta
            else:
            #Profundidad de la LN (valores negativos)
                x = -e_sup/(e_su-e_sup)*d1
                
            #ACERO
                #Deformaciones en cada camada de Acero
                Epsilon_s = np.array([
                        (d-x)/(d1-x)*e_su for d in cantos_utiles
                                      ])
                #Tensiones en MPa           
                Ss = np.array([sigma_s(e) for e in Epsilon_s])
                #Fuerzas en cada camada N
                Ts = Ss * areas_varillas
                #Fuerza total en las armaduras en N
                T = np.sum(Ts)
                
                #Momento
                Ms = np.sum(Ts*(cantos_utiles-h/2))
                
            #HORMIGÓN
                if traccion:
                    #Deformación de la fibra inferior                
                    e_inf = (h-x)/(d1-x)*e_su 
                    #Fuerzas en kN                
                    C = b*(d1-x)/e_su*area(e_sup, e_inf)
                    #Momento respecto a la LN en Nm
                    Mn = -b*(d1-x)**2/e_su**2 * \
                    momento_estatico(e_sup, e_inf)
                    #excentricidad respecto a la LN                
                    ec = Mn/C
                    #excentricidad respecto al CG                
                    yc = h/2 - x + ec
                    #Momento respecto al CG
                    Mc = -C*yc
                else:
                    C = 0                
                    Mc = 0
                    
            #FUERZA TOTAL
                Nu.append(T + C)
                
            #MOMENTO TOTAL
                Mu.append(Ms + Mc)
    
    #DOMINIO 2
        #Límite de prof. de LN
        x2 = -e_cu*d1/(e_su - e_cu)
        for x in np.linspace(0, x2, numx):
            if x > 0:
                #Deformación de la fibra superior
                e_sup = -x/(d1-x)*e_su
                
            #ACERO
                #Deformaciones en cada camada de Acero
                Epsilon_s = np.array([
                        (d-x)/(d1-x)*e_su for d in cantos_utiles
                                      ])
                #Fuerzas
                #Tensiones [MPa]
                Ss = np.array([sigma_s(e) for e in Epsilon_s])
                Ts = Ss * areas_varillas #fuerzas en c/ camada [N]
                T = np.sum(Ts) #Fuerza total en las armaduras [N]
                #Momento
                Ms = np.sum(Ts*(cantos_utiles - h/2))
                
            #HORMIGÓN
                #Deformación de la fibra inferior
                e_inf = (h-x)/(d1-x)*e_su
                #Fuerzas [kN]
                C = b*(d1-x)/e_su*area(e_sup, e_inf)
                #Momento respecto a la LN en Nm
                Mn = -b*(d1-x)**2/e_su**2\
                * momento_estatico(e_sup, e_inf)
                #Excentricidad respecto a la LN
                ec = Mn/C
                #excentricidad respecto al CG
                yc = h/2 - x + ec
                #Momento respecto al CG
                Mc = -C*yc
                    
            #FUERZA TOTAL
                Nu.append(T + C)
                
            #MOMENTO TOTAL
                Mu.append(Ms + Mc)
    
    #DOMINIO 3
        #Límite de prof. de LN
        x_lim = -e_cu*d1/(e_y - e_cu)
        for x in np.linspace(x2, x_lim, numx):
            if x > x2:          
            #ACERO
                #Deformaciones en cada camada de Acero
                Epsilon_s = np.array([
                        -e_cu*(d-x)/x for d in cantos_utiles
                                      ])
                #Fuerzas            
                #Tensiones [MPa]
                Ss = np.array([sigma_s(e) for e in Epsilon_s])
                Ts = Ss * areas_varillas #fuerzas en c/ camada [N]
                T = np.sum(Ts) #Fuerza total en las armaduras [N]
                #Momento
                Ms = np.sum(Ts*(cantos_utiles - h/2))
                
            #HORMIGÓN
                #Fuerzas
                e_inf = -(h-x)/x*e_cu
                C = -b*x/e_cu*area(e_cu, e_inf) #[N]
                
                #Momento respecto a la LN [kNm]
                Mn = -b*x**2/e_cu**2\
                *momento_estatico(e_cu, e_inf)
                ec = Mn/C #excentricidad respecto a la LN
                yc = h/2 - x + ec #excentricidad respecto al CG
                #Momento respecto al CG
                Mc = -C*yc
                    
            #FUERZA TOTAL
                Nu.append(T + C)
                
            #MOMENTO TOTAL
                Mu.append(Ms + Mc)
    
    #DOMINIO 4 y 4a
        for x in np.linspace(x_lim, h, numx):
            if x > x_lim:
            #ACERO
                #Deformaciones en cada camada de Acero
                Epsilon_s = np.array([
                        -e_cu*(d-x)/x for d in cantos_utiles
                                      ])
                #Fuerzas
                #Tensiones [MPa]
                Ss = np.array([sigma_s(e) for e in Epsilon_s]) 
                Ts = Ss * areas_varillas #fuerzas en c/ camada [N]
                T = np.sum(Ts) #Fuerza total en las armaduras [N]
                #Momento
                Ms = np.sum(Ts*(cantos_utiles - h/2))
                
            #HORMIGÓN
                #Deformación unitaria en la fibra inferior
                e_inf = -(h-x)/x*e_cu
                #Fuerza total en el hormigón [N]
                C = -b*x/e_cu*area(e_cu, e_inf)
                
                #Momento respecto a la LN [Nm]
                Mn = -b*x**2/e_cu**2\
                *momento_estatico(e_cu, e_inf)
                ec = Mn/C #excentricidad respecto a la LN
                yc = h/2 - x + ec #excentricidad respecto al CG
                #Momento respecto al CG
                Mc = -C*yc
                    
            #FUERZA TOTAL
                Nu.append(T + C)
                
            #MOMENTO TOTAL
                Mu.append(Ms + Mc)
    
    #DOMINIO 5
        xc = (e_cu-e_c1)/e_cu*h
        for e_sup in np.linspace(e_cu, e_c1, numx):
            if e_sup > e_cu:
                if e_sup < e_c1:
                    
                #Profundidad de la LN
                    x = -e_sup/(-e_sup+e_c1)*xc
                    
                #ACERO
                    #Deformaciones en cada camada de Acero
                    Epsilon_s = np.array([
                            -e_sup*(d-x)/x for d in cantos_utiles
                                          ])
                    #Tensiones [MPa]            
                    Ss = np.array([sigma_s(e) for e in Epsilon_s])
                    #Fuerzas en cada camada [N]                
                    Ts = Ss * areas_varillas
                    #Fuerza total en las armaduras [N]
                    T = np.sum(Ts)
                    #Momento
                    Ms = np.sum(Ts*(cantos_utiles - h/2))
                    
                #HORMIGÓN
                    #Fuerzas
                    e_inf = -(h-x)/x*e_sup
                    C = -b*x/e_sup*area(e_sup, e_inf) #[N]
                    
                    #Momento respecto a la LN [Nm]
                    Mn = -b*x**2/e_sup**2\
                    *momento_estatico(e_sup, e_inf)
                    ec = Mn/C #excentricidad respecto a la LN
                    yc = h/2 - x + ec #excent. respecto al CG
                    #Momento respecto al CG
                    Mc = -C*yc
                        
                #FUERZA TOTAL
                    Nu.append(T + C)
                    
                #MOMENTO TOTAL
                    Mu.append(Ms + Mc)
                else:
                #ACERO
                    #Tensión de compresión en todas las varillas
                    Ss = sigma_s(e_c1) 
                    #Fuerza de compresión en las varillas
                    Ts = Ss * areas_varillas
                    #Fuerza total en el acero
                    T = np.sum(Ts)
                #HORMIGÓN
                    #Fuerzas
                    C = b * h * sigma_c(e_c1) #[N]
                
                #FUERZA TOTAL
                    Nu.append(T + C)
                    Mu.append(0)
        
        N = np.array(Nu)
        M = np.array(Mu)
                
        return N, M

    def Mu(self):
        '''Momento flector último que puede soportar una sección
        de HºAº en ausencia de cargas axiales (flexión simple)'''
        N, M = self.diag_interaccion()
        
        if 0 in N:
            i = np.where(N == 0)
            return M[i]
        else:
            i1 = np.where(N>0)[0][-1] #Ubicación del último
                                     #elemento positivo
            i2 = i1 + 1 #Ubicación del primer elemento negativo
            N1 = N[i1] #Menor valor a tracción
            N2 = N[i2] #Menor valor a compresión
            M1 = M[i1] #Momento correspondiente a N1
            M2 = M[i2] #Momento correspondiente a N2
            Mu = ut.interpL(N1, M1, N2, M2, 0) #Interpolación
        return Mu
        
    #----------------------------------------------------#
    # Los siguientes dependen de la carga sobre la viga  #
    #----------------------------------------------------#
    
    def diag_curvatura(self, cargas_nodales_ext=0):
        '''
        Devuelve un numpy.array de curvaturas dada una lista o un
        numpy.array de momentos. Utilizando los valores de
        curvaturas y momentos del diagrama de momento-curvatura
        obtenido en viga.dmc()
        El objetivo es: dado el diagrama de momento flector de una
        viga obtener el diagrama de curvaturas de la citada viga.
        
        Parámetros:
        ----------
        >>> viga.diag_curvatura(cargas_nodales_ext, numx=10)
        cargas_nodales_ext: numpy.array, valores de las cargas
        nodales externas
        numx = número de iteraciones para el diagrama de momento
                curvatura, por defecto = 250.
                
        IMPORTANTE:
            - Notar que esto es válido solo para vigas de sección
            rectangular constante en toda su longitud.
        '''        
        #Valores del diagrama de momento curvatura
        k, M = self.dmc()
        DMF = self.DMF(cargas_nodales_ext)
        
        #Obtenemos las curvaturas haciendo interpolación lineal
        cur = [ut.yLista(M, k, m) for m in DMF]
        
        return np.array(cur)
    
    def elastica_dmc(self, cargas_nodales_ext=0):
        '''
        Devuelve la curva elástica de una viga de hormigón armado
        de sección transversal rectangular constante y de longitud
        dada, utilizando para el cálculo de las deflexiones el
        diagrama de momento curvatura.
        
        Parámetros:
        ----------
        DMF: numpy.array, valores de momentos del diagrama de
        momento flector
        
        Esta función devuelve un numpy.array con las deflexiones
        verticales a lo largo del eje de la viga.
        El cálculo se realiza por el método de la carga unitaria,
        en donde cada deflexión es:
            \delta = \int_0^{L} \kappa_(x) \bar{M} \ dx, es decir
            la integral en toda la longitud de la viga del
            diagrama de curvatura multiplicado por el diagrama de
            momento flector de la carga unitaria en la posición
            de la deflexión a ser calculada.
        '''
        L = self.L
        curvaturas = self.diag_curvatura(cargas_nodales_ext)
        x_centros = self.espacio.x_centros
        h = self.espacio.le #paso del método del trapecio
        tipo = self.viga_tipo
        
        deflexiones = np.array([])
        for i in range(self.espacio.NE):
            z = x_centros[i] #Posición de la carga unitaria
            M_carga_unit = np.array([
                    ut.M_unit(x, z, L, tipo) for x in x_centros])

            y = -curvaturas * M_carga_unit #Integrando
            
            #Integración por el método del trapecio
            delta = h*((y[0]+y[-1])/2 + np.sum(y[1:-1]))
            deflexiones = np.append(deflexiones, delta)

        return deflexiones

    def flecha_dmc(self, cargas_nodales_ext=0):
        '''
        Deflexión máxima calculada con el diagrama de momento
        curvatura.
        Devuelve una tupla (flecha, posición).
        Flecha: es la deflexión máxima de la viga.
        Posición: es la ubicación de la flecha.
        '''
        #Curva elástica de la viga
        deflexiones = self.elastica_dmc(cargas_nodales_ext)
        flecha = np.min(deflexiones)
        x_flecha = self.espacio.x_centros[np.where(deflexiones ==\
                                                   flecha)]
        return abs(flecha), x_flecha[0]
    
    #################################
    ## V E R I F I C A C I O N E S ##
    #################################
    
    def verif_ELU(self, cargas_nodales_ext=0):
        '''Variable booleana. 
        1: Si Mu >= Md, verifica
        0: Si Mu < Md, no verifica
        Parámetro:
            cargas_nodales_ext: cargas nodales externas,
            np.array((gdl,1))'''
        return self.Mu() >= self.Md(cargas_nodales_ext)
    
    def verif_ELS(self, cargas_nodales_ext=0, dmc=True, div=250):
        '''Variable booleana. 
        Parámetros:
        ----------
            cargas_nodales_ext: cargas nodales externas
            dmc: para considerar la flecha calculada con el
            diagrama momento-curvatura. Por defecto TRUE, en caso
            contrario, se toma en cuenta la flecha calculada por
            el Método de Elementos Finitos
            div: divisor para el límite de la fleca f < L/div'''
            
        f_max = self.L/div #flecha máxima
        if dmc:
            return self.flecha_dmc(cargas_nodales_ext)[0] <= f_max
        else:
            return self.flecha_fem(cargas_nodales_ext)[0] <= f_max
