# -*- coding: utf-8 -*-
"""
Permite definir propiedades del hormigón y el acero (pasivo) según
el Eurocódigo 2: Design of concrete structures EN 1992-1-1:2004
Part 1-1: General rules and rules for buildings.
-----------------------------------------
Created on Sat May 28 14:14:57 2016
@author: Fredy Gabriel Ramírez Villanueva
ESTÁ ESTO TERMINADO?: Aún no
-----------------------------------------

Símbolos utilizados:
        H, c: hormigón
        s: acero
        f: resistencia
        m: medio
        d: de diseño
        c: compresión
        t: tracción
        005: 5% inferior
        095: 95% superior
        u: último
        cr: crítico

Ejemplos de uso:
---------------

Asumimos la importanción del módulo de la siguiente manera:  
>>> import materiales as mat

Ejm 1: Definir un hormigón para análisis estructural no lineal,
con valores por defecto:
>>> H1 = mat.Hnl()

Los valores por defecto son: estado_limite='ELU', fck=25,
alpha_cc=1.0, alpha_ct=1.0, agregado=1.2, nu=0.2, traccion=False,
n=0.4

Donde:
    estado_limite = 'ELU', 'ELS' o 'VMS' (valores medios)
    fck: resistencia característica del hormigón.
         Solo se pueden usar las resistencias de hormigón dadas en
         la Tabla 3.1 de 12 MPa hasta 50 MPa.
         Se puede ver la lista de clases de hormigón haciendo:
             >>> H1.claseH
                 [12, 16, 20, 25, 30, 35, 40, 45, 50]
    alpha_cc: coeficiente de cansancio a compresión
    alpha_ct: coeficiente de cansancio a tracción
    agregado: tipo de agregado para el módulo de elasticidad
    (EC2 3.1.3 (2)) caliza: 0.90, arenisca: 0.70, cuarcita: 1.00,
    basalto: 1.20
    nu: Módulo de Poisson

Cada propiedad se puede modificar, por ejm. se puede cambiar el
coef. seg:
    >>> H1.gamma_c = 1.5
    
Ejm 2: Definir un hormigón con diagrama parábola-rectángulo
>>> H2 = mat.Hpr()

- - - - - - - FALTA IR MEJORANDO LA AYUDA - - - - - - - -

"""
#Módulos
import numpy as np #Vectores, matrices y más
from scipy.integrate import quad #Para integrales
import matplotlib.pyplot as plt

costo_Ho = 220 #[USD/m3] debe incluir mano de obra
costo_Ao = 1.00 #[USD/kg] debe incluir mano de obra

#MATERIAL genérico
class Material:
    '''Material resistente.
    UNIDADES
        Todos en el SI sin múltiplos ni submúltiplos
    '''
    def __init__(self, estado_limite, fk, densidad, costo_unit):
        '''Parámetros:
          -----------
        estado_limite: último (ELU), de servicio (ELS) o
        utilización de valores medios (VMS)
        fk: resistencia característica del material [Pa]
        densidad: densidad [kg/m3]
        costo_unit: costo unitario [USD]
        ''' 
        self.estado_limite = estado_limite
        self.fk = fk #resistencia característica
        self.densidad = densidad #densidad del material [kg/m3]
        self.costo_unit = costo_unit #costo unitario
        
    #Para verificar la correcta definición de los estados límite
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

#HORMIGÓN
class Hormigon(Material):
    '''
    Clase hormigón según el Eurocódigo Nº 2
    UNIDADES para datos de entrada y salida
        Todos el SI sin múltiplos ni submúltiplos.
    '''
    def __init__(self, alpha_cc=1.0, alpha_ct=1.0,
                 agregado='basalto', nu=0.2, estado_limite='ELU',
                 fk=25e6, densidad=2400, costo_unit=costo_Ho):
        
        Material.__init__(self, estado_limite, fk, densidad,
                          costo_unit)
        '''
        Hormigon(alpha_cc=1.0, alpha_ct=1.0, agregado='basalto',
        nu=0.2, costo_unit=250.00)
        
        Parámetros:
        ----------
        fk: resistencia característica
          Para el hormigón fck (EC2 Tabla 3.1) (Mpa)
          resist. de norma: [12, 16, 20, 25, 30, 35, 40, 45, 50]
        alpha_cc: coeficiente de cansancio a compresión
        alpha_ct: coeficiente de cansancio a tracción
        agregado: tipo de agregado para el módulo de elasticidad
        (EC2 3.1.3(2)) {'caliza': 0.90, 'arenisca': 0.70,
        'cuarcita': 1.00, 'basalto': 1.20}. Por defecto: 'basalto'
        nu: Módulo de Poisson. Por defecto = 0.2
        costo_unit: [USD] Precio de un m3 de hormigón en masa
        densidad: Peso específco. Por defecto 2400 kg/m3
        (EC1 Tabla A.1)
        '''
        self.alpha_cc = alpha_cc #Coef. de cansancio a compresión
        self.alpha_ct = alpha_ct #Coefi. de cansancio a tracción
        self.agregado = agregado #Coef. según el agregado grueso
        self.nu = nu #Módulo de Poisson
        self.densidad = densidad #Peso específico [kN/m3]
        self.costo_unit = costo_unit #[USD] Precio de 1m3 de Hº
    
    @property
    def peso_esp(self):
        '''Peso específico del hormigón [N/m3]'''
        return self.densidad*10
    
    @property
    def claseH(self):
        '''Clases de hormigón definidos en Tabla 3.1 EC2 en MPa'''
        return np.array([12, 16, 20, 25, 30, 35, 40, 45, 50])*1e6
    
    #Para evitar que fck asuma valores distintos a claseH
    @property
    def fk(self):
        '''Resistencia característica del hormigón'''
        return self.__fk
    @fk.setter
    def fk(self, fk):
        assert fk in self.claseH, "fck debe pertenecer a claseH"
        self.__fk = fk
        
    @property
    def fck(self):
        '''Resistencia característica del hormigón'''
        return self.fk
        
    @property
    def gamma_c(self):
        '''Coeficiente parcial de seguridad para el hormigón.
        Tabla 2.1N'''
        if self.estado_limite == 'ELU':
            return 1.5
        else:
            return 1.0
        
    @property
    def gamma_cE(self):
        '''Coeficiente parcial de seguridad del módulo de Young.
        5.8.6 (3) eq.(5.20)'''
        if self.estado_limite == 'ELU':
            return 1.2
        else:
            return 1.0
        
    #Para verificar la correcta elección del tipo de agregado
    @property
    def agregados(self):
        '''Factores por tipo de agregado'''
        return {'arenisca': 0.7, 'basalto': 1.2, 'caliza': 0.9,
                'cuarcita': 1.0}
    @property
    def agregado(self):
        '''Factor por tipo de agregado'''
        return self.__agregado
    @agregado.setter
    def agregado(self, agregado):
        assert agregado in self.agregados, \
        "debe ser arenisca, basalto o caliza"
        self.__agregado = agregado
    
    @property
    def fcm(self):
        '''Resistencia media a compresión [Pa] en función a la
        resistencia característica. Tabla 3.1'''
        return self.fk + 8e6
    
    @property
    def fctm(self):
        '''Resistencia media a tracción [Pa] en función a la
        resistencia característica. Tabla 3.1.'''
        return 0.30*(self.fk*1e-6)**(2/3)*1e6

    @property
    def fcd(self):
        '''Resistencia de calculo [Pa]'''
        return self.alpha_cc * self.fck / self.gamma_c
        
    @property
    def fctk005(self):
        '''Resistencia característica inferior a tracción [Pa]'''
        return 0.7*self.fctm
        
    @property
    def fctk095(self):
        '''Resistencia característica superior a tracción [Pa]'''
        return 1.3*self.fctm
        
    @property
    def fctd(self):
        '''Resistencia de cálculo a tracción [Pa]'''
        return self.alpha_ct * self.fctk005 / self.gamma_c
    
    @property
    def Ecm(self):
        '''Modulo secante. Tabla 3.1 Ecm [Pa]'''
        #fcm debe estar en MPa por eso el 1e-6 en la fórmula y
        #luego el 1e9 para pasar a Pa el resultado
        return self.agregados[self.agregado]*22  \
                * (self.fcm*1e-6/10)**0.3*1e9
    
    @property
    def Ecd(self):
        '''Módulo de elasticidad de cálculo [Pa].
        EC2 5.8.6 (3) eq. (5.20)
        '''
        return self.Ecm / self.gamma_cE
    
    @property
    def Ec0(self):
        '''Módulo de elasticidad tagente en el origen [Pa]'''
        return 1.05 * self.Ecm
    
    @property
    def epsilon_c1(self):
        '''Deformación máx a compresión simple en tanto por uno'''
        #En la fórmula: fcm en MPa por eso el 1e-6
        e = 0.7 * (self.fcm*1e-6)**0.31 #Tabla 3.1 [o/oo]
        if e < 2.8:
            return e/1000 #Pasamos a tanto por uno
        else:
            return 2.8/1000 #divido mil para pasar a tanto por uno
    
    @property
    def epsilon_cu1(self):
        '''Def. unitaria última del hormigón en tanto por uno'''
        return 3.5/1000
        
    @property
    def epsilon_c2(self):
        '''Deformación máxima a compresión simple en el diagrama
        parábola-rectángulo en tanto por uno'''
        return 2.0/1000
        
    @property
    def epsilon_cu2(self):
        '''Deformación unitaria última del hormigón en el diagrama
        parábola-rectángulo en tanto por uno'''
        return 3.5/1000
    
    @property
    def epsilon_c3(self):
        '''Deformación máxima a compresión simple en el diagrama
        bilineal en tanto por uno'''
        return 1.75/1000
        
    @property
    def epsilon_cu3(self):
        '''Deformación unitaria última del hormigón en el diagrama
        bilineal en tanto por uno'''
        return 3.5/1000
        
class Hnl(Hormigon):
    '''Hormigón para para análisis no lineal'''
    def __init__(self, traccion=True, n=0.4, alpha_cc=1.0,
                 alpha_ct=1.0, agregado='basalto', nu=0.2,
                 estado_limite='VMS', fk=25e6, densidad=2400,
                 costo_unit=costo_Ho):
        '''Hnl(traccion=True, n=0.4, estado_limite='VMS', fck=25,
        alpha_cc=1.0, alpha_ct=1.0, agregado='basalto', nu=0.2,
        costo_unit=150.00, peso_esp=24.)
        traccion: True or False, considerar la tracción en el Ho.
        n: exp. del diagrama de tracción Wang & Hsu, por def 0.4
        alpha_cc: coeficiente de cansancio a compresión
        alpha_ct: coeficiente de cansancio a tracción
        agregado: tipo de agregado para el módulo de elasticidad
                (EC2 3.1.3 (2))                
        nu: Coeficiente de Poisson, por defecto 0.2
        estado_limite: 'ELU', 'ELS' o 'VMS' para valores medios
        fk: resistencia característica
        costo_unit: costo unitario del hormigón
        densidad: densidad del hormigón'''
        
        Hormigon.__init__(self, alpha_cc, alpha_ct, agregado, nu,
                          estado_limite, fk, densidad, costo_unit)
        self.traccion = traccion #Considerar tracción del Ho
        self.n = n #exponente del diagrama de tracción W&S
        
    @property
    def fc(self):
        if self.estado_limite == 'VMS':
            return self.fcm
        else:
            return self.fcd
    
    @property
    def fct(self):
        if self.estado_limite == 'VMS':
            return self.fctm
        else:
            return self.fctd
    
    @property
    def Ec(self):
        if self.estado_limite == 'VMS':
            return self.Ecm
        else:
            return self.Ecd
        
    @property
    def epsilon_cr(self):
        '''Deformación crítica a tracción en tanto por uno'''
        return self.fct / self.Ec
        
    def __str__(self):
        datos_hormigon = "Hº para análisis estructural: \nfck: "\
        + str(self.fck*1e-6) + "MPa\tEstado: "+\
        str(self.estado_limite)+"\t\tgamma_c: " +\
        str(self.gamma_c) + "\tfcd: "+str(round(self.fc*1e-6,2))+\
        " MPa\nfct = " + str(round(self.fct*1e-6,2)) +\
        " MPa\tfctk005 = " + str(round(self.fctk005*1e-6,2)) +\
        "MPa\tEcm = "+str(round(self.Ecm*1e-9,2))+"GPa\tEcd = "+ \
        str(round(self.Ecd*1e-9,2)) + "GPa\tEc0 = " + \
        str(round(self.Ec0*1e-9,2)) + "GPa\ne_c1 = " + \
        str(round(self.epsilon_c1*1000,2))+"o/oo"+"\te_cu1 = " +\
        str(round(self.epsilon_cu1*1000,2))+"o/oo"+"\t\te_cr = "+\
        str(round(self.epsilon_cr*1000,3)) + "o/oo"
                      
        return datos_hormigon

    def sigma_cc(self, epsilon_c):
        '''Devuelve la COMPRESIÓN del hormigón en Pa (negativo: 
            compresión)
        para la resistencia media fcm.
        Dada deformación (negativo: acortamiento).
        UNIDAD: Pa, positivo: tracción
        PARÁMETROS:
        epsilon_c: deformación unitaria del hormigón en tanto por
        uno'''
        if epsilon_c < -self.epsilon_cu1:
            return 0
        elif epsilon_c < 0:
            E = self.Ecd
            fc = self.fc
            epsilon_c1 = self.epsilon_c1
            
            eta = -epsilon_c/epsilon_c1 #epsilon_c entra en o/oo
            km= 1.05*E*abs(epsilon_c1)/fc #Unidades correctas
            sigmaC_comp = -(fc)*(km*eta-eta**2)/(1+(km-2)*eta)
            return sigmaC_comp
        else:
            return 0

    def sigma_ct(self, epsilon_c):
        '''Devuelve la TRACCIÓN (Wang&Tsu)del hormigón en valores
        MEDIOS [Pa] (positivo: tracción) dada la deformación 
        negativo: acortamiento).
        Cargas de corta duración.
        UNIDAD: MPa, positivo: tracción
        PARÁMETROS:
        epsilon_c: deformación unitaria del hormigón tanto por uno
        '''
        if epsilon_c > 0:
            Ec = self.Ec
            fct = self.fct
            epsilon_cr = self.epsilon_cr
            if epsilon_c < epsilon_cr:
                return Ec*epsilon_c
            else:
                return fct*(epsilon_cr/epsilon_c)**self.n
        else:
            return 0
            
    def sigma_c(self, epsilon_c):
        '''Devuelve la tensión en valores MEDIOS del hormigón
        UNIDAD: Pa, positivo: tracción     
        PARÁMETROS:
            epsilon_c: deformación unitaria del hormigón tanto por
            uno, positivo: alargamiento'''
        if epsilon_c < 0: #Compresión
            return self.sigma_cc(epsilon_c)
        else: #Tracción o cero
            if self.traccion:
                return self.sigma_ct(epsilon_c)
            else:
                return 0.0
                          
    def area_c(self, a, b):
        '''Devuelve el área bajo la curva tensión-deformación del
        hormigón en valores MEDIOS, desde 'a' hasta 'b' ('a < b').
        UNIDAD: Pa
        PARÁMETROS:
            a: límite inferior en tanto por uno
            b: límite superior en tanto por uno'''
        sigma_cc = self.sigma_cc
        epsilon_cr = self.epsilon_cr
        traccion = self.traccion
        Ec = self.Ec
        fct = self.fct
        n = self.n
        if a < 0 and b <= 0:
            return quad(sigma_cc, a, b)[0]
        elif a < 0:
            if b <= epsilon_cr:
                if traccion:
                    return quad(sigma_cc, a, 0)[0] + Ec*b**2/2
                else:
                    return quad(sigma_cc, a, 0)[0]
            else:
                if traccion:
                    return quad(sigma_cc, a, 0)[0] + \
                    Ec*epsilon_cr**2/2 + \
                    fct*epsilon_cr**n/(1-n)*\
                    (b**(1-n) - epsilon_cr**(1-n))
                else: 
                    return quad(sigma_cc, a, 0)[0]
        elif a < epsilon_cr:
            if traccion:
                if b <= epsilon_cr:
                    return Ec/2*(b**2-a**2)
                else:
                    return Ec/2*(epsilon_cr**2-a**2) + \
                    fct*epsilon_cr**n/(1-n)*\
                    (b**(1-n) - epsilon_cr**(1-n))
            else:
                return 0.0
        else:
            if traccion:
                return fct*epsilon_cr**n/(1-n)*\
                (b**(1-n)-a**(1-n))
            else:
                return 0.0
                         
    def momento_estatico_c(self, a, b):
        '''Devuelve el momento estático respecto al origen en
        valores MEDIOS de la curva del hormigón comprendida entre
        'a' y 'b' ('a' y 'b' son valores extremos de 'epsilon_c').
        UNIDAD: Pa.
        PARÁMETROS:
            a: límite inferior en tanto por uno
            b: límite superior en tanto por uno'''
        def dMC(epsilon_c): #Función auxiliar
            return self.sigma_c(epsilon_c)*epsilon_c
        return quad(dMC, a, b)[0]
                                   
    def abcisas_cc(self, numx=100):
        '''Devuelve las abcisas del diagrama de compresión
        (deformaciones unitarias).
        abcisas_c(numx = 100)
        numx: números de puntos en el eje x'''
        return np.linspace(-self.epsilon_cu1, 0, numx)
        
    def abcisas_ct(self, numx=50):
        '''Devuelve las abcisas del diagrama de tracción
        (deformaciones unitarias).
        abcisas_c(numx = 100)
        numx: números de puntos en el eje x'''
        return np.linspace(0, 10*self.epsilon_cr, numx)
        
    def abcisas_c(self, numT=50, numC=100):
        '''Devuelve las abcisas del diagrama de Hº. Tracción y
        Compresión (deformaciones unitarias)
         numT: puntos del diagrama de tracción
         numC: puntos del diagrama de compresión'''
        return np.append(self.abcisas_cc(numC),
                         self.abcisas_ct(numT)[1:])
        
    def ordenadas_cc(self, Epsilon):
        '''Devuelve las ordenadas del diagrama de compresión del
        hormigón de cálculo para valores de fck comprendidos en
        las clases establecidas en el EC2. Dadas las abcisas para
        los cuales se desea el valor de la compresión.
        ordenadas_c(Epsilon).
        Epsilon: numpy.array con los valores de las deformaciones
        unitarias. Deben ser negativos'''
        sigma_con = 0 * Epsilon
        c = 0 #contador
        for e in Epsilon:
            sigma_con[c] = self.sigma_cc(e) #Tensión en hormigón
            c += 1
        return sigma_con
        
    def ordenadas_ct(self, Epsilon):
        '''Devuelve las ordenadas del diagrama de tracción de 
        álculo. Dadas las abcisas para las cuales se desea el
        valor de la tracción.
        ordenadas_t(Epsilon)
        Epsilon: numpy.array con los valores de las deformaciones
        unitarias. Deben ser positivos'''
        sigma_con = 0 * Epsilon
        c = 0 #contador
        for e in Epsilon:
            sigma_con[c] = self.sigma_ct(e) #Tensión en concreto
            c += 1
        return sigma_con
    
    def ordenadas_c(self, Epsilon):
        '''Devuelve las ordenadas del diagrama de Hº para valores
        fck establecidos en la norma (tensiones de tracción y
        compresión).
        Epsilon: (numpy.array) abcisas dadas'''
        Epsilon_c = Epsilon[Epsilon<0]
        Epsilon_t = Epsilon[Epsilon >= 0]
        return np.append(self.ordenadas_cc(Epsilon_c), \
        self.ordenadas_td(Epsilon_t))


######## LO QUE SIGUE FALTA VERIFICAR
class Hpr(Hormigon):
    '''Hormigón que se comporta según del diagrama Parábola-
    Rectángulo'''

    def __init__(self, n=2, alpha_cc=1.0, alpha_ct=1.0,
                 agregado='basalto', nu=0.2, estado_limite='ELU',
                 fk=25e6, densidad=2400, costo_unit=costo_Ho):
        '''
        Hormigon(fck=25, gamma_c=1.5)        
        
        fck: resistencia característica (Tabla 3.1) (Mpa)
            Para probetas cilíndricas: [12, 16, 20, 25, 30, 35,
            40, 45, 50] MPa
        gamma_c: coeficiente de seguridad (Tabla 2.1N) (adimensional)
        alpha_cc: coeficiente de cansancio a compresión
        alpha_ct: coeficiente de cansancio a tracción
        agregado: coeficiente para el módulo de elasticidad
        dependiente del tipo de agregado (EC2 3.1.3 (2))
        nu: coeficiente de Poisson
        gamma_cE: coef. de seguridad para el módulo de Young
        costo_unit: costo de 1m3 de hormigón en Gs.
        n:  exponente de la parábola
        '''
        Hormigon.__init__(self, alpha_cc, alpha_ct, agregado, nu,
                          estado_limite, fk, densidad, costo_unit)
        self.n = 2 #Tabla 3.1 Exponente de la parábola
       
    def __str__(self):
        datos_H = "Diagrama Parábola-Rectángulo: \n\t fck: " + \
        str(self.fck*1e-6) + " MPa \t\t" + \
        " gamma_c: " + str(self.gamma_c) + "\t\t fcd: " + \
        str(round(self.fcd*1e-6,2)) + " MPa\n\t Ecm = " + \
        str(round(self.Ecm*1e-9,2)) + " GPa \t Ecd = " + \
        str(round(self.Ecd*1e-9,2)) + " GPa\t Ec0 = " + \
        str(round(self.Ec0*1e-9,2)) + " GPa \n\t epsilon_c2 = "+\
        str(self.epsilon_c2*1000) + "\t epsilon_cu2 = " + \
        str(self.epsilon_cu2*1000) + "\t alpha_cc = "+\
        str(self.alpha_cc)
        return datos_H
        
        
    def sigma_cd(self, epsilon_c):
        '''Devuelve la tensión del hormigón en MPa (negativo:
            compresión) dada
        la deformación (negativo: acortamiento), del diagrama
        Parábola-Rectángulo.
        PARÁMETROS:
        epsilon_c: deformación unitaria del hormigón en tanto por
        uno'''
        
        if epsilon_c >= 0:
            return 0
        elif epsilon_c > -self.epsilon_c2: #Parábola
            return -self.fcd*(1-(1-epsilon_c/(-self.epsilon_c2))\
                              **self.n)
        else: #Rectángulo
            return -self.fcd
            
    def abcisas(self, numx=50):
        '''Devuelve las abcisas del diagrama'''
        #deformación del concreto
        return np.linspace(-self.epsilon_cu2, 0, numx)
        
    def ordenadas(self, Epsilon):
        '''Devuelve las ordenadas del diagrama de compresión del
        hormigón para valores de fck comprendidos en las clases
        establecidas en el EC2. Dadas las abcisas para los cuales
        se desea el valor de la compresión.
        ordenadas_c(Epsilon).
        Epsilon: numpy.array con los valores de las deformaciones
        unitarias. Deben ser negativos'''
        sigma_con = 0 * Epsilon
        c = 0 #contador
        for e in Epsilon:
            sigma_con[c] = self.sigma_cm(e) #Tensión en concreto
            c += 1
        return sigma_con
            
    def diagTD(self):
        '''Diagrama Parábola-Rectángulo del hormigón'''
        plt.figure("PR")
        plt.plot(self.abcisas(), self.ordenadas(), linewidth=2)
        plt.title("Diagrama Parábola-Rectángulo del Hormigón\n\
        $f_{{ck}}={0}MPa \qquad \\gamma_c={1}\qquad\\alpha={2}$".\
        format(self.fck, self.gamma_c, self.alpha))
        plt.xlabel("$\\varepsilon_c \, (o/oo)$", size=14)
        plt.ylabel("$\\sigma_{cd} \, (MPa)$", size=14)
        plt.gca().invert_xaxis() #invierte el eje x
        plt.gca().invert_yaxis() #invierte el eje y
        plt.axhline(color='r')
        plt.axvline(color='r')
        plt.grid()


class Hbl(Hormigon):
    '''Hormigón que se comporta según del diagrama Bilineal'''
    def __init__(self, alpha_cc=1.0, alpha_ct=1.0,
                 agregado='basalto', nu=0.2, estado_limite='ELU',
                 fk=25e6, densidad=2400, costo_unit=costo_Ho):
        Hormigon.__init__(self, alpha_cc, alpha_ct, agregado, nu,
                          estado_limite, fk, densidad, costo_unit)
                  
    @property
    def Ecd(self):
        '''Módulo de elasticidad de cálculo en la zona elástica'''
        return self.fcd / self.epsilon_c3
        
    def __str__(self):
        datos_H = "Diagrama Bilineal: \n\tfck: " + \
        str(self.fck*1e-6) + " MPa \t\t" + \
        "gamma_c: " + str(self.gamma_c) + "\t\tfcd: " + \
        str(round(self.fcd*1e-6,2)) + " MPa\n\tEc0 = " + \
        str(round(self.Ec0*1e-9,2)) + " GPa\t\tEcd = " + \
        str(round(self.Ecd*1e-9,2)) + " GPa\t\talpha = " + \
        str(self.alpha) + "\n\tepsilon_c3 = " + \
        str(self.epsilon_c3*1000) + "\tepsilon_cu3 = " + \
        str(self.epsilon_cu3*1000)
        return datos_H
        
    def sigma_cm(self, epsilon_c):
        '''Devuelve la tensión del hormigón en MPa (negativo:
            compresión) dada la deformación (negativo:
                acortamiento), del diagrama bilineal.
        PARÁMETROS:
        epsilon_c: deformación unitaria del hormigón en tanto por
        uno'''
        if epsilon_c >= 0:
            return 0
        elif epsilon_c > self.epsilon_c1: #Línea 1
            return epsilon_c/self.epsilon_c3*self.fcd
        else: #Línea 2 (horizontal)
            return -self.fcd

    def abcisas(self):
        '''Devuelve las abcisas del diagrama'''
        #deformación del concreto
        return np.array([0,-self.epsilon_c3,-self.epsilon_cu3])
        
    def ordenadas(self):
        '''Devuelve las ordenadas del diagrama'''
        return np.array([0,-self.fcd,-self.fcd])
            
    def diagTD(self):
        '''Diagrama bilineal del hormigón'''
        plt.figure("BLH")
        plt.plot(self.abcisas(), self.ordenadas(), linewidth=2)
        plt.title("Diagrama BiLineal del Hormigón\n\
                  $f_{{ck}}={0}MPa$\t\$\\gamma_c={1}$\t\t$\
                  \alpha={2}$".format(self.fck*1e-6, self.gamma_c,
        self.alpha))
        plt.xlabel("$\\varepsilon_c \, (o/oo)$", size=14)
        plt.ylabel("$\\sigma_{cd} \, (MPa)$", size=14)
        plt.gca().invert_xaxis() #invierte el eje x
        plt.gca().invert_yaxis() #invierte el eje y
        plt.axhline(color='r')
        plt.axvline(color='r')
        plt.grid()
    
#Diagramas combinados en un solo gráfico
def diagComb(Xcd, Ycd, Xpr, Ypr, Xbl, Ybl):
    '''Diagramas del hormigón
    Xcd, Ycd: abcisas y ordenadas del diagrama para cargas de
    corta duración
    Xpr, Ypr: abcisas y ordenadas del diagrama parábola-rectángulo
    Xbl, Ybl: abcisas y ordenadas del diagrama bilineal del
    hormigón
    '''
    plt.figure("CH")
    td, pr, bl = plt.plot(Xcd, Ycd, Xpr, Ypr, Xbl, Ybl, 
                          inewidth=2)
    plt.title("Diagramas del Hormigón")
    plt.xlabel("$\\varepsilon_c \, (o/oo)$", size=14)
    plt.ylabel("$\\sigma_{cd} \, (MPa)$", size=14)
    plt.legend((td, pr, bl),("Cargas de corta duración",
               "Parábola-Rectángulo","Bilineal"), loc=7,
            fontsize=12)
    plt.gca().invert_xaxis() #invierte el eje x
    plt.gca().invert_yaxis() #invierte el eje y
    plt.axhline(color='r')
    plt.axvline(color='r')
    plt.grid()


### Está verificado desde aquí, funciona, pero hay que mejorar

class Acero(Material):
    '''Clase Acero según el Eurocódigo Nº 2'''
    def __init__(self, epsilon_uk=0.10, epsilon_su = 0.01, k=1.1,
                 Es=200e9, estado_limite='ELU', fk=500e6,
                 densidad=7850, costo_unit=costo_Ao):
        '''
        Acero(fyk=500, gamma_s=1.15, epsilon_uk=10, k=1.1, Es=200)        
        
        fyk: resistencia característica del acero en Pa
        gamma_s: coeficiente de seguridad del acero (Cuadro 2.3)
        (adimensional)
        epsilon_uk: deformación última del acero en %
        epsilon_su: deformación última permitida del acero en %,
            en el interior del hormigón.
        k: fs/fy Carga de rotura / límite elástico (adimensional)
        Es: módulo de elasticidad en Pa
        densidad = por defecto 7850 kg/m3 3.2.7(3)
        '''
        Material.__init__(self, estado_limite, fk, densidad,
                          costo_unit)
        self.fk = fk
        self.epsilon_uk = epsilon_uk
        self.epsilon_su = epsilon_su
        self.k = k
        self.Es = Es
        self.costo_unit = costo_unit #[Gs/kg] Costo 1kg de acero
        
    @property
    def fk(self):
        return self.__fk
    @fk.setter
    def fk(self, fk):
        assert 400e6 <= fk <= 600e6, \
        "Debe estar entre 400MPa y 600MPa (EC2 Tabla C.1)"
        self.__fk = fk

    @property
    def fyk(self):
        '''Resistencia característica del acero [Pa]'''
        return self.fk
        
    @property
    def gamma_s(self):
        '''Coeficiente parcial de seguridad para el acero'''
        if self.estado_limite == 'ELU':
            return 1.15
        else:
            return 1.00
    
    @property
    def fym(self):
        '''Resistencia media del acero en Pa'''
        return self.fyk + 10e6

    @property
    def fyd(self):
        '''Resistencia de cálculo del acero en Pa'''
        return self.fyk / self.gamma_s
    
    @property
    def fy(self):
        if self.estado_limite == 'VMS': #Valores medios
            return self.fym
        else:
            return self.fyd
        
    @property
    def epsilon_y(self):
        '''Deformación unitaria en el límite elástico en tanto por
        uno'''
        return self.fy/self.Es
        
    @property
    def epsilon_ud(self):
        '''Deformación unitaria última de cálculo'''
        return 0.9 * self.epsilon_uk
        
    @property
    def fyud(self):
        '''Tensión máxima de cálculo en rotura'''
        return self.fyd * self.k #Tensión máxima de rotura
        
    def __str__(self):
        datos_acero = "Acero de armar:\n\tfyk = " + \
        str(self.fyk*1e-6) + " MPa\tgamma_s = " + \
        str(self.gamma_s) + "\t\tfym = " + \
        str(round(self.fy*1e-6,2)) + " MPa\t\tfs/fy= " +\
        str(self.k) + "\n\tEs = " + str(self.Es*1e-9) +\
        " GPa\tepsilon_uk = " + str(self.epsilon_uk*100) + "%" +\
        "\tepsilon_ym = " + str(self.epsilon_y*100) +\
        "%" "\tepsilon_yd = "
        return datos_acero
        
    def area(self, d):
        '''Área en cm2 de las secciones transversales de varillas
        area(d)
        d: diámetro de la varilla en m
        '''
        return np.pi*(d)**2/4
        
    def sigma_s(self, epsilon_s):
        '''Devuelve la tensión de cálculo del acero en Pa a partir
        de la deformación unitaria introducida en tanto por uno'''
        Es = self.Es
        fy = self.fy
        epsilon_y = self.epsilon_y
        epsilon_uk = self.epsilon_uk
        epsilon_ud = self.epsilon_ud
        epsilon_su = self.epsilon_su
        k = self.k
        e_s = abs(epsilon_s)
        T1 = Es * e_s #Pa
        T2 = fy*(1+(e_s-epsilon_y)/(epsilon_uk-epsilon_y)*(k-1))
        if epsilon_s < -epsilon_su:
            return 0.0
        elif epsilon_s <= -epsilon_y:
            return -T2
        elif epsilon_s < 0:
            return -T1
        elif epsilon_s <= epsilon_y:
            return T1
        elif epsilon_s <= epsilon_ud:
            return T2
        else:
            return 0.0
                  
    def diagTD(self):
        '''Diagrama trilineal del acero'''
        plt.figure("TDA")
        x1 = self.epsilon_uk*1e3
        x2 = self.epsilon_y*1e3
        X_s = [-x1, -x2, x2, x1]
        Y_s = [-self.fyud, -self.fy, self.fy, self.fyud]
        plt.plot(X_s, Y_s, linewidth=2)
        plt.title("Diagrama TriLineal del Acero\n$f_{{yk}}={0}\
                  MPa$\t$\\gamma_s={1}$\t\t$f_s/f_y={2}$"\
                  .format(self.fyk, self.gamma_s,
        self.k), size=16)
        plt.xlabel("$\\varepsilon_s \, (o/oo)$", size=14)
        plt.ylabel("$\\sigma_{s} \, (MPa)$", size=14)
        plt.ylim(-1.05*self.fyud, 1.05*self.fyud)
        plt.axhline(color='r')
        plt.axvline(color='r')
        plt.grid()
        plt.savefig('diagAcero.png', dpi=300)