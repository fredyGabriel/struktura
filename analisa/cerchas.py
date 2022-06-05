# Análisis estático y dinámico de reticulados 1D, 2D o 3D
#
# @author: Fredy Gabriel Ramírez Villanueva
# Inicio de código: 06 de mayo de 2022
# Realese 1: 18 de mayo de 2022
# Control git: 01 de junio de de 2022
# Versión actual: 0.1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
from scipy.linalg import eigh

import dsm  # Librería casera "Direct Stiffness Method"

# Versores del sistema global de coordenadas
IX = np.array([1, 0, 0])
IY = np.array([0, 1, 0])
IZ = np.array([0, 0, 1])
VERSORES = np.array([IX, IY, IZ])


@dataclass
class Nudo:
    """Nudo de reticulado 2D o 3D.

    Attributes:
        id: número asignado
        coord: Coordenadas de los nudos. Puede contener 1, 2 o 3 valores según
            la dimensión considerada.

        restr: (opcional) Restricciones en dirección a los ejes. Si no
                es dado, se sopondrá que el nudo no está restringido.
                1: restringido; 0: libre
                - Ejemplo en 3D:
                    (0, 1, 1): libre en x, restringido en y y z
                - Ejemplo en 2D
                    (1, 0): restringido en x, libre en y
                - Ejemplo en 1D
                    1: restringido en x

        enum_gdl: (opcional) Enumeración correspondiente de los grados de
                            libertad.

    """
    id: int  # Número asignado al nudo
    coord: tuple[float]                 # Coordenadas
    restr: tuple[int] = field(init=False)  # Restricciones
    cargas: tuple[float] = field(init=False)  # Cargas en los nudos
    enum_gdl: Optional[tuple[int]] = None  # Enumeración de los gdl's

    def __post_init__(self):
        dim = len(self.coord)
        self.restr = tuple([0]*dim)
        self.cargas = tuple([0]*dim)

    @property
    def dim(self) -> int:
        """Dimensión vectorial del nudo"""
        return len(self.coord)

    def desplaz(self, D: np.ndarray) -> np.ndarray:
        """Desplazamiento del nudo.

        Args:
            D: Desplazamientos nodales de la estructura en los grados de
                libertad

        Returns:
            Vector dim x 1. Tipo np.array (dim,)
        """
        gdl = self.enum_gdl  # Grados de libertad
        n = len(D)  # Dimensión del vector D

        d = np.zeros(self.dim)  # Inicialización

        for i, g in enumerate(gdl):
            if g <= n:
                d[i] = D[g - 1]

        return d

    def posicion(self, D: np.ndarray, amp=1.0) -> np.ndarray:
        """Posición del nudo, antes o después de la aplicación de las cargas.

        Args:
            D: Desplazamientos nodales de la estructura en los grados de
                libertad
            amp: factor amplificador de desplazamientos

        Returns:
            Coordenadas de la nueva posición del nudo.
        """
        u = amp*self.desplaz(D)  # Desplazamientos
        c = np.array(self.coord)  # Coordenadas iniciales

        return c + u

    def dibuja2D(self, num=True):
        """Dibuja el nudo.

        Args:
            num: Para incluir o no el número identificador del nudo
        """
        X, Y = self.coord
        r = sum(self.restr)

        if r == 0:  # Nudo libre
            c1 = 'r'
        elif r == 1:  # Nudo con una restricción
            c1 = 'y'
        else:  # Nudo retringido
            c1 = 'k'

        plt.scatter(X, Y, c=c1)

        if num:
            plt.text(X, Y, str(self.id + 1))
            
    def dibuja3D(self, ax, zdir='z'):
        X, Y, Z = self.coord
        ax.scatter(X, Y, Z, zdir=zdir)
        


@dataclass
class Material:
    """Propiedades del material.

    Hipótesis: material isotrópico, homogéneo, elástico, lineal.

    El módulo de elasticidad transversal (G) es calculado con los atributos
    dados.

    Attributes:
        densidad: densidad del material
        elast_long: módulo de elasticidad longitudinal (módulo de Young)
        poisson: (opcional) módulo de Poisson
    """

    densidad: float  # Densidad del material
    elast_long: float  # Módulo de elasticidad longitudinal (Young)
    poisson: Optional[float] = None  # Módulo de Poisson
    _elast_transv: float = field(init=False, default=None)

    def __post_init__(self):
        """Inicialización de la elasticidad transversal G."""
        E = self.elast_long
        nu = self.poisson
        if nu is not None:
            self._elast_transv = E/2/(1 + nu)

    @property
    def elast_transv(self) -> float:
        return self._elast_transv


@dataclass
class SeccionTransversal:
    """Propiedades de la sección transversal de una barra recta.

    La sección transversal debe ser constante a lo largo del elemento.
    """

    area: float  # Área de la sección transversal
    inercia_y: Optional[float] = None  # Inercia alrededor del eje y
    inercia_z: Optional[float] = None  # Inercia alrededor del eje z
    area_cortante_y: Optional[float] = None  # Área efectiva a cortante en y
    area_cortante_z: Optional[float] = None  # Área efectiva a cortante en z

    def inercia_polar(self) -> float:
        """Momento polar de inercia."""
        Iy = self.inercia_y
        Iz = self.inercia_z

        assert Iy is not None | Iz is not None, "Faltan datos de inercia"

        return Iy + Iz


@dataclass
class Barra2F:
    """Barra de dos fuerzas para reticulado plano o espacial.

    Adoptar el critero de que el nudo inicial tenga menor enumeración que el
    nudo final, esto es: Ni.N < Nf.N

    Sistema de coordenadas locales:
        x: eje de la barra posicionada horizontalmente
        y: eje vertical
        z: eje perpendicular a x e y

    Attributes:
        nudo_inicial: nudo inicial
        nudo_final: nudo final
        material: datos del material, definidos en la clase 'Material'
        prop_seccion: propiedades de la sección transversal, clase
                    'SeccionTransversal'
    """
    nudo_inicial: Nudo                # Nudo inicial
    nudo_final: Nudo                  # Nudo final
    material: Material                # Propiedades del material
    prop_seccion: SeccionTransversal  # Propiedades geométricas de la sección
    _tension: float = 0.0  # Tensión en la barra
    _def_unitaria: float = 0.0  # Deformación unitaria

    @property
    def dim(self) -> int:
        """Dimensión vectorial de la barra (1D, 2D o 3D)"""
        di = self.nudo_inicial.dim
        df = self.nudo_final.dim
        assert di == df, "Verificar dimensión de los nudos"
        return di

    @property
    def longitud(self) -> float:
        """Longitud de la barra."""
        ci = np.array(self.nudo_inicial.coord)  # Coordenadas del nudo inicial
        cf = np.array(self.nudo_final.coord)    # Coordenadas del nudo final
        return np.linalg.norm(ci - cf)

    @property
    def gdl_barra(self) -> list:
        """Grados de libertad de los nudos extremos de la barra."""
        gdl_i = list(self.nudo_inicial.enum_gdl)  # gdl del nudo inicial
        gdl_f = list(self.nudo_final.enum_gdl)  # gdl del nudo final
        return gdl_i + gdl_f

    @property
    def rigidez_local(self) -> np.ndarray:
        """Matriz de rigidez en coordenadas locales"""
        E = self.material.elast_long
        A = self.prop_seccion.area
        L = self.longitud

        return dsm.rigidez_b2f(L, E, A)

    @property
    def masa_local(self) -> np.ndarray:
        """Matriz de masa en coordenadas locales"""
        L = self.longitud
        A = self.prop_seccion.area
        rho = self.material.densidad

        return dsm.masa_b2f(L, A, rho, self.dim)

    @property
    def matriz_cos_dirs(self) -> np.ndarray:
        """Matriz de cosenos directores respecto de los ejes globales.
        
        Las filas contienen los cosenos directores de los ejes locales
        respecto a los globales.
        
        En 1D se supone que el eje local x coincide con el global X

        Basado en Aslam Kassimalli, Matrix Analysis of Structures. Cap. 8
        
        Returns:
            - 1D: 1
            - 2D: Matriz de 2 x 2
            - 3D: Matriz de 3 x 3
        """
        dim = self.dim

        if dim == 1:
            return 1

        else:  # Para 2D o 3D
            ci = np.array(self.nudo_inicial.coord)  # Coords del nudo inicial
            cf = np.array(self.nudo_final.coord)  # Coords del nudo final
            L = self.longitud
            v = cf - ci  # Barra como vector

            # Cosenos directores del eje local x
            ix = np.array([np.dot(v, VERSORES[i][:dim]) for i in range(dim)])/L

            if dim == 2:  # 2D
                c, s = ix[0], ix[1]
                return np.array([
                    [c, s],
                    [-s, c]
                ])

            else:  # 3D
                # Cosenos directores del eje local z
                z = np.cross(ix, IY)  # Kassimalli (8.59)
                nz = np.linalg.norm(z) # Norma de z
                if nz > 0.0: # Si ix no es paralelo IY
                    iz = z / nz  # Kassimalli (8.60)
                else:
                    iz = IZ # Kassimalli p477 2do párrafo.

                # Cosenos directores del eje local y
                iy = np.cross(iz, ix)  # Kassimalli (8.61)
                return np.array([ix, iy, iz])  # Kassimalli (8.62)

    @property
    def transf_coord_rigidez(self) -> np.ndarray:
        """Matriz de transformación de coordenadas globales a locales.
        
        Returns:
            Matriz de 2 x 2dim, en donde dim puede ser 1, 2 o 3.
        """

        ix = self.matriz_cos_dirs[0]  # Cosenos directores del eje local x

        return dsm.transf_coord_b2f(ix)  # Matriz de transf. de coord.

    @property
    def transf_coord_masas(self) -> np.ndarray:
        """Matriz de transformación globales a locales, para masas."""

        # Matriz de cos directores de los ejes locales respecto a los globales
        lam = self.matriz_cos_dirs

        return dsm.transf_coord_m2f(lam)  # Matriz de transf. de coordenadas

    @property
    def rigidez_global(self) -> np.ndarray:
        """Matriz de rigidez en coordenadas globales"""

        T = self.transf_coord_rigidez  # Matriz de transf. de coord.
        ke = self.rigidez_local  # Matriz de rigidez en coordenadas locales.

        K = T.T @ ke @ T  # Cálculo de la matriz de rigidez en coord. globales
        return K

    @property
    def masa_global(self) -> np.ndarray:
        """Matriz de masa en coordenadas globales"""
        T = self.transf_coord_masas  # Matriz de transf. de coord.
        me = self.masa_local  # Matriz de masa en coordenadas locales

        M = T.T @ me @ T  # Cálculo de la matriz de masa en coord. globales
        return M

    def desplaz_global(self, D: np.ndarray) -> np.ndarray:
        """Desplazamientos nodales de la barra en coordenadas globales.

        Args:
            D: Vector de desplazamientos nodales de la estructura en los grados
            de libertad.

        Returns:
            Vector 2dim x 1 de desplazamientos nodales. Tipo np.ndarray (2dim,)
        """
        ni = self.nudo_inicial
        nf = self.nudo_final
        di = ni.desplaz(D)  # Desplazamientos del nudo inicial
        df = nf.desplaz(D)  # Desplazamientos del nudo final
        dim = self.dim

        U = np.zeros(2*dim)  # Inicialización
        U[:dim] = di
        U[dim:] = df

        return U

    def fuerza_y_desplaz_local(self, D: np.ndarray) -> tuple:
        """Fuerzas y deformaciones nodales en coordenadas locales.

        Args:
            desplaz_nodales: desplazamientos nodales en coordenadas globales.

        Returns:
            Vector de fuerzas y vector de deformaciones nodales en coordenadas
            locales, ambos de orden 2x1. Tipo np.ndarray (2,)
        """
        U = self.desplaz_global(D)  # Desplaz. nodales en coordenadas globales
        T = self.transf_coord_rigidez  # Matriz de transf. de coordenadas
        u = T @ U  # Desplazamientos nodales en coordenadas locales
        Q = self.rigidez_local @ u  # Fuerzas nodales en coordenadas locales

        return Q, u

    def tension_y_defunit(self, D: np.ndarray) -> tuple:
        """Tensión y deformación unitaria de la barra.

        Args:
            D: Vector de desplazamientos nodales de la estructura.

        Returns:
            El valor de la tensión normal (sigma) y el valor del desplazamiento
            unitario (epsilon).
        """
        Q, u = self.fuerza_y_desplaz_local(D)
        N = Q[1]  # Fuerza normal
        delta = u[1]  # Desplazamiento
        sigma = N / self.prop_seccion.area  # Tensión
        epsilon = delta / self.longitud  # Deformación unitaria

        return sigma, epsilon

    def poner_tension(self, sigma):
        self._tension = sigma

    def ver_tension(self):
        return self._tension

    def poner_def_unit(self, epsilon):
        self._def_unitaria = epsilon

    def ver_def_unit(self):
        return self._def_unitaria

    def dibuja2D(self, espesor_area=True) -> None:
        """Dibuja la barra de 2D.

        Args:
            espesor_area: Considerar o no las áreas para dibujuar
            proporcionalmente a ellas el espesor de las líneas.
        """
        if espesor_area:
            lw = self.prop_seccion.area * 1000
        else:
            lw = 1

        x0, y0 = self.nudo_inicial.coord  # Coordenadas del nudo inicial
        xf, yf = self.nudo_final.coord  # Coordenadas del nudo final
        X = [x0, xf]
        Y = [y0, yf]
        sigma = self.ver_tension()
        if sigma > 0:  # Tracción
            c = 'b'
        elif sigma < 0:  # Compresión
            c = 'r'
        else:
            c = '0.5'  # Gris
        plt.plot(X, Y, color=c, linewidth=lw)
        
    def dibuja3D(self, ax, zdir='z', espesor_area = True):
        """Dibuja la barra en 3D.
        
        Args:
            ax: axes object
            zdir: Qué dirección usar como z ('x', 'y', 'z', '-x', '-y', '-z')
            espesor_area: modifica el espesor de las líneas proporcionalmente
                al área de la sección transversal de la barra correspondiente.
        """
        
        if espesor_area:
            lw = self.prop_seccion.area * 1000
        else:
            lw = 1
        
        X0, Y0, Z0 = self.nudo_inicial.coord
        Xf, Yf, Zf = self.nudo_final.coord
        
        XX = [X0, Xf]
        YY = [Y0, Yf]
        ZZ = [Z0, Zf]
        
        sigma = self.ver_tension()
        if sigma > 0:  # Tracción
            c = 'b'
        elif sigma < 0:  # Compresión
            c = 'r'
        else:
            c = '0.5'  # Gris
        
        ax.plot(XX, YY, ZZ, zdir= zdir, color=c, linewidth=lw)

    def dibuja2D_deform(self, D: np.ndarray, amp=1, colorear=False):
        area = self.prop_seccion.area
        Nid = self.nudo_inicial.posicion(D, amp)  # Nudo inicial desplazado
        Nfd = self.nudo_final.posicion(D, amp)  # Nudo final desplazado
        X = [Nid[0], Nfd[0]]
        Y = [Nid[1], Nfd[1]]

        if colorear:
            sigma = self.tension_y_defunit(D)[0]
            if sigma > 0:  # Tracción
                color = 'b'
            elif sigma < 0:  # Compresión
                color = 'r'
            else:
                color = 'k'
        else:
            color = 'g'

        plt.plot(X, Y, color=color, linewidth=area*1000)
        
    def dibuja3D_deform(self, ax, zdir, D: np.ndarray, amp=1, colorear=False):
        area = self.prop_seccion.area
        Nid = self.nudo_inicial.posicion(D, amp)  # Nudo inicial desplazado
        Nfd = self.nudo_final.posicion(D, amp)  # Nudo final desplazado
        X = [Nid[0], Nfd[0]]
        Y = [Nid[1], Nfd[1]]
        Z = [Nid[2], Nfd[2]]

        if colorear:
            sigma = self.tension_y_defunit(D)[0]
            if sigma > 0:  # Tracción
                color = 'b'
            elif sigma < 0:  # Compresión
                color = 'r'
            else:
                color = 'k'
        else:
            color = 'g'

        ax.plot(X, Y, Z, zdir=zdir, color=color, linewidth=area*1000)

##################################
## RESOLUCIÓN DE UNA ESTRUCTURA ##
##################################


def config(coords: dict, restricciones: dict, cargas_nodales: dict,
           elementos: list) -> dict:
    """Configuración de estructura reticulada

    Args:
        coordenadas: diccionario de coordenadas de los nudos
        retricciones: diccionario con nudos restringidos y sus restricciones
        elementos: lista con datos de las barras

    Returns:
        Diccionario con:
            dim: dimension considerada de la estructura (1, 2 o 3)
            nudos: lista de nudos de tipo adr.Nudo
            barras: lista de barras de tipo adr.Barra2F
            Ngdl: número de grados de libertad de la estructura
    """
    coordenadas = list(coords.values())
    dim = len(coordenadas[0])  # Dimensión de los nudos
    assert dim < 4, 'Las coordenadas de un nudo no pueden ser más de 3 valores'

    Nnudos = len(coordenadas)  # Número de nudos
    ngdr = sum([sum(v) for v in restricciones.values()])  # Número de gdr
    ngdl = dim*Nnudos - ngdr  # Número de grados de libertad

    # Generación de nudos
    nudos = [Nudo(i, c) for i, c in enumerate(coordenadas)]

    # Asignación de restricciones
    for rr in restricciones.keys():  # Recorre el diccionario de restricciones
        nudos[rr - 1].restr = restricciones[rr]

    # Asignación de cargas
    for qq in cargas_nodales.keys():  # Recorre el diccionario de cargas
        nudos[qq - 1].cargas = cargas_nodales[qq]

    # Enumeración de grados de libertad
    gdl_lista = [[0]*dim for _ in range(Nnudos)]  # Para guardar los gld's
    contador = 0  # Contador de grados de libertad

    # Asignación de los primeros ngdl grados de libertad
    for ii, nudo in enumerate(nudos):  # Recorre los nudos
        if nudo.restr[0] == 0:  # Si no hay restricción en X
            contador += 1  # Se incrementa contador
            gdl_lista[ii][0] = contador  # Se guarda el valor del contador
        if dim == 2 or dim == 3:  # Si existe Y
            if nudo.restr[1] == 0:  # Si no hay restricción en Y
                contador += 1  # Se incrementa contador
                gdl_lista[ii][1] = contador  # Se guarda el valor del contador
        if dim == 3:  # Si existe Z
            if nudo.restr[2] == 0:  # Si no hay restricción en Z
                contador += 1  # Se incrementa contador
                gdl_lista[ii][2] = contador  # Se guarda el valor del contador
        nudo.enum_gdl = tuple(gdl_lista[ii])  # Se agrega como atributo al nudo

    # Asignación de los grados de restricción
    for ii, nudo in enumerate(nudos):  # Recorre los nudos nuevamente
        if nudo.restr[0] == 1:  # Si hay restricción en X
            contador += 1  # Se incrementa contador
            gdl_lista[ii][0] = contador  # Se guarda el valor del contador
        if dim == 2 or dim == 3:  # Si existe Y
            if nudo.restr[1] == 1:  # Si hay restricción en Y
                contador += 1  # Se incrementa contador
                gdl_lista[ii][1] = contador  # Se guarda el valor del contador
        if dim == 3:  # Si existe Z
            if nudo.restr[2] == 1:  # Si hay restricción en Z
                contador += 1  # Se incrementa contador
                gdl_lista[ii][2] = contador  # Se guarda el valor del contador
        nudo.enum_gdl = tuple(gdl_lista[ii])  # Se agrega como atributo al nudo

    # Generación de las barras
    barras = [Barra2F(nudos[e[0]-1], nudos[e[1]-1], e[2], e[3]) for e in
              elementos]

    return {'Dimension': dim, 'Nudos': nudos, 'Barras': barras, 'Ngdl': ngdl}


def matrices_globales(propiedades) -> tuple[np.ndarray]:
    """Ensamble de las matrices de masa, rigidez y vector de fuerzas.

    Args:
        Diccionario con propiedades de la estructura

    Returns:
        - Matriz de masa ncg x ncg. Tipo numpy.ndarray (ncg, ncg)
        - Matriz de rigidez ncg x ncg. Tipo numpy.ndarray (ncg, ncg)
        - Vector de fuerzas ncg x 1. Tipo numpy.ndarray (ncg,)
        Siendo ncg: número total de coordenadas globales de la estructura
                    (incluyendo los apoyos)
    """
    # Datos
    dim = propiedades['Dimension']
    nudos = propiedades['Nudos']  # Lista de datos de tipo Nudo
    barras = propiedades['Barras']

    Nnudos = len(nudos)  # Número de nudos
    ngda = dim*Nnudos  # Número de grados de acción

    # Inicialización de las matrices
    MG = np.zeros((ngda, ngda))   # Masa global
    KG = np.zeros((ngda, ngda))   # Rigidez global
    FG = np.zeros(ngda)          # Vector de fuerzas nodales

    # Ensamble de matrices de masa y rigidez
    for barra in barras:  # Recorre todas las barras
        Me = barra.masa_global  # Matriz de masa del elemento en c. globales
        Ke = barra.rigidez_global  # Matriz de rigidez del elemento c. globales
        # Números de gdl asignados a los nudos de la bar.
        gdl = barra.gdl_barra

        # Truco
        B = np.zeros((2*dim, ngda))
        for i in range(2*dim):
            B[i, gdl[i] - 1] = 1.0

        # Se agrega la colaboración de cada barra
        MG += B.T @ Me @ B
        KG += B.T @ Ke @ B

    # Ensamble del vector de fuerzas
    for nudo in nudos:  # Recorre todos los nudos
        g = np.array(nudo.enum_gdl)  # Grados de libertad del nudo
        indices = g - 1
        P = np.array(nudo.cargas)  # Lista con cargas en los nudos
        FG[indices] = P

    return MG, KG, FG


def matrices_gdl(MG, KG, FG, ngdl) -> tuple[np.ndarray]:
    """Matrices en los grados de libertad con coordenadas globales."""

    M = MG[:ngdl, :ngdl]
    K = KG[:ngdl, :ngdl]
    F = FG[:ngdl]

    return M, K, F

def norm_uno(modos):
    """Normaliza autovectores tal que la componente máxima sea igual a uno.
    
    Args:
        modos: matriz de vectores característicos
        
    Returns:
        Matriz de vectores propios normalizados con este criterio.
    """
    # Que cada autovector tenga su máxima componente igual a 1
    max_abs = np.abs(np.max(modos, axis=0))  # Valores absolutos de máximos
    min_abs = np.abs(np.min(modos, axis=0))  # Valores absolutos de mínimos
    comp = max_abs > min_abs  # comparación de valores absolutos
    
    S = modos.T*0 # Modos normalizados en las filas
    for i, v in enumerate(modos.T):
        if comp[i]:
            S[i] = v / max_abs[i]
        else:
            S[i] = v / min_abs[i]
    return S.T


def desplaz_t(PhiP: np.ndarray, wf: float, wn: np.ndarray, modos: np.ndarray,
              t: float) -> np.ndarray:
    """Desplazamientos nodales en función del tiempo.

    Devuelve el vector de desplazamientos nodales u(t). Se supone condiciones 
    iniciales homogéneas u(0)=0, du/dt(0) = 0

    Ecuaciones según: Anil K. Chopra "Dinámica de estructuras" 4ta. Ed.

    Args:
        PhiP: Amplitudes de las fuerzas en los gdls de la estructura en
            coordenadas modales
        wf: frecuencia angular de vibración forzada (de las fuerzas externas)
        wn: vector de frecuencias angulares naturales
        modos: matriz de autovectores de la estructura normalizados con la
                matriz de masas.
        t: tiempo (segundos).

    Returns:
        Vector de desplazamientos en el tiempo t.
    """
    q = [PhiP[i]/(w**2 - wf**2)*(np.sin(wf*t) - wf/w*np.sin(w*t)) for i, w in
         enumerate(wn)]
    q = np.array(q)
    u = modos @ q

    return u


def desplaz_lapso(P: np.ndarray, ff: float, w2: np.ndarray, modos: np.ndarray,
                  lapso: float, mps=1.0) -> np.ndarray:
    """Desplazamientos de los gdls de la estructura en el lapso 0 a T.

    Calcula desplazamientos cada 1/mps segundos durante el tiempo T.

    La fila 't' contiene desplazamientos de todos los nudos en el tiempo 't'
    La columna 'g' contiene los desplaz. del gdl 'g' en el lapso 0 a T.

    Args:
        P: Amplitudes de las fuerzas en los gdls de la estructura
        ff: frecuencia de vibración forzada (frec. de las fuerzas externas)
        w2: vector de autovalores de la estructura (cuadrados de las
            frecuencias angulares)
        modos: matriz de autovectores de la estructura normalizados con la
                matriz de masas.
        lapso: período de análisis (segundos).
        mps: muestreo por segundo. Cantidad de veces por segundo que se hará
            el cálculo de los desplazamientos.

    Returns:
        Matriz (int(mps*T), ngdl) de desplazamientos de los nudos de la
        estructura en el lapso de tiempo t0 = 0, tf = T.
    """
    PhiP = modos @ P  # Fuerzas en coordenadas modales
    wf = 2*np.pi*ff  # Frecuencia angular forzada
    wn = np.sqrt(w2)  # Vector de frecuencias angulares naturales

    tiempos = np.linspace(0, lapso, int(mps*lapso))  # Valores de tiempo
    m = len(tiempos)  # Número de filas = cantidad de tiempos analizados
    n = len(P)  # Número de columnas = número de grados de libertad
    UU = np.zeros((m, n))  # Inicialización de la matriz de desplazamientos
    for i, t in enumerate(tiempos):
        u = desplaz_t(PhiP, wf, wn, modos, t)
        UU[i] = u

    return UU


def desplaz_minymax(P: np.ndarray, ff: float, w2: np.ndarray, modos: np.ndarray,
                    lapso=10, mps=1.0) -> np.ndarray:
    """Envolvente de desplazamientos dinámicos nodales (min y max).

    Obs: no son concomitantes.

    Args:
        P: Amplitudes de las fuerzas en los gdls de la estructura
        ff: frecuencia de vibración forzada (frec. de las fuerzas externas)
        avals: vector de autovalores de la estructura
                (cuadrados de las frecuencias angulares)
        avecs: matriz de autovectores de la estructura normalizados con la
                matriz de masas.
        T: período de análisis (segundos).

    Returns:
        Lista con valores mínimos y máximos de desplazamientos en los grados
        de libertad de la estructura.
    """
    UU = desplaz_lapso(P, ff, w2, modos, lapso, mps)
    maximos = np.max(UU, axis=0)
    minimos = np.min(UU, axis=0)

    return minimos, maximos


def obtener_fuerzas(propiedades: dict, D: np.ndarray) -> np.ndarray:
    """Devuelve las fuerzas normales en las barras.

        Args:
        props: Obtenido de config()
        D: vector de desplazamientos nodales en los grados de libertad

    Returns:
        Vector de fuerzas en las barras.
    """
    barras = propiedades['Barras']
    F = np.array([barra.fuerza_y_desplaz_local(D)[0][1] for barra in barras])

    return F


def obtener_tensiones(propiedades: dict, D: np.ndarray) -> np.ndarray:
    """Devuelve las tensiones en las barras."""
    barras = propiedades['Barras']

    tensiones = np.zeros(len(barras))
    for i, b in enumerate(barras):
        Ts, _ = b.tension_y_defunit(D)
        tensiones[i] = Ts

    return tensiones


def poner_tensiones(propiedades: dict, tensiones: list) -> None:
    """Agrega la tensión correspondiente a cada barra.

    Args:
        propiedades: Obtenido de config()
        tensiones: lista o array de tensiones de las barras
    """
    barras = propiedades['Barras']
    for i, b in enumerate(barras):
        b.poner_tension(tensiones[i])


def tensiones_minymax(propiedades: dict, P: np.ndarray, ff: float,
                      w2: np.ndarray, modos: np.ndarray, lapso=10.0,
                      mps=1.0) -> tuple[np.ndarray]:
    """Envolvente de tensiones dinámicas en las barras (mínimos y máximos)

    Obs: no son concomitantes.

    Args:
        propiedades: Obtenido de config()
        P: Amplitudes de las fuerzas en los gdls de la estructura
        ff: frecuencia de vibración forzada (frec. de las fuerzas externas)
        w2: vector de autovalores de la estructura
            (cuadrados de las frecuencias angulares)
        modos: matriz de autovectores de la estructura normalizados con la
                matriz de masas.
        T: período de análisis (segundos).

    Returns:
        Tupla con valores mínimos y máximos de las tensiones.
    """
    UU = desplaz_lapso(P, ff, w2, modos, lapso,
                       mps)  # Desplaz. en diferentes ts
    nb = len(propiedades['Barras'])  # Número de barras
    nt = int(mps*lapso)  # Número de tiempos
    FF = np.zeros((nt, nb))

    for i, U in enumerate(UU):  # Recorre las deformaciones
        FF[i] = obtener_fuerzas(propiedades, U)  # Fuerzas normales en barras
    Fmin = np.min(FF, axis=0)  # Fuerzas mínimas en las barras
    Fmax = np.max(FF, axis=0)  # Fuerzas máximas en las barras

    barras = propiedades['Barras']
    areas = np.array([b.prop_seccion.area for b in barras])
    Tmin = Fmin / areas
    Tmax = Fmax / areas

    return Tmin, Tmax


def obtener_reacciones(KG, D, ngdl: int) -> np.ndarray:
    """Devuelve las reacciones en los grados de restricción"""
    K10 = KG[ngdl:, :ngdl]
    R = K10 @ D
    return R


def reacciones_minymax(KG: np.ndarray, ngdr: int, P, ff, w2, modos, lapso=10.0,
                       mps=1.0) -> np.ndarray:
    """Envolvente de reacciones (mínimos y máximos).

    Args:
        KG: matriz de rigidez global
        ngdl: número de grados de libertad
        T: período de tiempo considerado desde t = 0.
        P: Amplitudes de las fuerzas en los gdls de la estructura
        ff: frecuencia de vibración forzada (frec. de las fuerzas externas)
        w2: vector de autovalores de la estructura
        modos: matriz de autovectores de la estructura
        mps: muestreo por segundo. Cantidad de veces por segundo que se 
        calcularán los desplazamientos.

    Returns:
        Tupla con valores de reacciones mínimos y máximos.
    """
    ngdl = len(P)  # Número de grados de libertad
    UU = desplaz_lapso(P, ff, w2, modos, lapso, mps)  # Desplazamientos
    nt = UU.shape[0]  # Número de tiempos
    RR = np.zeros((nt, ngdr))  # Inicialización

    for i, U in enumerate(UU):
        RR[i] = obtener_reacciones(KG, U, ngdl)

    reacciones_minimas = np.min(RR, axis=0)
    reacciones_maximas = np.max(RR, axis=0)

    return reacciones_minimas, reacciones_maximas


def respuesta_estatica(D: np.ndarray, fuerzas: np.ndarray,
                       tensiones: np.ndarray, reacciones: np.ndarray):

    Ds = np.round(D*1000, 2)  # Desplazamientos en mm
    Ns = np.round(fuerzas*1e-3, 2)  # Fuerzas normales en kN
    Ts = np.round(tensiones*1e-6, 2)  # Tensiones en MPa
    Rs = np.round(reacciones*1e-3, 2)  # Reacciones en kN

    print('RESPUESTA ESTATICA')
    print('- Desplazamientos nodales (mm):', Ds)
    print('- Fuerzas normales (kN):', Ns)
    print('- Tensiones normales (MPa):', Ts)
    print('- Reacciones (kN):', Rs)
    print()


def respuesta_dinamica(frecuencias: np.ndarray, Uenv: tuple[np.ndarray],
                       Tenv: tuple[np.ndarray], Renv: tuple[np.ndarray],
                       lapso: float):
    """Resultados del análisis dinámico.

    Los valores no son concomitantes, es decir no aparecen al mismo tiempo.

    Args:
        frecuencias: frecuencias naturales de la estructura
        Uenv: Envolvente de desplazamientos (mín y máx) de los glds
        Fenv: Fuerzas normales máximas en las barras
        Tenv: Envolvente de tensiones
        Renv: Envolvente de reacciones
        T: período de tiempo analizado, desde t=0s

    Returns:
        Imprime los resultados
    """
    freq = np.round(frecuencias, 2)  # Frecuencias
    Umin = np.round(Uenv[0]*1000, 2)  # Desplazamientos mínimos en mm
    Umax = np.round(Uenv[1]*1000, 2)  # Desplazamientos máximos en mm
    Tmin = np.round(Tenv[0]*1e-6, 2)  # Tensiones mínimas en MPa
    Tmax = np.round(Tenv[1]*1e-6, 2)  # Tensiones máximas en MPa
    Rmin = np.round(Renv[0]*1e-3, 2)  # Reacciones mínimas en kN
    Rmax = np.round(Renv[1]*1e-3, 2)  # Reacciones maximas en kN

    print('RESULTADOS DINÁMICOS:')
    print('- Período de tiempo analizado (s):', lapso, 's')
    print('- Frecuencias naturales (Hz):', freq)
    print('- Envolvente de desplazamientos (mm):', list(zip(Umin, Umax)))
    print('- Envolvente de tensiones (MPa):', list(zip(Tmin, Tmax)))
    print('- Envolvente de reacciones (MPa):', list(zip(Rmin, Rmax)))
    print()


def procesamiento(coord: list[tuple[float]], restr: dict, cargas: dict,
                  barras: list[Barra2F], ff=1, lapso=10.0, mps=1.0,
                  tipo_analisis: str = 'estatico'):
    """Procesamiento estático o dinámico.

    Args:
        coord: lista de coordenadas de los nudos
        retr: diccionario con nudos restringidos y sus restricciones
        cargas: dicccionario con
        barras: lista con datos de las barras
        ff: frecuencia de vibración forzada (frec. de las cargas externas
            armónicas)
        lapso: período de tiempo de análisis desde t=0s
        mps: muestreo por segundo
        tipo_analisis:
            'estatico' para análisis estático
            'dinamico' (o cualquier otra palabra) para análisis dinámico

    Returns:
        - Diccionario con las propiedades de la estructura
        Además:
        En ANÁLISIS ESTÁTICO:
            - Vector de desplazamientos en los grados de libertad
            - Vector de fuerzas en las barras;
            - Vector de tensiones en las barras;
            - Vector de reacciones.
            - Tipo de análisis
        En ANÁLISIS DINÁMICO:
            - Vector de frecuencias naturales;
            - Matriz con datos de vibración de los grados de libertad
            - Lista de mínimos y máximos desplazamientos nodales
            - Lista de mínimas y máximas fuerzas en las barras;
            - Lista de mínimas y máximas tensiones en las barras.
            - Tipo de análisis
    """
    # Configuración de las propiedades de la estructura
    propiedades = config(coord, restr, cargas, barras)
    ngdl = propiedades['Ngdl']  # Número de grados de libertad

    # Matrices globales
    MG, KG, FG = matrices_globales(propiedades)

    # Matrices en los grados de libertad
    M, K, F = matrices_gdl(MG, KG, FG, ngdl)

    if tipo_analisis == 'estatico':
        # Desplazamientos nodales estáticos
        Xs = np.linalg.inv(K) @ F

        # Fuerzas normales estáticas
        Fs = obtener_fuerzas(propiedades, Xs)

        # Tensiones normales estáticas
        Ts = obtener_tensiones(propiedades, Xs)

        # Poner en cada barra las tensiones
        poner_tensiones(propiedades, Ts)

        # Reacciones estáticas
        Rs = obtener_reacciones(KG, Xs, ngdl)

        return propiedades, Xs, Fs, Ts, Rs, tipo_analisis
    else:  # tipo_analisis = Dinámico
        # Número de grados de restricción
        Nnudos = len(coord)  # Número de nudos
        dim = propiedades['Dimension']  # Dimensión
        ngdr = dim*Nnudos - ngdl  # Número de grados de restricción

        # Resolución del problema de autovalores
        w2, modos = eigh(K, M)
        freqs = np.sqrt(w2)/(2*np.pi)  # Frecuencias en Hz
        S = norm_uno(modos)  # Normalización a la componente máxima = 1

        # Vibración de los grados de libertad
        UU = desplaz_lapso(F, ff, w2, modos, lapso, mps)

        # Mínimos y máximos desplazamientos en los grados de libertad
        Uenv = desplaz_minymax(F, ff, w2, modos, lapso, mps)

        # Envolvente de tensiones
        Tenv = tensiones_minymax(propiedades, F, ff, w2, modos, lapso, mps)

        # Reacciones mínimas y máximas dinámicas
        Renv = reacciones_minymax(KG, ngdr, F, ff, w2, modos, lapso, mps)

        return propiedades, freqs, S, UU, Uenv, Tenv, Renv, lapso, \
            tipo_analisis


def mostrar_resultados(resultados):
    """Resultado del análisis estático o dinámico.

    Args:
        coord: lista de coordenadas de los nudos
        retr: diccionario con nudos restringidos y sus restricciones
        cargas: dicccionario con
        barras: lista con datos de las barras
        ff: frecuencia de vibración forzada (frec. de las cargas externas
            armónicas)
        lapso: período de tiempo de análisis desde t=0s
        mps: muestreo por segundo
        tipo_analisis:
            'estatico' para análisis estático
            'dinamico' (o cualquier otra palabra) para análisis dinámico

    Returns:
        Imprime resultados.
    """
    tipo_analisis = resultados[-1]
    if tipo_analisis == 'estatico':
        _, Xs, Fs, Ts, Rs, _ = resultados
        # Resultados estáticos
        respuesta_estatica(Xs, Fs, Ts, Rs)
    else:
        _, frecuencias, modos, UU, Uenv, Tenv, Renv, lapso, _ = resultados

        # Resultados dinámicos
        respuesta_dinamica(frecuencias, Uenv, Tenv, Renv, lapso)


def main():
    puntos_pilar = pd.read_csv('nudos_pilar.csv')
    barras_pilar = pd.read_csv('barras_pilar.csv')
    ps = puntos_pilar.to_numpy()
    bs = barras_pilar.to_numpy()
    coordenadas = {}
    for i, p in enumerate(ps):
        coordenadas[i] = tuple(p)
    
    # Nº de nudo y restricciones en X, Y, Z
    restr = {1: (1, 1, 1), 2: (1, 1, 1), 3: (1, 1, 1), 4: (1, 1, 1)}
    cargas = {49: (0, -400e3, 0), 83:(800e3, -400e3, 0)} # No está correcto aún
    
    freq = 15.0 # (Hz) Frecuencia de vibración de las cargas (solo viento)
    m1 = Material(7850, 200e9) # material
    
    ## Secciones transversales
    s1 = SeccionTransversal(2550*1e-6/2) # Cordón superior e inferior
    s2 = SeccionTransversal(4936.8*1e-6/2) # Cordones zona de refuerzo y pilar
    s3 = SeccionTransversal(693.6*1e-6/2) # Diagonal vigas
    s4 = SeccionTransversal(940.44*1e-6/2) # Diagonal vigas zona de refuerzo
    s5 = SeccionTransversal(7486.8*1e-6/2) # Vertical conexión zona de refuerzo
    s6 = SeccionTransversal(1270*1e-6) # Planchuela horizontal
    s7 = SeccionTransversal(1270*1e-6) # Planchuela diagonal

    secciones = [s1, s2, s3, s4, s5, s6, s7]
    
    elementos = []
    for i, b in enumerate(bs):
        Ni, Nf, m, s = tuple(b)
        elementos.append([Ni, Nf, m1, secciones[s-1]])
        
    mps = 10
    resultados = procesamiento(coordenadas, restr, cargas, elementos, freq,
                                   lapso=10, mps=mps, tipo_analisis='dinamico')
    
    mostrar_resultados(resultados)


if __name__ == "__main__":
    main()
