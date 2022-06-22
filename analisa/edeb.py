# Estática y Dinámica de Estructuras de Barras 1D, 2D, 3D
#
# @author: Fredy Gabriel Ramírez Villanueva
# Inicio de escritura del código: 06 de mayo de 2022
# Release 1: 18 de mayo de 2022
# Control git: 01 de junio de de 2022
# Versión actual: 0.1
# Versión 0.2: 17 de junio de 2022

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh

'''
Resolución de los problemas estático y dinámico para estructuras de barras.

EDEB: Estática y Dinámica de Estructuras de Barras.

Se consideran barras con diferentes grados de libertad a partir de la de 12
grados de libertad, 6 por cada nudo, con la siguiente convención:

Sistema de coordenadas locales:
    x: eje de la barra
    y: eje vertical
    z: eje perpendicular a x e y

Grados de libertad por nudo:
    1: translación en x
    2: translación en y
    3: translación en z
    4: rotación alrededor de x
    5: rotación alrededor de y
    6: rotación alrededor de z

Código asignado a los tipos estructurales y sus grados de libertad
    0: Barra unidimensional, grado de libertad: 1
    1: Reticulado plano, grados de libertad: (1, 2)
    2: Viga, grados de libertad: (2, 6)
    3: Pórtico plano, grados de libertad: (1, 2, 6)
    4: Reticulado espacial, grados de libertad: (1, 2, 3)
    5: Grilla, grados de libertad: (2, 4, 6)
    6: Pórtico espacial, grados de libertad: (1, 2, 3, 4, 5, 6)
'''
# Variables globales
DIMS = {1, 2, 3}  # Conjunto de dimensiones posibles
TIPO_ESTRUCTURA = {0: 'Barra unidimensional', 1: 'Reticulado plano',
                   2: 'Viga', 3: 'Pórtico plano', 4: 'Reticulado espacial',
                   5: 'Grilla', 6: 'Pórtico espacial'}

TE = TIPO_ESTRUCTURA  # Para acortar

# Dimensiones espaciales de las estructuras definidas en TIPO_ESTRUCTURA
DIM_ESTRUCTURA = {TE[0]: 1, TE[1]: 2, TE[2]: 2, TE[3]: 2, TE[4]: 3, TE[5]: 3,
                  TE[6]: 3}

# Son reticulados (estructuras con barras de dos fuerzas):
RETICULADOS = {0: TE[0], 1: TE[1], 4: TE[4]}

# Número de grados de libertad por nudo
NGDL_NUDO = {TE[0]: 1, TE[1]: 2, TE[2]: 2, TE[3]: 3, TE[4]: 3, TE[5]: 3,
             TE[6]: 6}

# Número de grados de libertad de translación por nudo
NGDL_TRANS_NUDO = {TE[0]: 1, TE[1]: 2, TE[2]: 1, TE[3]: 2, TE[4]: 3, TE[5]: 1,
                   TE[6]: 3}

# Versores del sistema global de coordenadas
VERSORES = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])


@dataclass
class Nudo:
    """Nudo de estructura de barras

    :arg
        num (int): número asignado;
        coord (tuple[float]): Coordenadas de los nudos. Puede contener 1, 2 o 3
            valores según la dimensión considerada;
        tipo (int): Uno de los valores del siguiente diccionario:
            {0: 'Barra unidimensional', 1: 'Reticulado plano', 2: 'Viga',
             3: 'Pórtico plano', 4: 'Reticulado espacial', 5: 'Grilla',
             6: 'Pórtico espacial'}

        restr (tuple[int]): Restricciones según los gdl's del nudo.
            Si no es dado, se sopondrá que el nudo no está restringido.
            1: restringido; 0: libre
            - Ejemplo en Reticulado espacial:
                (0, 1, 1): libre en x, restringido en y y z
            - Ejemplo en Viga:
                (1, 0): restringido en y, libre en el giro
            - Ejemplo en Barra unidimensional:
                1: restringido en x
    """
    # Variables públicas
    num: int  # Número asignado al nudo
    coord: tuple[float]  # Coordenadas
    tipo: int  # Tipo de nudo
    restr: tuple[int] = field(init=False)  # Restricciones
    cargas: tuple[float] = field(init=False)  # Cargas en los nudos

    # Variables protegidas
    _gdl: tuple[int] = field(init=False)  # Enumeración de los gdl's
    _desplaz: np.ndarray = field(init=False)  # Desplaz. del nudo s/ ngdl.

    def __post_init__(self):
        tipo = self.tipo
        assert tipo in TIPO_ESTRUCTURA, "Verificar tipo de estructura"

        ngdl = NGDL_NUDO[TIPO_ESTRUCTURA[tipo]]
        self.restr = tuple([0] * ngdl)
        self.cargas = tuple([0.0] * ngdl)
        self._desplaz = np.zeros(ngdl)

    @property
    def ngdl(self) -> int:
        """Número de grados de libertad del nudo."""
        return NGDL_NUDO[TIPO_ESTRUCTURA[self.tipo]]

    @property
    def ntr(self) -> int:
        """Número de grados de libertad de translación"""
        ntr = NGDL_TRANS_NUDO[TIPO_ESTRUCTURA[self.tipo]]
        return ntr

    @property
    def nrot(self) -> int:
        """Número de grados de libertad de rotación"""
        return self.ngdl - self.ntr

    @property
    def dim(self) -> int:
        """Dimensión espacial del nudo."""
        if self.tipo == 0:  # Barra unidimensional
            return 1
        elif self.tipo in {1, 2, 3}:  # Reticulado plano, viga y pórtico plano
            return 2
        elif self.tipo in {4, 5, 6}:  # Ret. espacial, grilla, pórtico espacial
            return 3

    def set_gdl(self, gdl: tuple) -> None:
        """Asigna números de grados de libertad al nudo.

        :param gdl: enumeración de los grados del libertad del nudo.
        :return: None
        """
        self._gdl = gdl

    def gdl(self) -> tuple:
        """Devuelve la enumeración de los grados de libertad del nudo."""
        return self._gdl

    def set_desplaz(self, u: np.ndarray) -> None:
        """Configura desplazamiento del nudo.

        :param u: Vector de desplazamientos del nudo
        """
        self._desplaz = u

    def desplaz(self) -> np.ndarray:
        """Devuelve desplazamiento nodal en los grados de libertad."""
        return self._desplaz

    def translacion(self) -> np.ndarray:
        """Translación del nudo luego de la acción de las cargas.

        :return: Desplazamientos en los grados de libertad de translación.
        """
        u_trans = self.desplaz()[:self.ntr]
        return u_trans

    def rotacion(self) -> np.ndarray:
        """Rotación del nudo luego de la acción de las cargas.

        :return: Desplazamientos en los grados de libertad de translación.
        """
        u_rot = self.desplaz()[self.ntr:]
        return u_rot

    def posicion(self, amp=1.0) -> np.ndarray:
        """Posición del nudo después de la aplicación de las cargas.

        Args:
            amp: factor amplificador de desplazamientos

        Returns:
            Coordenadas de la nueva posición del nudo (puede estar amplificado)
        """
        u_amp = amp * self.translacion()  # Translación amplificada
        c = np.array(self.coord)  # Coordenadas iniciales

        return c + u_amp

    def dibujar(self, num=False, ax=None, **kwargs) -> None:
        """Dibuja el nudo.

        Args:
            num (bool): Para incluir o no el número identificador del nudo.
            ax (Axes): De matplotlib para el caso de dibujo en 3D
            **kwargs: Argumentos que se pasarán a scatter 2D o 3D
        Returns:
            Nada, solo dibuja.
        """
        if self.dim == 2:
            [x, y] = self.coord
            r = sum(self.restr)
            mk = 'o'

            if r == 0:  # Nudo libre
                c1 = 'r'
            elif r == 1:  # Nudo con alguna restricción
                c1 = 'y'
                mk = '^'
            else:  # Nudo restringido
                c1 = 'k'
                mk = '^'

            plt.scatter(x, y, marker=mk, c=c1, **kwargs)

            if num:  # Agregar o no número asignado al nudo
                plt.text(x, y, str(self.num + 1))

        elif self.dim == 3:
            [x, y, z] = self.coord
            ax.scatter(x, y, z, **kwargs)


@dataclass
class Material:
    """Propiedades del material.

    Hipótesis: material isótropo, homogéneo, elástico, lineal.

    :arg
        elast_long (float): módulo de elasticidad longitudinal (de Young);
        densidad (float): densidad del material;
        poisson (float): (opcional) módulo de Poisson.
    """

    elast_long: float = 200e9  # Módulo de elasticidad longitudinal (Young)
    densidad: float = 7850.0  # Densidad del material
    poisson: Optional[float] = None  # Módulo de Poisson
    _elast_transv: Optional[float] = None  # Módulo de cizalladura

    def __post_init__(self):
        if self.poisson is not None:
            E = self.elast_long
            nu = self.poisson
            self._elast_transv = E / 2 / (1 + nu)

    @property
    def elast_transv(self) -> float:
        """Módulo de cizalladura (G)."""
        return self._elast_transv

    @elast_transv.setter
    def elast_transv(self, ge):
        self._elast_transv = ge

    @property
    def lame1(self) -> float:
        """Primer parámetro de Lamé (lambda)."""
        E = self.elast_long
        nu = self.poisson
        if nu is not None:
            return E*nu / ((1+nu)*(1-2*nu))
        else:
            raise AttributeError("Coeficiente de Poisson desconocido")

    @property
    def lame2(self) -> float:
        """Segundo parámetro de Lamé (mu)."""
        return self.elast_transv

    @property
    def compres(self) -> float:
        """Módulo de compresibilidad (K)."""
        E = self.elast_long
        nu = self.poisson
        if nu is not None:
            return E / (3*(1-2*nu))
        else:
            raise AttributeError("Coeficiente de Poisson desconocido")

    @property
    def modulo_onda(self) -> float:
        """Módulo de onda (M)."""
        E = self.elast_long
        nu = self.poisson
        if nu is not None:
            return E*(1-nu) / ((1+nu)*(1-2*nu))
        else:
            raise AttributeError("Coeficiente de Poisson desconocido")


@dataclass
class Seccion:
    """Propiedades de la sección transversal de una barra recta.

    La sección transversal debe ser constante a lo largo del elemento.
    """

    area: float  # Área de la sección transversal
    inercia_y: Optional[float] = None  # Inercia alrededor del eje y
    inercia_z: Optional[float] = None  # Inercia alrededor del eje z
    _inercia_polar: Optional[float] = None  # Inercia polar, Ix
    area_cortante_y: Optional[float] = None  # Área efectiva a cortante en y
    area_cortante_z: Optional[float] = None  # Área efectiva a cortante en z

    def __post_init__(self):
        Iy = self.inercia_y
        Iz = self.inercia_z
        if Iy is not None:
            if Iz is not None:
                self._inercia_polar = Iy + Iz

    @property
    def inercia_polar(self) -> float:
        """Momento polar de inercia."""
        return self._inercia_polar

    @inercia_polar.setter
    def inercia_polar(self, jota: float):
        self._inercia_polar = jota


#############################
# Tratamiento de las barras #
#############################

@dataclass
class Barra(ABC):
    """Barra estructural.

    Adoptar el critero de que el nudo inicial tenga menor enumeración que el
    nudo final.

    Sistema de coordenadas locales:
        x: eje de la barra posicionada horizontalmente
        y: eje vertical o definido por el ángulo roll.
        z: eje perpendicular a x e y

    :arg
        tipo (int): Tipo de estructura según se define en TIPO_ESTRUCTURA
        nudo_inicial (Nudo): nudo inicial
        nudo_final (Nudo): nudo final
        material (Material): datos del material
        seccion (Seccion): propiedades de la sección transversal
        versores (np.ndarray): Matriz cuyas columnas son los versores del
            sistema de coordenadas global adoptado.
    """

    # Métodos abstractos:
    # 1. __post_init__
    # 2. rigidez_local
    # 3. masa_local
    # 4. transf_coord

    # Públicas
    tipo: int  # Tipo de estructura al que pertenece la barra
    nudo_inicial: Nudo  # Nudo inicial
    nudo_final: Nudo  # Nudo final
    material: Material  # Propiedades del material
    seccion: Seccion  # Propiedades geométricas de la sección
    versores: np.ndarray  # Matriz de versores adoptados

    @abstractmethod
    def __post_init__(self):
        # Para verificar el tipo correcto de la estructura entre otras cosas
        pass

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
        cf = np.array(self.nudo_final.coord)  # Coordenadas del nudo final
        return np.linalg.norm(ci - cf)

    def gdl(self) -> list:
        """Grados de libertad de los nudos extremos de la barra."""
        gdl_i = list(self.nudo_inicial.gdl())  # gdl del nudo inicial
        gdl_f = list(self.nudo_final.gdl())  # gdl del nudo final
        return gdl_i + gdl_f

    @property
    def ngdl(self) -> int:
        """Número de grados de libertad de la barra."""
        return len(self.gdl())

    @property
    def ngdl_nudo(self):
        """Número de grados de libertad por nudo"""
        return NGDL_NUDO[TE[self.tipo]]

    @property
    def ngdl_trans_nudo(self) -> int:
        """Número de grados de libertad de translación por nudo"""
        return NGDL_TRANS_NUDO[TIPO_ESTRUCTURA[self.tipo]]

    @property
    @abstractmethod
    def rigidez_local(self) -> np.ndarray:
        """Matriz de rigidez en coordenadas locales."""
        pass

    @property
    @abstractmethod
    def masa_local(self) -> np.ndarray:
        """Matriz de masa en coordenadas locales."""
        pass

    @abstractmethod
    def transf_coord(self) -> np.ndarray:
        """Matriz de transformación de coordenadas globales a locales."""
        pass

    @property
    def rigidez_global(self) -> np.ndarray:
        """Matriz de rigidez en coordenadas globales"""

        T = self.transf_coord()  # Matriz de transf. de coord.
        ke = self.rigidez_local  # Matriz de rigidez en coordenadas locales.

        K = T.T @ ke @ T  # Cálculo de la matriz de rigidez en coord. globales
        return K

    @property
    def masa_global(self) -> np.ndarray:
        """Matriz de masa en coordenadas globales"""
        T = self.transf_coord()  # Matriz de transf. de coord.
        me = self.masa_local  # Matriz de masa en coordenadas locales

        M = T.T @ me @ T  # Cálculo de la matriz de masa en coord. globales
        return M

    def desplaz_global(self) -> np.ndarray:
        """Desplazamientos nodales de la barra en coordenadas globales.

        Returns:
            Vector de desplazamientos nodales. Tipo np.ndarray.
        """
        n = self.ngdl_nudo  # Número de grados de libertad por nudo

        u = np.zeros(self.ngdl)  # Inicialización
        u[:n] = self.nudo_inicial.desplaz()  # Desplazamientos del nudo inicial
        u[n:] = self.nudo_final.desplaz()  # Desplazamientos del nudo final

        return u

    def fuerza_global(self) -> np.ndarray:
        """Vector de fuerzas en los nudos de la barra en coord. globales."""

        U = self.desplaz_global()
        K = self.rigidez_global
        return K @ U

    def desplaz_local(self) -> np.ndarray:
        """Vector de desplazamientos nodales de la barra en coord. locales."""
        U = self.desplaz_global()
        T = self.transf_coord()
        return T @ U

    def fuerza_local(self) -> np.ndarray:
        """Vector de fuerzas nodales de la barra en coordenadas locales."""
        u = self.desplaz_local()
        k = self.rigidez_local
        return k @ u

    def dibujar(self, espesor_area=True, color='k', ax=None, **kwargs) -> None:
        """Dibuja la barra en 2D o 3D.

        Args:
            espesor_area (bool): Considerar o no las áreas para dibujuar
                proporcionalmente a ellas el espesor de las líneas.
            color: color de las barras.
            ax (Axes): Axes object (matplotlib)
            **kwargs: Argumentos que se pasan a plt.plot o a ax.plot
        """
        if espesor_area:
            lw = self.seccion.area * 1000
        else:
            lw = 1

        X0 = self.nudo_inicial.coord  # Coordenadas del nudo inicial
        Xf = self.nudo_final.coord  # Coordenadas del nudo final
        coord_separadas = list(zip(X0, Xf))
        X = coord_separadas[0]
        Y = coord_separadas[1]
        if self.dim == 2:
            plt.plot(X, Y, color=color, linewidth=lw, **kwargs)

        elif self.dim == 3:
            Z = coord_separadas[2]
            ax.plot(X, Y, Z, color=color, linewidth=lw, **kwargs)

    def dibujar_deform(self, espesor_area=True, color='k', amp=1, ax=None,
                       **kwargs) -> None:
        """Dibuja la barra en su posición desplazada.

        :param espesor_area: Considerar en el dibujo el espesor
            relativo de la barra según su área.
        :param color: color de las barras
        :param amp: Factor ae amplificación de desplazamientos
        :param ax: Axes 3D matplotlib para el dibujo en 3D
        :param kwargs: Argumentos para plt.plot o ax.plot
        """
        if espesor_area:
            lw = self.seccion.area * 1000
        else:
            lw = 1

        X0d = self.nudo_inicial.posicion(amp)  # Coord. nudo inicial desplazado
        Xfd = self.nudo_final.posicion(amp)  # Coord. nudo final desplazado
        coord_separadas = list(zip(X0d, Xfd))
        X = coord_separadas[0]
        Y = coord_separadas[1]

        if self.dim == 2:
            plt.plot(X, Y, color=color, linewidth=lw, **kwargs)

        if self.dim == 3:
            Z = coord_separadas[2]
            ax.plot(X, Y, Z, color=color, linewidth=lw, **kwargs)


@dataclass
class BarraReticulado(Barra):
    """Barra de reticulado, unidimensional, plano o espacial.

    Barra de 2 fuerzas.

    Adoptar el critero de que el nudo inicial tenga menor enumeración que el
    nudo final.

    Sistema de coordenadas locales:
        x: eje de la barra posicionada horizontalmente
        y: eje vertical.
        z: eje perpendicular a x e y

    :arg
        tipo (int): Tipo de estructura según se define en TIPO_ESTRUCTURA
        nudo_inicial (Nudo): nudo inicial
        nudo_final (Nudo): nudo final
        material (Material): datos del material
        seccion (Seccion): propiedades de la sección transversal
        versores (np.ndarray): Matriz cuyas columnas son los versores del
            sistema de coordenadas global adoptado.
    """

    # Métodos sobreescritos:
    # 1. __post_init__
    # 2. rigidez_local
    # 3. masa_local
    # 4. transf_coord

    # Protegidas
    _tension: float = 0.0  # Tensión en la barra

    # Sobre-escritura de la clase Barra
    def __post_init__(self):
        assert self.tipo in RETICULADOS, 'Verificar tipo de estructura'

    # Sobre-escritura de la clase barra
    @property
    def rigidez_local(self) -> np.ndarray:
        """Matriz de rigidez cuadrada en coordenadas locales.

        Returns:
            Matriz cuadrada de orden igual al número de grados de libertad de
            la barra.
        """
        n = self.ngdl_nudo
        k = np.zeros((self.ngdl, self.ngdl))  # Inicialización
        k[0, 0], k[0, n] = 1.0, -1.0  # Unos de la primera fila
        k[n] = -k[0]  # Filas iguales con signo cambiado
        factor = self.seccion.area * self.material.elast_long / self.longitud
        return factor * k

    @property  # Sobre-escritura de la clase Barra
    def masa_local(self) -> np.ndarray:
        """Matriz de masa en coordenadas locales

        Returns:
            Matriz cuadrada de orden igual al número de grados de libertad de
            la barra.
        """
        n = self.ngdl_nudo
        rho = self.material.densidad
        area = self.seccion.area
        longitud = self.longitud

        M = 2*np.identity(self.ngdl)  # Inicialización
        II = np.identity(n)  # Matriz identidad
        M[:n, n:] = II
        M[n:, :n] = II

        return rho*area*longitud/6 * M

    def matriz_cos_dir(self) -> np.ndarray:
        """Matriz de cosenos directores.

        Las filas contienen los cosenos directores de los ejes locales
        respecto a los globales.

        En 1D se supone que el eje local x coincide con el global X

        Basado en Aslam Kassimalli, Matrix Analysis of Structures. Cap. 8

        Returns:
            - 1D: Matriz de 1 x 1 (identidad 1x1)
            - 2D: Matriz de 2 x 2
            - 3D: Matriz de 3 x 3
        """
        dim = self.dim
        versT = self.versores.T  # Para tomar las columnas y no las filas

        if dim == 1:
            return np.array([1])

        else:  # Para 2D o 3D
            ci = np.array(self.nudo_inicial.coord)  # Coords. del nudo inicial
            cf = np.array(self.nudo_final.coord)  # Coords. del nudo final
            L = self.longitud
            v = cf - ci  # Barra como vector

            # Cosenos directores del eje local x
            ix = np.array([np.dot(v, versT[i][:dim]) for i in range(dim)]) \
                / L

            if dim == 2:  # 2D
                c, s = ix[0], ix[1]
                return np.array([
                    [c, s],
                    [-s, c]
                ])

            else:  # 3D
                # Adaptación para ángulo roll = 0
                # Versores j, k del sistema global
                jota = versT[1]
                ka = versT[2]

                # Cosenos directores del eje local z
                z = np.cross(ix, jota)  # Kassimalli (8.59)
                nz = np.linalg.norm(z)  # Norma de z
                if nz > 0.0:  # Si ix no es paralelo IY
                    izb = z / nz  # Kassimalli (8.60)
                    # Cosenos directores del eje local y
                    iyb = np.cross(izb, ix)  # Kassimalli (8.61)
                    iy = iyb
                    # Cosenos directores del eje local z
                    iz = izb
                else:  # si ix es paralelo a IY
                    iz = ka  # Kassimalli p477 2do párrafo.
                    iy = np.cross(iz, ix)

                return np.array([ix, iy, iz])  # Kassimalli (8.62)

    # Sobre-escritura de la clase Barra
    def transf_coord(self) -> np.ndarray:
        """Matriz de transformación de coordenadas globales a locales.

        Returns:
            Matriz ngdl_barra x ngdl_barra
        """
        d = self.dim
        mcd = self.matriz_cos_dir()  # Matriz de cosenos directores
        T = np.identity(self.ngdl)  # Inicialización
        T[:d, :d] = mcd
        T[d:, d:] = mcd

        return T

    def set_tension(self) -> None:
        """Guarda la tensión axial de la barra como atributo de clase."""
        Ui = self.nudo_inicial.desplaz()  # Desplazamientos en el nudo inicial
        Uf = self.nudo_final.desplaz()  # Desplazamientos en el nudo final
        U = np.append(Ui, Uf)  # Desplaz. globales en los nudos de la barra
        u = self.transf_coord() @ U  # Desplazamientos en coord. locales
        Q = self.rigidez_local @ u  # Fuerzas nodales en coord. locales
        N = -Q[0]  # Fuerza normal en la barra
        A = self.seccion.area
        self._tension = N / A

    def tension(self) -> float:
        """Esfuerzo axial de la barra."""
        return self._tension

    def def_unit(self) -> float:
        """Deformación unitaria de la barra.

        Positivo: alargamiento; Negativo: acortamiento.
        """
        sigma = self.tension()
        E = self.material.elast_long
        return sigma / E

    def fuerza_normal(self) -> float:
        """Fuerza normal en la barra.

        Positivo: tracción; Negativo: compresión.
        """
        return self.tension() * self.seccion.area

    def elongacion(self) -> float:
        """Elongación de la barra.

        Positivo: alargamiento; Negativo: acortamiento.
        """
        return self.def_unit() * self.longitud

    def dibujar(self, espesor_area=True, color='tension', ax=None, **kwargs)\
            -> None:
        """Dibuja la barra en 2D o 3D.

        Args:
            espesor_area (bool): Considerar o no las áreas para dibujuar
                proporcionalmente a ellas el espesor de las líneas.
            color (str): 'tension' para colorear según tensión en la barra, o
                cualquier valor válido del sistema de colores.
            ax (Axes): Axes object (matplotlib)
            **kwargs: Argumentos que se pasan a plt.plot o ax.plot
        """
        if espesor_area:
            lw = self.seccion.area * 1000
        else:
            lw = 1

        if color == 'tension':  # Colorear la barra según su tensión
            sigma = self.tension()
            if sigma > 0:  # Tracción
                c = 'b'
            elif sigma < 0:  # Compresión
                c = 'r'
            else:
                c = '0.5'  # Gris para tensión cero
        else:
            c = color

        X0 = self.nudo_inicial.coord  # Coordenadas del nudo inicial
        Xf = self.nudo_final.coord  # Coordenadas del nudo final
        coord_separadas = list(zip(X0, Xf))
        X = coord_separadas[0]
        Y = coord_separadas[1]
        if self.dim == 2:
            plt.plot(X, Y, color=c, linewidth=lw, **kwargs)

        elif self.dim == 3:
            Z = coord_separadas[2]
            ax.plot(X, Y, Z, color=c, linewidth=lw, **kwargs)

    def dibujar_deform(self, espesor_area=True, amp=1, colorear=False, ax=None,
                       **kwargs) -> None:
        """Dibuja la barra en su posición desplazada

        :param espesor_area: Considerar en el dibujo el espesor
            relativo de la barra según su área.
        :param amp: Factor ae amplificación de desplazamientos
        :param colorear: Colorear la barra según su tensión
        :param ax: Axes 3D matplotlib para el dibujo en 3D
        :param kwargs: Argumentos que se pasarán a plt.plot o ax.plt
        """
        if espesor_area:
            lw = self.seccion.area * 1000
        else:
            lw = 1

        if colorear:
            sigma = self.tension()
            if sigma > 0:  # Tracción
                color = 'b'
            elif sigma < 0:  # Compresión
                color = 'r'
            else:  # Tensión cero
                color = 'k'
        else:
            color = 'g'

        X0d = self.nudo_inicial.posicion(amp)  # Coord. nudo inicial desplazado
        Xfd = self.nudo_final.posicion(amp)  # Coord. nudo final desplazado
        coord_separadas = list(zip(X0d, Xfd))
        X = coord_separadas[0]
        Y = coord_separadas[1]

        if self.dim == 2:
            plt.plot(X, Y, color=color, linewidth=lw, **kwargs)

        if self.dim == 3:
            Z = coord_separadas[2]
            ax.plot(X, Y, Z, color=color, linewidth=lw, **kwargs)


@dataclass
class BarraPortico(Barra):
    """Barra de pórtico plano o espacial.

    Adoptar el critero de que el nudo inicial tenga menor enumeración que el
    nudo final.

    Sistema de coordenadas locales:
        x: eje de la barra.
        y: definido por el ángulo roll.
        z: eje perpendicular a x e y

    :arg
        tipo (int): Tipo de estructura según se define en TIPO_ESTRUCTURA
        nudo_inicial (Nudo): nudo inicial
        nudo_final (Nudo): nudo final
        material (Material): datos del material
        seccion (Seccion): propiedades de la sección transversal
        versores (np.ndarray): Matriz cuyas columnas son los versores del
            sistema de coordenadas global adoptado.
        roll (float): Ángulo en radianes medido en sentido horario cuando se
            mira en la dirección negativa del eje x local, con el cual el
            sistema xyz rota alrededor de x, tal que el xy quede vertical con
            el eje y apuntando hacia arriba (i.e. en la dirección positiva del
            eje Y global). Kassimali p477.
        cortante (bool): Consideración de la rigidez a cortante.
    """

    roll: float = 0.0  # Ángulo roll en radianes
    cortante: bool = False  # Considerar el efecto de la cortante

    # Métodos sobreescritos:
    # 1. __post_init__
    # 2. rigidez_local
    # 3. masa_local
    # 4. transf_coord

    # Sobreescrito
    def __post_init__(self):
        assert self.tipo in {2, 3, 5, 6}, 'Verifica el tipo de barra'

    @property  # Sobreescrito
    def masa_local(self) -> np.ndarray:
        """Matriz de masas en coordenadas locales.

            No se considera el efecto de las deformaciones por cortante.

        Returns:
            2D: Matriz 6x6; 3D: Matriz 12x12
        """

        dim = self.dim
        L = self.longitud
        A = self.seccion.area
        Iz = self.seccion.inercia_z
        rho = self.material.densidad

        if dim == 2:  # Barra de pórtico plano
            m = rho*A*L/420 * np.array([
                [140, 0, 0, 70, 0, 0],
                [0, 156, 22*L, 0, 54, -13*L],
                [0, 22*L, 4*L**2, 0, 13*L, -3*L**2],
                [70, 0, 0, 140, 0, 0],
                [0, 54, 13*L, 0, 156, -22*L],
                [0, -13*L, -3*L**2, 0, -22*L, 4*L**2]
            ])
            return m

        elif dim == 3:
            Jx = self.seccion.inercia_polar
            Iy = self.seccion.inercia_y

            m = rho*A*L*np.array([
                [1/3, 0, 0, 0, 0, 0, 1/6, 0, 0, 0, 0, 0],
                [0, 13/35 + 6*Iz/(5*A*L**2), 0, 0, 0, 11*L/210 + Iz/(10*A*L),
                 0, 9/70 - 6*Iz/(5*A*L**2), 0, 0, 0, -13*L/420 + Iz/(10*A*L)],
                [0, 0, 13/35 + 6*Iy/(5*A*L**2), 0, -11*L/210 - Iy/(10*A*L), 0,
                 0, 0, 9/70 - 6*Iy/(5*A*L**2), 0, 13*L/420 - Iy/(10*A*L), 0],
                [0, 0, 0, Jx/3/A, 0, 0, 0, 0, 0, Jx/6/A, 0, 0],
                [0, 0, -11*L/210 - Iy/(10*A*L), 0, L**2/105 + 2*Iy/15/A, 0, 0,
                 0, -13*L/420 + Iy/(10*A*L), 0, -L**2/140 - Iy/30/A, 0],
                [0, 11*L/210 + Iz/(10*A*L), 0, 0, 0, L**2/105 + 2*Iz/15/A, 0,
                 13*L/420 - Iz/10/A/L, 0, 0, 0, -L**2/140 - Iz/30/A],
                [1/6, 0, 0, 0, 0, 0, 1/3, 0, 0, 0, 0, 0],
                [0, 9/70 - 6*Iz/(5*A*L**2), 0, 0, 0, 13*L/420 - Iz/10/A/L, 0,
                 13/35 + 6*Iz/(5*A*L**2), 0, 0, 0, -11*L/210 - Iz/(10*A*L)],
                [0, 0, 9/70 - 6*Iy/(5*A*L**2), 0, -13*L/420 + Iy/(10*A*L), 0,
                 0, 0, 13/35 + 6*Iz/(5*A*L**2), 0, 11*L/210 + Iy/(10*A*L), 0],
                [0, 0, 0, Jx/6/A, 0, 0, 0, 0, 0, Jx/3/A, 0, 0],
                [0, 0, 13*L/420 - Iy/(10*A*L), 0, -L**2/140 - Iy/30/A, 0, 0, 0,
                 11*L/210 + Iy/(10*A*L), 0, L**2/105 + 2*Iy/15/A, 0],
                [0, -13*L/420 + Iz/10/A/L, 0, 0, 0, -L**2/140 - Iz/30/A, 0,
                 -11*L/210 - Iz/10/A/L, 0, 0, 0, L**2/105 + 2*Iz/15/A]
            ])
            return m

        else:
            raise Exception("Verifica la dimensión")

    def rigidez_p2d(self) -> np.ndarray:
        """Matriz de rigidez para pórtico plano.

        Elemento de barra de 3 grados de libertad por nudo (translación en x,
        translación en y, giro en z)

        Returns:
            Matriz 6x6 de tipo np.ndarray
        """
        L = self.longitud
        E = self.material.elast_long
        A = self.seccion.area
        Iz = self.seccion.inercia_z
        G = self.material.elast_transv
        Ac = self.seccion.area_cortante_y
        if self.cortante:  # Se considera la rigidez al corte
            assert Ac is not None, "Falta dato de área efectiva al corte"
            f = 12*E*Iz/(G*Ac*L**2)
        else:  # No se toma en cuenta la rigidez al corte
            f = 0

        k = E*Iz/(L**3*(1 + f)) * np.array([
            [A*L**2/Iz, 0, 0, -A*L**2/Iz, 0, 0],
            [0, 12, 6*L, 0, -12, 6*L],
            [0, 6*L, (4 + f)*L**2, 0, -6*L, (2 - f)*L**2],
            [-A*L**2/Iz, 0, 0, A*L**2/Iz, 0, 0],
            [0, -12, -6*L, 0, 12, -6*L],
            [0, 6*L, (2 - f)*L**2, 0, -6*L, (4 + f)*L**2]
        ])
        return k

    def rigidez_p3d(self) -> np.ndarray:
        """Matriz de rigidez para barra de pórtico 3D.

        Elemento de barra de 6 grados de libertad por nudo (translaciones y
        rotaciones en x, y, z).

        Returns:
            Matriz 12x12 de tipo np.ndarray
        """
        L = self.longitud
        E = self.material.elast_long
        A = self.seccion.area
        [Iy, Iz] = self.seccion.inercia_y, self.seccion.inercia_z
        J = self.seccion.inercia_polar
        G = self.material.elast_transv
        [Acy, Acz] = self.seccion.area_cortante_y, self.seccion.area_cortante_z

        if self.cortante:  # Se toma en cuenta la rigidez al corte
            fy = 12*E*Iz/(G*Acy*L**2)
            fz = 12*E*Iy/(G*Acz*L**2)
        else:  # No se considera la rigidez al corte
            fy, fz = 0, 0

        k = np.array([
            [E*A/L, 0, 0, 0, 0, 0, -E*A/L, 0, 0, 0, 0, 0],
            [0, 12*E*Iz, 0, 0, 0, 6*E*Iz/L**2/(1 + fy), 0, -12*E*Iz/L**3/(1 +
                                                                          fy),
             0, 0, 0, 6*E*Iz/L**2/(1 + fy)],
            [0, 0, 12*E*Iy/L**3/(1 + fz), 0, -6*E*Iy/L**2/(1 + fz), 0, 0, 0,
             -12*E*Iy/L**3/(1 + fy), 0, -6*E*Iy/L**2/(1 + fz), 0],
            [0, 0, 0, G*J/L, 0, 0, 0, 0, 0, -G*J/L, 0, 0],
            [0, 0, -6*E*Iy/L**2/(1 + fz), 0, (4 + fz)*E*Iy/L/(1 + fz), 0, 0, 0,
             6*E*Iy/L**2/(1 + fz), 0, (2 - fz)*E*Iy/L/(1 + fz), 0],
            [0, 6*E*Iy/L**2/(1 + fz), 0, 0, 0, (4 + fy)*E*Iz/L/(1 + fy), 0,
             -6*E*Iz/L**2/(1 + fy), 0, 0, 0, (2 - fy)*E*Iz/L/(1 + fy)],
            [-E*A/L, 0, 0, 0, 0, 0, E*A/L, 0, 0, 0, 0, 0],
            [0, -12*E*Iz/L**3/(1 + fy), 0, 0, 0, -6*E*Iz/L**2/(1 + fy), 0,
             12*E*Iz/L**3/(1 + fy), 0, 0, 0, -6*E*Iz/L**2/(1 + fy)],
            [0, 0, -12*E*Iy/L**3/(1 + fz), 0, 6*E*Iy/L**2/(1 + fz), 0, 0, 0,
             12*E*Iy/L**3/(1 + fz), 0, 6*E*Iy/L**2/(1 + fz), 0],
            [0, 0, 0, G*J/L, 0, 0, 0, 0, 0, G*J/L, 0, 0],
            [0, 0, -6*E*Iy/L**2/(1 + fz), 0, (2 - fz)*E*Iy/L/(1 + fz), 0, 0, 0,
             6*E*Iy/L**2/(1 + fz), 0, (4 + fz)*E*Iy/L/(1 + fz), 0],
            [0, 6*E*Iz/L**3/(1 + fy), 0, 0, 0, (2 - fy)*E*Iz/L/(1 + fy), 0,
             -6*E*Iz/L**2/(1 + fy), 0, 0, 0, (4 + fy)*E*Iz/L/(1 + fy)]
        ])
        return k

    @property  # Sobreescrito
    def rigidez_local(self) -> np.ndarray:
        """Matriz de rigidez en coordenadas locales."""

        dim = self.dim
        if dim == 2:  # Pórtico plano
            return self.rigidez_p2d()
        return self.rigidez_p3d()  # Pórtico espacial

    def matriz_cos_dir(self) -> np.ndarray:
        """Matriz de cosenos directores.

        Las filas contienen los cosenos directores de los ejes locales
        respecto a los globales.

        En 1D se supone que el eje local x coincide con el global X

        Basado en Aslam Kassimalli, Matrix Analysis of Structures. Cap. 8

        Returns:
            - 1D: Matriz de 1 x 1 (identidad)
            - 2D: Matriz de 2 x 2
            - 3D: Matriz de 3 x 3
        """
        dim = self.dim
        roll = self.roll
        versores = self.versores

        if dim == 1:
            return np.array([1])

        else:  # Para 2D o 3D
            ci = np.array(self.nudo_inicial.coord)  # Coords. del nudo inicial
            cf = np.array(self.nudo_final.coord)  # Coords. del nudo final
            L = self.longitud
            v = cf - ci  # Barra como vector

            # Cosenos directores del eje local x
            ix = np.array([np.dot(v, versores[i][:dim]) for i in range(dim)]) \
                / L

            if dim == 2:  # 2D
                c, s = ix[0], ix[1]
                return np.array([
                    [c, s],
                    [-s, c]
                ])

            else:  # 3D
                # Versores j, k del sistema global
                jota = versores.T[1]
                ka = versores.T[2]

                # Cosenos directores del eje local z
                z = np.cross(ix, jota)  # Kassimalli (8.59)
                nz = np.linalg.norm(z)  # Norma de z
                if nz > 0.0:  # Si ix no es paralelo IY
                    izb = z / nz  # Kassimalli (8.60)
                    # Cosenos directores del eje local y
                    iyb = np.cross(izb, ix)  # Kassimalli (8.61)
                    iy = np.cos(roll)*iyb + np.sin(roll)*izb
                    # Cosenos directores del eje local z
                    iz = -np.sin(roll)*iyb + np.cos(roll)*izb
                else:  # si ix es paralelo a IY
                    iz = ka  # Kassimalli p477 2do párrafo.
                    iy = np.cross(iz, ix)

                return np.array([ix, iy, iz])  # Kassimalli (8.62)

    # Sobreescrito
    def transf_coord(self) -> np.ndarray:
        """Matriz de transformación de coordenadas para pórticos planos.

        Transforma las coordenadas globales a locales.

        Returns:
            2D: Matriz de 6x6. Tipo np.ndarray.
            3D: Matriz de 12x12. Tipo np.ndarray.
        """
        m = self.matriz_cos_dir()  # Matriz de cosenos directores
        dim = self.dim  # Dimensión espacial
        ngdl = self.ngdl  # Número de grados de libertad de la barra

        T = np.identity(ngdl)  # Inicialización
        if dim == 2:
            T[:2, :2] = m
            T[3:5, 3:5] = m
            return T  # Notar que hay un uno en T[2,2] y T[5,5]

        elif dim == 3:
            for i in range(4):
                T[3*i:3*(i+1), 3*i:3*(i+1)] = m
            return T

    ############################################
    # Vectores de fuerzas nodales equivalentes #
    ############################################

    def rne_qud_vig(self, q: float) -> np.ndarray:
        """Reacción nodal equivalente, carga uniformemente distribuida, viga.

        La carga debe estar distribuida en toda la longitud del elemento.

        Args:
            q: carga uniformemente distribuida en toda la longitud de la viga

        Returns:
            Vector de 4 componentes, tipo np.ndarray (4,)
        """
        L = self.longitud
        v = q*L/12*np.array([6, L, 6, -L])
        return v

    def rne_qpt_vig(self, p: float, a: float) -> np.ndarray:
        """Reacción nodal equivalente, carga puntual, viga plana.

        Args:
            p: carga puntual
            a: posición de la carga puntual respecto al apoyo izquierdo

        Returns:
            Vector de 4 componentes, tipo np.ndarray (4,)
        """
        L = self.longitud

        b = L - a  # Distancia de la carga puntual al apoyo derecho
        v1 = p*b**2/L**3 * (3*a+b)
        v2 = p*a*b**2/L**2
        v3 = p*a/L**3 * (a+3*b)
        v4 = -p*a**2*b/L**2
        return np.array([v1, v2, v3, v4])

    def rne_mcd_vig(self, momento: float, a: float) -> np.ndarray:
        """Reacción nodal equivalente, momento concentrado, viga plana.

        Args:
            momento: momento concentrado
            a: posición del momento concentrado respecto al apoyo izquierdo

        Returns:
            Vector de 4 componentes, tipo np.ndarray (4,)
        """
        M = momento
        L = self.longitud

        b = L - a  # Distancia del momento concentrado al apoyo derecho
        v1 = -6*M*a*b/L**3
        v2 = M*b/L**2 * (b-2*a)
        v3 = 6*M*a*b/L**3
        v4 = M*a/L**2 * (a-2*b)

        return np.array([v1, v2, v3, v4])


class BarraViga(BarraPortico):
    """Barra de pórtico plano o espacial.

    Adoptar el critero de que el nudo inicial tenga menor enumeración que el
    nudo final.

    Sistema de coordenadas locales:
        x: eje horizontal
        y: eje vertical
        z: eje perpendicular a x e y

    :arg
        tipo (int): Tipo de estructura según se define en TIPO_ESTRUCTURA
        nudo_inicial (Nudo): nudo inicial
        nudo_final (Nudo): nudo final
        material (Material): datos del material
        seccion (Seccion): propiedades de la sección transversal
        roll (float): Ángulo en radianes medido en sentido horario cuando se
            mira en la dirección negativa del eje x local, con el cual el
            sistema xyz rota alrededor de x, tal que el xy quede vertical con
            el eje y apuntando hacia arriba (i.e. en la dirección positiva del
            eje Y global). Kassimali p477.
        versores (np.ndarray): Matriz cuyas columnas son los versores del
            sistema de coordenadas global adoptado.
    """

    @property  # Sobreescrito
    def masa_local(self) -> np.ndarray:
        """Matriz de masa local de la viga.

            Viga de Euler-Bernoulli o Timoshenko según el parámetro 'cortante'
        de la estructura.

        Si la inercia Iz es dada, se considerará la inercia rotacional. Si Iz
        no es dada, se despreciará la inercia rotacional.

        :return: Matriz 4x4
        """
        L = self.longitud
        rho = self.material.densidad
        E = self.material.elast_long
        G = self.material.elast_transv
        A = self.seccion.area
        Iz = self.seccion.inercia_z
        Ac = self.seccion.area_cortante_y

        if Iz is None:  # No se considera la inercia rotacional
            Iz = 0

        if self.cortante:  # Se considera el efecto de las cortantes
            assert E is not None or Ac is not None, "Revisa los datos"
            fs = 12*E*Iz/(G*Ac*L**2)
        else:  # No se considera el efecto de la cortante
            fs = 0.0

        # Inercia translacional
        m1 = rho*A*L/(1 + fs)**2*np.array([
            [13/35 + 7/10*fs + 1/3*fs**2, (11/210 + 11/120*fs + 1/24*fs**2)*L,
             9/70 + 3/10*fs + 1/6*fs**2, -(13/420 + 3/40*fs + 1/24*fs**2)*L],
            [(11/210 + 11/120*fs + 1/24*fs**2)*L, (1/105 + 1/60*fs +
                                                   1/120*fs**2)*L**2,
             (13/420 + 3/40*fs + 1/24*fs**2)*L, -(1/140 + 1/60*fs +
                                                  1/120*fs**2)*L**2],
            [9/70 + 3/10*fs + 1/6*fs**2, (13/420 + 3/40*fs + 1/24*fs**2)*L,
             13/35 + 7/10*fs + 1/3*fs**2, -(11/210 + 11/120*fs +
                                            1/24*fs**2)*L],
            [-(13/420 + 3/40*fs + 1/24*fs**2)*L,
             -(1/140 + 1/60*fs + 1/120*fs**2)*L**2,
             -(11/210 + 11/120*fs + 1/24*fs**2)*L,
             (1/105 + 1/60*fs + 1/120*fs**2)*L**2]
        ])

        # Inercia rotacional
        m2 = rho*A*L/(1 + fs)**2*Iz/A/L**2*np.array([
            [6/5, (1/10 - 1/2*fs)*L, -6/5, (1/10 - 1/2*fs)*L],
            [(1/10 - 1/2*fs)*L, (2/15 + 1/6*fs + 1/3*fs**2)*L**2,
             (-1/10 + 1/2*fs)*L, (-1/30 - 1/6*fs + 1/6*fs**2)*L**2],
            [-6/5, (-1/10 + 1/2*fs)*L, 6/5, (-1/30 - 1/6*fs +
                                             1/6*fs**2)*L**2],
            [(1/10 - 1/2*fs)*L, (-1/30 - 1/6*fs + 1/6*fs**2)*L**2,
             (-1/10 + 1/2*fs)*L, (2/15 + 1/6*fs + 1/3*fs**2)*L**2]
        ])
        return m1 + m2

    @property  # Sobreescrito
    def rigidez_local(self) -> np.ndarray:
        """Matriz de rigidez para elemento de viga.

        Elemento de barra de 2 grados de libertad por nudo, vertical (y) y giro
        (z).

        Si se da el valor de G (módulo de corte), se considerará la rigidez a
        cortante. Si G no es dado, no se considerará el aporte de la rigidez a
        cortante.

        Returns:
            Matriz 4x4 de tipo np.ndarray
        """
        L = self.longitud
        E = self.material.elast_long
        Iz = self.seccion.inercia_z
        G = self.material.elast_transv
        Ac = self.seccion.area_cortante_y

        if G is None:  # No se considera la rigidez al corte
            f = 0
        else:  # Se toma en cuenta la rigidez al corte
            assert Ac is not None, "Falta dato del área afectiva al corte"
            f = 12*E*Iz/(G*Ac*L**2)

        k = E*Iz/(L**3*(1 + f))*np.array([
            [12, 6*L, -12, 6*L],
            [6*L, (4 + f)*L**2, -6*L, (2 - f)*L**2],
            [-12, -6*L, 12, 6*L],
            [6*L, (2 - f)*L**2, -6*L, (4 + f)*L**2]
        ])
        return k

    # Sobreescrito
    def transf_coord(self) -> np.ndarray:
        return np.identity(4)


@dataclass
class Estructura(ABC):
    """Estructura hecha de barras.

    :arg
        tipo (int): tipo de estructura según se define en TIPO_ESTRUCTURA
        coords (dict): Coordenadas de los nudos.
            Tiene la forma: {1:C_1, 2:C_2, ...}, en donde las coordenadas C_i:
                1D: X
                2D: (X, Y)
                3D: (X, Y, Z)
            Las coordenadas deben expresarse en el sistema global adoptado
            aunque no sea el tradicional: X: horizontal; Y: vertical;
            Z: saliente.
        restricciones (dict): Nudos restringidos y sus restricciones según se
            define en la clase Nudo.
        cargas_nodales (dict): cargas nodales
        elementos (list): lista con datos de las barras para configurar los
            instancias de la clase Barra
    """
    # Públicas
    coords: dict  # Coordenadas de los nudos
    restricciones: dict  # Nudos restringidos y sus restricciones
    elementos: list  # Datos de las barras

    # Protegidas
    _tipo: ClassVar[int]  # Tipo de estructura
    _nudos: list = field(init=False)  # Lista con objetos de tipo Nudo
    _barras: list = field(init=False)  # Lista con objetos de tipo Barra
    _masa_global: np.ndarray = field(init=False)  # Matriz de masa global
    _rigidez_global: np.ndarray = field(init=False)  # Matriz de rigidez global
    _fuerza_global: np.ndarray = field(init=False)  # Vector de fuerzas nodales
    _desplaz_gdl: np.ndarray = field(init=False)  # Desplazamientos nodales
    _freqs: np.ndarray = field(init=False)  # Frecuencias naturales (Hz)
    _modos: np.ndarray = field(init=False)  # Modos naturales de vibración
    _versores: np.ndarray = VERSORES

    # Con valores por defecto
    cargas_nodales: Optional[dict] = None  # No necesarias en cálculo dinámico

    @abstractmethod
    def __post_init__(self):
        # Para verificar tipo de estructura entre otras cosas.
        pass

    @property
    def versores(self):
        return self._versores

    @versores.setter
    def versores(self, nuevos_versores):
        self._versores = nuevos_versores

    @property
    def tipo(self):
        """Tipo de estructura.

        Según lo siguiente:
            0: Barra unidimensional
            1: Reticulado plano
            2: Viga
            3: Pórtico plano
            4: Reticulado espacial
            5: Grilla
            6: Pórtico espacial
        """
        return self._tipo

    @property
    def n_nudos(self) -> int:
        """Número de nudos."""
        return len(self.coords)

    @property
    def n_barras(self) -> int:
        """Número de barras."""
        return len(self.elementos)

    @property
    def dim(self) -> int:
        """Dimensión espacial de la estructura."""
        if self.tipo == 0:  # Barra unidimensional
            return 1
        elif self.tipo in {1, 2, 3}:  # Reticulado plano, viga y pórtico plano
            return 2
        elif self.tipo in {4, 5, 6}:  # Ret. espacial, grilla, pórtico espacial
            return 3

    @property
    def ngdr(self):
        """Número de grados de restricción de la estructura."""
        return sum([sum(v) for v in self.restricciones.values()])

    @property
    def ngdl_nudo(self):
        """Número de grados de libertad por nudo."""
        return NGDL_NUDO[TIPO_ESTRUCTURA[self.tipo]]

    @property
    def ngdl_barra(self):
        """Número de grados de libertad por barra."""
        return 2*self.ngdl_nudo

    @property
    def ngda(self):
        """Número de grados de acción de la estructura."""
        return self.ngdl_nudo * self.n_nudos

    @property
    def ngdl(self):
        """Número de grados de libertad de la estructura."""
        return self.ngda - self.ngdr

    def config_nudos(self) -> None:
        """Configuración de los nudos y las barras de la estructura.

        !Ejecución independiente necesaria.

        Asigna a los nudos de la estructura:
            - Restricciones
            - Cargas nodales
            - Enumeración de grados de libertad
            - Enumeración de grados de restricción

        Crea y guarda lista de objetos de tipo Nudo y otra lista de tipo Barra
        correspondiente a la estructura.
        """
        tipo = self.tipo
        coords = self.coords
        restricciones = self.restricciones
        cargas_nodales = self.cargas_nodales
        ngdl_nudo = self.ngdl_nudo
        Nnudos = self.n_nudos  # Número de nudos

        coordenadas = list(coords.values())

        # Generación de nudos
        nudos = [Nudo(i, c, tipo) for i, c in enumerate(coordenadas)]

        # Asignación de restricciones
        for rr in restricciones.keys():  # Recorre el diccionario de restr.
            nudos[rr - 1].restr = restricciones[rr]

        # Asignación de cargas
        if cargas_nodales is not None:
            for qq in cargas_nodales.keys():  # Recorre diccionario de cargas
                nudos[qq - 1].cargas = cargas_nodales[qq]

        # Enumeración de grados de libertad
        gdl_lista = [[0] * ngdl_nudo for _ in range(Nnudos)]  # Inicialización
        contador = 0  # Contador de grados de libertad

        # Asignación de los primeros ngdl grados de libertad
        for ii, nudo in enumerate(nudos):  # Recorre los nudos
            for jj, r in enumerate(nudo.restr):  # Recorre las restricciones
                if r == 0:  # Si no hay restricción
                    contador += 1
                    gdl_lista[ii][jj] = contador
            nudo.set_gdl(tuple(gdl_lista[ii]))  # Se agrega dato al nudo

        # Asignación de los grados de restricción
        for ii, nudo in enumerate(nudos):  # Recorre los nudos nuevamente
            for jj, r in enumerate(nudo.restr):  # Recorre las restricciones
                if r == 1:  # Si hay restricción
                    contador += 1
                    gdl_lista[ii][jj] = contador
            nudo.set_gdl(tuple(gdl_lista[ii]))  # Se agrega dato al nudo

        self._nudos = nudos  # Guarda la lista nudos

    @abstractmethod
    def config(self):
        """Para configurar nudos y barras"""
        pass

    def nudos(self) -> list:
        """Lista de nudos de la estructura."""
        return self._nudos

    def barras(self) -> list:
        """Lista de barras de la estructura."""
        return self._barras

    def ensambladora(self, barra) -> np.ndarray:
        """Matriz B de ensamble global.

        :arg
            barra: Objeto de tipo Barra

        :returns
            Matriz de ensamble B.
        """
        m = self.ngdl_barra  # Número de grados de libertad por barra
        n = self.ngda  # Número de grados de acción
        gdl = barra.gdl()  # Grados de liberad de la barra
        B = np.zeros((m, n))  # Inicialización
        for i in range(m):  # Por cada grado de libertad de la barra
            B[i, gdl[i] - 1] = 1.0
        return B

    def set_matrices_globales(self, nudos: list, barras: list) -> None:
        """Ensambla y guarda matriz de masa, rigidez y vector de fuerzas.
         En los grados de acción de la estructura.

         !Ejecución independiente necesaria.
         """
        # Datos
        ngda = self.ngda  # Número de grados de acción

        # Inicialización de las matrices
        MG = np.zeros((ngda, ngda))  # Masa global
        KG = np.zeros((ngda, ngda))  # Rigidez global
        FG = np.zeros(ngda)  # Vector de fuerzas nodales

        # Ensamble de matrices de masa y rigidez
        for barra in barras:  # Recorre todas las barras
            Me = barra.masa_global  # Matriz masa del elemento en c. globales
            Ke = barra.rigidez_global  # Matriz rigidez del elemento c. glob.
            B = self.ensambladora(barra)  # Truco

            # Se agrega la colaboración de cada barra
            MG += B.T @ Me @ B
            KG += B.T @ Ke @ B

        # Ensamble del vector de fuerzas
        for nudo in nudos:  # Recorre todos los nudos
            g = np.array(nudo.gdl())  # Grados de libertad del nudo
            indices = g - 1
            P = np.array(nudo.cargas)  # Lista con cargas en los nudos
            FG[indices] = P

        self._masa_global = MG
        self._rigidez_global = KG
        self._fuerza_global = FG

    def masa_global(self) -> np.ndarray:
        """Matriz de masa de la estructura global."""
        return self._masa_global

    def rigidez_global(self) -> np.ndarray:
        """Matriz de masa de la estructura global."""
        return self._rigidez_global

    def fuerza_global(self) -> np.ndarray:
        """Vector de fuerzas nodales de la estructura global."""
        return self._fuerza_global

    def masa_gdl(self) -> np.ndarray:
        """Matriz de masa en los grados de libertad de la estructura."""
        n = self.ngdl  # Número de grados de libertad
        return self.masa_global()[:n, :n]

    def rigidez_gdl(self) -> np.ndarray:
        """Matriz de rigidez en los grados de libertad de la estructura."""
        n = self.ngdl  # Número de grados de libertad
        return self.rigidez_global()[:n, :n]

    def fuerza_gdl(self) -> np.ndarray:
        """Fuerzas nodales en los grados de libertad de la estructura."""
        return self.fuerza_global()[:self.ngdl]

    def set_desplaz_gdl(self) -> None:
        """Desplazamientos nodales en los grados de libertad.

        Solución del problema estático.
        Calcula y guarda el vector desplazamientos, además asigna los
        desplazamientos a los nudos correspondientes.

        !Ejecución independiente necesaria.
        """
        S = self.rigidez_gdl()
        F = self.fuerza_gdl()
        d = np.linalg.inv(S) @ F
        self._desplaz_gdl = d  # Guarda d

        # Asignación de los desplazamientos a los nudos correspondientes
        ngdl_nudo = self.ngdl_nudo  # Número de grados de libertad por nudo
        ngdl = self.ngdl  # Número de grados de libertad de la estructura
        for nudo in self.nudos():
            gdl = nudo.gdl()  # Grados de libertad del nudo
            dd = np.zeros(ngdl_nudo)  # Inicialización
            for i, g in enumerate(gdl):  # Recorre gdl's de cada nudo
                if g <= ngdl:  # Menor porque se utiliza desplaz_gdl
                    dd[i] = d[g - 1]  # Desplaz. del nudo
            nudo.set_desplaz(dd)  # Asigna el desplazamiento al nudo

    def desplaz_gdl(self) -> np.ndarray:
        """Desplazamientos nodales en los grados de libertad de la estructura.
        """
        return self._desplaz_gdl

    def reacciones(self) -> np.ndarray:
        """Devuelve las reacciones en los grados de restricción."""
        K10 = self.rigidez_global()[self.ngdl:, :self.ngdl]
        d = self.desplaz_gdl()
        R = K10 @ d
        return R

    def set_autos(self) -> None:
        """Valores y vectores característicos.

        Frecuencias naturales (Hz) y modos de vibración.
        """
        w2, S = eigh(self.rigidez_gdl(), self.masa_gdl())
        f = np.sqrt(w2) / (2*np.pi)  # Frecuencias en Hz
        modos_norm_uno = norm_uno(S)  # Normalización a |max| = 1
        self._freqs = f
        self._modos = modos_norm_uno

    def freqs(self) -> np.ndarray:
        """Frecuencias naturales de vibración."""
        return self._freqs

    def modos(self) -> np.ndarray:
        """Modos naturales de vibración.

        Normalizado tal que el valor absoluto del máximo es igual a 1.
        """
        return self._modos

    def procesamiento(self, tipo_analisis='estatico') -> tuple:
        """Procesamiento estático o dinámico.

        Args:
            tipo_analisis:
                'estatico' para análisis estático
                'dinamico' (o cualquier otra palabra) para análisis dinámico

        Returns:
            En ANÁLISIS ESTÁTICO:
                - Desplazamientos en los grados de libertad
                - Reacciones en los grados de restricción.

            En ANÁLISIS DINÁMICO:
                - Frecuencias naturales;
                - Modos de vibración normalizados a |max| = uno.
        """
        # Configuración de nudos y barras
        print("Configuración de nudos y barras...")
        self.config()  # Realiza configuración
        nudos = self.nudos()
        barras = self.barras()

        # Ensamble de matrices globales
        print("Ensamble de matrices globales...")
        self.set_matrices_globales(nudos, barras)

        # noinspection SpellCheckingInspection
        if tipo_analisis == 'estatico':
            if self.cargas_nodales is None:
                print("Esta estructura no tiene cargas.")
            else:
                print("Resolución del problema estático...")
                # Desplazamientos nodales estáticos
                self.set_desplaz_gdl()  # Resuelve el problema estático
                Xs = self.desplaz_gdl()

                # Reacciones estáticas
                Rs = self.reacciones()

                print("Listo, preguntá lo que quieras.")
                return Xs, Rs

        else:  # tipo_analisis = dinámico

            # Resolución del problema de autovalores
            print("Resolución del problema de autovalores...")
            self.set_autos()
            freqs = self.freqs()
            modos = self.modos()
            print("Listo.")
            return freqs, modos

    def dibujar_nudos(self, num=False, ax=None, **kwargs):
        """Dibuja los nudos como scatter usando matplotlib."""

        nudos = self.nudos()
        dim = nudos[0].dim
        if dim == 3 and ax is None:
            raise Exception('Debe proveer -ax- para el dibujo 3D')

        # Dibuja los nudos como scatter
        for n in nudos:
            n.dibujar(num=num, ax=ax, **kwargs)

    def dibujar_barras(self, espesor_area=False, color='k', ax=None, **kwargs):
        """Dibuja la estructura de barras usando matplotlib."""

        barras = self.barras()
        if self.dim == 3 and ax is None:
            raise Exception('Debe proveer -ax- para el dibujo 3D')

        # Dibuja las barras en líneas
        for b in barras:
            b.dibujar(espesor_area=espesor_area, color=color, ax=ax, **kwargs)


@dataclass
class Reticulado(Estructura):

    def __post_init__(self):
        """
        Define el tipo de reticulado en función a la dimensión espacial de las
        coordenadas nodales.
        """
        dim_nudo = len(self.coords[1])
        self._tipo = list(RETICULADOS.keys())[dim_nudo-1]

    def config(self) -> None:
        """Configuración de nudos y barras."""
        self.config_nudos()  # Configuración de nudos
        nudos = self.nudos()
        versores = self.versores

        # Configuración de las barras
        barras = [BarraReticulado(self.tipo, nudos[e[0] - 1], nudos[e[1] - 1],
                                  e[2], e[3], versores) for e in
                  self.elementos]

        self._barras = barras  # Guarda la lista barras

    def set_tensiones(self) -> None:
        """Guarda la tensión correspondiente de cada barra.

        Obs: Requiere ejecución independiente.
        """
        barras = self.barras()  # Objetos BarraReticulado
        for barra in barras:  # Recorre las barras
            barra.set_tension()  # Asigna la tensión a cada barra

    def tensiones(self) -> np.ndarray:
        """Tensiones en las barras."""
        tensiones = np.zeros(self.n_barras)
        for i, b in enumerate(self.barras()):
            sigma = b.tension()
            tensiones[i] = sigma

        return tensiones

    def normales(self) -> np.ndarray:
        """Fuerza normal de cada barrra."""
        normales = np.zeros(self.n_barras)
        for i, b in enumerate(self.barras()):
            normales[i] = b.fuerza_normal()

        return normales

    def def_units(self) -> np.ndarray:
        """Deformación unitaria de cada barra."""
        def_units = np.zeros(self.n_barras)
        for i, b in enumerate(self.barras()):
            def_units[i] = b.def_unit()

        return def_units

    def elongaciones(self) -> np.ndarray:
        """Elongación de cada barra."""
        elongs = np.zeros(self.n_barras)
        for i, b in enumerate(self.barras()):
            elongs[i] = b.elongacion()

        return elongs

    # Sobreescrito
    def procesamiento(self, tipo_analisis='estatico') -> tuple:
        """Procesamiento estático o dinámico.

        Args:
            tipo_analisis:
                'estatico' para análisis estático
                'dinamico' (o cualquier otra palabra) para análisis dinámico

        Returns:
            - Diccionario con las propiedades de la estructura
            Además:
            En ANÁLISIS ESTÁTICO:
                - Desplazamientos en los grados de libertad
                - Fuerzas en las barras;
                - Tensiones en las barras;
                - Reacciones en los grados de restricción

            En ANÁLISIS DINÁMICO:
                - Frecuencias naturales;
                - Modos de vibración normalizados a max = uno.
        """
        # Configuración de nudos y barras
        print("Configuración de nudos y barras...")
        self.config()  # Realiza configuración
        nudos = self.nudos()
        barras = self.barras()

        # Ensamble de matrices globales
        print("Ensamble de matrices globales...")
        self.set_matrices_globales(nudos, barras)

        # noinspection SpellCheckingInspection
        if tipo_analisis == 'estatico':
            if self.cargas_nodales is None:
                print("Esta estructura no tiene cargas.")
            else:
                print("Resolución del problema estático...")
                # Desplazamientos nodales estáticos
                self.set_desplaz_gdl()  # Resuelve el problema estático
                self.set_tensiones()  # Calcula y guarda las tensiones axiales

                # Resultados a mostrar
                Xs = self.desplaz_gdl()
                Ts = self.tensiones()
                Rs = self.reacciones()

                print("Listo, preguntá lo que quieras.")
                return Xs, Ts, Rs

        else:  # tipo_analisis = dinámico

            # Resolución del problema de autovalores
            print("Resolución del problema de autovalores...")
            self.set_autos()
            freqs = self.freqs()
            modos = self.modos()
            print("Listo.")
            return freqs, modos


@dataclass
class Viga(Estructura):

    def __post_init__(self):
        """Define el tipo de estructura."""
        self._tipo = 2

    def config(self):
        """AÚN NO IMPLEMENTADO"""
        self.config_nudos()  # Configuración de nudos
        nudos = self.nudos()
        return nudos


@dataclass
class Portico(Estructura):
    """Estructura tipo pórtico 2D o 3D.

    :param tipo (int): tipo de estructura según se define en TIPO_ESTRUCTURA
    :param coords (dict): Coordenadas de los nudos.
        Tiene la forma: {1:C_1, 2:C_2, ...}, en donde las coordenadas C_i:
            1D: X
            2D: (X, Y)
            3D: (X, Y, Z)
            Las coordenadas deben expresarse en el sistema global adoptado
            aunque no sea el tradicional: X: horizontal; Y: vertical;
            Z: saliente.
    :param restricciones (dict): Nudos restringidos y sus restricciones según
        se define en la clase Nudo.
    :param cargas_nodales (dict): cargas nodales
    :param elementos (list): lista con datos de las barras para configurar los
        instancias de la clase Barra.
    :param cortante (bool): Considerar o no la rigidez a cortante.
     """

    cortante: bool = False  # Considerar o no la rigidez a cortante.

    def __post_init__(self):
        """Define el tipo de estructura."""
        dim = len(self.coords[1])
        self._tipo = 3 if dim == 2 else 6

    def config(self):
        """Configuración de nudos y barras."""
        self.config_nudos()  # Configuración de nudos
        nudos = self.nudos()  # Lista de nudos del tipo Nudo
        versores = self.versores  # Versores del sistema de coord. global
        Q = self.cortante  # Consideración de la rigidez a cortante

        # Configuración de las barras
        elementos = self.elementos  # Obtenido del preprocesamiento
        bs = []  # Inicialización de la lista de barras

        for e in elementos:
            ni = e[0] - 1  # Índice del nudo inicial
            nf = e[1] - 1  # Índice del nudo final
            if len(e) == 4:  # Si no se da el roll
                bs.append(BarraPortico(self.tipo, nudos[ni], nudos[nf], e[2],
                                       e[3], versores, cortante=Q))
            else:  # Si roll es distinto de cero.
                r = e[4]/180*np.pi  # Conversión a radianes
                bs.append(BarraPortico(self.tipo, nudos[ni], nudos[nf], e[2],
                                       e[3], versores, r, cortante=Q))

        self._barras = bs  # Guarda la lista de barras
        return nudos


@dataclass
class Grilla(Estructura):

    def __post_init__(self):
        """Define el tipo de estructura."""
        self._tipo = 5

    def config(self):
        """AÚN NO IMPLEMENTADO"""
        self.config_nudos()  # Configuración de nudos
        nudos = self.nudos()
        return nudos


######################
# Funciones globales #
######################

def preprocesamiento(datos_nudos, datos_barras, materiales, secciones):
    """Preprocesamiento. Preparación de datos para la clase Estructura.

    :param datos_nudos: Una de las siguientes:
        - Archivo csv con las coordenadas de los nudos en orden.
        Se asume que la primera fila tiene los nombres de las coordenadas.
        - Diccionario: con el número de nudo como key y las coordenadas como
            values.
    :param datos_barras: Una de las siguientes:
        - Archivo csv con los datos de las barras. Se asume que la primera
        fila es de encabezado. Los datos en cada columna deben ser:
            - Primera columna: Número de nudo inicial
            - Segunda columna: Número de nudo final
            - Tercera columna: Número de material
            - Cuarta columna: Número de sección
        - Diccionario: con el número de barra como key y los mismos datos del
            caso csv como values.
    :param materiales: Lista con la clase Material
    :param secciones: Lista con la clase Seccion
    :return:
        - Diccionario de coordenadas nodales de la forma: {1: (X, Y, Z), ...}
        - Lista de elementos con datos de barras de la forma:
        [numero_nudo_inicial, numero_nudo_final, Material, Seccion]
    """
    if type(datos_nudos) is str:
        npd = pd.read_csv(datos_nudos)  # Nudos en pandas dataframe
        nnp = npd.to_numpy()  # Nudos en numpy ndarray
    elif type(datos_nudos) is dict:
        nnp = np.array(list(datos_nudos.values()))
    else:
        raise Exception("Verifica datos de nudos.")

    if type(datos_barras) == str:
        bpd = pd.read_csv(datos_barras)  # Barras en pandas dataframe
        bnp = bpd.to_numpy()  # Barras en numpy ndarray
    elif type(datos_barras) == dict:
        bnp = np.array(list(datos_barras.values()))
    else:
        raise Exception("Verifica datos de barras.")

    coordenadas = {}  # Inicialización para coordenadas nodales
    for i, n in enumerate(nnp):
        coordenadas[i] = tuple(n)

    elementos = []
    for i, b in enumerate(bnp):
        c = len(b)
        if c == 4:  # Si no se da el roll (pórticos) o no importa (reticulados)
            Ni, Nf, m, s = tuple(b)
            elementos.append([Ni, Nf, materiales[m-1], secciones[s-1]])
        elif c == 5:  # Si se da el dato de roll
            Ni, Nf, m, s, r = tuple(b)
            elementos.append([Ni, Nf, materiales[m-1], secciones[s-1], r])
        else:
            raise ValueError('Verificar datos de barras.')

    return coordenadas, elementos


def norm_uno(modos):
    """Normaliza autovectores tal que la componente máx/mín sea igual a +1/-1.

    Args:
        modos: matriz de vectores característicos

    Returns:
        Matriz con filas de vectores propios normalizados con este criterio.
    """
    # Que cada autovector tenga su máxima componente igual a 1
    max_abs = np.abs(np.max(modos, axis=0))  # Valores absolutos de máximos
    min_abs = np.abs(np.min(modos, axis=0))  # Valores absolutos de mínimos
    comp = np.array(max_abs > min_abs)  # comparación de valores absolutos

    S = modos.T * 0  # Modos normalizados en las filas
    for i, v in enumerate(modos.T):
        if comp[i]:
            S[i] = v / max_abs[i]
        else:
            S[i] = v / min_abs[i]
    return S.T


# def desplaz_t(phi_p: np.ndarray, wf: float, wn: np.ndarray, modos:np.ndarray,
#               t: float) -> np.ndarray:
#     """Desplazamientos nodales en función del tiempo.
#
#     Devuelve el vector de desplazamientos nodales u(t).Se suponen condiciones
#     iniciales homogéneas u(0)=0, du/dt(0) = 0
#
#     Ecuaciones según: Anil K. Chopra "Dinámica de estructuras" 4ta. Ed.
#
#     Args:
#         phi_p: Amplitudes de las fuerzas en los gdl's de la estructura en
#             coordenadas modales
#         wf: frecuencia angular de vibración forzada (de las fuerzas externas)
#         wn: vector de frecuencias angulares naturales
#         modos: matriz de autovectores de la estructura normalizados con la
#                 matriz de masas.
#         t: tiempo (segundos).
#
#     Returns:
#         Vector de desplazamientos en el tiempo t.
#     """
#     q = [phi_p[i] / (w ** 2 - wf ** 2) * (np.sin(wf * t) - wf / w *
#                                           np.sin(w * t)) for i, w in
#          enumerate(wn)]
#     q = np.array(q)
#     u = modos @ q
#
#     return u
#
#
# def desplaz_lapso(p: np.ndarray, ff: float, w2: np.ndarray, modos:np.ndarray,
#                   lapso: float, mps=1.0) -> np.ndarray:
#     """Desplazamientos de los gdl's de la estructura en el periodo 0 a lapso.
#
#     Calcula desplazamientos cada 1/mps segundos durante el tiempo T.
#
#     La fila 't' contiene desplazamientos de todos los nudos en el tiempo 't'
#     La columna 'g' contiene los desplaz. del gdl 'g' en el lapso 0 a T.
#
#     Args:
#         p: Amplitudes de las fuerzas en los gdl's de la estructura
#         ff: frecuencia de vibración forzada (freq. de las fuerzas externas)
#         w2: vector de autovalores de la estructura (cuadrados de las
#             frecuencias angulares)
#         modos: matriz de autovectores de la estructura normalizados con la
#                 matriz de masas.
#         lapso: período de análisis (segundos).
#         mps: muestreo por segundo. Cantidad de veces por segundo que se hará
#             el cálculo de los desplazamientos.
#
#     Returns:
#         Matriz (int(mps*T), ngdl) de desplazamientos de los nudos de la
#         estructura en el lapso de tiempo t0 = 0, tf = lapso.
#     """
#     PhiP = modos @ p  # Fuerzas en coordenadas modales
#     wf = 2 * np.pi * ff  # Frecuencia angular forzada
#     wn = np.sqrt(w2)  # Vector de frecuencias angulares naturales
#
#     tiempos = np.lin-space(0, lapso, int(mps * lapso))  # Valores de tiempo
#     m = len(tiempos)  # Número de filas = cantidad de tiempos analizados
#     n = len(p)  # Número de columnas = número de grados de libertad
#     UU = np.zeros((m, n))  # Inicialización de la matriz de desplazamientos
#     for i, t in enumerate(tiempos):
#         u = desplaz_t(PhiP, wf, wn, modos, t)
#         UU[i] = u
#
#     return UU
#
#
# def desplaz_min y max(p: np.ndarray, ff: float, w2: np.ndarray,
#                     modos: np.ndarray, lapso=10.0, mps=1.0) ->\
#         tuple[np.ndarray, np.ndarray]:
#     """Envolvente de desplazamientos dinámicos nodales (min y max).
#
#     Obs: no son concomitantes.
#
#     Args:
#         p: Amplitudes de las fuerzas en los gdl's de la estructura
#         ff: frecuencia de vibración forzada (freq. de las fuerzas externas)
#         w2: vector de autovalores de la estructura
#             (cuadrados de las frecuencias angulares)
#         modos: matriz de autovectores de la estructura normalizados con la
#                 matriz de masas.
#         lapso: período de análisis (segundos).
#         mps: muestreo por segundo
#
#     Returns:
#         Lista con valores mínimos y máximos de desplazamientos en los grados
#         de libertad de la estructura.
#     """
#     UU = desplaz_lapso(p, ff, w2, modos, lapso, mps)
#     máximos = np.max(UU, axis=0)
#     mínimos = np.min(UU, axis=0)
#
#     return mínimos, máximos
#
#
# def tensiones_min y max(propiedades: dict, p: np.ndarray, ff: float,
#                       w2: np.ndarray, modos: np.ndarray, lapso=10.0,
#                       mps=1.0) -> tuple[np.ndarray, np.ndarray]:
#     """Envolvente de tensiones dinámicas en las barras (mínimos y máximos)
#
#     Obs: no son concomitantes.
#
#     Args:
#         propiedades: Obtenido de config()
#         p: Amplitudes de las fuerzas en los gdl's de la estructura
#         ff: frecuencia de vibración forzada (freq. de las fuerzas externas)
#         w2: vector de autovalores de la estructura
#             (cuadrados de las frecuencias angulares)
#         modos: matriz de autovectores de la estructura normalizados con la
#                 matriz de masas.
#         lapso: período de análisis (segundos).
#         mps: muestreo por segundo.
#
#     Returns:
#         Tupla con valores mínimos y máximos de las tensiones.
#     """
#     UU = desplaz_lapso(p, ff, w2, modos, lapso, mps)  # Desplaz.diferentes ts
#     nb = len(propiedades['Barras'])  # Número de barras
#     nt = int(mps * lapso)  # Número de tiempos
#     FF = np.zeros((nt, nb))
#
#     for i, U in enumerate(UU):  # Recorre las deformaciones
#         FF[i] = obtener_fuerzas(propiedades)  # Fuerzas normales en barras
#     F min = np.min(FF, axis=0)  # Fuerzas mínimas en las barras
#     F max = np.max(FF, axis=0)  # Fuerzas máximas en las barras
#
#     barras = propiedades['Barras']
#     areas = np.array([b.prop_sección.area for b in barras])
#     T min = F min / areas
#     T max = F max / areas
#
#     return T min, T max
#
#
# def obtener_reacciones(kg:np.ndarray, d:np.ndarray, ngdl: int) -> np.ndarray:
#     """Devuelve las reacciones en los grados de restricción
#
#     :arg
#         kg: matriz de rigidez global
#         d: vector de desplazamientos en los gdl's
#         ngdl: número de grados de libertad
#
#     :returns
#         Reacciones en los apoyos.
#     """
#     K10 = kg[ngdl:, :ngdl]
#     R = K10 @ d
#     return R
#
#
# def reacciones_min y max(kg:np.ndarray, ng dr:int, p,ff,w2, modos,
# lapso=10.0,
#                        mps=1.0) -> tuple[np.ndarray, np.ndarray]:
#     """Envolvente de reacciones (mínimos y máximos).
#
#     :arg
#         kg: matriz de rigidez global
#         ng dr: número de grados de restricción
#         T: período de tiempo considerado desde t = 0.
#         p: Amplitudes de las fuerzas en los gdl's de la estructura
#         ff: frecuencia de vibración forzada (freq. de las fuerzas externas)
#         w2: vector de autovalores de la estructura
#         modos: matriz de autovectores de la estructura
#         mps: muestreo por segundo. Cantidad de veces por segundo que se
#         calcularán los desplazamientos.
#
#     :returns
#         Tupla con valores de reacciones mínimos y máximos.
#     """
#     ngdl = len(p)  # Número de grados de libertad
#     UU = desplaz_lapso(p, ff, w2, modos, lapso, mps)  # Desplazamientos
#     nt = UU.shape[0]  # Número de tiempos
#     RR = np.zeros((nt, ng dr))  # Inicialización
#
#     for i, U in enumerate(UU):
#         RR[i] = obtener_reacciones(kg, U, ngdl)
#
#     reacciones_mínimas = np.min(RR, axis=0)
#     reacciones_máximas = np.max(RR, axis=0)
#
#     return reacciones_mínimas, reacciones_máximas

def mostrar_resultados(resultados):
    n_results = len(resultados)  # Número de resultados
    if n_results == 4:  # Respuesta estática
        Xs, Fs, Ts, Rs = resultados  # Resultados de 'procesamiento' estático
        Ds = np.round(Xs * 1000, 2)  # Desplazamientos en mm
        Ns = np.round(Fs * 1e-3, 2)  # Fuerzas normales en kN
        Ts = np.round(Ts * 1e-6, 2)  # Tensiones en MPa
        Rs = np.round(Rs * 1e-3, 2)  # Reacciones en kN

        print('RESPUESTA ESTÁTICA')
        print('- Desplazamientos nodales (mm):', Ds)
        print('- Fuerzas normales (kN):', Ns)
        print('- Tensiones normales (MPa):', Ts)
        print('- Reacciones (kN):', Rs)
        print()

    elif n_results == 2:  # Respuesta dinámica
        fs, S = resultados  # Resultados de 'procesamiento' dinámico
        freqs = np.round(fs, 2)  # Frecuencias en Hz
        modos = np.round(S, 2)  # Modos de vibración

        print('RESPUESTA DINÁMICA')
        print('- Frecuencias naturales (Hz):', freqs)
        print('- Modos de vibración:', modos)
        print()

    else:
        print('Verificar la cantidad de resultados de -procesamiento-')
        print()

#
# def respuesta_dinámica(frecuencias: np.ndarray, u_env: tuple[np.ndarray],
#                        s_env: tuple[np.ndarray], r_env: tuple[np.ndarray],
#                        lapso: float) -> None:
#     """Resultados del análisis dinámico.
#
#     Los valores no son concomitantes, es decir no aparecen al mismo tiempo.
#
#     Args:
#         frecuencias: frecuencias naturales de la estructura
#         u_env: Envolvente de desplazamientos (mín y máx) de los gdl's
#         s_env: Envolvente de tensiones
#         r_env: Envolvente de reacciones
#         lapso: período de tiempo analizado, desde t=0s
#
#     Returns:
#         Imprime los resultados
#     """
#     freq = np.round(frecuencias, 2)  # Frecuencias
#     U min = np.round(u_env[0] * 1000, 2)  # Desplazamientos mínimos en mm
#     U max = np.round(u_env[1] * 1000, 2)  # Desplazamientos máximos en mm
#     T min = np.round(s_env[0] * 1e-6, 2)  # Tensiones mínimas en MPa
#     T max = np.round(s_env[1] * 1e-6, 2)  # Tensiones máximas en MPa
#     R min = np.round(r_env[0] * 1e-3, 2)  # Reacciones mínimas en kN
#     R max = np.round(r_env[1] * 1e-3, 2)  # Reacciones máximas en kN
#
#     print('RESULTADOS DINÁMICOS:')
#     print('- Período de tiempo analizado (s):', lapso, 's')
#     print('- Frecuencias naturales (Hz):', freq)
#     print('- Envolvente de desplazamientos (mm):', list(zip(U min, U max)))
#     print('- Envolvente de tensiones (MPa):', list(zip(T min, T max)))
#     print('- Envolvente de reacciones (MPa):', list(zip(R min, R max)))
#     print()
#
#
# def mostrar_resultados(resultados):
#     """Resultado del análisis estático o dinámico.
#
#     Args:
#         resultados: tupla devuelta por el método 'procesamiento'
#
#     Returns:
#         Imprime resultados.
#     """
#     tipo_análisis = resultados[-1]
#     if tipo_análisis == 'estático':
#         _, Xs, Fs, Ts, Rs, _ = resultados
#         # Resultados estáticos
#         respuesta_estática(Xs, Fs, Ts, Rs)
#     else:
#         _, frecuencias, modos, UU, U env, Tenv, Renv, lapso, _ = resultados
#
#         # Resultados dinámicos
#         respuesta_dinámica(frecuencias, U env, Tenv, Renv, lapso)


def grafica_de_modo(barras, n, amp):

    fg = plt.figure()
    axs = fg.add_subplot(projection='3d')
    axs.set_box_aspect((1, 1, 6))
    axs.set_xlim3d(-.5, .5)
    axs.set_ylim3d(-.5, .5)
    axs.set_zlim3d(0, 6)

    axs.set_title(f"Modo de vibración Nº {n+1}\n")
    axs.set_xlabel("X (m)")
    axs.set_ylabel("Y (m)")
    axs.set_zlabel("Z (m)")
    axs.tick_params(axis='x', labelsize=6)
    axs.tick_params(axis='y', labelsize=6)
    axs.tick_params(axis='z', labelsize=10)
    axs.zaxis.labelpad = 30

    # Estructura deformada
    for b in barras:
        b.dibuja3d_deform(axs, amp)


def main():
    return


if __name__ == "__main__":
    main()
