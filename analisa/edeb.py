# Estática y Dinámica de Estructuras de Barras 1D, 2D, 3D
#
# @author: Fredy Gabriel Ramírez Villanueva
# Inicio de escritura del código: 06 de mayo de 2022
# Release 1: 18 de mayo de 2022
# Control git: 01 de junio de de 2022
# Versión 0.1: 02 de julio de 2022

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh

"""
Resolución de problemas estático y dinámico para estructuras de barras.

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
"""
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


# %%
@dataclass
class Nudo:
    """Nudo de estructura de barras

    :arg
        num (int): número asignado al nudo, inicia en 0;
        coord (tuple[float]): Coordenadas de los nudos. Puede contener 1, 2 o 3
            valores según la dimensión considerada;
        tipo (int): Uno de los valores del siguiente diccionario:
            {0: 'Barra unidimensional', 1: 'Reticulado plano', 2: 'Viga',
             3: 'Pórtico plano', 4: 'Reticulado espacial', 5: 'Grilla',
             6: 'Pórtico espacial'}

        restr (tuple[int]): Restricciones según los gdl's del nudo.
            Si no es dado, se supondrá que el nudo no está restringido.
            1: restringido; 0: libre
            - Ejemplo en Reticulado espacial:
                (0, 1, 1): libre en x, restringido en y y z
            - Ejemplo en Viga:
                (1, 0): restringido en y, libre en el giro
            - Ejemplo en Barra unidimensional:
                1: restringido en x
    """
    # Variables públicas
    num: int  # Número asignado al nudo, inicia en 0.
    coord: tuple[float]  # Coordenadas
    tipo: int  # Tipo de estructura para el nudo
    restr: tuple[int] = field(init=False)  # Restricciones

    # Variables protegidas
    _cargas: np.ndarray = field(init=False)  # Cargas en los nudos
    _gdl: tuple[int] = field(init=False)  # Enumeración de los gdl's
    _desplaz: np.ndarray = field(init=False)  # Desplaz. del nudo s/ ngdl.

    def __post_init__(self):
        tipo = self.tipo
        assert tipo in TIPO_ESTRUCTURA, \
            f"Invalid structure type: {tipo}.\
            Must be one of {list(TIPO_ESTRUCTURA.keys())}"

        ngdl = NGDL_NUDO[TIPO_ESTRUCTURA[tipo]]
        self.restr = tuple([0] * ngdl)
        self._cargas = np.array([0.0] * ngdl)
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

    def cargas(self) -> np.ndarray:
        """Cargas en el nudo."""
        return self._cargas

    def set_cargas(self, cargas) -> None:
        self._cargas = cargas

    def set_desplaz(self, u: np.ndarray) -> None:
        """Configura desplazamiento del nudo.

        :param u: Vector de desplazamientos del nudo
        """
        self._desplaz = u

    def desplaz(self) -> np.ndarray:
        """Devuelve desplazamiento nodal en los grados de libertad."""
        return self._desplaz

    def translacion(self) -> np.ndarray:
        """Translación del nudo luego de la acción de las _cargas.

        :return: Desplazamientos en los grados de libertad de translación.
        """
        u_trans = self.desplaz()[:self.ntr]
        return u_trans

    def rotacion(self) -> np.ndarray:
        """Rotación del nudo luego de la acción de las _cargas.

        :return: Desplazamientos en los grados de libertad de translación.
        """
        u_rot = self.desplaz()[self.ntr:]
        return u_rot

    def posicion(self, amp=1.0) -> np.ndarray:
        """Posición del nudo después de la aplicación de las _cargas.

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
                plt.text(x, y, str(self.num))

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
        coef_dilat (float): (opcional) coeficiente de dilatación lineal
    """

    # Públicos
    elast_long: float = 200e9  # Módulo de elasticidad longitudinal (Young)
    densidad: float = 7850.0  # Densidad del material
    poisson: Optional[float] = None  # Módulo de Poisson
    coef_dilat: Optional[float] = None  # Coeficiente de dilatación lineal

    # Protegidos
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

    area: float = 1.0  # Área de la sección transversal
    inercia_y: Optional[float] = None  # Inercia alrededor del eje y
    inercia_z: Optional[float] = None  # Inercia alrededor del eje z
    modulo_torsion: Optional[float] = None  # Módulo de torsión
    area_cortante_y: Optional[float] = None  # Área efectiva a cortante en y
    area_cortante_z: Optional[float] = None  # Área efectiva a cortante en z


# Función global para instancias de Sección
# TODO: ver introducir dentro de la clase Seccion o devolver también material
def perfil_aisc(nombre: str, aisc: pd.DataFrame) -> Seccion:
    """Instancia de Seccion para el perfil dado.

    Args:
        nombre: Nombre del perfil según convención AISC
        asic: Base de datos de perfiles AISC, por ejemplo:
            "aisc-shapes-database-v15.0.xlsx"

    Ver:
    https://www.aisc.org/publications/steel-construction-manual-resources/#37584

    Returns:
        Instancia de tipo Seccion
    """
    try:
        idx = aisc.index[aisc['EDI_Std_Nomenclature'] == nombre][0]
    except IndexError:
        raise ValueError(f"Profile '{nombre}' not found in AISC database.")

    A = aisc.loc[idx, 'A.1']*1e-3**2  # Área
    Iy = aisc.loc[idx, 'Iy.1']*1e-3**4*1e6  # Inercia en y
    Iz = aisc.loc[idx, 'Ix.1']*1e-3**4*1e6  # Inercia en z
    J_temp = aisc.loc[idx, 'J.1']  # Módulo de torsión

    # Si no se da el valor de J se asume igual a cero.
    if isinstance(J_temp, int) or isinstance(J_temp, float):
        J = J_temp*1e-3**4*1e3
    else:
        J = 0.0
        print(f'Chequear el valor de J para {nombre}. Se asume igual a cero.')

    s = Seccion(A, Iy, Iz, J)
    return s


# Función global para instancias de Sección
# TODO: probablemente es mejor que sea parte de la clase Seccion
def perfil_gerdau(nombre: str, gerdau: pd.DataFrame) -> Seccion:
    """Instancia de Seccion para el perfil dado.

    :param nombre: Nombre del perfil según convención AISC

    Perfiles L de alas iguales Gerdau Corsa
        https://www.gerdaucorsa.com.mx/sites/mx_gerdau/files/PDF/
        Manual_Perfiles_Estructurales_2019_new%20Validado-min_8.pdf
        f2 = "../analisa/database/gerdaucorsa.xlsx"
        gerdau = pd.read_excel(io=f2, sheet_name=1)


    :returns Instancia de tipo Seccion
    """
    idx = gerdau.index[gerdau['Designación'] == nombre][0]
    try:
        idx = gerdau.index[gerdau['Designación'] == nombre][0]
    except IndexError:
        raise ValueError(f"Profile '{nombre}' not found in GERDAU database.")

    A = gerdau.loc[idx, 'A']*1e-2**2  # Área
    Iy = gerdau.loc[idx, 'Iy']*1e-2**4  # Inercia en y
    Iz = gerdau.loc[idx, 'Ix']*1e-2**4  # Inercia en z
    J = gerdau.loc[idx, 'J']*1e-2**4  # Módulo de torsión
    s = Seccion(A, Iy, Iz, J)
    return s


#############################
# Tratamiento de las barras #
#############################

class InvalidStructureType(Exception):
    '''Custom exception class'''
    pass


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
        num (int): Número asignado a la barra;
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
    """

    # Métodos abstractos:
    # 1. __post_init__
    # 2. rigidez_local
    # 3. masa_local
    # 4. transf_coord

    # Públicas
    num: int  # Número asignado a la barra
    tipo: int  # Tipo de estructura al que pertenece la barra
    nudo_inicial: Nudo  # Nudo inicial
    nudo_final: Nudo  # Nudo final
    material: Material  # Propiedades del material
    seccion: Seccion  # Propiedades geométricas de la sección

    # Matriz de versores del sistema de coordenadas global
    versores: np.ndarray = field(default_factory=lambda: VERSORES)

    # Ángulo roll en radianes. Ver Kassimalli Cap8
    roll: float = 0.0

    # Protegido
    # Carga distribuida en toda la barra, en coord. locales.
    # En gral.: np.array([qx, qy, qz]). Para vigas (float): q
    _carga: Optional[np.ndarray] = None

    @abstractmethod
    def __post_init__(self):
        # Para verificar el tipo correcto de estructura entre otras cosas
        assert self.nudo_inicial.dim == self.nudo_final.dim, \
            "Initial and final nodes must have the same dimension."

    @property
    def peso_propio(self):
        '''Peso propio de la barra

        Returns:
            peso propio de la barra.
        '''
        g = 9.80665  # aceleración de la gravedad
        L = self.longitud
        A = self.seccion.area
        peso_esp = self.material.densidad * g

        return A * L * peso_esp

    @property
    def carga(self) -> Union[np.ndarray, float]:
        """Carga uniformemente distribuida en toda la barra."""
        if self._carga is None:
            # Si la carga no es dada, se asume igual a cero.
            if self.tipo in (0, 2, 5):  # Barra unid, viga y grilla
                return 0
            elif self.tipo == 3:  # Pórtico plano
                return np.zeros(2)
            elif self.tipo == 6:  # Pórtico espacial
                return np.zeros(3)
        else:
            return self._carga

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
        """Grados de libertad de los nudos extremos de la barra.
        Inicia en 0.
        """
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
        return NGDL_NUDO[TIPO_ESTRUCTURA[self.tipo]]

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
        versT = self.versores.T  # para facilitar

        if dim == 1:
            return np.array([1])

        else:  # Para 2D o 3D
            ci = np.array(self.nudo_inicial.coord)  # Coords. del nudo inicial
            cf = np.array(self.nudo_final.coord)  # Coords. del nudo final
            L = self.longitud
            v = cf - ci  # Barra como vector

            # Cosenos directores del eje local x
            ix = np.array([np.dot(v, versT[i][:dim]) for i in range(dim)]) / L

            if dim == 2:  # 2D
                c, s = ix[0], ix[1]
                return np.array([
                    [c, s],
                    [-s, c]
                ])

            else:  # 3D
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
                    iy = np.cos(roll)*iyb + np.sin(roll)*izb
                    # Cosenos directores del eje local z
                    iz = -np.sin(roll)*iyb + np.cos(roll)*izb
                else:  # si ix es paralelo a IY
                    iz = ka  # Kassimalli p477 2do párrafo.
                    iy = np.cross(iz, ix)

                return np.array([ix, iy, iz])  # Kassimalli (8.62)

    @abstractmethod
    def transf_coord(self) -> np.ndarray:
        """Matriz de transformación de coordenadas globales a locales."""
        pass

    @abstractmethod
    def carga_equiv_local(self) -> np.ndarray:
        """Vector de cargas nodales equivalentes en coordenadas locales."""
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

    def carga_equiv_global(self) -> np.ndarray:
        """Vector de cargas nodales equivalentes en coordenadas globales."""
        Qf = self.carga_equiv_local()  # Cargas en coord. locales
        T = self.transf_coord()  # Matriz de transf. de coordenadas
        Ff = T.T @ Qf  # Cargas en coord. globales
        return Ff

    def desplaz_global(self) -> np.ndarray:
        """Desplazamientos nodales de la barra en coordenadas globales.

        Returns:
            Vector de desplazamientos nodales. Tipo np.ndarray.
        """
        n = self.ngdl_nudo  # Número de grados de libertad por nudo

        v = np.zeros(self.ngdl)  # Inicialización
        v[:n] = self.nudo_inicial.desplaz()  # Desplazamientos del nudo inicial
        v[n:] = self.nudo_final.desplaz()  # Desplazamientos del nudo final

        return v

    def fuerza_global(self) -> np.ndarray:
        """Vector de fuerzas en los nudos de la barra en coord. globales."""

        v = self.desplaz_global()
        K = self.rigidez_global
        Ff = self.carga_equiv_global()
        F = K @ v - Ff
        return F

    def desplaz_local(self) -> np.ndarray:
        """Vector de desplazamientos nodales de la barra en coord. locales."""
        v = self.desplaz_global()
        T = self.transf_coord()
        u = T @ v
        return u

    def fuerza_local(self) -> np.ndarray:
        """Vector de fuerzas en nudos de la barra en coordenadas locales."""
        k = self.rigidez_local
        u = self.desplaz_local()
        Qf = self.carga_equiv_local()
        Q = k @ u - Qf
        return Q

    def dibujar(self, num=False, espesor_area=True,
                color='k', ax=None, **kwargs) -> None:
        """Dibuja la barra en 2D o 3D.

        Args:
            cuadro: medidas para el cuadro de la figura
            num (int): agregar número asignado a la barra
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

        X0 = np.asarray(self.nudo_inicial.coord)  # Coords. del nudo inicial
        Xf = np.asarray(self.nudo_final.coord)  # Coordenadas del nudo final
        coord_separadas = list(zip(X0, Xf))
        X = coord_separadas[0]
        Y = coord_separadas[1]
        Xm = (X0 + Xf)/2  # Coordenadas del punto medio
        if self.dim == 2:
            if ax is None:
                plt.plot(X, Y, color=color, linewidth=lw, **kwargs)
                plt.gca().set_aspect('equal', adjustable='box')
                if num:  # Agregar número asignado al nudo
                    plt.text(Xm[0], Xm[1], str(self.num))
            else:
                ax.plot(X, Y, color=color, linewidth=lw, **kwargs)
                ax.set_aspect('equal', adjustable='box')
                if num:  # Agregar número asignado al nudo
                    ax.text(Xm[0], Xm[1], str(self.num))

        elif self.dim == 3:
            Z = coord_separadas[2]
            ax.plot(X, Y, Z, color=color, linewidth=lw, **kwargs)
            if num:
                ax.text(Xm[0], X[1], X[2], str(self.num))

    def dibujar_deform(self, num=False, espesor_area=True, color='k', amp=1,
                       ax=None, **kwargs) -> None:
        """Dibuja la barra en su posición desplazada.

        :param num: imprimir número de barra
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

        X0d = np.asarray(self.nudo_inicial.posicion(amp))
        Xfd = np.asarray(self.nudo_final.posicion(amp))
        coord_separadas = list(zip(X0d, Xfd))
        X = coord_separadas[0]
        Y = coord_separadas[1]
        Xm = (X0d + Xfd)/2  # Coordenadas del punto medio

        if self.dim == 2:
            plt.plot(X, Y, color=color, linewidth=lw, **kwargs)
            if num:  # Agregar número asignado al nudo
                plt.text(Xm[0], Xm[1], str(self.num))

        if self.dim == 3:
            Z = coord_separadas[2]
            ax.plot(X, Y, Z, color=color, linewidth=lw, **kwargs)
            if num:
                ax.text(Xm[0], X[1], X[2], str(self.num))


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
        num (int): Número asignado a la barra;
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
        if self.tipo not in RETICULADOS:
            raise InvalidStructureType(
                f"Invalid structure type for truss element: {self.tipo}")

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

    def carga_equiv_local(self) -> np.ndarray:
        """Vector de _cargas equivalentes nodales en coordenadas locales.

        En el caso de barra de reticulado, este vector es nulo.
        """
        return np.zeros(self.ngdl)

    # Sobre-escritura del método abstracto de la clase Barra
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

    def dibujar(self, num=False, espesor_area=True,
                color='tension', ax=None, **kwargs) -> None:
        """Dibuja la barra en 2D o 3D.

        Args:
            num (bool): imprimir número de barra
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
            if sigma > 1e-4:  # Tracción
                c = 'b'
            elif sigma < -1e-4:  # Compresión
                c = 'r'
            else:
                c = '0.5'  # Gris para tensión cero
        else:
            c = color

        X0 = np.asarray(self.nudo_inicial.coord)  # Coords. del nudo inicial
        Xf = np.asarray(self.nudo_final.coord)  # Coordenadas del nudo final
        coord_separadas = list(zip(X0, Xf))
        X = coord_separadas[0]
        Y = coord_separadas[1]
        Xm = (X0 + Xf) / 2  # Coordenadas del punto medio
        if self.dim == 2:
            if ax is not None:
                ax.plot(X, Y, color=c, linewidth=lw, **kwargs)
                ax.set_aspect('equal', adjustable='box')
                if num:
                    ax.text(Xm[0], Xm[1], str(self.num))
            else:
                plt.plot(X, Y, color=c, linewidth=lw, **kwargs)
                plt.gca().set_aspect('equal', adjustable='box')
                if num:
                    plt.text(Xm[0], Xm[1], str(self.num))

        elif self.dim == 3:
            assert ax is not None, 'Debe proveer ax para el dibujo en 3D'
            Z = coord_separadas[2]
            ax.plot(X, Y, Z, color=c, linewidth=lw, **kwargs)
            if num:
                ax.text(Xm[0], X[1], X[2], str(self.num))

    def dibujar_deform(self, num=False, espesor_area=1, amp=1,
                       colorear=False, ax=None, **kwargs) -> None:
        """Dibuja la barra en su posición desplazada

        :param num: imprimir número de barra
        :param espesor_area: Considerar en el dibujo el espesor
            relativo de la barra según su área multiplicado por el valor dado.
        :param amp: Factor ae amplificación de desplazamientos
        :param colorear: Colorear la barra según su tensión
        :param ax: Axes 3D matplotlib para el dibujo en 3D
        :param kwargs: Argumentos que se pasarán a plt.plot o ax.plt
        """

        lw = np.sqrt(self.seccion.area) * 2 * espesor_area

        if colorear:
            sigma = self.tension()
            if sigma > 1e-5:  # Tracción
                color = 'b'
            elif sigma < -1e-5:  # Compresión
                color = 'r'
            else:  # Tensión cero
                color = 'k'
        else:
            color = 'g'

        X0d = np.asarray(self.nudo_inicial.posicion(amp))
        Xfd = np.asarray(self.nudo_final.posicion(amp))
        coord_separadas = list(zip(X0d, Xfd))
        X = coord_separadas[0]
        Y = coord_separadas[1]
        Xm = (X0d + Xfd) / 2  # Coordenadas del punto medio

        if self.dim == 2:
            if ax is not None:
                ax.plot(X, Y, color=color, linewidth=lw, **kwargs)
                ax.set_aspect('equal', adjustable='box')
                if num:
                    ax.text(Xm[0], Xm[1], str(self.num))
            else:
                plt.plot(X, Y, color=color, linewidth=lw, **kwargs)
                plt.gca().set_aspect('equal', adjustable='box')
                if num:
                    plt.text(Xm[0], Xm[1], str(self.num))

        if self.dim == 3:
            assert ax is not None, 'Debe proveer ax para el dibujo en 3D'
            Z = coord_separadas[2]
            ax.plot(X, Y, Z, color=color, linewidth=lw, **kwargs)
            if num:
                ax.text(Xm[0], X[1], X[2], str(self.num))


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
            sistema de coordenadas global adoptado, usualmente identidad.
        roll (float): Ángulo en radianes medido en sentido horario cuando se
            mira en la dirección negativa del eje x local, con el cual el
            sistema xyz rota alrededor de x, tal que el xy quede vertical con
            el eje y apuntando hacia arriba (i.e. en la dirección positiva del
            eje Y global). Kassimali p477.
        cortante (bool): Consideración de la rigidez a cortante.
    """

    cortante: bool = False  # Considerar el efecto de la cortante

    # Métodos sobreescritos:
    # 1. __post_init__
    # 2. rigidez_local
    # 3. masa_local
    # 4. transf_coord

    # Sobreescrito
    def __post_init__(self):
        if self.tipo not in {2, 3, 5, 6}:
            raise InvalidStructureType(
                f"Invalid structure type for frame element: {self.tipo}")

        if self.cortante and self.seccion.area_cortante_y is None:
            raise ValueError(
                "Shear area (area_cortante_y) must be provided when \
                considering shear stiffness.")

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
            Jx = self.seccion.modulo_torsion
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
        Acy = self.seccion.area_cortante_y
        if self.cortante:  # Se considera la rigidez al corte
            assert Acy is not None, "Falta dato de área efectiva al corte"
            fy = 12*E*Iz/(G*Acy*L**2)
        else:  # No se toma en cuenta la rigidez al corte
            fy = 0

        k = np.array([
            [E*A/L, 0, 0, -E*A/L, 0, 0],
            [0, 12*E*Iz/L**3/(1+fy), 6*E*Iz/L**2/(1+fy), 0,
             -12*E*Iz/L**3/(1+fy), 6*E*Iz/L**2/(1+fy)],
            [0, 6*E*Iz/L**2/(1+fy), (4+fy)*E*Iz/L/(1+fy), 0,
             -6*E*Iz/L**2/(1+fy), (2-fy)*E*Iz/L/(1+fy)],
            [-E*A/L, 0, 0, E*A/L, 0, 0],
            [0, -12*E*Iz/L**3/(1+fy), -6*E*Iz/L**2/(1+fy), 0,
             12*E*Iz/L**3/(1+fy), -6*E*Iz/L**2/(1+fy)],
            [0, 6*E*Iz/L**2/(1+fy), (2-fy)*E*Iz/L/(1+fy), 0,
             -6*E*Iz/L**2/(1+fy), (4+fy)*E*Iz/L/(1+fy)]
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
        J = self.seccion.modulo_torsion
        G = self.material.elast_transv
        [Acy, Acz] = self.seccion.area_cortante_y, self.seccion.area_cortante_z

        if self.cortante:  # Se toma en cuenta la rigidez al corte
            fy = 12*E*Iz/(G*Acy*L**2)
            fz = 12*E*Iy/(G*Acz*L**2)
        else:  # No se considera la rigidez al corte
            fy, fz = 0, 0

        k = np.array([
            [E*A/L, 0, 0, 0, 0, 0, -E*A/L, 0, 0, 0, 0, 0],
            [0, 12*E*Iz/(L**3*(1+fy)), 0, 0, 0, 6*E*Iz/L**2/(1+fy), 0,
             -12*E*Iz/L**3/(1+fy), 0, 0, 0, 6*E*Iz/L**2/(1+fy)],
            [0, 0, 12*E*Iy/L**3/(1+fz), 0, -6*E*Iy/L**2/(1+fz), 0, 0, 0,
             -12*E*Iy/L**3/(1+fz), 0, -6*E*Iy/L**2/(1+fz), 0],
            [0, 0, 0, G*J/L, 0, 0, 0, 0, 0, -G*J/L, 0, 0],
            [0, 0, -6*E*Iy/L**2/(1+fz), 0, (4+fz)*E*Iy/L/(1+fz), 0, 0, 0,
             6*E*Iy/L**2/(1+fz), 0, (2-fz)*E*Iy/L/(1+fz), 0],
            [0, 6*E*Iz/L**2/(1+fy), 0, 0, 0, (4+fy)*E*Iz/L/(1+fy), 0,
             -6*E*Iz/L**2/(1+fy), 0, 0, 0, (2-fy)*E*Iz/L/(1+fy)],
            [-E*A/L, 0, 0, 0, 0, 0, E*A/L, 0, 0, 0, 0, 0],
            [0, -12*E*Iz/L**3/(1+fy), 0, 0, 0, -6*E*Iz/L**2/(1+fy), 0,
             12*E*Iz/L**3/(1+fy), 0, 0, 0, -6*E*Iz/L**2/(1+fy)],
            [0, 0, -12*E*Iy/L**3/(1+fz), 0, 6*E*Iy/L**2/(1 + fz), 0, 0, 0,
             12*E*Iy/L**3/(1+fz), 0, 6*E*Iy/L**2/(1+fz), 0],
            [0, 0, 0, -G*J/L, 0, 0, 0, 0, 0, G*J/L, 0, 0],
            [0, 0, -6*E*Iy/L**2/(1+fz), 0, (2-fz)*E*Iy/L/(1+fz), 0, 0, 0,
             6*E*Iy/L**2/(1+fz), 0, (4+fz)*E*Iy/L/(1+fz), 0],
            [0, 6*E*Iz/L**2/(1+fy), 0, 0, 0, (2-fy)*E*Iz/L/(1+fy), 0,
             -6*E*Iz/L**2/(1+fy), 0, 0, 0, (4+fy)*E*Iz/L/(1+fy)]
        ])
        return k

    @property  # Sobreescrito
    def rigidez_local(self) -> np.ndarray:
        """Matriz de rigidez en coordenadas locales."""

        dim = self.dim
        if dim == 2:  # Pórtico plano
            return self.rigidez_p2d()
        return self.rigidez_p3d()  # Pórtico espacial

    def carga_equiv_local(self) -> np.ndarray:
        """Vector de cargas nodales equivalentes en coordenadas locales.

        Admite solo cargas distribuidas en toda la longitud de la barra.
        Los signos de las cargas deben coincidir con el sistema de coordenadas
        global.
        """
        # TODO verificar validez de resultados
        Q = self.carga  # Cargas en coordenadas globales
        Qf = np.zeros(self.ngdl)

        if Q is not None:
            dim = self.dim
            M = self.matriz_cos_dir()
            q = M @ Q  # Cargas en coordenadas locales
            L = self.longitud
            if dim == 1:
                Qf = q*L/2*np.array([1, 1])
            elif dim == 2:
                qx, qy = q
                Qf = np.array([qx*L/2,
                               qy*L/2,
                               qy*L**2/12,
                               qx*L/2,
                               qy*L/2,
                               -qy*L**2/12])
            elif dim == 3:
                qx, qy, qz = q
                Qf = np.array([qx*L/2,
                               qy*L/2,
                               qz*L/2,
                               0,
                               qz*L**2/12,
                               qy*L**2/12,
                               qx*L/2,
                               qy*L/2,
                               qz*L/2,
                               0,
                               -qz*L**2/12,
                               -qy*L**2/12])
        return Qf

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

    def abscisas(self, num: int = 50):
        """Valores de abcisas en coordenadas locales para los diagramas.

        :param num: número de puntos
        """
        return np.linspace(0, self.longitud, num)


class BarraViga(BarraPortico):
    """Barra de viga simple o continua.

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
        versores (np.ndarray): Matriz cuyas columnas son los versores del
            sistema de coordenadas global adoptado.
        roll (float): Ángulo en radianes medido en sentido horario cuando se
            mira en la dirección negativa del eje x local, con el cual el
            sistema xyz rota alrededor de x, tal que el xy quede vertical con
            el eje y apuntando hacia arriba (i.e. en la dirección positiva del
            eje Y global). Kassimali p477.
        cortante (bool): Consideración de la rigidez a cortante.
    """
    # Protegido
    # Sobreescrito
    _carga: float = 0.0  # Carga distribuida en toda la barra

    # Sobreescrito
    def __post_init__(self):
        if self.tipo not in {2}:
            raise InvalidStructureType(
                f"Invalid structure type for beam element: {self.tipo}")

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

        k1 = self.rigidez_p2d()  # Rigidez de pórtico plano

        # Elimina la 1ra y 3ra filas y columnas
        k = np.delete(np.delete(k1, [0, 2], 0), [0, 2], 1)

        return k

    # Sobreescrito
    def carga_equiv_local(self) -> np.ndarray:
        """Vector de cargas nodales equivalentes en coordenadas locales.

        Admite solo cargas distribuidas en toda la longitud de la barra.
        El signo de la carga distribuida es positivo hacia arriba.
        """
        L = self.longitud
        q = self.carga  # Positivo hacia arriba

        # Acciones nodales equivalentes
        Qf = q*L/12*np.array([6, L, 6, -L])  # Notar los signos
        return Qf

    # Sobreescrito
    def transf_coord(self) -> np.ndarray:
        return np.identity(4)

    def cortantes(self, num: int = 50) -> np.ndarray:
        """Valores de la fuerza cortante para diagrama de fuerza cortante.

        :param num: número de puntos para el diagrama
        """
        Q = self.fuerza_local()  # Vector de fuerzas nodales
        V1 = Q[0]  # Cortante a la izq. de la barra
        q = self.carga  # Carga uniformemente distribuida en toda la barra
        return V1 + q*self.abscisas(num)

    def flectores(self, num: int = 50) -> np.ndarray:
        """Valores de momentos flectores para el diagrama.

        :param num: número de puntos para el diagrama
        """
        Q = self.fuerza_local()  # Vector de fuerzas nodales
        V1 = Q[0]  # Cortante a la izq. de la barra
        M1 = Q[1]  # Flector a la izq. de la barra
        q = self.carga  # Carga uniformemente distribuida en toda la barra
        x = self.abscisas(num)
        return V1*x - M1 + q*x**2/2


@dataclass
class BarraGrilla(BarraPortico):
    """Barra de grilla.

    Adoptar el critero de que el nudo inicial tenga menor enumeración que el
    nudo final.

    Sistema de coordenadas locales (xyz):
        x: eje de la barra.
        y: ortogonal al plano de la grilla.
        z: eje ortogonal a x e y

    Sistema de coordenadas globales (XYZ):
        - La grilla se desarrolla en el plano XZ.
        - El eje y local es paralelo al Y global.

    Params:
        num (int): número asignado a la barra;
        tipo (int): Tipo de estructura según se define en TIPO_ESTRUCTURA;
        nudo_inicial (Nudo): nudo inicial;
        nudo_final (Nudo): nudo final;
        material (Material): datos del material;
        seccion (Seccion): propiedades de la sección transversal;
        versores (np.ndarray): Matriz cuyas columnas son los versores del
            sistema de coordenadas global adoptado, usualmente identidad;
        cortante (bool): Consideración de la rigidez a cortante.
    """
    # Aún no implementado

    # Sobreescrito
    def __post_init__(self):
        if self.tipo not in {5}:
            raise InvalidStructureType(
                f"Invalid structure type for grid element: {self.tipo}")

        if self.cortante and self.seccion.area_cortante_y is None:
            raise ValueError(
                "Shear area (area_cortante_y) must be provided when \
                considering shear stiffness.")

        # Verificación de que la barra esté en el plano XZ
        if self.nudo_final.coord[1] != self.nudo_inicial.coord[1]:
            raise ValueError("La barra debe estar en el plano XZ")

    # Sobreescrito
    # TODO: se necesitan inercias en todos los ejes.
    @property
    def masa_local(self) -> np.ndarray:
        """Matriz de masa local de la barra de grilla.

        Teoría de Euler-Bernoulli.

        Si la inercia Iz es dada, se considerará la inercia rotacional. Si Iz
        no es dada, se despreciará la inercia rotacional.

        Returns:
            Matriz 6x6
        """
        if self.seccion.inercia_y is None:
            self.seccion.inercia_y = 1

        matriz_portico = super().masa_local  # Matriz de barra pórtico 12x12

        filas_a_aliminar = [0, 2, 4, 6, 8, 10]
        columnas_a_eliminar = [0, 2, 4, 6, 8, 10]

        # Matriz de 6x6
        matriz_grilla = np.delete(matriz_portico, filas_a_aliminar, axis=0)
        matriz_grilla = np.delete(matriz_portico, columnas_a_eliminar, axis=1)

        return matriz_grilla


@dataclass
class Estructura(ABC):
    """Estructura hecha de barras.

    La enumeración de los nudos inicia en 0.

    :arg
        coords: Union[str, dict, list]: Coordenadas de los nudos.
            - Formato dict: {0:C_1, 1:C_2, ...}, en donde las coordenadas C_i:
                1D: X
                2D: (X, Y)
                3D: (X, Y, Z)
            - Formato lista: [C_1, C_2, ..., C_n], en donde las coords C_i:
                1D: X
                2D: (X, Y)
                3D: (X, Y, Z)
            Las coordenadas deben expresarse en el sistema global adoptado
            aunque no sea el tradicional: X: horizontal; Y: vertical;
            Z: saliente.
        restricciones (dict): Nudos restringidos y sus restricciones según se
            define en la clase Nudo. La enumeración de nudos inicia en 0.
        datos_barras (list): lista con datos de las barras para configurar los
            instancias de la clase Barra
        cargas_nodales (dict): cargas nodales
        cargas_barras (dict): cargas en las barras
        frac_amortig (float): fracción de amortiguamiento
    """
    # Públicas
    datos_nudos: Union[str, dict, list]  # Coordenadas de los nudos
    restricciones: dict  # Nudos restringidos y sus restricciones
    datos_barras: Union[str, dict, list]  # Datos de las barras
    materiales: Union[list[Material], Material]
    secciones: Union[list[Seccion], Seccion]

    # Protegidas
    _tipo: ClassVar[int]  # Tipo de estructura
    _elementos: list = field(init=False)  # Lista con datos de barras
    _nudos: list = field(init=False)  # Lista con objetos de tipo Nudo
    _barras: list = field(init=False)  # Lista con objetos de tipo Barra
    _masa_global: np.ndarray = field(init=False)  # Matriz de masa global
    _rigidez_global: np.ndarray = field(init=False)  # Matriz de rigidez global
    _fuerza_global: np.ndarray = field(init=False)  # Vector de fuerzas nodales
    _desplaz_gdl: np.ndarray = field(init=False)  # Desplazamientos nodales
    _omega2: np.ndarray = field(init=False)  # Freqs. angulares al cuadrado
    _modos: np.ndarray = field(init=False)  # Modos normalizados con M
    _versores: np.ndarray = field(default_factory=lambda: VERSORES)
    _nudos_dibujados: bool = False  # Si ya se dibujaron los nudos de la estr.
    _barras_dibujadas: bool = False  # Si ya se dibujaron las barras
    _peso_propio: Optional[float] = None  # Peso propio de la estructura
    _tipo_analisis: str = field(init=False)  # Análisis estático o dinámico

    # No protegidas y con valores por defecto
    cargas_nodales: Optional[dict] = None  # Cargas en los nudos
    cargas_barras: Optional[dict] = None  # Cargas en las barras
    frac_amortig: float = 0.025  # Fracción de amortiguamiento

    @abstractmethod
    def __post_init__(self):
        # Para configuraciones
        pass

    def preprocesamiento(self):
        """Preprocesamiento. Preparación de datos.

        - Convierte los datos de nudos en un diccionario.
        - Convierte los datos de barras en una lista.

        Configura atributos:
            datos_nudos: Diccionario de coordenadas nodales de la forma:
                {0: (X, Y, Z), ...}
            datos_barras Lista de datos_barras con datos de barras de la forma:
                [numero_nudo_inicial, numero_nudo_final, Material, Seccion]
        """
        datos_nudos = self.datos_nudos
        datos_barras = self.datos_barras
        materiales = self.materiales
        secciones = self.secciones
        # Nudos
        coordenadas = {}  # Inicialización para coordenadas nodales
        if isinstance(datos_nudos, str):  # Si es un archivo csv
            npd = pd.read_csv(datos_nudos)  # Nudos pandas dataframe
            nnp = npd.to_numpy()  # Nudos en numpy ndarray
            for i, n in enumerate(nnp):
                coordenadas[i] = tuple(n)
        elif isinstance(datos_nudos, dict):  # Si es un diccionario
            coordenadas = dict(sorted(datos_nudos.items()))
            if list(coordenadas.keys())[0] != 0:
                raise ValueError("La enumeración de nudos debe iniciar en 0")
        elif isinstance(datos_nudos, list):  # Si es una lista
            for i, n in enumerate(datos_nudos):
                coordenadas[i] = tuple(n)
        else:
            raise TypeError("Verifica datos de nudos.")

        # Barras
        if isinstance(datos_barras, str):  # Si es un archivo csv
            bpd = pd.read_csv(datos_barras)  # Barras pandas dataframe
            bnp = bpd.to_numpy()  # Barras en numpy ndarray (vectorización)
        elif isinstance(datos_barras, dict):  # Si es un diccionario
            ordenado = dict(sorted(datos_barras.items()))
            bnp = np.array(list(ordenado.values()))
        elif isinstance(datos_barras, list):  # Si es una lista
            bnp = np.array(datos_barras)
        else:
            raise TypeError("Verifica datos de barras.")

        # Si hay un solo material se convierte en lista de ese único material
        if isinstance(materiales, Material):
            materiales = [materiales]

        # Si hay una sola sección se convierte en lista de esa única sección
        if isinstance(secciones, Seccion):
            secciones = [secciones]

        elementos = []
        # Todo conteo inicia en 0.
        for i, b in enumerate(bnp):
            c = len(b)  # Cantidad de datos en cada barra
            if c == 2:  # Si solo se dan los datos de nudos se adopta el
                # primer material y la primera sección de las listas
                Ni, Nf = tuple(b)
                elementos.append([Ni, Nf, materiales[0], secciones[0]])
            elif c == 4:  # Si no se da el roll
                Ni, Nf, m, s = tuple(b)
                elementos.append([Ni, Nf, materiales[m], secciones[s]])
            elif c == 5:  # Si se da el dato de roll
                Ni, Nf, m, s, r = tuple(b)
                elementos.append([Ni, Nf, materiales[m], secciones[s], r])
            else:
                raise ValueError('Verifica datos de barras.')

        self.datos_nudos = coordenadas  # datos_nudos ahora es un dict ordenado
        self._elementos = elementos  # guarda datos_barras en una lista

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
        return len(self.datos_nudos)

    @property
    def n_barras(self) -> int:
        """Número de barras."""
        return len(self.datos_barras)

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
        coords = self.datos_nudos
        restricciones = self.restricciones
        cargas_nodales = self.cargas_nodales
        ngdl_nudo = self.ngdl_nudo
        Nnudos = self.n_nudos  # Número de nudos

        coordenadas = list(coords.values())  # Lista ordenada de nudos

        # Instanciación de nudos    
        nudos = [Nudo(i, c, tipo) for i, c in enumerate(coordenadas)]

        # Asignación de restricciones
        for rr in restricciones.keys():  # Recorre el diccionario de restr.
            nudos[rr].restr = restricciones[rr]

        # Asignación de cargas nodales
        if cargas_nodales is not None:
            for qq in cargas_nodales.keys():  # Recorre diccionario de _cargas
                nudos[qq].set_cargas(np.array(cargas_nodales[qq]))

        # Enumeración de grados de libertad
        gdl_lista = [[0] * ngdl_nudo for _ in range(Nnudos)]  # Inicialización
        contador = 0  # Contador de grados de libertad

        # Asignación de los primeros ngdl grados de libertad
        for ii, nudo in enumerate(nudos):  # Recorre los nudos
            for jj, r in enumerate(nudo.restr):  # Recorre las restricciones
                if r == 0:  # Si no hay restricción
                    gdl_lista[ii][jj] = contador
                    contador += 1
            nudo.set_gdl(tuple(gdl_lista[ii]))  # Se agrega dato al nudo

        # Asignación de los grados de restricción
        for ii, nudo in enumerate(nudos):  # Recorre los nudos nuevamente
            for jj, r in enumerate(nudo.restr):  # Recorre las restricciones
                if r == 1:  # Si hay restricción
                    gdl_lista[ii][jj] = contador
                    contador += 1
            nudo.set_gdl(tuple(gdl_lista[ii]))  # Se agrega dato al nudo

        self._nudos = nudos  # Guarda la lista nudos

    @abstractmethod
    def config_barras(self):
        """Para configurar barras"""
        pass

    def nudos(self) -> list:
        """Lista de nudos de la estructura."""
        try:
            return self._nudos
        except AttributeError:
            print("AttributeError: Pruebe ejecutando primero config_nudos().")

    def barras(self) -> list:
        """Lista de barras de la estructura."""
        try:
            return self._barras
        except AttributeError:
            print("AttributeError: Pruebe ejecutando primero config_barras().")

    def peso_propio(self):
        '''Peso propio de la estructura completa'''
        return sum([b.peso_propio for b in self.barras()])

    def longitudes(self) -> np.ndarray:
        """Longitudes de las barras"""
        return np.array([b.longitud for b in self.barras()])

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
            B[i, gdl[i]] = 1.0
        return B

    def tipo_analisis(self) -> str:
        try:
            return self._tipo_analisis
        except AttributeError:
            print("AttributeError: Pruebe ejecutando primero procesamiento().")

    # TODO: separar los cálculos estáticos de los dinámicos
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
            P = np.array(nudo.cargas())  # Lista con cargas en los nudos
            FG[g] = P

        # Guarda las matrices globales
        self._masa_global = MG
        self._rigidez_global = KG
        self._fuerza_global = FG

    def masa_global(self) -> np.ndarray:
        """Matriz de masa de la estructura global."""
        try:
            return self._masa_global
        except AttributeError:
            print("AttributeError: Verifique la ejecución de procesamiento()")

    def rigidez_global(self) -> np.ndarray:
        """Matriz de rigidez de la estructura global."""
        try:
            return self._rigidez_global
        except AttributeError:
            print("AttributeError: Verifique la ejecución de procesamiento()")

    def fuerza_global(self) -> np.ndarray:
        """Vector de fuerzas nodales de la estructura global."""
        try:
            return self._fuerza_global
        except AttributeError:
            print("AttributeError: Verifique la ejecución de procesamiento()")

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

    # TODO: agregar verificación de flechas.
    def set_desplaz_gdl(self) -> None:
        """Desplazamientos nodales en los grados de libertad.

        Solución del problema estático.
        Calcula y guarda el vector desplazamientos, además asigna los
        desplazamientos a los nudos correspondientes.

        !Ejecución independiente necesaria.
        """
        S = self.rigidez_gdl()
        P = self.fuerza_gdl()
        d = np.linalg.inv(S) @ P
        self._desplaz_gdl = d  # Guarda d

    def set_desplaz_nudos(self, desplaz_gdl=None) -> None:
        """Asignación de los desplazamientos en los nudos de la estructura.

        :param desplaz_gdl: Vector de desplazamientos en los grados de libertad

        !Llamada a la función necesaria.
        """
        if desplaz_gdl is None:  # Si no se da el vector de desplazamientos
            if self._desplaz_gdl is None:  # Si no hay desplaz. estático
                desplaz_gdl = np.zeros(self.ngdl)  # Se asume zeros
            else:  # Si existe desplaz. estático se lo toma.
                desplaz_gdl = self.desplaz_gdl()

        d = desplaz_gdl  # Vector de desplazamientos
        ngdl_nudo = self.ngdl_nudo  # Número de grados de libertad por nudo
        ngdl = self.ngdl  # Número de grados de libertad de la estructura
        for nudo in self.nudos():
            gdl = nudo.gdl()  # Grados de libertad del nudo
            dd = np.zeros(ngdl_nudo)  # Inicialización
            for i, g in enumerate(gdl):  # Recorre gdl's de cada nudo
                if g < ngdl:  # Menor porque se utiliza desplaz_gdl
                    dd[i] = d[g]  # Desplaz. del nudo
            nudo.set_desplaz(dd)  # Asigna el desplazamiento al nudo

    def desplaz_gdl(self) -> np.ndarray:
        """Desplazamientos nodales en los grados de libertad de la estructura.
        """
        return self._desplaz_gdl

    def reacciones(self) -> np.ndarray:
        """Devuelve las reacciones en los grados de restricción."""
        K10 = self.rigidez_global()[self.ngdl:, :self.ngdl]
        P1 = self.fuerza_global()[self.ngdl:]
        d = self.desplaz_gdl()
        R = K10 @ d - P1
        return R

    #####################
    # Análisis dinámico #
    #####################

    def set_autos(self) -> None:
        """Valores y vectores característicos.

        Frecuencias naturales (Hz) y modos de vibración.
        """
        w2, S = eigh(self.rigidez_gdl(), self.masa_gdl())
        self._omega2 = w2  # Valores característicos
        self._modos = S  # Vectores característicos

    @property
    def omega2(self) -> np.ndarray:
        """Frecuencias angulares naturales al cuadrado."""
        return self._omega2

    @property
    def freqs(self) -> np.ndarray:
        """Frecuencias naturales de vibración."""
        return np.sqrt(self.omega2)/2/np.pi

    @property
    def modos(self) -> np.ndarray:
        """Modos naturales de vibración.

        Vectores normalizados con la matriz de masas (Phi.T M Phi = I)
        """
        return self._modos

    def modos_max_uno(self) -> np.ndarray:
        """Modos naturales de vibración.

           Normalizados tal que el valor absoluto máximo de cada modo es 1.
        """
        return norm_uno(self._modos)

    def desplaz_arm(self, f_forzada: float, lapso: tuple[float, float])\
            -> np.ndarray:
        """Desplazamientos debidos a cargas armónicas.

        Se asume que los desplazamientos estáticos son los valores de la
        deformación en t = 0, y que la velocidad de los desplazamientos en t=0
        es 0.

        :param f_forzada: Frecuencia de vibración forzada en Hz.
        :param lapso: periodo de análisis

        :returns Matriz de desplazamientos nodales dinámicos en el lapso dado.
            Cada fila contiene los valores de los desplazamientos en los grados
            de libertad para un valor de tiempo t fijo.
        """
        ngdl = self.ngdl  # Número de grados de libertad de la estructura
        phi = self.modos  # Matriz de modos de vibración normalizada con M
        p = self.fuerza_gdl()  # Vector de fuerzas en los grados de libertad
        omega2 = self.omega2  # Vector de frecuencias angulares al cuadrado
        wn = np.sqrt(omega2)  # Frecuencias angulares naturales
        w = 2*np.pi*f_forzada  # Frecuencia angular de vibración forzada
        zeta = self.frac_amortig  # Fracción de amortiguamiento
        wD = wn*np.sqrt(1 - zeta**2)  # Frecuencias amortiguadas
        t0, tf = lapso  # tiempo inicial y tiempo final
        periodo = np.linspace(t0, tf, 100)  # 100 vals. en período de análisis

        # Desplaz. estáticos como valores de la deformación en t=0
        Xs = self.desplaz_gdl()

        # Coeficientes A, B, C, D para cálculo de desplazamientos
        # Chopra ec. (3.2.5)
        P = phi.T @ p  # Vector de _cargas generalizadas
        r = w / np.sqrt(omega2)  # Vector de ngdl valores
        denominador = ((1-r**2)**2 + (2*zeta*r)**2)  # Vector
        C = P/omega2*(1 - r**2) / denominador  # Vector
        D = P/omega2 * (-2*zeta*r / denominador)  # Vector
        A = D - Xs  # Vector
        B = (A*wn*zeta - C*w) / wD  # Vec

        m = len(periodo)  # Número de filas = valores de tiempo
        n = ngdl  # Número de columnas = número de grados de libertad
        Qn = np.zeros((m, n))  # Ini. matriz de desplazamientos generalizados
        Un = np.zeros((m, n))  # Ini. matriz de desplazamientos

        for i, t in enumerate(periodo):
            # Desplaz. generalizados en los gdl's para un t dado
            transitoria = np.e**(-zeta*wn*t)*(A*np.cos(wD*t) + B*np.sin(wD*t))
            estacionaria = C*np.sin(w*t) + D*np.cos(w*t)
            Qn[i] = transitoria + estacionaria

            Un[i] = phi @ Qn[i]  # Desplazamientos nodales en tiempo = t

        return Un

    def desplaz_esc(self, lapso: tuple[float, float]) -> np.ndarray:
        """Desplazamientos debidos a cargas escalonadas.

        Una carga escalonada salta de repente de cero a un valor constante.

        :param lapso: periodo de análisis

        :returns Matriz de desplazamientos nodales dinámicos en el lapso dado.
            Cada fila contiene los valores de los desplazamientos en los grados
            de libertad para un valor de tiempo t fijo.
        """
        ngdl = self.ngdl  # Número de grados de libertad de la estructura
        phi = self.modos  # Matriz de modos de vibración normalizada con M
        omega2 = self.omega2  # Vector de frecuencias angulares al cuadrado
        wn = np.sqrt(omega2)  # Frecuencias angulares naturales
        dseta = self.frac_amortig  # Fracción de amortiguamiento
        t0, tf = lapso  # tiempo inicial y tiempo final
        periodo = np.linspace(t0, tf, 100)  # 100 vals de t en per. analizado
        wD = wn*np.sqrt(1-dseta**2)
        Xs = self.desplaz_gdl()  # Desplazamientos del cálculo estático

        m = len(periodo)  # Número de filas = valores de tiempo
        n = ngdl  # Número de columnas = número de grados de libertad
        Qn = np.zeros((m, n))  # Ini. matriz de desplazamientos generalizados
        Un = np.zeros((m, n))  # Ini. matriz de desplazamientos

        for i, t in enumerate(periodo):
            # Desplaz. generalizados en los gdl's para un t dado
            Qn[i] = 2*Xs*(1 - np.e**(-dseta*wn*t) *
                          (np.cos(wD*t) + dseta/np.sqrt(1-dseta**2) *
                           np.sin(wD*t)))

            Un[i] = phi @ Qn[i]  # Desplazamientos nodales en tiempo = t

        return Un

    def desplaz_estaticos_equiv(self, desplaz_din: np.ndarray) -> np.ndarray:
        """Desplazamientos estáticos equivalentes.

        Para el cálculo de fuerzas debidas a cargas dinámicas.

        :param desplaz_din: Matriz de desplazamientos dinámicos en un lapso de
            tiempo. Cada fila debe contener los desplazamientos en los grados
            de libertad para un tiempo t dado.

        :returns Vector de desplazamientos estáticos equivalentes.
        """
        Un = desplaz_din
        m, n = Un.shape
        kdim = np.diag(self.omega2)  # Matriz de rigidez dinámica normalizada
        S = self.rigidez_gdl()  # Matriz de rigidez estática en los gdl's
        invS = np.linalg.inv(S)  # Inversa de la matriz de rigidez
        Dn = np.zeros((m, n))
        for i in range(m):
            F = kdim @ Un[i]  # Fuerzas estáticas equivalentes
            Dn[i] = invS @ F  # Desplazamientos estáticos equivalentes

        Xequiv = Dn.sum(axis=0)  # Suma la contribución de cada modo

        return Xequiv

    ###############
    # Parte final #
    ###############

    @abstractmethod
    def procesamiento(self, tipo_analisis='estatico', verbose=False) -> None:
        """Realiza todos los cálculos necesarios.

        Args:
            tipo_analisis (str): análisis estático o dinámico;
            verbose (bool): mensajes en el proceso de cálculo.

        Returns:
            None
        """
        self._tipo_analisis = tipo_analisis
        pass

    @abstractmethod
    def mostrar_resultados(self, prefijos_SI=False):
        """Muestra los resultados más importantes.

        Args:
            prefijos_SI (bool): si los datos utilizados tienen o no prefijos
                                del Sistema Internacional de Unidades.
                - True: datos con prefijos como KN, mm, MPa, etc.
                - False: datos sin prefijos como N, m, Pa, etc.

        Returns:
            Imprime resultados.
        """
        pass

    def dibujar_nudos(self, cuadro: tuple = None, num=False, ax=None,
                      **kwargs):
        """Dibuja los nudos como scatter usando matplotlib."""

        nudos = self.nudos()
        dim = nudos[0].dim
        if dim == 3 and ax is None:
            raise Exception('Debe proveer -ax- para el dibujo 3D')

        # Dibuja los nudos como scatter
        if not self._barras_dibujadas:  # Si las barras no están dibujadas
            if cuadro is not None:  # Si se da el dato del cuadro
                plt.figure(figsize=cuadro)
        for n in nudos:
            n.dibujar(num=num, ax=ax, **kwargs)

        self._nudos_dibujados = True  # Confirma el plot de los nudos
        plt.grid(True)

    def dibujar_barras(self, cuadro: tuple = None, num=False,
                       espesor_area=False, color='k', ax=None,
                       **kwargs):
        """Dibuja la estructura de barras usando matplotlib."""

        barras = self.barras()
        if self.dim == 3 and ax is None:
            raise Exception('Debe proveer -ax- para el dibujo 3D')

        # Dibuja las barras en líneas
        if not self._nudos_dibujados:  # Si los nudos no están dibujados
            if cuadro is not None:  # Si se da el dato del cuadro
                plt.figure(figsize=cuadro)
        for b in barras:
            b.dibujar(num=num, espesor_area=espesor_area, color=color, ax=ax,
                      **kwargs)

        self._barras_dibujadas = True  # Confirma el plot de las barras

        if self.dim == 2:
            if ax is not None:
                ax.grid(True)
            else:
                plt.grid(True)

    def dibujar_deform(self, num=False, espesor_area=False, amp=1, ax=None,
                       **kwargs):
        """Dibuja la estructura deformada usando matplotlib."""

        barras = self.barras()
        if self.dim == 3 and ax is None:
            raise Exception('Debe proveer -ax- para el dibujo 3D')

        # Dibuja las barras en líneas
        for b in barras:
            b.dibujar_deform(num=num, espesor_area=espesor_area, amp=amp,
                             ax=ax, **kwargs)

    def grafica_de_modo(self, espesor_area=False, n=0, amp=1, **kwargs)\
            -> None:
        """Gráfica de modo de vibración.

        :param espesor_area: Considerar el espersor de las líneas proporcional
            al área de las barras.
        :param n: número de modo de vibración
        :param amp: factor de amplifiación de las deformaciones
        :param kwargs: parámetros que se pasan a matplotlib 3D
        """

        # Estructura deformada según modo de vibración establecido
        normalizado = self.modos_max_uno()
        self.set_desplaz_nudos(normalizado[n])

        # Definición del título del gráfico
        if n == 0:
            titulo = "Modo fundamental\n"
        else:
            titulo = f"Modo de vibración Nº {n+1}\n"

        # Gráfico para estructuras 2D
        if self.dim == 2:
            plt.axis('equal')
            plt.title(titulo)
            plt.text(.1, 1.1, f"freq = {self.freqs[n]:.1f} Hz", fontsize=12)
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")

            self.dibujar_deform(espesor_area=espesor_area, amp=amp, **kwargs)

        # Gráfico para estructuras 3D
        elif self.dim == 3:

            fg = plt.figure()
            axs = fg.add_subplot(projection='3d')

            axs.set_title(titulo)
            axs.set_xlabel("X (m)")
            axs.set_ylabel("Y (m)")
            axs.set_zlabel("Z (m)")
            axs.text(-.5, -.5, 6, f"freq = {self.freqs[n]:.1f} Hz")

            axs.set_box_aspect((1, 1, 6))
            axs.set_xlim3d(-.5, .5)
            axs.set_ylim3d(-.5, .5)
            axs.set_zlim3d(0, 5.5)
            axs.tick_params(axis='x', labelsize=5)
            axs.tick_params(axis='y', labelsize=5)
            axs.tick_params(axis='z', labelsize=10)
            axs.zaxis.labelpad = 30

            self.dibujar_deform(espesor_area=espesor_area, ax=axs, amp=amp,
                                **kwargs)


@dataclass
class Reticulado(Estructura):
    """Estructura tipo Reticulado.

    IMPORTANTE:
    - No se considera el peso propio en los cálculos.
    - Todos los conteos inician en 0.

    Args:
        coords: Union[str, dict, list]: Coordenadas de los nudos.
            - Formato dict: {0:C_1, 1:C_2, ...}, en donde las coordenadas C_i:
                1D: X
                2D: (X, Y)
                3D: (X, Y, Z)
            - Formato lista: [C_1, C_2, ..., C_n], en donde las coords C_i:
                1D: X
                2D: (X, Y)
                3D: (X, Y, Z)
            Las coordenadas deben expresarse en el sistema global adoptado
            aunque no sea el tradicional: X: horizontal; Y: vertical;
            Z: saliente.
        restricciones (dict): Nudos restringidos y sus restricciones según se
            define en la clase Nudo.
            Ejemplo:
                {2: (1, 0), 4: (1, 1)}
                El nudo 2, está restringido en X
                El nudo 4, está restringido en X e Y
        datos_barras (list): lista con datos de las barras para configurar los
            instancias de la clase Barra
        cargas_nodales (dict): cargas nodales
        cargas_barras (dict): cargas en las barras
        frac_amortig (float): fracción de amortiguamiento
    """

    def __post_init__(self):
        """Definición del tipo de reticulado en función a la
         dimensión espacial de las coordenadas nodales.
        """
        dim_nudo = len(self.datos_nudos[1])
        self._tipo = list(RETICULADOS.keys())[dim_nudo-1]
        self.preprocesamiento()
        self.config_nudos()
        self.config_barras()

    def config_barras(self) -> None:
        """Configuración de barras."""
        # self.config_nudos()  # Configuración de nudos
        nudos = self.nudos()
        versores = self.versores

        # Configuración de las barras
        barras = [BarraReticulado(
            i, self.tipo, nudos[e[0]], nudos[e[1]], e[2], e[3], versores
            ) for i, e in enumerate(self._elementos)]

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
    def procesamiento(self, tipo_analisis='estatico', verbose=False):
        """Procesamiento estático o dinámico.

        Realiza todos los cálculos necesarios.

        Args:
            tipo_analisis:
                'estatico' para análisis estático
                'dinamico' (o cualquier otra palabra) para análisis dinámico
            devolver_resultados (bool): para análisis estático devuelve:
                - Desplazamientos en los grados de libertad;
                - Tensiones normales en las barras;
                - Elongaciones de las barras;
                - Reacciones en los grados de restricción.
            verbose (bool): mensajes en el proceso de cálculo.
        """
        self._tipo_analisis = tipo_analisis
        nudos = self.nudos()
        barras = self.barras()

        # Ensamble de matrices globales
        if verbose:
            print("Ensamble de matrices globales...")
        self.set_matrices_globales(nudos, barras)

        # noinspection SpellCheckingInspection
        if tipo_analisis == 'estatico':
            if self.cargas_nodales is None:
                raise ValueError("Esta estructura no tiene cargas.")
            else:
                if verbose:
                    print("Resolución del problema estático...")
                # Desplazamientos nodales estáticos
                self.set_desplaz_gdl()  # Resuelve el problema estático
                self.set_desplaz_nudos()  # Asigna desplaz. a los nudos
                self.set_tensiones()  # Calcula y guarda las tensiones axiales
                if verbose:
                    print("Listo, preguntá lo que quieras.")

        else:  # tipo_analisis = dinámico
            # Resolución del problema de autovalores
            if verbose:
                print("Resolución del problema de autovalores...")
            self.set_autos()
            if verbose:
                print("Listo.")

    # Sobre-escrito
    def mostrar_resultados(self, prefijos_SI=False):
        """Muestra los resultados más importantes.

        Args:
            prefijos_SI (bool): si los datos utilizados tienen o no prefijos
                                del Sistema Internacional de Unidades.
                - True: datos con prefijos como KN, mm, MPa, etc., con unidades
                    anglosajonas u otras unidades especiales.
                - False: datos sin prefijos como N, m, Pa, etc.

        Returns:
            Imprime resultados.
        """
        if self.tipo_analisis() == 'estatico':
            Xs = self.desplaz_gdl()
            Ts = self.tensiones()
            Es = self.elongaciones()
            Rs = self.reacciones()

            # TODO: chequear si haría falta redondeos.
            if prefijos_SI:
                print('RESPUESTA ESTÁTICA en unidades de los datos')
                print('- Desplazamientos nodales:', Xs)
                print('- Tensiones normales:', Ts)
                print('- Elongaciones axiales:', Es)
                print('- Reacciones:', Rs)
                print()

            else:
                Ds = np.round(Xs * 1e3, 2)  # Desplazamientos en mm
                Ts = np.round(Ts * 1e-6, 2)  # Tensiones en MPa
                Es = np.round(Es * 1e3, 2)  # Elongaciones en mm
                Rs = np.round(Rs * 1e-3, 2)  # Reacciones en kN
                print('RESPUESTA ESTÁTICA')
                print('- Desplazamientos nodales (mm):', Ds)
                print('- Tensiones normales (MPa):', Ts)
                print('- Elongaciones axiales (mm):', Es)
                print('- Reacciones (kN):', Rs)
                print()

        else:
            fs = self.freqs()
            S = self.modos()
            freqs = np.round(fs, 2)  # Frecuencias en Hz
            modos = np.round(S, 2)  # Modos de vibración

            print('RESPUESTA DINÁMICA')
            print('- Frecuencias naturales (Hz):', freqs)
            print('- Modos de vibración:', modos)
            print()


@dataclass
class Portico(Estructura):
    """Estructura tipo pórtico 2D o 3D.

    :param tipo (int): tipo de estructura según se define en TIPO_ESTRUCTURA
    :param coords (dict): Coordenadas de los nudos.
        Tiene la forma: {0:C_1, 2:C_2, ...}, en donde las coordenadas C_i:
            1D: X
            2D: (X, Y)
            3D: (X, Y, Z)
            Las coordenadas deben expresarse en el sistema global adoptado
            aunque no sea el tradicional: X: horizontal; Y: vertical;
            Z: saliente.
    :param restricciones (dict): Nudos restringidos y sus restricciones según
        se define en la clase Nudo.
    :param cargas_nodales (dict): _cargas nodales
    :param datos_barras (list): lista con datos de las barras para configurar
        instancias de la clase Barra.
    :param cortante (bool): Considerar o no la rigidez a cortante.
    :param cargas_barras (Optional[dict]): Cargas en las barras (AÚN NO
        IMPLEMENTADO)
     """

    cortante: bool = False  # Considerar o no la rigidez a cortante.
    cargas_barras: Optional[dict] = None  # Cargas en las barras

    def __post_init__(self):
        """Definición del tipo de estructura y preprocesamiento."""
        dim = len(self.datos_nudos[1])
        self._tipo = 3 if dim == 2 else 6
        self.preprocesamiento()
        self.config_nudos()
        self.config_barras()

    def config_barras(self) -> None:
        """Configuración de barras."""
        tipo = self.tipo  # Tipo de estructura
        nudos = self.nudos()  # Lista de nudos del tipo Nudo
        versores = self.versores  # Versores del sistema de coord. global
        Q = self.cortante  # Consideración de la rigidez a cortante
        cargas = self.cargas_barras  # Cargas en las barras
        n = self.ngdl_nudo  # Número de grados de libertad por nudo

        # Configuración de las barras
        elementos = self._elementos  # Obtenido del preprocesamiento
        bs = []  # Inicialización de la lista de barras

        # Tipo de estructura
        if tipo == 2:  # Viga
            Bar = BarraViga
        elif tipo == 5:  # Grilla
            Bar = BarraGrilla
        else:  # Si es un pórtico
            Bar = BarraPortico

        # Instanciación de las barras
        for i, e in enumerate(elementos):
            ni = e[0] - 1  # Índice del nudo inicial
            nf = e[1] - 1  # Índice del nudo final
            if len(e) == 4:  # Si no se da el roll
                bs.append(Bar(i, self.tipo, nudos[ni], nudos[nf], e[2], e[3],
                              versores, cortante=Q))
            else:  # Si roll es distinto de cero.
                r = e[4]/180*np.pi  # Conversión a radianes
                bs.append(Bar(i, self.tipo, nudos[ni], nudos[nf], e[2], e[3],
                              versores, r, cortante=Q))

        self._barras = bs  # Guarda la lista de barras

        # Configuración de cargas en las barras
        # Por ahora solo cargas distribuidas paralelas a y-y
        # TODO: Incorporar cargas en otras direcciones
        if cargas is not None:
            for nb in cargas.keys():  # Recorre diccionario de cargas
                barra = bs[nb-1]  # Barra actual
                q = cargas[nb]  # Cargas en la barra
                barra._carga = q  # Asignación de cargas distribuidas a barra

                # Vector de cargas nodales eq. en coords. globales
                F = barra.carga_equiv_global()

                # Suma cargas nodales equiv. a cargas nodales existentes
                Ni = barra.nudo_inicial
                Nf = barra.nudo_final
                F0 = Ni.cargas() + F[:n]  # Carga total en nudo inicial
                F1 = Nf.cargas() + F[n:]  # Carga total en nudo final
                Ni.set_cargas(F0)  # Asigna carga total al nudo inicial
                Nf.set_cargas(F1)  # Asigna carga total al nudo final

    def procesamiento(self, tipo_analisis='estatico') -> None:
        """Procesamiento estático o dinámico.

        Realiza todos los cálculos necesarios.

        Args:
            tipo_analisis:
                'estatico' para análisis estático
                'dinamico' (o cualquier otra palabra) para análisis dinámico
        """
        nudos = self.nudos()
        barras = self.barras()

        # Ensamble de matrices globales
        # print("Ensamble de matrices globales...")
        self.set_matrices_globales(nudos, barras)

        # noinspection SpellCheckingInspection
        if tipo_analisis == 'estatico':
            if self.cargas_nodales is None and self.cargas_barras is None:
                print("Esta estructura no tiene cargas.")
            else:
                # print("Resolución del problema estático...")
                # Desplazamientos nodales estáticos
                self.set_desplaz_gdl()  # Resuelve el problema estático
                self.set_desplaz_nudos()  # Asigna desplaz. a los nudos
                # print("Listo, preguntá lo que quieras.")

        else:  # tipo_analisis = dinámico
            # Resolución del problema de autovalores
            # print("Resolución del problema de autovalores...")
            self.set_autos()
            # print("Listo.")


@dataclass
class Viga(Portico):

    """Viga simple o continua.

    :param cargas_barras (Opcional[dict]): Cargas en las barras.
        De la forma: {#tramo: q, ...}

        CARGAS:
        ------
        Solo se admiten _cargas distribuidas verticales.
        Para considerar _cargas puntuales o momentos concetrados se deberá
        introducir nudos en sus posiciones.
        Si las _cargas son variables, los nudos deben aproximarse entre sí, tal
        que puedan suponerse _cargas uniformemente distribuidas.

        El peso propio es una carga que debe ser agregada como cualquier otra,
        con su correspondiente coef. de seguridad.
    """
    cargas_barras: Optional[dict] = None  # Cargas en las barras

    def __post_init__(self):
        """Definición del tipo de estructura y preprocesamiento."""
        self._tipo = 2
        self.preprocesamiento()  # Adaptación de los datos
        self.config_nudos()  # Configuración de los nudos
        self.config_barras()  # Configuración de las barras

    def abscisas(self, num: int = 50) -> np.ndarray:
        """Valores de abscisas para los diagramas.

        :param num: número de puntos por barra para el diagrama.
        """
        longs = self.longitudes()
        # Longitud acumulada, desde cero hasta el penúltimo tramo
        lacum = np.concatenate([np.array([0.0]), longs.cumsum()])[:-1]

        # Valores de las abscisas
        X = np.concatenate([b.abscisas(num) + lacum[i] for i, b in
                            enumerate(self.barras())])
        return X

    def cortantes(self, num: int = 50):
        """Ordenadas para el diagrama de fuerzas cortantes.

        :param num: número de puntos por barra para el diagrama.
        """
        return np.concatenate([b.cortantes(num) for b in self.barras()])

    def flectores(self, num: int = 50):
        """Ordenadas para el diagrama de momentos flectores.

        :param num: número de puntos por barra para el diagrama.
        """
        return np.concatenate([b.flectores(num) for b in self.barras()])

    # TODO: Verificar diagrama de fuerza cortante
    def diag_cortante(self, num: int = 50):
        """Diagrama de fuerza cortante de la viga.

        :param num: número de puntos por barra para el diagrama
        """
        Ltotal = sum(self.longitudes())
        abscisas = self.abscisas(num)
        cortantes = self.cortantes(num)
        X = np.concatenate([np.array([0.0]), abscisas, np.array([Ltotal])])
        Y = np.concatenate([np.array([0.0]), cortantes, np.array([0.0])])
        plt.title("Diagrama de fuerza cortante")
        plt.xlabel("x")
        plt.ylabel("Q")
        plt.hlines(y=0.0, xmin=0, xmax=Ltotal, color='k')
        plt.plot(X, Y)
        plt.grid()

    # TODO: Verificar diagrama de momento flector.
    def diag_flector(self, num: int = 50):
        """Diagrama de momento flector de la viga.

        :param num: número de puntos por barra para el diagrama
        """
        Ltotal = sum(self.longitudes())
        abscisas = self.abscisas(num)
        flectores = self.flectores(num)
        X = np.concatenate([np.array([0.0]), abscisas, np.array([Ltotal])])
        Y = np.concatenate([np.array([0.0]), flectores, np.array([0.0])])
        plt.title("Diagrama de momento flector")
        plt.xlabel("x")
        plt.ylabel("M")
        plt.hlines(y=0.0, xmin=0, xmax=Ltotal, color='k')
        plt.plot(X, Y)
        plt.grid()
        plt.gca().invert_yaxis()


class Grilla(Estructura):
    _tipo = 5

    def __post_init__(self):
        """Aún no implementado"""
        pass

    def config_barras(self):
        """AÚN NO IMPLEMENTADO"""
        self.config_nudos()  # Configuración de nudos
        nudos = self.nudos()
        return nudos

    def procesamiento(self, tipo_analisis='estatico') -> None:
        pass


######################
# Funciones globales #
######################

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


def mostrar_resultados(resultados):
    n_results = len(resultados)  # Número de resultados
    if n_results == 4:  # Respuesta estática
        Xs, Ts, Es, Rs = resultados  # Resultados de 'procesamiento' estático
        Ds = np.round(Xs, 4)  # Desplazamientos
        Ts = np.round(Ts, 0)  # Tensiones
        Es = np.round(Es, 4)  # Elongaciones
        Rs = np.round(Rs, 2)  # Reacciones

        print('RESPUESTA ESTÁTICA')
        print('- Desplazamientos en los grados de libertad:', Ds)
        print('- Tensiones normales:', Ts)
        print('- Elongaciones axiales:', Es)
        print('- Reacciones:', Rs)
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


def main():
    # Prueba con Kassimali Example 6.7
    # Datos de los nudos
    coords = {0: (0, 0), 1: (9, 0), 2: (0, 6), 3: (9, 6), 4: (0, 12)}
    restricciones = {0: (1, 1, 1), 1: (1, 1, 1)}  # Apoyos

    # Materiales
    E = 30e9
    m = Material(E)

    # Datos de las secciones
    A = 75e-3  # área
    inercia = 4.8e-4
    s = Seccion(area=A, inercia_z=inercia)

    # Conectividad
    datos_barras = {0: (1, 3), 1: (2, 4), 2: (3, 5), 3: (3, 4), 4: (4, 5)}

    # Cargas
    cargas_nodales = {2: (80e3, 0, 0), 4: (40e3, 0, 0)}

    # Cargas en barras
    ang = np.arctan(6/9)
    q = 12e3
    qx = q*np.sin(ang)
    qy = q*np.cos(ang)
    cargas_barras = {5: (qx, qy)}
    # Instancia de la estructura
    p = Portico(datos_nudos=coords, restricciones=restricciones,
                datos_barras=datos_barras, materiales=m, secciones=s,
                cargas_nodales=cargas_nodales, cargas_barras=cargas_barras)
    p.procesamiento()


if __name__ == "__main__":
    main()
