# Direct Stiffness Method
# Matrices y elementos varios del método directo de rigidez
#
# @author: Fredy Gabriel Ramírez Villanueva
# Inicio de codificación: 12 de mayo de 2022
# Realese 1: 15 de mayo de 2022
# Control git: 01 de junio de 2022

import numpy as np

DIMS = {1, 2, 3} # Conjunto de dimensiones posibles

#########################
## Matrices de rigidez ##
#########################

'''
Matrices y elementos varios del método directo de rigidez.

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

Elementos estructurales:
    b2f: Barra de dos fuerzas, grado de libertad 1
    vig: Viga, grados de libertad 2 y 6
    vgc: Viga-columna, grados de libertad 1, 2 y 6
    v3d: viga en 3 dimensiones, se consideran 6 grados de libertad por nudo.
'''


def rigidez_b2f(L: float, E: float, A: float) -> np.ndarray:
    """Matriz de rigidez para barras de dos fuerzas.

    Elemento de barra de 1 grado de libertad por nudo, en dirección del eje de
    la barra.

    Args:
        L: longitud de la barra
        E: módulo de elasticidad;
        A: área de la sección transversal;

    Returns:
        Matriz 2x2 de tipo np.ndarray.
    """
    k = A*E/L*np.array([
        [1, -1],
        [-1, 1]
    ])
    return k


def rigidez_uni_cuad(L: float, E: float, A: float) -> np.ndarray:
    """Matriz de rigidez para problemas unidimensionales con funciones de forma
    cuadrática.

    Args:
        L: longitud de la barra
        E: módulo de elasticidad;
        A: área de la sección transversal;

    Returns:
        Matriz 3x3 de tipo np.ndarray.
    """
    k = E*A/3/L*np.array([
        [7, 1, -8],
        [1, 7, -8],
        [-8, -8, 16]
    ])
    return k


def rigidez_vig(L: float, E: float, I: float, G: float = None,
                Ac: float = None) -> np.ndarray:
    """Matriz de rigidez para elemento de viga.

    Elemento de barra de 2 grados de libertad por nudo, vertical (y) y giro
    (z).

    Si se da el valor de G (módulo de corte), se considerará la rigidez a
    cortante. Si G no es dado, no se considerará el aporte de la rigidez a
    cortante.

    Args:
        L: longitud de la barra
        E: módulo de elasticidad
        I: inercia alrededor de z
        G: módulo de corte (opcional)
        Ac: área efectiva al corte (opcional)

    Returns:
        Matriz 4x4 de tipo np.ndarray
    """
    if G is None:  # No se considera la rigidez al corte
        f = 0
    else:  # Se toma en cuenta la rigidez al corte
        assert Ac is not None, "Falta dato del área afectiva al corte"
        f = 12*E*I/(G*Ac*L**2)

    k = E*I/(L**3*(1 + f))*np.array([
        [12, 6*L, -12, 6*L],
        [6*L, (4 + f)*L**2, -6*L, (2 - f)*L**2],
        [-12, -6*L, 12, 6*L],
        [6*L, (2 - f)*L**2, -6*L, (4 + f)*L**2]
    ])
    return k


def rigidez_vgc(L: float, E: float, A: float, I: float, G: float = None,
                Ac: float = None) -> np.ndarray:
    """Matriz de rigidez para viga-columna.

    Elemento de barra de 3 grados de libertad por nudo (translación en x,
    translación en y, giro en z)

    Si se da el valor de G (módulo de corte), se considerará la rigidez a
    cortante. Si G no es dado, no se considerará el aporte de la rigidez a
    cortante.

    Args:
        L: longitud de la barra
        E: módulo de elasticidad
        A: área de la sección transversal
        I: inercias alrededor de z
        G: (opcional) módulo de elasticidad al corte
        Ac: (opcional) área efectiva a cortante

    Returns:
        Matriz 6x6 de tipo np.ndarray
    """
    if G is None:  # No se considera la rigidez al corte
        f = 0
    else:  # Se toma en cuenta la rigidez al corte
        assert Ac is not None, "Falta dato de área efectiva al corte"
        f = 12*E*I/(G*Ac*L**2)

    k = E*I/(L**3*(1 + f)) * np.array([
        [A*L**2/I, 0, 0, -A*L**2/I, 0, 0],
        [0, 12, 6*L, 0, -12, 6*L],
        [0, 6*L, (4 + f)*L**2, 0, -6*L, (2 - f)*L**2],
        [-A*L**2/I, 0, 0, A*L**2/I, 0, 0],
        [0, -12, -6*L, 0, 12, -6*L],
        [0, 6*L, (2 - f)*L**2, 0, -6*L, (4 + f)*L**2]
    ])
    return k


def rigidez_v3d(L: float, E: float, A: float, I: tuple[float], G: float,
                Ac: tuple[float] = None) -> np.ndarray:
    """Matriz de rigidez para barra de pórtico 3D.

    Elemento de barra de 6 grados de libertad por nudo (translaciones y
    rotaciones en x, y, z).

    Si se dan los valores de las áreas a cortante, se considerará la rigidez a
    cortante. En caso contrario, no se considerará el aporte de la rigidez a
    cortante.

    Args:
        L: longitud de la barra
        E: módulo de elasticidad
        I: inercias, tupla con 2 valores (Iy, Iz), inercias en y, z de la
        sección transversal
        A: área de la sección transversal
        G: módulo de elasticidad al corte
        Ac: (opcional) área efectiva a cortante, tuplas con 2 valores
            (Acy, Acz), área a cortante en y, área a cortante en z.

    Returns:
        Matriz 12x12 de tipo np.ndarray
    """
    Iy, Iz = I  # Inercias
    J = Iy + Iz

    if Ac is None:  # No se considera la rigidez al corte
        fy, fz = 0, 0
    else:  # Se toma en cuenta la rigidez al corte
        Acy, Acz = Ac
        fy = 12*E*Iz/(G*Acy*L**2)
        fz = 12*E*Iy/(G*Acz*L**2)

    k = np.array([
        [E*A/L, 0, 0, 0, 0, 0, -E*A/L, 0, 0, 0, 0, 0],
        [0, 12*E*Iz, 0, 0, 0, 6*E*Iz/L**2/(1 + fy), 0, -12*E*Iz/L**3/(1 + fy),
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


######################
## Matrices de masa ##
######################

def masa_b2f(L: float, A: float, rho: float, dim: int = 1) -> np.ndarray:
    """Matriz de masa para barra de dos fuerzas.

    Esta matriz es invariante con respecto al sistema de ejes adoptado.

    Args:
        L: longitud de la barra
        A: área de la sección transversal
        rho: densidad del material
        dim: dimensión considerada
            dim = 1: Movimiento en dirección del eje de la barra
            dim = 2: Movimiento en un plano (default)
            dim = 3: Movimiento en el espacio

    Returns:
        Matriz de 2dim x 2dim, tipo np.ndarray
    """
    assert dim in DIMS, "La dimensión debe ser 1, 2 o 3"

    M = 2*np.identity(2*dim)  # Inicialización
    I = np.identity(dim)
    M[:dim, dim:] = I
    M[dim:, :dim] = I

    return rho*A*L/6 * M


def masa_vig(L: float, A: float, rho: float, E: float = None, I: float = None,
             G: float = None, Ac: float = None) -> np.ndarray:
    """Matriz de masa para viga-columna como elmento de pórtico plano.

    Si se da el valor de G (módulo de corte), se considerarán las deformaciones
    a cortante. Si G no es dado, no se considerará el aporte de las
    deformaciones a cortante. Notar que si G es dado, también deben darse los
    valores de E, I y Ac.

    Si la inercia I es dada, se considerará la inercia rotacional. Si I no es
    dado, se despreciará la inercia rotacional.

    Args:
        L: longitud de la barra
        A: área de la sección transversal
        rho: densidad del material
        E: (opcional) módulo de elasticidad
        I: (opcional) inercia de la sección transversal
        G: (opcional) módulo de elasticidad a cortante
        Ac: (opcional) área efectiva a cortante

    Returns:
        Matriz de 4x4, tipo np.ndarray
    """
    if G is None:  # No se considera el efecto de las deformaciones a cortante
        fs = 0.0
    else:  # Se considera el efecto de la cortante
        assert E is not None | Ac is not None, "Revisa los datos"
        fs = 12*E*I/(G*Ac*L**2)

    if I is None:  # No se considera la inercia rotacional
        I = 0

    # Inercia translacional
    m1 = rho*A*L/(1 + fs)**2*np.array([
        [13/35 + 7/10*fs + 1/3*fs**2, (11/210 + 11/120*fs + 1/24*fs**2)*L,
         9/70 + 3/10*fs + 1/6*fs**2, -(13/420 + 3/40*fs + 1/24*fs**2)*L],
        [(11/210 + 11/120*fs + 1/24*fs**2)*L,
         (1/105 + 1/60*fs + 1/120*fs**2)*L**2,
         (13/420 + 3/40*fs + 1/24*fs**2)*L,
         -(1/140 + 1/60*fs + 1/120*fs**2)*L**2],
        [9/70 + 3/10*fs + 1/6*fs**2, (13/420 + 3/40*fs + 1/24*fs**2)*L,
         13/35 + 7/10*fs + 1/3*fs**2, -(11/210 + 11/120*fs + 1/24*fs**2)*L],
        [-(13/420 + 3/40*fs + 1/24*fs**2)*L,
         -(1/140 + 1/60*fs + 1/120*fs**2)*L**2,
         -(11/210 + 11/120*fs + 1/24*fs**2)*L,
         (1/105 + 1/60*fs + 1/120*fs**2)*L**2]
    ])

    # Inercia rotacional
    m2 = rho*A*L/(1 + fs)**2*I/A/L**2*np.array([
        [6/5, (1/10 - 1/2*fs)*L, -6/5, (1/10 - 1/2*fs)*L],
        [(1/10 - 1/2*fs)*L, (2/15 + 1/6*fs + 1/3*fs**2)*L**2,
         (-1/10 + 1/2*fs)*L, (-1/30 - 1/6*fs + 1/6*fs**2)*L**2],
        [-6/5, (-1/10 + 1/2*fs)*L, 6/5, (-1/30 - 1/6*fs + 1/6*fs**2)*L**2],
        [(1/10 - 1/2*fs)*L, (-1/30 - 1/6*fs + 1/6*fs**2)*L**2,
         (-1/10 + 1/2*fs)*L, (2/15 + 1/6*fs + 1/3*fs**2)*L**2]
    ])
    return m1 + m2


def masa_v3d(L: float, A: float, rho: float, I: tuple[float]) -> np.ndarray:
    """Matriz de masa para viga-columna como elemento de pórtico espacial.

    No se considera el efecto de las deformaciones por cortante.

    Args:
        L: longitud de la barra
        A: área de la sección transversal
        rho: densidad del material
        I: inercias de la sección transversal (Jx, Iy, Iz)

    Returns:
        Matriz de 12x12, tipo np.ndarray
    """
    J, Iy, Iz = I  # Inercias

    m = rho*A*L*np.array([
        [1/3, 0, 0, 0, 0, 0, 1/6, 0, 0, 0, 0, 0],
        [0, 13/35 + 6*Iz/(5*A*L**2), 0, 0, 0, 11*L/210 + Iz/(10*A*L), 0,
         9/70 - 6*Iz/(5*A*L**2), 0, 0, 0, -13*L/420 + Iz/(10*A*L)],
        [0, 0, 13/35 + 6*Iy/(5*A*L**2), 0, -11*L/210 - Iy/(10*A*L), 0, 0, 0,
         9/70 - 6*Iy/(5*A*L**2), 0, 13*L/420 - Iy/(10*A*L), 0],
        [0, 0, 0, J/3/A, 0, 0, 0, 0, 0, J/6/A, 0, 0],
        [0, 0, -11*L/210 - Iy/(10*A*L), 0, L**2/105 + 2*Iy/15/A, 0, 0, 0,
         -13*L/420 + Iy/(10*A*L), 0, -L**2/140 - Iy/30/A, 0],
        [0, 11*L/210 + Iz/(10*A*L), 0, 0, 0, L**2/105 + 2*Iz/15/A, 0,
         13*L/420 - Iz/10/A/L, 0, 0, 0, -L**2/140 - Iz/30/A],
        [1/6, 0, 0, 0, 0, 0, 1/3, 0, 0, 0, 0, 0],
        [0, 9/70 - 6*Iz/(5*A*L**2), 0, 0, 0, 13*L/420 - Iz/10/A/L, 0,
         13/35 + 6*Iz/(5*A*L**2), 0, 0, 0, -11*L/210 - Iz/(10*A*L)],
        [0, 0, 9/70 - 6*Iy/(5*A*L**2), 0, -13*L/420 + Iy/(10*A*L), 0, 0, 0,
         13/35 + 6*Iz/(5*A*L**2), 0, 11*L/210 + Iy/(10*A*L), 0],
        [0, 0, 0, J/6/A, 0, 0, 0, 0, 0, J/3/A, 0, 0],
        [0, 0, 13*L/420 - Iy/(10*A*L), 0, -L**2/140 - Iy/30/A, 0, 0, 0,
         11*L/210 + Iy/(10*A*L), 0, L**2/105 + 2*Iy/15/A, 0],
        [0, -13*L/420 + Iz/10/A/L, 0, 0, 0, -L**2/140 - Iz/30/A, 0,
         -11*L/210 - Iz/10/A/L, 0, 0, 0, L**2/105 + 2*Iz/15/A]
    ])
    return m

###############################################
## Matrices de transformación de coordenadas ##
###############################################


def transf_coord_b2f(cos_dirs: np.ndarray) -> np.ndarray:
    """Matriz de transformación de coordenadas para barras de dos fuerzas.

    Transforma las coordenadas globales a locales.

    Args:
        cos_dirs: cosenos directores de la barra. 2 valores para 2D, 3 valores
        para 3D.

    Returns:
        Matriz 2x4 para barras 2D, 3x4 para barras 3D, tipo np.ndarray.
    """
    dim = len(cos_dirs)  # Dimensión considerada
    T = np.zeros((2, 2*dim))  # Inicialización
    T[0, :dim] = cos_dirs  # Primera fila
    T[1, dim:] = cos_dirs  # Segunda fila

    return T


def transf_coord_m2f(matriz_cos_dirs: np.ndarray) -> np.ndarray:
    """Matriz de transformación de coordenadas para masas.

    Transforma las coordenadas globales a locales de la matriz de masa.
    Notar quer depende de la dimensión.

    Args:
        matriz_cos_dirs: matriz de cosenos directores de la barra.
            Cada fila contiene los cosenos directores de cada uno de los ejes
            coordenados locales.

    Returns:
        Matriz 2x2 (identidad) para barras en 1D, 4x4 para barras en 2D y 6x6
        para barras en 3D. Tipo np.ndarray.
    """
    a, b = matriz_cos_dirs.shape
    assert a == b, "La matriz de cosenos directores debe ser cuadrada"

    # La dimensión es igual a ´a´
    assert a in DIMS, "La matriz debe ser 1x1, 2x2 o 3x3"

    T = np.zeros((2*a, 2*a))
    T[:a, :a] = matriz_cos_dirs
    T[a:, a:] = matriz_cos_dirs
    return T


def transf_coord_vgc(cos_dirs: tuple[float]) -> np.ndarray:
    """Matriz de transformación de coordenadas para vigas-columna planas.

    Transforma las coordenadas globales a locales.

    Args:
        cos_dirs: cosenos directores del eje de la viga-columna

    Returns:
        Matriz de 6x6. Tipo np.ndarray.
    """
    c, s = cos_dirs[0], cos_dirs[1]  # Coseno y seno
    T = np.array([
        [c, s, 0, 0, 0, 0],
        [-s, c, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, c, s, 0],
        [0, 0, 0, -s, c, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    return T


def transf_coord_v3d(cos_dirs: np.ndarray) -> np.ndarray:
    """Matriz de transformación de coordenadas para vigas-columna espaciales.

    Transforma las coordenadas globales a locales.

    Args:
        cos_dirs: matriz con los cosenos directores del elemento.
            Primera fila: cosenos directores del eje local x
            Segunda fila: cosenos directores del eje local y
            Tercera fila: cosenos directores del eje local z

    Returns:
        Matriz de 12x12. Tipo np.ndarray.
    """
    T = np.zeros((12, 12))
    T[:3, :3] = cos_dirs
    T[3:6, 3:6] = cos_dirs
    T[6:9, 6:9] = cos_dirs
    T[9:, 9:] = cos_dirs
    return T

##############################################
## Vectores de fuerzas nodales equivalentes ##
##############################################


def rne_qud_vig(q: float, L: float) -> np.ndarray:
    """Reacción nodal equivalente, carga uniformemente distribuida, viga plana.

    La carga debe estar distribuida en toda la longitud del elemento.

    Args:
        q: carga uniformemente distribuida en toda la longitud de la viga
        L: longitud del elemento

    Returns:
        Vector de 4 componentes, tipo np.ndarray (4,)
    """
    v = q*L/12*np.array([6, L, 6, -L])
    return v


def rne_qpt_vig(P: float, L: float, a: float) -> np.ndarray:
    """Reacción nodal equivalente, carga puntual, viga plana.

    Args:
        P: carga puntual
        L: longitud de la viga
        a: posición de la carga puntual respecto al apoyo izquierdo

    Returns:
        Vector de 4 componentes, tipo np.ndarray (4,)
    """
    b = L - a  # Distancia de la carga puntual al apoyo derecho
    v1 = P*b**2/L**3*(3*a + b)
    v2 = P*a*b**2/L**2
    v3 = P*a/L**3*(a + 3*b)
    v4 = -P*a**2*b/L**2
    return np.array([v1, v2, v3, v4])


def rne_mcd_vig(M: float, L: float, a: float) -> np.ndarray:
    """Reacción nodal equivalente, momento concentrado, viga plana.

    Args:
        M: momento concentrado
        L: longitud de la viga
        a: posición del momento concentrado respecto al apoyo izquierdo

    Returns:
        Vector de 4 componentes, tipo np.ndarray (4,)
    """
    b = L - a  # Distancia del momento concentrado al apoyo derecho
    v1 = -6*M*a*b/L**3
    v2 = M*b/L**2*(b - 2*a)
    v3 = 6*M*a*b/L**3
    v4 = M*a/L**2*(a - 2*b)
    return np.array([v1, v2, v3, v4])
