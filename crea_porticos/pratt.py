import numpy as np
import matplotlib.pyplot as plt


def generate_pratt_truss(L, H, N, slope, VL_izq, VL_der):
    """
    Genera las coordenadas de nodos y elementos de una cabriada Pratt.

    Args:
        L (float): Luz de la cabriada.
        H (float): Altura de la cabriada en el punto medio.
        N (int): Número de paneles.
        slope (float): Pendiente del techo.
        VL_izq (float): Voladizo izquierdo.
        VL_der (float): Voladizo derecho.

    Returns:
        tuple: Una tupla que contiene dos listas:
            - nodes: Lista de tuplas (x, y), coordenadas de los nodos.
            - elements: Lista de tuplas (nodo_inicio, nodo_fin), elementos.
    """

    theta = np.arctan(slope)
    panel_length = (L + VL_izq + VL_der) / N

    nodes = []
    elements = []

    # Nodos del cordón inferior, incluyendo voladizos
    for i in range(N + 1):
        x = VL_izq + i * panel_length
        y = 0
        nodes.append((x, y))

    # Nodos del cordón superior
    for i in range(N + 1):
        x = VL_izq + i * panel_length
        if i == 0 or i == N:
            y = 0  # Nodos de los extremos en el nivel del cordón inferior
        else:
            y = (x - VL_izq) * np.tan(theta)
        nodes.append((x, y))

    # Elementos del cordón inferior y superior
    for i in range(N):
        elements.append((i, i + 1))
        elements.append((N + 1 + i, N + 1 + i + 1))

    # Elementos verticales y diagonales
    for i in range(N):
        elements.append((i, N + 1 + i))
        if i < N - 1:  # Evitar diagonal en el último panel
            elements.append((i + 1, N + 1 + i))

    # Visualización
    for i, (x, y) in enumerate(nodes):
        plt.plot(x, y, 'bo')
        plt.text(x, y, str(i))

    for start, end in elements:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        plt.plot([x1, x2], [y1, y2], 'k-')

    plt.title('Cabriada Pratt')
    plt.xlabel('Distancia horizontal')
    plt.ylabel('Distancia vertical')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    return nodes, elements
