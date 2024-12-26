from pyautocad import Autocad


def obtener_coordenadas_nudos(ruta_dwg):
    """
    Obtiene las coordenadas de los nudos de un gráfico de barras en un archivo
    .dwg, considerando que un nudo siempre está al inicio o al final de una
    línea.

    Args:
      ruta_dwg: La ruta al archivo .dwg.

    Returns:
      Un diccionario con las coordenadas de los nudos.
    """

    acad = Autocad(create_if_not_exists=True)
    acad.prompt("Hola, por favor abre el archivo .dwg con el gráfico de barras.")
    doc = acad.ActiveDocument

    # 1. Identificar las barras (líneas en diferentes capas)
    barras = [entity for entity in doc.ModelSpace if entity.ObjectName == "AcDbLine"]

    # 2. Encontrar los puntos de intersección y filtrar los que no son nudos
    nudos = []
    for i in range(len(barras)):
        for j in range(i + 1, len(barras)):
            intersecciones = barras[i].IntersectWith(barras[j], 0)
            if intersecciones:
                for punto in intersecciones:
                    # Verificar si el punto es inicio o fin de alguna línea
                    if (punto == barras[i].StartPoint or punto == barras[i].EndPoint or
                        punto == barras[j].StartPoint or punto == barras[j].EndPoint):
                        nudos.append(punto)

    # 3. Ordenar los puntos de intersección
    nudos.sort(key=lambda punto: (punto[0], punto[1]))  # Ordenar por x, luego por y

    # 4. Crear el diccionario de coordenadas
    coordenadas = {i: (punto[0], punto[1]) for i, punto in enumerate(nudos)}

    return coordenadas


# Ejemplo de uso
ruta_dwg = "coords_from_dwg.dwg"  # Reemplaza con la ruta a tu archivo
coordenadas = obtener_coordenadas_nudos(ruta_dwg)
print(coordenadas)