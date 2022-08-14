import unittest
import numpy as np
import analisa.edeb as edb


class MyTestCase(unittest.TestCase):
    def test_valores(self):
        S = pruebas_k67()
        Pf = np.array([0, 0, 0, -36, -54, 117, -36, -54, -117])*1e3
        P = np.array([80, 0, 0, 0, 0, 0, 40, 0, 0])*1e3
        d = np.linalg.inv(S) @ (P - Pf)
        self.assertAlmostEqual(d[5], -0.0260283, 6)

    def test_equilibrio_local(self, estructura):

        return

    def test_equilibrio_global_reticulado(self, reticulado):
        reticulado.procesamiento()
        dim = reticulado.dim
        cargas_nodales = reticulado.cargas_nodales
        reacciones = reticulado.reacciones()

        # Suma de fuerzas
        F = np.zeros(dim)
        for n in cargas_nodales.keys():  # Recorre diccionario cargas_nodales
            F += np.asarray(cargas_nodales[n])

        # Identificación de la dirección de las reacciones
        rX, rY, rZ = [], [], []
        cX, cY, cZ = 0, 0, 0  # Números de restricciones en X, Y, Z
        sum_r = 0  # Suma de las restricciones
        for r in reticulado.restricciones.values():
            if r[0] == 1:  # Si hay restricción en X
                cX += 1 + sum_r
                rX.append(cX)
            if dim > 1:
                if r[1] == 1:  # Si hay restricción en Y
                    cY += 1 + cX
                    rY.append(cY)
            if dim > 2:
                if r[2] == 1:  # Si hay restricción en Z
                    cZ += 1 + cY
                    rZ.append(cZ)

            sum_r += sum(r)

        # Vector sumatoria de reacciones
        R = np.zeros(dim)
        R[0] = sum(reacciones[cX])
        if dim > 1:
            R[1] = sum(reacciones[cY])
        if dim > 2:
            R[2] = sum(reacciones[cZ])

        self.assertAlmostEqual(np.linalg.norm(F+R), 0.0, 6)


def desplaz_test():
    nudos_csv = 'csv/nudos_k39.csv'
    barras_csv = 'csv/barras_k39.csv'
    materiales = edb.Material(70e9)  # Lista de materiales
    A = 4000*1e-3**2
    secciones = edb.Seccion(area=A)  # Lista de secciones
    restricciones = {1: (1, 1), 2: (1, 1), 3: (1, 0)}  # Restricciones
    cargas = {3: (0, -400e3), 4: (800e3, -400e3)}
    ret = edb.Reticulado(datos_nudos=nudos_csv, restricciones=restricciones,
                         datos_barras=barras_csv, materiales=materiales,
                         secciones=secciones, cargas_nodales=cargas)
    ret.procesamiento()
    return ret.desplaz_gdl()


def pruebas_k67():
    nudos_csv = 'csv/nudos_k67.csv'
    barras_csv = 'csv/barras_k67.csv'
    m = edb.Material(30e9)
    m.elast_transv = 11500
    materiales = m  # Lista de materiales
    A = 75e-3
    Iz = 4.8e8*1e-3**4
    s = edb.Seccion(area=A, inercia_z=Iz)
    secciones = s  # Lista de secciones
    restricciones = {1: (1, 1, 1), 2: (1, 1, 1)}
    cargas = {3: (80e3, 0, 0), 5: (40e3, 0, 0)}
    p = edb.Portico(datos_nudos=nudos_csv, restricciones=restricciones,
                    datos_barras=barras_csv, materiales=materiales,
                    secciones=secciones,cargas_nodales=cargas)
    p.procesamiento()
    return p.rigidez_gdl()


def pruebas_k81():
    nudos_csv = 'csv/nudos_k81.csv'
    barras_csv = 'csv/barras_k81.csv'
    materiales = [est.Material(10000)]  # Lista de materiales
    A = 8.4
    secciones = [est.Seccion(area=A)]  # Lista de secciones
    restricciones = {1: (1, 1, 1), 2: (1, 1, 1), 3: (1, 1, 1), 4: (1, 1, 1)}
    cargas = {5: (0, -100, -50)}
    coords, elementos = est.preprocesamiento(nudos_csv, barras_csv, materiales,
                                             secciones)
    p = est.Reticulado(coords, restricciones, elementos, cargas_nodales=cargas)
    resultados = p.procesamiento()
    return p.rigidez_gdl()


def pruebas_k84():
    nudos_csv = 'csv/nudos_k84.csv'
    barras_csv = 'csv/barras_k84.csv'
    m = est.Material(29000)
    m.elast_transv = 11500
    materiales = [m]  # Lista de materiales
    A = 32.9
    Iz = 716
    Iy = 213
    s = est.Seccion(area=A, inercia_y=Iy, inercia_z=Iz)
    s.inercia_polar = 15.1
    secciones = [s]  # Lista de secciones
    restricciones = {2: (1, 1, 1, 1, 1, 1), 3: (1, 1, 1, 1, 1, 1), 4: (1, 1, 1,
                                                                       1, 1, 1)
                     }
    cargas = {1: (0, -360, 0, -1800, 0, 1800+1200), 2: (0, -360, 0, 0, 0,
                                                        -1200)}
    coords, elementos = est.preprocesamiento(nudos_csv, barras_csv, materiales,
                                             secciones)
    p = est.Reticulado(coords, restricciones, elementos, cargas_nodales=cargas)
    resultados = p.procesamiento()
    return p.reacciones()


if __name__ == '__main__':
    unittest.main()
