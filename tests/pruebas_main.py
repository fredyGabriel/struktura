import analisa.edeb as edb


def pruebas_k67():
    nudos_csv = 'csv/nudos_k67.csv'
    barras_csv = 'csv/barras_k67.csv'
    m = edb.Material(30e9)
    m.elast_transv = 11500
    A = 75e-3
    Iz = 4.8e8*1e-3**4
    s = edb.Seccion(area=A, inercia_z=Iz)
    restricciones = {1: (1, 1, 1), 2: (1, 1, 1)}
    cargas = {3: (80e3, 0, 0), 5: (40, 0, 0)}
    p = edb.Portico(nudos_csv, restricciones, barras_csv,
                    cargas_nodales=cargas, materiales=m, secciones=s)
    p.procesamiento()
    return p.reacciones()


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
    cargas = {1: (0, -30, 0, -1800, 0, 1800+1200), 2: (0, -30, 0, 0, 0,
                                                       -1200)}
    coords, elementos = est.preprocesamiento(nudos_csv, barras_csv, materiales,
                                             secciones)
    p = est.Portico(coords, restricciones, elementos, cargas_nodales=cargas)
    resultados = p.procesamiento()
    return p.rigidez_gdl()


def main():
    print("Reacciones")
    print(pruebas_k67()/1000)


if __name__ == '__main__':
    main()

#%%
