"""Normalización óptica ligera y opcional para escenas oscuras.

No elimina sombras ni demuestra detección fiable: realza luminancia local
sin alterar coordenadas, tamaño o canales de la imagen BGR.
"""
import cv2


def mejorar_iluminacion(imagen, habilitado=True):
    """Devuelve (imagen_para_inferencia, fue_ajustada).

    En escenas bien iluminadas no hace trabajo adicional. La evaluación
    geométrica y las cajas siguen en las coordenadas originales.
    """
    if not habilitado:
        return imagen, False
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    fraccion_sombra = float((gris < 65).mean())
    if float(gris.mean()) >= 110 and fraccion_sombra < 0.28:
        return imagen, False
    lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(l)
    mejorada = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return mejorada, True
