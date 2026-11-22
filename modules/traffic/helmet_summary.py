"""Resumen del detector de cascos independiente del análisis de motocicletas.

Nunca interpreta apariciones por frame como personas únicas ni infracciones.
"""


def resumen_cascos(fotogramas: int, conteos: dict) -> dict:
    if fotogramas < 0:
        raise ValueError("fotogramas negativos")
    con_casco = int(conteos.get("con_casco", 0))
    sin_casco = int(conteos.get("sin_casco", 0))
    placa = int(conteos.get("placa", 0))
    if min(con_casco, sin_casco, placa) < 0:
        raise ValueError("conteos negativos")
    estado = (
        "sin_fotogramas" if fotogramas == 0 else
        "sin_detecciones" if con_casco + sin_casco == 0 else
        "detecciones_de_casco_sin_asociacion_vial"
    )
    return {
        "modo": "solo_cascos",
        "estado": estado,
        "fotogramas_analizados": fotogramas,
        "apariciones_con_casco": con_casco,
        "apariciones_sin_casco": sin_casco,
        "apariciones_placa": placa,
        "motocicletas_analizadas": None,
        "infraccion_confirmada": False,
        "advertencia": (
            "Las apariciones se suman por fotograma; no son personas únicas. "
            "No se relacionaron con motocicletas ni se verificaron infracciones. "
            "Evaluar falsos positivos visualmente antes de concluir precisión."
        ),
    }
