"""Clasifica la cobertura del monitor vial sin emitir conclusiones legales.

No utiliza videos, modelos ni identificadores de personas.
Las cifras son apariciones por fotograma y no sujetos únicos.
"""
from collections.abc import Mapping


def resumen_cobertura(fotogramas: int, conteos: Mapping[str, int]) -> dict:
    """Devuelve una advertencia de validez independiente de los eventos."""
    if fotogramas < 0:
        raise ValueError("fotogramas no puede ser negativo")
    personas = int(conteos.get("personas", 0))
    motos = int(conteos.get("motos", 0))
    con_casco = int(conteos.get("deteccion_con_casco", 0))
    sin_casco = int(conteos.get("deteccion_sin_casco", 0))
    relaciones = int(conteos.get("relaciones", 0))
    candidatos = int(conteos.get("candidatos", 0))
    if any(x < 0 for x in (personas, motos, con_casco, sin_casco, relaciones, candidatos)):
        raise ValueError("conteos no pueden ser negativos")

    if fotogramas == 0:
        estado = "no_evaluable_sin_fotogramas"
        detalle = "No se procesaron fotogramas de la fuente."
    elif motos == 0:
        estado = "no_evaluable_sin_motocicletas"
        detalle = (
            "No se detectaron motocicletas con el modelo y umbral actuales. "
            "Las detecciones de casco se registraron por separado, pero no "
            "pueden vincularse a una motocicleta. Cero alertas NO significa "
            "que no existan personas sin casco. Verificar encuadre y modelo COCO."
        )
    elif relaciones == 0:
        estado = "no_evaluable_sin_asociaciones"
        detalle = (
            "Se detectaron motocicletas, pero no hubo asociaciones persona-moto. "
            "Cero alertas NO demuestra ausencia de personas sin casco. "
            "Revisar cajas y encuadre antes de modificar umbrales."
        )
    else:
        estado = "asociaciones_experimentales_para_revision"
        detalle = (
            "Hay asociaciones geométricas tentativas, no identificación "
            "de conductores ni confirmación de infracciones. "
            "Revisar visualmente candidatos y casos no asociados."
        )

    return {
        "estado": estado,
        "detalle": detalle,
        "fotogramas_analizados": fotogramas,
        "apariciones_persona": personas,
        "apariciones_motocicleta": motos,
        "detecciones_con_casco": con_casco,
        "detecciones_sin_casco": sin_casco,
        "asociaciones": relaciones,
        "candidatos_para_revision": candidatos,
        "observaciones_temporales_inciertas": int(conteos.get("estados_inciertos", 0)),
        "contradicciones_temporales": int(conteos.get("contradicciones_temporales", 0)),
        "resultado_es_infraccion_confirmada": False,
    }
