"""Diseño de las vistas del detector conductual, sin dependencias gráficas.

No modifica los datos de entrada de los modelos ni el significado de eventos.
"""
from math import isfinite


def dimensiones_anotadas(ancho: int, alto: int) -> tuple[int, int, int]:
    """Dimensiones de la imagen presentada + panel fuera de la escena.

    Las imágenes KTH de 160 x 120 se presentan a 640 x 480 (no se
    reescalan antes de YOLO). La ampliación es sólo para lectura humana.
    """
    if not isinstance(ancho, int) or not isinstance(alto, int):
        raise ValueError("Dimensiones inválidas")
    if ancho < 1 or alto < 1:
        raise ValueError("Dimensiones inválidas")
    escala = max(1.0, 640 / ancho, 480 / alto)
    salida = (round(ancho * escala), round(alto * escala))
    return salida[0], salida[1], 108


def texto_evento_resumido(evento, modo: str) -> str:
    """Resumen corto para el panel; no etiqueta conductas violentas."""
    if modo == "brazos":
        brazo = "izq" if evento.brazo == "izquierdo" else "der"
        accion = (
            "elevado persistente"
            if evento.tipo == "brazo_elevado_persistente"
            else "movimiento rapido aparente"
        )
        return f"ID {evento.track_id} {brazo}: {accion} (revisar)"
    if modo == "proximidad":
        a, b = evento.track_ids
        return f"IDs {a}-{b}: proximidad observada (revisar)"
    if modo == "posturas":
        return f"ID {evento.track_id}: postura horizontal (revisar)"
    raise ValueError("Modo desconocido")
