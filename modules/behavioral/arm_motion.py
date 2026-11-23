"""Movimientos observables de brazos con YOLO Pose, sin inferir intenciones.

Señala brazos levantados y desplazamientos aparentes repetidos. No identifica
golpes, amenazas, agresiones ni armas. Lógica pura, sin OpenCV ni Ultralytics.
"""
from dataclasses import dataclass
from math import hypot, isfinite

from .events import Persona


# COCO 17 keypoints: hombro, codo, muñeca y cadera por lado.
BRAZOS = {"izquierdo": (5, 7, 9, 11), "derecho": (6, 8, 10, 12)}


@dataclass(frozen=True)
class EventoBrazo:
    fotograma: int
    track_id: int
    tipo: str
    brazo: str


@dataclass(frozen=True)
class ObservacionBrazo:
    posicion_relativa: tuple[float, float]
    elevado: bool


@dataclass(frozen=True)
class HistoriaBrazo:
    fotograma: int
    tiempo_s: float
    caja: tuple[float, float, float, float]
    observacion: ObservacionBrazo
    cuenta_elevados: int
    cuenta_movimientos: int


def observar_brazos(
    persona: Persona, confianza_minima: float = 0.55
) -> dict[str, ObservacionBrazo]:
    """Usa posición de muñeca relativa al hombro medida en longitudes de torso."""
    if len(persona.puntos) < 13 or not 0 < confianza_minima <= 1:
        return {}
    x1, y1, x2, y2 = persona.caja
    ancho, alto = x2 - x1, y2 - y1
    if ancho <= 0 or alto <= 0:
        return {}
    resultado = {}
    for lado, (h, c, m, hip) in BRAZOS.items():
        puntos = [persona.puntos[i] for i in (h, c, m, hip)]
        if any(
            len(p) != 3
            or not all(isfinite(v) for v in p)
            or p[2] < confianza_minima
            for p in puntos
        ):
            continue
        hombro, _, muneca, cadera = puntos
        torso = hypot(cadera[0] - hombro[0], cadera[1] - hombro[1])
        if not 0.10 * alto <= torso <= 0.85 * alto:
            continue
        pos = (
            (muneca[0] - hombro[0]) / torso,
            (muneca[1] - hombro[1]) / torso,
        )
        resultado[lado] = ObservacionBrazo(pos, pos[1] < -0.20)
    return resultado


def _caja_continua(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    aw, ah, bw, bh = a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]
    if min(aw, ah, bw, bh) <= 0:
        return False
    if min(aw * ah, bw * bh) / max(aw * ah, bw * bh) < 0.5:
        return False
    ac = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    bc = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    return (
        abs(ac[0] - bc[0]) <= 0.5 * max(aw, bw)
        and abs(ac[1] - bc[1]) <= 0.5 * max(ah, bh)
    )


class DetectorMovimientosBrazos:
    """Rachas independientes por track y brazo, con enfriamiento de eventos."""

    def __init__(
        self,
        min_elevados: int = 4,
        min_movimientos: int = 2,
        velocidad_minima: float = 1.5,
        separacion_alertas: int = 90,
        pausa_maxima_s: float = 5.0,
    ) -> None:
        if (
            min_elevados < 2 or min_movimientos < 1
            or not isfinite(velocidad_minima) or velocidad_minima <= 0
            or separacion_alertas < 1
            or not isfinite(pausa_maxima_s) or pausa_maxima_s <= 0
        ):
            raise ValueError("Parámetros de movimiento inválidos")
        self.min_elevados = min_elevados
        self.min_movimientos = min_movimientos
        self.velocidad_minima = velocidad_minima
        self.separacion_alertas = separacion_alertas
        self.pausa_maxima_s = pausa_maxima_s
        self._historial: dict[tuple[int, str], HistoriaBrazo] = {}
        self._ultima_alerta: dict[tuple[int, str, str], int] = {}

    def procesar(
        self, fotograma: int, personas: list[Persona], tiempo_s: float
    ) -> list[EventoBrazo]:
        if fotograma < 1 or not isfinite(tiempo_s) or tiempo_s < 0:
            raise ValueError("Fotograma o tiempo inválido")
        siguiente = {}
        eventos = []
        vistos = set()
        for persona in personas:
            identificador = persona.track_id
            if identificador is None or identificador in vistos:
                continue
            vistos.add(identificador)
            for brazo, observacion in observar_brazos(persona).items():
                clave = (identificador, brazo)
                previo = self._historial.get(clave)
                consecutivo = (
                    previo is not None
                    and previo.fotograma == fotograma - 1
                    and 0 < tiempo_s - previo.tiempo_s <= self.pausa_maxima_s
                    and _caja_continua(previo.caja, persona.caja)
                )
                elevados = (
                    (previo.cuenta_elevados + 1 if consecutivo else 1)
                    if observacion.elevado else 0
                )
                velocidad = (
                    hypot(
                        observacion.posicion_relativa[0] - previo.observacion.posicion_relativa[0],
                        observacion.posicion_relativa[1] - previo.observacion.posicion_relativa[1],
                    ) / (tiempo_s - previo.tiempo_s)
                    if consecutivo else 0.0
                )
                movimientos = (
                    previo.cuenta_movimientos + 1
                    if consecutivo and velocidad >= self.velocidad_minima
                    else 0
                )
                siguiente[clave] = HistoriaBrazo(
                    fotograma, tiempo_s, persona.caja, observacion,
                    elevados, movimientos,
                )
                for tipo, cumple in (
                    ("brazo_elevado_persistente", elevados >= self.min_elevados),
                    ("desplazamiento_rapido_aparente_de_brazo",
                     movimientos >= self.min_movimientos),
                ):
                    evento_clave = (identificador, brazo, tipo)
                    if cumple and (
                        fotograma - self._ultima_alerta.get(evento_clave, -10**9)
                        >= self.separacion_alertas
                    ):
                        eventos.append(EventoBrazo(fotograma, identificador, tipo, brazo))
                        self._ultima_alerta[evento_clave] = fotograma
        self._historial = siguiente
        for clave, fotograma_anterior in list(self._ultima_alerta.items()):
            if fotograma - fotograma_anterior > max(600, self.separacion_alertas * 2):
                del self._ultima_alerta[clave]
        return eventos
