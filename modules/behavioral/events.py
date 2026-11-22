"""Reglas conservadoras para generar candidatos de eventos a partir de pose.

Esta lógica no determina violencia, intención, delitos ni emergencias.
La postura horizontal puede deberse a una acción totalmente inocua.
"""
from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class Persona:
    """Observación de una persona con puntos COCO (x, y, confianza)."""

    track_id: int | None
    caja: tuple[float, float, float, float]
    puntos: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class Evento:
    fotograma: int
    track_id: int
    tipo: str


def postura_horizontal(persona: Persona, confianza_minima: float = 0.5) -> bool:
    """Candidato geométrico, no reconocimiento de una caída real."""
    if len(persona.puntos) < 13:
        return False
    indices = (5, 6, 11, 12)  # hombros y caderas en COCO Pose
    if any(persona.puntos[i][2] < confianza_minima for i in indices):
        return False

    x1, y1, x2, y2 = persona.caja
    ancho, alto = x2 - x1, y2 - y1
    if alto <= 0 or ancho <= 0 or ancho / alto < 1.2:
        return False

    izq_h, der_h, izq_c, der_c = (persona.puntos[i] for i in indices)
    hombro = ((izq_h[0] + der_h[0]) / 2, (izq_h[1] + der_h[1]) / 2)
    cadera = ((izq_c[0] + der_c[0]) / 2, (izq_c[1] + der_c[1]) / 2)
    dx, dy = cadera[0] - hombro[0], cadera[1] - hombro[1]
    if hypot(dx, dy) < max(10.0, ancho * 0.12):
        return False
    return abs(dx) >= 1.5 * max(1.0, abs(dy))


class DetectorEventos:
    """Confirma observaciones consecutivas con track ID y limita duplicados."""

    def __init__(
        self,
        min_fotogramas: int = 6,
        separacion_alertas: int = 90,
        caducidad: int = 30,
    ) -> None:
        if min_fotogramas < 2 or separacion_alertas < 1 or caducidad < 1:
            raise ValueError("Parámetros de seguimiento inválidos")
        self.min_fotogramas = min_fotogramas
        self.separacion_alertas = separacion_alertas
        self.caducidad = caducidad
        self._racha: dict[int, tuple[int, int]] = {}
        self._ultima_alerta: dict[int, int] = {}

    def procesar(self, numero_fotograma: int, personas: list[Persona]) -> list[Evento]:
        if numero_fotograma < 1:
            raise ValueError("El número de fotograma debe ser positivo")
        alertas: list[Evento] = []
        vistos: set[int] = set()

        for persona in personas:
            identificador = persona.track_id
            if identificador is None or identificador in vistos:
                continue
            vistos.add(identificador)
            if not postura_horizontal(persona):
                self._racha.pop(identificador, None)
                continue

            anterior, previo = self._racha.get(identificador, (0, -1))
            consecutivos = anterior + 1 if previo == numero_fotograma - 1 else 1
            self._racha[identificador] = (consecutivos, numero_fotograma)

            ultima = self._ultima_alerta.get(identificador, -10**9)
            if (
                consecutivos >= self.min_fotogramas
                and numero_fotograma - ultima >= self.separacion_alertas
            ):
                alertas.append(
                    Evento(numero_fotograma, identificador, "postura_horizontal")
                )
                self._ultima_alerta[identificador] = numero_fotograma

        for identificador, (_, ultimo) in list(self._racha.items()):
            if numero_fotograma - ultimo > self.caducidad:
                del self._racha[identificador]
        for identificador, ultimo in list(self._ultima_alerta.items()):
            if numero_fotograma - ultimo > max(600, self.caducidad * 10):
                del self._ultima_alerta[identificador]
        return alertas
