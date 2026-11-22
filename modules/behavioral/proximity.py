"""Observación conservadora de proximidad persistente entre personas.

Proximidad NO implica pelea, violencia, amenaza ni intención.
Se usan sólo cajas y track IDs temporales de YOLO Pose, no identidades.
"""
from dataclasses import dataclass
from itertools import combinations

from .events import Persona


@dataclass(frozen=True)
class EventoProximidad:
    fotograma: int
    track_ids: tuple[int, int]
    tipo: str = "proximidad_persistente"


def estan_cerca(
    primero: Persona,
    segundo: Persona,
    margen_horizontal: float = 0.35,
    solape_vertical: float = 0.40,
) -> bool:
    """Proximidad geométrica normalizada por tamaño, no píxeles fijos.

    Sólo compara observaciones en escala similar, con cuerpos en una
    franja vertical compartida. Sigue habiendo falsos positivos normales
    en multitudes, filas, abrazos o escenas con perspectiva.
    """
    ax1, ay1, ax2, ay2 = primero.caja
    bx1, by1, bx2, by2 = segundo.caja
    aw, ah = ax2 - ax1, ay2 - ay1
    bw, bh = bx2 - bx1, by2 - by1
    if min(aw, ah, bw, bh) <= 0:
        return False
    if max(ah, bh) / min(ah, bh) > 2.5:
        return False
    vertical = max(0.0, min(ay2, by2) - max(ay1, by1))
    if vertical / min(ah, bh) < solape_vertical:
        return False
    horizontal = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    return horizontal <= margen_horizontal * min(aw, bw)


class DetectorProximidad:
    """Marca observaciones persistentes de proximidad como candidatos.

    Cada pareja de IDs debe cumplir la condición en fotogramas
    consecutivos. Un cambio de ID reinicia el conteo. Sólo emite
    evidencias para revisión humana, no clasificación de acciones.
    """

    def __init__(
        self,
        min_fotogramas: int = 6,
        separacion_alertas: int = 90,
        margen_horizontal: float = 0.35,
        solape_vertical: float = 0.40,
    ) -> None:
        if min_fotogramas < 2 or separacion_alertas < 1:
            raise ValueError("Ventana o separación de alertas inválida")
        if not 0 <= margen_horizontal <= 2 or not 0 < solape_vertical <= 1:
            raise ValueError("Umbral geométrico inválido")
        self.min_fotogramas = min_fotogramas
        self.separacion_alertas = separacion_alertas
        self.margen_horizontal = margen_horizontal
        self.solape_vertical = solape_vertical
        self._rachas: dict[tuple[int, int], tuple[int, int]] = {}
        self._ultima_alerta: dict[tuple[int, int], int] = {}

    def procesar(
        self, fotograma: int, personas: list[Persona]
    ) -> list[EventoProximidad]:
        if fotograma < 1:
            raise ValueError("Fotograma inválido")
        por_id: dict[int, Persona] = {}
        for persona in personas:
            if persona.track_id is not None:
                por_id.setdefault(persona.track_id, persona)

        siguientes: dict[tuple[int, int], tuple[int, int]] = {}
        eventos: list[EventoProximidad] = []
        for a, b in combinations(sorted(por_id), 2):
            if not estan_cerca(
                por_id[a], por_id[b],
                self.margen_horizontal, self.solape_vertical,
            ):
                continue
            par = (a, b)
            anterior = self._rachas.get(par)
            cantidad = (
                anterior[0] + 1
                if anterior is not None and anterior[1] == fotograma - 1
                else 1
            )
            siguientes[par] = (cantidad, fotograma)
            ultima = self._ultima_alerta.get(par, -10**9)
            if (
                cantidad >= self.min_fotogramas
                and fotograma - ultima >= self.separacion_alertas
            ):
                eventos.append(EventoProximidad(fotograma, par))
                self._ultima_alerta[par] = fotograma

        self._rachas = siguientes
        # Acotar memoria sin alterar el enfriamiento de eventos activos.
        for par, ultimo in list(self._ultima_alerta.items()):
            if (
                par not in siguientes
                and fotograma - ultimo > max(600, self.separacion_alertas * 2)
            ):
                del self._ultima_alerta[par]
        return eventos
