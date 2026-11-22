"""Consistencia temporal conservadora para detecciones de cascos sobre motos.

Mantiene zonas de cabeza *aproximadas*, nunca identidades humanas. Una misma
zona observada como CON_CASCO y SIN_CASCO en una ventana corta pasa a
INCIERTO; no se supone que una etiqueta verde sea protección verificada.
Conserva la observación original en Relacion.casco para auditoría y video.

Sólo usa geometría y clases normalizadas; no carga YOLO ni imágenes.
"""
from dataclasses import dataclass, field
from .association import Relacion, _firma_relativa, _misma_zona_relativa


@dataclass(frozen=True)
class ContradiccionCasco:
    fotograma: int
    moto_id: int
    persona_id: int
    cabeza_id: int | None
    clase_anterior: str
    clase_actual: str
    confianza_actual: float
    zona_local_id: int


@dataclass
class _Zona:
    local_id: int
    moto_id: int
    firma: tuple[float, float, float, float]
    cabeza_id: int | None
    ultimo_paso: int
    observaciones: list[tuple[int, str]] = field(default_factory=list)
    incierta: bool = False


def _coincidencia(
    zona: _Zona, firma: tuple[float, float, float, float],
    cabeza_id: int | None,
) -> float | None:
    """La geometría evita asociar IDs reciclados a otra cabeza."""
    # _misma_zona_relativa utiliza una cota conservadora (0.13) y escala.
    if not _misma_zona_relativa(zona.firma, firma):
        return None
    dx, dy = abs(zona.firma[0] - firma[0]), abs(zona.firma[1] - firma[1])
    misma_id = cabeza_id is not None and cabeza_id == zona.cabeza_id
    # Un ID distinto exige una coincidencia espacial aún más estrecha.
    if not misma_id and (dx > 0.085 or dy > 0.085):
        return None
    return (10.0 if misma_id else 0.0) - dx - dy


class ConsistenciaTemporalCabezas:
    """Agrupa tentativamente clases de una cabeza por moto y región relativa.

    Las zonas sólo viven unos pocos fotogramas analizados. Cada observación
    del fotograma actual utiliza una zona distinta, de forma que dos cabezas
    presentes a la vez no comparten historial. No altera etiquetas crudas.
    """

    def __init__(self, ventana: int = 12) -> None:
        if ventana < 2:
            raise ValueError("La ventana temporal debe tener al menos 2 pasos")
        self.ventana = ventana
        self._zonas: dict[int, list[_Zona]] = {}
        self._siguiente_id = 1
        self.contradicciones: list[ContradiccionCasco] = []

    def procesar(
        self, paso: int, fotograma: int, relaciones: list[Relacion]
    ) -> list[Relacion]:
        if paso < 1 or fotograma < 1:
            raise ValueError("Paso y fotograma deben ser positivos")
        self.contradicciones = []
        limite = paso - self.ventana + 1
        for moto_id in list(self._zonas):
            vivas = [
                z for z in self._zonas[moto_id] if z.ultimo_paso >= limite
            ]
            if vivas:
                self._zonas[moto_id] = vivas
            else:
                del self._zonas[moto_id]

        # Reservar cada zona UNA vez por fotograma para no unir ocupantes
        # simultáneos, incluso si YOLO da una sola persona COCO.
        ocupadas: set[int] = set()
        salida: list[Relacion] = []
        for r in relaciones:
            if (
                r.casco is None
                or r.estado not in ("con_casco", "sin_casco")
                or r.moto.track_id is None
                or r.persona.track_id is None
            ):
                salida.append(r)
                continue
            firma = _firma_relativa(r)
            if firma is None:
                salida.append(r)
                continue
            moto_id = r.moto.track_id
            cabeza_id = r.casco.track_id
            zonas = self._zonas.setdefault(moto_id, [])
            opciones = [
                (puntaje, zona)
                for zona in zonas
                if zona.local_id not in ocupadas
                for puntaje in [_coincidencia(zona, firma, cabeza_id)]
                if puntaje is not None
            ]
            if opciones:
                # Preferir ID estable y la coincidencia espacial más cercana.
                zona = max(opciones, key=lambda x: x[0])[1]
            else:
                zona = _Zona(self._siguiente_id, moto_id, firma, cabeza_id, paso)
                self._siguiente_id += 1
                zonas.append(zona)

            ocupadas.add(zona.local_id)
            zona.observaciones = [
                (p, clase) for p, clase in zona.observaciones if p >= limite
            ]
            previo_incierto = zona.incierta
            clase_anterior = next(
                (clase for _, clase in reversed(zona.observaciones)
                 if clase != r.estado), None
            )
            zona.observaciones.append((paso, r.estado))
            zona.incierta = len({c for _, c in zona.observaciones}) > 1
            zona.ultimo_paso = paso
            zona.firma = firma
            zona.cabeza_id = cabeza_id

            if zona.incierta and not previo_incierto and clase_anterior:
                self.contradicciones.append(
                    ContradiccionCasco(
                        fotograma, moto_id, r.persona.track_id,
                        cabeza_id, clase_anterior, r.estado,
                        r.casco.confianza, zona.local_id,
                    )
                )

            if zona.incierta:
                salida.append(
                    Relacion(r.moto, r.persona, r.casco, "incierto")
                )
            else:
                salida.append(r)
        return salida
