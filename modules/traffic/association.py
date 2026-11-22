"""Asociación geométrica conservadora entre motos, personas y detecciones de casco.

No identifica infracciones confirmadas. Cada relación es una hipótesis para
revisión: una persona cerca de una moto no necesariamente la conduce.
No requiere OpenCV, Ultralytics ni datos personales.
"""
from dataclasses import dataclass
from math import hypot
from unicodedata import normalize


@dataclass(frozen=True)
class Objeto:
    clase: str
    caja: tuple[float, float, float, float]
    confianza: float
    track_id: int | None = None


@dataclass(frozen=True)
class Relacion:
    moto: Objeto
    persona: Objeto
    casco: Objeto | None
    estado: str  # con_casco | sin_casco | desconocido


@dataclass(frozen=True)
class EventoVial:
    fotograma: int
    moto_id: int
    persona_id: int
    confianza: float
    personas_asociadas: int
    tipo: str = "posible_persona_sin_casco"


def normalizar_clase(nombre: str) -> str:
    texto = normalize("NFKD", nombre).encode("ascii", "ignore").decode()
    texto = texto.lower().replace("_", " ").replace("-", " ").strip()
    texto = " ".join(texto.split())
    if texto in ("without helmet", "no helmet", "sin casco"):
        return "sin_casco"
    if texto in ("with helmet", "helmet", "con casco"):
        return "con_casco"
    if texto in ("motorcycle", "motorbike", "motocicleta"):
        return "moto"
    if texto in ("person", "persona"):
        return "persona"
    if texto in ("licence", "license plate", "placa", "matricula"):
        return "placa"
    return "otro"


def _dimensiones(caja: tuple[float, float, float, float]) -> tuple[float, float]:
    return max(0.0, caja[2] - caja[0]), max(0.0, caja[3] - caja[1])


def _superposicion_x(a: Objeto, b: Objeto) -> float:
    ancho_a, _ = _dimensiones(a.caja)
    ancho_b, _ = _dimensiones(b.caja)
    if ancho_a == 0 or ancho_b == 0:
        return 0.0
    inter = max(0.0, min(a.caja[2], b.caja[2]) - max(a.caja[0], b.caja[0]))
    return inter / min(ancho_a, ancho_b)


def _puntuacion_persona_moto(persona: Objeto, moto: Objeto) -> float:
    """Relaciona un torso sobre la moto, sin equipararlo a conducción real."""
    pw, ph = _dimensiones(persona.caja)
    mw, mh = _dimensiones(moto.caja)
    if min(pw, ph, mw, mh) <= 0:
        return 0.0
    px1, py1, px2, py2 = persona.caja
    mx1, my1, mx2, my2 = moto.caja
    cx_persona = (px1 + px2) / 2
    cx_moto = (mx1 + mx2) / 2
    if not (mx1 - 0.10 * mw <= cx_persona <= mx2 + 0.10 * mw):
        return 0.0
    if py1 >= my1 + 0.65 * mh:
        return 0.0
    if not (my1 - 0.35 * mh <= py2 <= my2 + 0.15 * mh):
        return 0.0
    solape = _superposicion_x(persona, moto)
    if solape < 0.35:
        return 0.0
    centro = max(0.0, 1.0 - abs(cx_persona - cx_moto) / max(mw, 1.0))
    distancia_base = abs(py2 - (my1 + 0.40 * mh))
    base = max(0.0, 1.0 - distancia_base / max(1.2 * mh, 1.0))
    return 0.45 * solape + 0.35 * centro + 0.20 * base


def _puntuacion_casco_persona(casco: Objeto, persona: Objeto) -> float:
    pw, ph = _dimensiones(persona.caja)
    hw, hh = _dimensiones(casco.caja)
    if min(pw, ph, hw, hh) <= 0:
        return 0.0
    x1, y1, x2, _ = persona.caja
    cx = (casco.caja[0] + casco.caja[2]) / 2
    cy = (casco.caja[1] + casco.caja[3]) / 2
    if not (x1 - 0.05 * pw <= cx <= x2 + 0.05 * pw):
        return 0.0
    if not (y1 - 0.08 * ph <= cy <= y1 + 0.52 * ph):
        return 0.0
    if hw > 1.20 * pw or hh > 0.70 * ph:
        return 0.0
    estimado_x, estimado_y = (x1 + x2) / 2, y1 + 0.20 * ph
    distancia = hypot(
        (cx - estimado_x) / max(pw, 1.0),
        (cy - estimado_y) / max(ph, 1.0),
    )
    return max(0.0, 1.0 - distancia)


def relacionar(
    motos: list[Objeto],
    personas: list[Objeto],
    cascos: list[Objeto],
    umbral_persona_moto: float = 0.63,
) -> list[Relacion]:
    """Cada persona se vincula a una moto como máximo.

    Si dos motos son prácticamente igual de probables, no se asigna.
    Si dos personas compiten por el mismo casco y no se distinguen claramente,
    la etiqueta se descarta para prevenir atribuciones peligrosas.
    """
    asignaciones: dict[int, int] = {}  # indice persona -> indice moto
    for pi, persona in enumerate(personas):
        opciones = sorted(
            ((_puntuacion_persona_moto(persona, moto), mi) for mi, moto in enumerate(motos)),
            reverse=True,
        )
        if not opciones or opciones[0][0] < umbral_persona_moto:
            continue
        if len(opciones) > 1 and opciones[0][0] - opciones[1][0] < 0.10:
            continue
        asignaciones[pi] = opciones[0][1]

    # Una observación de casco se asigna, como máximo, a una cabeza.
    cascos_por_persona: dict[int, list[Objeto]] = {}
    for casco in cascos:
        if normalizar_clase(casco.clase) not in ("con_casco", "sin_casco"):
            continue
        puntuaciones = sorted(
            ((_puntuacion_casco_persona(casco, personas[pi]), pi) for pi in asignaciones),
            reverse=True,
        )
        if not puntuaciones or puntuaciones[0][0] < 0.50:
            continue
        if len(puntuaciones) > 1 and puntuaciones[0][0] - puntuaciones[1][0] < 0.12:
            continue
        cascos_por_persona.setdefault(puntuaciones[0][1], []).append(casco)

    relaciones: list[Relacion] = []
    for pi, mi in sorted(asignaciones.items()):
        etiquetas = cascos_por_persona.get(pi, [])
        tipos = {normalizar_clase(item.clase) for item in etiquetas}
        if len(tipos) != 1 or not tipos:
            estado, casco_asociado = "desconocido", None
        else:
            tipo = next(iter(tipos))
            estado = tipo
            casco_asociado = max(etiquetas, key=lambda item: item.confianza)
        relaciones.append(Relacion(motos[mi], personas[pi], casco_asociado, estado))
    return relaciones


class ConfirmadorEventosViales:
    """Alertas por moto tras observaciones consecutivas del MISMO rider.

    Los eventos se emiten una vez por track de motocicleta por ejecución,
    para no contar el mismo vehículo múltiples veces.
    """

    def __init__(self, observaciones_minimas: int = 4) -> None:
        if observaciones_minimas < 2:
            raise ValueError("observaciones_minimas debe ser al menos 2")
        self.observaciones_minimas = observaciones_minimas
        self._rachas: dict[tuple[int, int], tuple[int, int]] = {}
        self._registradas: set[int] = set()

    def procesar(
        self, paso: int, fotograma: int, relaciones: list[Relacion]
    ) -> list[EventoVial]:
        if paso < 1 or fotograma < 1:
            raise ValueError("paso y fotograma deben ser positivos")
        candidatas: dict[tuple[int, int], Relacion] = {}
        por_moto: dict[int, int] = {}
        for relacion in relaciones:
            moto_id = relacion.moto.track_id
            persona_id = relacion.persona.track_id
            if moto_id is None or persona_id is None:
                continue
            por_moto[moto_id] = por_moto.get(moto_id, 0) + 1
            if relacion.estado == "sin_casco" and relacion.casco is not None:
                clave = (moto_id, persona_id)
                actual = candidatas.get(clave)
                if actual is None or relacion.casco.confianza > actual.casco.confianza:
                    candidatas[clave] = relacion

        siguientes: dict[tuple[int, int], tuple[int, int]] = {}
        eventos: list[EventoVial] = []
        for clave, relacion in sorted(candidatas.items()):
            previo = self._rachas.get(clave)
            cantidad = previo[1] + 1 if previo and previo[0] == paso - 1 else 1
            siguientes[clave] = (paso, cantidad)
            moto_id, persona_id = clave
            if cantidad >= self.observaciones_minimas and moto_id not in self._registradas:
                self._registradas.add(moto_id)
                eventos.append(
                    EventoVial(
                        fotograma,
                        moto_id,
                        persona_id,
                        relacion.casco.confianza,
                        por_moto.get(moto_id, 0),
                    )
                )
        self._rachas = siguientes
        return eventos
