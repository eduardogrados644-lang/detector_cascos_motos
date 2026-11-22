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
    cabeza_id: int | None = None


def normalizar_clase(nombre: str) -> str:
    texto = normalize("NFKD", nombre).encode("ascii", "ignore").decode()
    texto = texto.lower().replace("_", " ").replace("-", " ").strip()
    texto = " ".join(texto.split())
    if texto in ("without helmet", "no helmet", "nohelmet", "sin casco"):
        return "sin_casco"
    if texto in ("with helmet", "helmet", "con casco"):
        return "con_casco"
    if texto in ("motorcycle", "motorbike", "motocicleta"):
        return "moto"
    if texto in ("person", "persona"):
        return "persona"
    if texto in ("licence", "license", "license plate", "numberplate", "number plate", "placa", "matricula"):
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
        if not etiquetas:
            relaciones.append(
                Relacion(motos[mi], personas[pi], None, "desconocido")
            )
            continue

        # Una caja COCO de persona puede incluir varios ocupantes superpuestos.
        # Agrupar sólo las etiquetas que parecen describir la misma cabeza.
        # Si hay etiquetas contradictorias para la misma cabeza, no emitir caso.
        grupos: list[list[Objeto]] = []
        for etiqueta in sorted(etiquetas, key=lambda x: x.confianza, reverse=True):
            grupo = next(
                (
                    grupo
                    for grupo in grupos
                    if any(_iou(etiqueta.caja, otro.caja) >= 0.40 for otro in grupo)
                ),
                None,
            )
            if grupo is None:
                grupos.append([etiqueta])
            else:
                grupo.append(etiqueta)
        for grupo in grupos:
            tipos = {normalizar_clase(item.clase) for item in grupo}
            if len(tipos) != 1:
                relaciones.append(
                    Relacion(motos[mi], personas[pi], None, "desconocido")
                )
            else:
                relaciones.append(
                    Relacion(
                        motos[mi],
                        personas[pi],
                        max(grupo, key=lambda x: x.confianza),
                        next(iter(tipos)),
                    )
                )
    return relaciones


def _iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersección de rectángulos / unión, para duplicados y continuidad."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    wa, ha = max(0.0, a[2] - a[0]), max(0.0, a[3] - a[1])
    wb, hb = max(0.0, b[2] - b[0]), max(0.0, b[3] - b[1])
    union = wa * ha + wb * hb - inter
    return inter / union if union > 0 else 0.0


def _continuidad(
    anterior: tuple[float, float, float, float],
    actual: tuple[float, float, float, float],
) -> bool:
    """Rechaza saltos bruscos de caja asociados a reciclaje de track ID.

    Puede reiniciar observaciones auténticas con cámaras móviles. Es preferible
    perder una alerta tentativa a atribuirla a la persona equivocada.
    """
    aw, ah = _dimensiones(anterior)
    bw, bh = _dimensiones(actual)
    if min(aw, ah, bw, bh) <= 0:
        return False
    area_anterior, area_actual = aw * ah, bw * bh
    if min(area_anterior, area_actual) / max(area_anterior, area_actual) < 0.35:
        return False
    if _iou(anterior, actual) >= 0.12:
        return True
    cx_a = (anterior[0] + anterior[2]) / 2
    cy_a = (anterior[1] + anterior[3]) / 2
    cx_b = (actual[0] + actual[2]) / 2
    cy_b = (actual[1] + actual[3]) / 2
    return (
        abs(cx_b - cx_a) <= 0.75 * max(aw, bw)
        and abs(cy_b - cy_a) <= 0.75 * max(ah, bh)
    )


class ConfirmadorEventosViales:
    """Confirma candidatos por cabeza rastreada, no una sola alerta por moto.

    Si hay track ID de cabeza se prefiere a la caja COCO de persona, que puede
    intercambiar IDs al solaparse varios ocupantes. Si el modelo de cabezas
    no entrega ID, se recurre al track de persona; se aplica continuidad de caja.
    Esto NO cuenta automáticamente ocupantes reales.
    """

    def __init__(self, observaciones_minimas: int = 4) -> None:
        if observaciones_minimas < 2:
            raise ValueError("observaciones_minimas debe ser al menos 2")
        self.observaciones_minimas = observaciones_minimas
        self._rachas: dict[
            tuple[int, int], tuple[int, int, tuple[float, float, float, float]]
        ] = {}
        self._registradas: set[tuple[int, int]] = set()

    def procesar(
        self, paso: int, fotograma: int, relaciones: list[Relacion]
    ) -> list[EventoVial]:
        if paso < 1 or fotograma < 1:
            raise ValueError("paso y fotograma deben ser positivos")

        candidatas: dict[tuple[int, int], Relacion] = {}
        por_moto: dict[int, set[int]] = {}
        for relacion in relaciones:
            moto_id = relacion.moto.track_id
            persona_id = relacion.persona.track_id
            if moto_id is None or persona_id is None:
                continue
            por_moto.setdefault(moto_id, set()).add(persona_id)
            if relacion.estado != "sin_casco" or relacion.casco is None:
                continue
            cabeza_id = relacion.casco.track_id
            # IDs negativos sólo en el modo de respaldo por persona.
            clave = (moto_id, cabeza_id if cabeza_id is not None else -persona_id - 1)
            actual = candidatas.get(clave)
            if actual is None or relacion.casco.confianza > actual.casco.confianza:
                candidatas[clave] = relacion

        siguientes: dict[
            tuple[int, int], tuple[int, int, tuple[float, float, float, float]]
        ] = {}
        eventos: list[EventoVial] = []
        for clave, relacion in sorted(candidatas.items()):
            previo = self._rachas.get(clave)
            cabeza_id = relacion.casco.track_id
            caja = relacion.casco.caja if cabeza_id is not None else relacion.persona.caja
            cantidad = (
                previo[1] + 1
                if previo and previo[0] == paso - 1 and _continuidad(previo[2], caja)
                else 1
            )
            siguientes[clave] = (paso, cantidad, caja)
            moto_id = relacion.moto.track_id
            persona_id = relacion.persona.track_id
            if cantidad >= self.observaciones_minimas and clave not in self._registradas:
                self._registradas.add(clave)
                eventos.append(
                    EventoVial(
                        fotograma=fotograma,
                        moto_id=moto_id,
                        persona_id=persona_id,
                        confianza=relacion.casco.confianza,
                        personas_asociadas=len(por_moto.get(moto_id, ())),
                        cabeza_id=cabeza_id,
                    )
                )
        self._rachas = siguientes
        return eventos
