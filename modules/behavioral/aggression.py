"""Candidatos visuales de gestos dirigidos entre personas, NO agresiones comprobadas.

Combina proximidad y desplazamientos de muñeca compensados por el hombro en
fotogramas consecutivos. Una mano alzada o dos personas cerca NO bastan.
No se deduce intención ni contacto físico real a partir de una imagen.
"""
from dataclasses import dataclass
from math import hypot, isfinite

from .arm_motion import BRAZOS, observar_brazos, desplazamiento_plausible, _caja_continua
from .events import Persona
from .proximity import estan_cerca


@dataclass(frozen=True)
class EventoGestoDirigido:
    fotograma: int
    track_id: int
    objetivo_id: int
    tipo: str  # posible_intento_agresion | posible_agresion
    brazo: str

    @property
    def track_ids(self) -> tuple[int, int]:
        return self.track_id, self.objetivo_id


def _direccion_objetivo(persona: Persona, objetivo: Persona, brazo: str):
    """Dirección del hombro al tronco ajeno en coordenadas absolutas."""
    indice_hombro = BRAZOS[brazo][0]
    if len(persona.puntos) <= indice_hombro:
        return None
    hombro = persona.puntos[indice_hombro]
    if len(hombro) != 3 or hombro[2] < 0.55:
        return None
    ax1, ay1, ax2, ay2 = objetivo.caja
    ancho, alto = ax2 - ax1, ay2 - ay1
    if ancho <= 0 or alto <= 0:
        return None
    dx = (ax1 + ax2) * 0.5 - hombro[0]
    dy = ay1 + 0.35 * alto - hombro[1]
    norma = hypot(dx, dy)
    if norma < 0.15 * (persona.caja[3] - persona.caja[1]):
        return None
    return (dx / norma, dy / norma)


def _mano_en_torso(persona: Persona, objetivo: Persona, brazo: str) -> bool:
    """La muñeca se proyecta dentro del torso ajeno (NO prueba contacto)."""
    indice = BRAZOS[brazo][2]
    if len(persona.puntos) <= indice:
        return False
    muneca = persona.puntos[indice]
    if len(muneca) != 3 or muneca[2] < 0.55:
        return False
    x1, y1, x2, y2 = objetivo.caja
    ancho, alto = x2 - x1, y2 - y1
    return (x1 + 0.08 * ancho <= muneca[0] <= x2 - 0.08 * ancho
            and y1 + 0.10 * alto <= muneca[1] <= y1 + 0.72 * alto)



def _punto(persona, indice, confianza_minima=0.55):
    if len(persona.puntos) <= indice:
        return None
    punto = persona.puntos[indice]
    if (len(punto) != 3 or punto[2] < confianza_minima
            or not all(isfinite(valor) for valor in punto)):
        return None
    return punto[:2]


def _movimiento_relativo(anterior, actual, brazo, dt):
    """Movimiento de muñeca con desplazamiento del hombro compensado.

    Ayuda a no confundir traslación del cuerpo/cámara con un gesto de brazo.
    """
    if dt <= 0 or not isfinite(dt):
        return None
    hombro_id, _, mano_id, cadera_id = BRAZOS[brazo]
    hp, mp = _punto(anterior, hombro_id), _punto(anterior, mano_id)
    ha, ma = _punto(actual, hombro_id), _punto(actual, mano_id)
    cp, ca = _punto(anterior, cadera_id), _punto(actual, cadera_id)
    if any(p is None for p in (hp, mp, ha, ma, cp, ca)):
        return None
    torso_previo = hypot(cp[0] - hp[0], cp[1] - hp[1])
    torso_actual = hypot(ca[0] - ha[0], ca[1] - ha[1])
    if torso_previo <= 0 or not 0.55 <= torso_actual / torso_previo <= 1.8:
        return None
    dx = (ma[0] - mp[0]) - (ha[0] - hp[0])
    dy = (ma[1] - mp[1]) - (ha[1] - hp[1])
    magnitud = hypot(dx, dy)
    escala = (torso_previo + torso_actual) * 0.5
    if magnitud < 0.12 * escala:
        return None
    return (dx / magnitud, dy / magnitud), magnitud / (escala * dt)


class DetectorGestosDirigidos:
    """Heurística temporal interpretable, no clasificador de violencia.

    Exige proximidad sostenida, movimiento plausible de muñeca HACIA
    otra persona y varias observaciones consecutivas. Se etiqueta como
    posible agresión solo si la muñeca se superpone al torso ajeno; esto
    NO demuestra contacto real. Una mano alzada sola no alerta.
    """

    def __init__(self, min_cercania=3, min_gestos=2, min_superposicion=2,
                 velocidad_minima=1.2, separacion_alertas=90,
                 pausa_maxima_s=1.0):
        if (type(min_cercania) is not int or min_cercania < 2
                or type(min_gestos) is not int or min_gestos < 2
                or type(min_superposicion) is not int or min_superposicion < 1
                or type(separacion_alertas) is not int or separacion_alertas < 1 or not isfinite(velocidad_minima)
                or velocidad_minima <= 0 or not isfinite(pausa_maxima_s)
                or pausa_maxima_s <= 0):
            raise ValueError("Configuracion de gestos invalida")
        self.min_cercania = min_cercania
        self.min_gestos = min_gestos
        self.min_superposicion = min_superposicion
        self.velocidad_minima = velocidad_minima
        self.separacion_alertas = separacion_alertas
        self.pausa_maxima_s = pausa_maxima_s
        self._anteriores = {}
        self._cercania = {}
        self._movimientos = {}
        self._ultima_alerta = {}
        self._ultimo_fotograma = 0

    def procesar(self, fotograma: int, personas: list[Persona], tiempo_s: float):
        if (type(fotograma) is not int or fotograma <= self._ultimo_fotograma
                or not isfinite(tiempo_s) or tiempo_s < 0):
            raise ValueError("Fotogramas o tiempos invalidos")
        self._ultimo_fotograma = fotograma
        por_id = {}
        for p in personas:
            if p.track_id is not None:
                por_id.setdefault(p.track_id, p)
        observaciones = {idx: observar_brazos(p) for idx, p in por_id.items()}
        siguiente_cercania = {}
        siguiente_mov = {}
        eventos = []
        for a in sorted(por_id):
            for b in sorted(por_id):
                if a == b or not estan_cerca(por_id[a], por_id[b]):
                    continue
                par = tuple(sorted((a, b)))
                anterior = self._cercania.get(par)
                continuidad_pareja = (
                    anterior is not None and anterior[1] == fotograma - 1
                    and _caja_continua(anterior[2], por_id[par[0]].caja)
                    and _caja_continua(anterior[3], por_id[par[1]].caja)
                )
                cerca = 1 + (anterior[0] if continuidad_pareja else 0)
                siguiente_cercania[par] = (
                    cerca, fotograma, por_id[par[0]].caja, por_id[par[1]].caja
                )
                if cerca < self.min_cercania:
                    continue
                persona = por_id[a]
                objetivo = por_id[b]
                for brazo, obs in observaciones[a].items():
                    anterior_brazo = self._anteriores.get((a, brazo))
                    if anterior_brazo is None:
                        continue
                    fprev, tprev, persona_previa, obs_prev = anterior_brazo
                    dt = tiempo_s - tprev
                    if (fprev != fotograma - 1 or not 0 < dt <= self.pausa_maxima_s
                            or not _caja_continua(persona_previa.caja, persona.caja)
                            or not desplazamiento_plausible(obs_prev, obs, dt)):
                        continue
                    direccion = _direccion_objetivo(persona, objetivo, brazo)
                    movimiento = _movimiento_relativo(persona_previa, persona, brazo, dt)
                    if direccion is None or movimiento is None:
                        continue
                    orientacion, velocidad = movimiento
                    hacia = (orientacion[0] * direccion[0]
                             + orientacion[1] * direccion[1])
                    if velocidad < self.velocidad_minima or hacia < 0.70:
                        continue
                    clave = (a, b, brazo)
                    previo = self._movimientos.get(clave)
                    pasos = 1 + (previo[0] if previo and previo[2] == fotograma - 1 else 0)
                    superpuestos = (1 + (previo[1] if previo and previo[2] == fotograma - 1 else 0)
                                    if _mano_en_torso(persona, objetivo, brazo) else 0)
                    siguiente_mov[clave] = (pasos, superpuestos, fotograma)
                    if pasos < self.min_gestos:
                        continue
                    tipo = ("posible_agresion" if superpuestos >= self.min_superposicion
                            else "posible_intento_agresion")
                    clave_alerta = (a, b, tipo)
                    if fotograma - self._ultima_alerta.get(clave_alerta, -10**9) >= self.separacion_alertas:
                        eventos.append(EventoGestoDirigido(fotograma, a, b, tipo, brazo))
                        self._ultima_alerta[clave_alerta] = fotograma

        self._cercania = siguiente_cercania
        self._movimientos = siguiente_mov
        self._anteriores = {
            (idx, brazo): (fotograma, tiempo_s, por_id[idx], obs)
            for idx, obs_por_brazo in observaciones.items()
            for brazo, obs in obs_por_brazo.items()
        }
        for k, ultima in list(self._ultima_alerta.items()):
            if fotograma - ultima > max(600, self.separacion_alertas * 2):
                del self._ultima_alerta[k]
        return eventos
