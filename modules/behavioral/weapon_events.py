"""Reglas de candidatos de objetos con persistencia espacial.

La misma clase apareciendo en lugares diferentes no crea una racha.
No determina posesion, amenazas, delitos ni intencion.
"""
from dataclasses import dataclass
from math import hypot, isfinite


@dataclass(frozen=True)
class CandidatoArma:
    fotograma: int
    objeto_id: int
    clase: str
    confianza: float
    caja: tuple[float, float, float, float]


def solape(a, b):
    """Interseccion sobre union (IoU) entre cajas."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2-x1) * max(0., y2-y1)
    area_a = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    area_b = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.


def cajas_compatibles(a, b, min_iou=0.25, max_centro_rel=0.35):
    """Admite variaciones moderadas sin unir objetos que saltan de posicion."""
    aw, ah = a[2]-a[0], a[3]-a[1]
    bw, bh = b[2]-b[0], b[3]-b[1]
    if min(aw, ah, bw, bh) <= 0:
        return False
    ratio = aw*ah/(bw*bh)
    if not 0.4 <= ratio <= 2.5:
        return False
    if solape(a, b) >= min_iou:
        return True
    dx = (a[0]+a[2]-b[0]-b[2]) / 2
    dy = (a[1]+a[3]-b[1]-b[3]) / 2
    return hypot(dx, dy) <= max_centro_rel*max(hypot(aw, ah), hypot(bw, bh))


class ConfirmadorArmas:
    """Confirma una caja en fotogramas consecutivos con procesar_objetos().

    El metodo antiguo procesar() permanece solo por compatibilidad; no
    debe usarse para alertar en videos porque no conoce las posiciones.
    """

    def __init__(self, consecutivos=3, enfriamiento=45):
        if (type(consecutivos) is not int or consecutivos < 1
                or type(enfriamiento) is not int or enfriamiento < 1):
            raise ValueError("Parámetros incorrectos")
        self.consecutivos = consecutivos
        self.enfriamiento = enfriamiento
        self.rachas = {}
        self.ultima_alerta = {}
        self.ultimo_fotograma = 0
        self._pistas = {}
        self._siguiente_id = 1


    def procesar_objetos(self, fotograma, objetos):
        """Recibe (clase, confianza, caja) y confirma la misma caja en N frames.

        Emparejamiento exclusivo por clase y cercania; ninguna racha de otra
        region contribuye. No asocia el objeto con una persona.
        """
        if type(fotograma) is not int or fotograma <= self.ultimo_fotograma:
            raise ValueError("El fotograma debe avanzar")
        self.ultimo_fotograma = fotograma
        datos = []
        for clase, confianza, caja in objetos:
            if clase not in ("gun", "knife"):
                raise ValueError("Solo clases gun/knife")
            if (not isfinite(confianza) or not 0 < confianza <= 1
                    or len(caja) != 4
                    or any(not isfinite(v) for v in caja)
                    or caja[0] >= caja[2] or caja[1] >= caja[3]):
                raise ValueError("Deteccion invalida")
            datos.append((clase, float(confianza), tuple(float(v) for v in caja)))
        nuevos = {}
        usados = set()
        candidatos = []
        for clase, confianza, caja in sorted(datos, key=lambda x: x[1], reverse=True):
            posibles = [
                (solape(caja, pista["caja"]), indice)
                for indice, pista in self._pistas.items()
                if indice not in usados and pista["clase"] == clase
                and pista["fotograma"] == fotograma - 1
                and cajas_compatibles(caja, pista["caja"])
            ]
            if posibles:
                _, objeto_id = max(posibles)
                anterior = self._pistas[objeto_id]
                racha = anterior["racha"] + 1
                ultima = anterior["ultima"]
            else:
                objeto_id = self._siguiente_id
                self._siguiente_id += 1
                racha, ultima = 1, -10**9
            usados.add(objeto_id)
            nuevos[objeto_id] = dict(
                clase=clase, confianza=confianza, caja=caja,
                fotograma=fotograma, racha=racha, ultima=ultima,
            )
            if racha >= self.consecutivos and fotograma - ultima >= self.enfriamiento:
                nuevos[objeto_id]["ultima"] = fotograma
                candidatos.append(CandidatoArma(
                    fotograma, objeto_id, clase, confianza, caja))
        self._pistas = nuevos
        return candidatos

    def procesar(self, fotograma, clases):
        if type(fotograma) is not int or fotograma <= self.ultimo_fotograma:
            raise ValueError("El fotograma debe avanzar")
        self.ultimo_fotograma = fotograma
        presentes = set(clases)
        if not presentes.issubset({"gun", "knife"}):
            raise ValueError("Solo clases gun/knife")
        for clase in ("gun", "knife"):
            self.rachas[clase] = self.rachas.get(clase, 0) + 1 if clase in presentes else 0
        avisos = []
        for clase in sorted(presentes):
            if (self.rachas[clase] >= self.consecutivos
                    and fotograma - self.ultima_alerta.get(clase, -(10**9)) >= self.enfriamiento):
                self.ultima_alerta[clase] = fotograma
                avisos.append(clase)
        return avisos


def nombres_armas(nombres):
    """Devuelve id->clase para un modelo Detect con etiquetas gun y knife."""
    if isinstance(nombres, (list, tuple)):
        nombres = dict(enumerate(nombres))
    if not isinstance(nombres, dict):
        raise ValueError("Las clases del modelo deben ser dict o lista")
    resultado = {}
    for indice, nombre in nombres.items():
        if not isinstance(nombre, str):
            continue
        normalizado = nombre.strip().lower()
        if normalizado in ("gun", "knife"):
            resultado[int(indice)] = normalizado
    if not resultado:
        raise ValueError(
            "Se requiere un modelo especializado con clase gun y/o knife. "
            "El modelo COCO común no detecta pistolas."
        )
    return resultado
