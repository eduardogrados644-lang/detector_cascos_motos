"""Memoria visual de cajas; NO convierte ausencias en detecciones reales."""
from dataclasses import dataclass
from math import isfinite

@dataclass(frozen=True)
class CajaVisual:
    id: int | None
    xyxy: tuple[float, float, float, float]
    confianza: float = 0.0
    ausente: int = 0

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    cruce = max(0., x2-x1) * max(0., y2-y1)
    aa = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    bb = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    return cruce / (aa+bb-cruce) if aa+bb-cruce > 0 else 0.

class MemoriaCajas:
    """Devuelve detecciones reales y últimas cajas conocidas durante N frames.

    Cajas ausentes son SOLO una referencia visual; nunca eventos del detector.
    """
    def __init__(self, conservar=6, umbral_duplicado=0.65):
        if type(conservar) is not int or not 0 <= conservar <= 60:
            raise ValueError("conservar debe ser entero entre 0 y 60")
        if not 0 <= umbral_duplicado <= 1:
            raise ValueError("umbral_duplicado inválido")
        self.conservar = conservar
        self.umbral_duplicado = umbral_duplicado
        self._ultimo = {}
        self._fotograma = 0

    def procesar(self, fotograma: int, reales: list[CajaVisual]):
        if type(fotograma) is not int or fotograma <= self._fotograma:
            raise ValueError("fotograma debe avanzar")
        self._fotograma = fotograma
        ids = set()
        for x in reales:
            a = x.xyxy
            if (len(a) != 4 or not all(isfinite(v) for v in a)
                    or a[2] <= a[0] or a[3] <= a[1] or x.ausente):
                raise ValueError("Caja real inválida")
            if x.id is not None:
                if x.id < 0 or x.id in ids:
                    raise ValueError("ID duplicado o inválido")
                ids.add(x.id)
                self._ultimo[x.id] = (fotograma, a, x.confianza)
        salida = list(reales)
        for ident, (visto, caja, conf) in list(self._ultimo.items()):
            if ident in ids:
                continue
            edad = fotograma - visto
            if edad > self.conservar:
                del self._ultimo[ident]
                continue
            # Si otro ID tomó la misma región, evitamos un duplicado ficticio.
            if any(iou(caja, x.xyxy) >= self.umbral_duplicado for x in reales):
                continue
            salida.append(CajaVisual(ident, caja, conf, ausente=edad))
        return salida
