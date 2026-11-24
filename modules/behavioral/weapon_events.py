"""Reglas puras para candidatos de objetos; no determinan delito o intención."""

class ConfirmadorArmas:
    """Espera N fotogramas consecutivos por clase para limitar falsos avisos.

    No asigna objetos a personas ni identifica un arma real sin validación.
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
    if isinstance(nombres, list):
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
    if set(resultado.values()) != {"gun", "knife"}:
        raise ValueError(
            "Este prototipo requiere un modelo especializado con ambas clases: gun y knife. "
            "El modelo COCO común no detecta pistolas."
        )
    return resultado
