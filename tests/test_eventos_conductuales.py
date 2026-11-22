"""Pruebas de la lógica conductual, ejecutables sin YOLO ni OpenCV."""
import unittest

from modules.behavioral.events import DetectorEventos, Persona, postura_horizontal


def persona(horizontal: bool, track_id: int | None = 7, confianza: float = 0.9) -> Persona:
    puntos = [(0.0, 0.0, 0.0)] * 17
    if horizontal:
        valores = ((20, 50), (28, 52), (70, 52), (78, 54))
        caja = (10.0, 20.0, 110.0, 90.0)
    else:
        valores = ((50, 20), (60, 20), (52, 80), (62, 80))
        caja = (35.0, 5.0, 78.0, 110.0)
    for indice, (x, y) in zip((5, 6, 11, 12), valores):
        puntos[indice] = (float(x), float(y), confianza)
    return Persona(track_id, caja, tuple(puntos))


class PruebasEventos(unittest.TestCase):
    def test_distingue_postura_vertical(self):
        self.assertFalse(postura_horizontal(persona(False)))
        self.assertTrue(postura_horizontal(persona(True)))

    def test_no_usa_puntos_poco_confiables(self):
        self.assertFalse(postura_horizontal(persona(True, confianza=0.2)))

    def test_no_emite_evento_sin_tracking(self):
        detector = DetectorEventos(min_fotogramas=2)
        self.assertEqual(detector.procesar(1, [persona(True, None)]), [])
        self.assertEqual(detector.procesar(2, [persona(True, None)]), [])

    def test_confirma_y_limita_duplicados(self):
        detector = DetectorEventos(min_fotogramas=3, separacion_alertas=5)
        self.assertEqual(detector.procesar(1, [persona(True)]), [])
        self.assertEqual(detector.procesar(2, [persona(True)]), [])
        eventos = detector.procesar(3, [persona(True)])
        self.assertEqual(len(eventos), 1)
        self.assertEqual(eventos[0].tipo, "postura_horizontal")
        self.assertEqual(detector.procesar(4, [persona(True)]), [])
        for fotograma in range(5, 8):
            self.assertEqual(detector.procesar(fotograma, [persona(True)]), [])
        self.assertEqual(len(detector.procesar(8, [persona(True)])), 1)

    def test_interrupcion_rompe_racha(self):
        detector = DetectorEventos(min_fotogramas=2)
        self.assertEqual(detector.procesar(1, [persona(True)]), [])
        self.assertEqual(detector.procesar(2, []), [])
        self.assertEqual(detector.procesar(3, [persona(True)]), [])
        self.assertEqual(len(detector.procesar(4, [persona(True)])), 1)

    def test_postura_vertical_resetea_racha(self):
        detector = DetectorEventos(min_fotogramas=2)
        detector.procesar(1, [persona(True)])
        self.assertEqual(detector.procesar(2, [persona(False)]), [])
        self.assertEqual(detector.procesar(3, [persona(True)]), [])
        self.assertEqual(len(detector.procesar(4, [persona(True)])), 1)


if __name__ == "__main__":
    unittest.main()
