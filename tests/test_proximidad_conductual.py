"""Regresión del análisis experimental de proximidad, sin YOLO/OpenCV."""
import unittest

from modules.behavioral.events import Persona
from modules.behavioral.proximity import DetectorProximidad, estan_cerca


def persona(ident, caja):
    return Persona(ident, caja, ())


A = persona(3, (10, 10, 50, 110))
B = persona(9, (58, 12, 98, 112))
LEJANA = persona(9, (180, 12, 220, 112))


class PruebasProximidad(unittest.TestCase):
    def test_proximidad_independiente_de_pixeles_absolutos(self):
        self.assertTrue(estan_cerca(A, B))
        a = persona(3, tuple(v * 4 for v in A.caja))
        b = persona(9, tuple(v * 4 for v in B.caja))
        self.assertTrue(estan_cerca(a, b))
        self.assertFalse(estan_cerca(A, LEJANA))

    def test_no_confunde_verticalmente_separados(self):
        self.assertFalse(estan_cerca(A, persona(9, (24, 140, 64, 230))))

    def test_no_confunde_escalas_muy_distintas(self):
        self.assertFalse(estan_cerca(A, persona(9, (40, 10, 45, 30))))

    def test_no_usa_cajas_invalidas(self):
        self.assertFalse(estan_cerca(persona(3, (10, 10, 10, 110)), B))

    def test_requiere_varios_fotogramas(self):
        d = DetectorProximidad(min_fotogramas=3)
        self.assertEqual(d.procesar(1, [A, B]), [])
        self.assertEqual(d.procesar(2, [A, B]), [])
        eventos = d.procesar(3, [A, B])
        self.assertEqual(len(eventos), 1)
        self.assertEqual(eventos[0].track_ids, (3, 9))
        self.assertEqual(eventos[0].tipo, "proximidad_persistente")

    def test_cercania_no_se_llama_violencia(self):
        d = DetectorProximidad(min_fotogramas=2)
        d.procesar(1, [A, B])
        self.assertEqual(d.procesar(2, [A, B])[0].tipo, "proximidad_persistente")

    def test_separacion_interrumpe_la_racha(self):
        d = DetectorProximidad(min_fotogramas=2)
        d.procesar(1, [A, B])
        d.procesar(2, [A, LEJANA])
        self.assertEqual(d.procesar(3, [A, B]), [])
        self.assertEqual(len(d.procesar(4, [A, B])), 1)

    def test_fotograma_omitido_interrumpe(self):
        d = DetectorProximidad(min_fotogramas=2)
        d.procesar(1, [A, B])
        self.assertEqual(d.procesar(3, [A, B]), [])
        self.assertEqual(len(d.procesar(4, [A, B])), 1)

    def test_sin_ids_no_genera_eventos(self):
        d = DetectorProximidad(min_fotogramas=2)
        a = persona(None, A.caja)
        for n in range(1, 5):
            self.assertEqual(d.procesar(n, [a, B]), [])

    def test_parejas_independientes_no_se_unen(self):
        d = DetectorProximidad(min_fotogramas=2)
        c = persona(12, (106, 15, 146, 115))
        d.procesar(1, [A, B, c])
        eventos = d.procesar(2, [A, B, c])
        self.assertEqual({e.track_ids for e in eventos}, {(3, 9), (9, 12)})

    def test_dedup_por_pareja(self):
        d = DetectorProximidad(min_fotogramas=2, separacion_alertas=5)
        d.procesar(1, [A, B])
        self.assertEqual(len(d.procesar(2, [A, B])), 1)
        for n in (3, 4, 5, 6):
            self.assertEqual(d.procesar(n, [A, B]), [])
        self.assertEqual(len(d.procesar(7, [A, B])), 1)

    def test_cambio_track_id_no_continua_racha(self):
        d = DetectorProximidad(min_fotogramas=2)
        d.procesar(1, [A, B])
        otra = persona(15, B.caja)
        self.assertEqual(d.procesar(2, [A, otra]), [])

    def test_parametros_invalidos(self):
        for kw in (
            {"min_fotogramas": 1}, {"separacion_alertas": 0},
            {"margen_horizontal": -1}, {"solape_vertical": 1.1},
        ):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                DetectorProximidad(**kw)
        with self.assertRaises(ValueError):
            DetectorProximidad().procesar(0, [])


if __name__ == "__main__":
    unittest.main()
