"""Pruebas unitarias del layout legible en videos de baja resolución."""
import unittest
from types import SimpleNamespace

from modules.behavioral.visual_layout import (
    dimensiones_anotadas,
    texto_evento_resumido,
)


class PruebasVisualizacionConductual(unittest.TestCase):
    def test_kth_se_amplia_sin_deformar_proporciones(self):
        self.assertEqual(dimensiones_anotadas(160, 120), (640, 480, 108))

    def test_videos_hd_mantienen_resolucion_original(self):
        self.assertEqual(dimensiones_anotadas(1280, 720), (1280, 720, 108))

    def test_resolucion_320_240_es_legible(self):
        self.assertEqual(dimensiones_anotadas(320, 240), (640, 480, 108))

    def test_dimensiones_invalidas(self):
        for w, h in ((0, 120), (-1, 5), (160, 0)):
            with self.subTest(w=w, h=h), self.assertRaises(ValueError):
                dimensiones_anotadas(w, h)

    def test_brazo_usa_descripcion_observable_y_corta(self):
        alerta = SimpleNamespace(
            track_id=1, brazo="izquierdo",
            tipo="desplazamiento_rapido_aparente_de_brazo",
        )
        texto = texto_evento_resumido(alerta, "brazos")
        self.assertIn("ID 1 izq", texto)
        self.assertIn("aparente", texto)
        self.assertNotIn("agresion", texto)
        self.assertLess(len(texto), 70)

    def test_modos_separados(self):
        p = SimpleNamespace(track_ids=(2, 5))
        self.assertIn("2-5", texto_evento_resumido(p, "proximidad"))
        self.assertIn(
            "postura", texto_evento_resumido(
                SimpleNamespace(track_id=2), "posturas"
            )
        )
        with self.assertRaises(ValueError):
            texto_evento_resumido(None, "golpes")


if __name__ == "__main__":
    unittest.main()
