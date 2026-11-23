"""Validaciones puras de cobertura, sin OpenCV ni pesos YOLO."""
import unittest

from modules.traffic.coverage import resumen_cobertura


class TestCoberturaVial(unittest.TestCase):
    def test_video_con_casco_sin_motocicletas_no_evaluable(self):
        r = resumen_cobertura(1230, {
            "personas": 1230, "motos": 0, "relaciones": 0,
            "deteccion_con_casco": 1201, "deteccion_sin_casco": 0,
            "candidatos": 0,
        })
        self.assertEqual(r["estado"], "no_evaluable_sin_motocicletas")
        self.assertEqual(r["detecciones_con_casco"], 1201)
        self.assertFalse(r["resultado_es_infraccion_confirmada"])

    def test_video_sin_casco_sin_motocicletas_tampoco_valida(self):
        r = resumen_cobertura(1211, {
            "personas": 1211, "motos": 0, "relaciones": 0,
            "deteccion_con_casco": 4, "deteccion_sin_casco": 1208,
            "candidatos": 0,
        })
        self.assertEqual(r["estado"], "no_evaluable_sin_motocicletas")
        self.assertIn("NO significa", r["detalle"])
        self.assertEqual(r["detecciones_sin_casco"], 1208)

    def test_motos_sin_asociaciones_no_evaluable(self):
        r = resumen_cobertura(90, {
            "personas": 62, "motos": 16, "relaciones": 0,
            "deteccion_sin_casco": 9,
        })
        self.assertEqual(r["estado"], "no_evaluable_sin_asociaciones")

    def test_con_asociaciones_solo_experimental(self):
        r = resumen_cobertura(92, {
            "personas": 130, "motos": 73, "relaciones": 81,
            "deteccion_sin_casco": 30, "candidatos": 1,
        })
        self.assertEqual(r["estado"], "asociaciones_experimentales_para_revision")
        self.assertFalse(r["resultado_es_infraccion_confirmada"])

    def test_estado_cobertura_registra_incertidumbre_temporal(self):
        r = resumen_cobertura(40, {
            "motos": 30, "personas": 25, "relaciones": 10,
            "estados_inciertos": 3, "contradicciones_temporales": 1,
        })
        self.assertEqual(r["observaciones_temporales_inciertas"], 3)
        self.assertEqual(r["contradicciones_temporales"], 1)
        self.assertFalse(r["resultado_es_infraccion_confirmada"])

    def test_cero_fotogramas(self):
        r = resumen_cobertura(0, {})
        self.assertEqual(r["estado"], "no_evaluable_sin_fotogramas")

    def test_rechaza_conteos_invalidos(self):
        with self.assertRaises(ValueError):
            resumen_cobertura(-1, {})
        with self.assertRaises(ValueError):
            resumen_cobertura(10, {"motos": -2})


if __name__ == "__main__":
    unittest.main()
