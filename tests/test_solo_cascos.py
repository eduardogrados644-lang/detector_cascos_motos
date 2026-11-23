"""Pruebas del modo de cascos independientes de cámara y modelos pesados."""
import sys
import unittest

import iniciar_cascos
from modules.traffic.helmet_summary import resumen_cascos


class TestSoloCascos(unittest.TestCase):
    def test_videos_p16_con_casco_sin_motos(self):
        r = resumen_cascos(1230, {"con_casco": 1201, "sin_casco": 0})
        self.assertEqual(r["estado"], "detecciones_de_casco_sin_asociacion_vial")
        self.assertIsNone(r["motocicletas_analizadas"])
        self.assertFalse(r["infraccion_confirmada"])

    def test_videos_p16_sin_casco_sin_motos(self):
        r = resumen_cascos(1211, {"con_casco": 4, "sin_casco": 1208})
        self.assertEqual(r["apariciones_sin_casco"], 1208)
        self.assertFalse(r["infraccion_confirmada"])

    def test_sin_detecciones_es_un_estado_explicito(self):
        self.assertEqual(resumen_cascos(25, {})["estado"], "sin_detecciones")
        self.assertEqual(resumen_cascos(0, {})["estado"], "sin_fotogramas")

    def test_cuentas_negativas_son_invalidas(self):
        with self.assertRaises(ValueError):
            resumen_cascos(20, {"sin_casco": -1})
        with self.assertRaises(ValueError):
            resumen_cascos(-1, {})

    def test_iniciador_no_invoca_detector_de_motos_en_modo_cascos(self):
        cmd = iniciar_cascos.construir_comando(
            "videos_prueba/With Helmet/p16.mp4",
            modo="cascos", guardar_video=True, mostrar=False,
        )
        self.assertEqual(cmd[:3], [sys.executable, "-m", "modules.traffic.helmet_monitor"])
        self.assertNotIn("--modelo-vehiculos", cmd)
        self.assertNotIn("modules.behavioral", " ".join(cmd))
        self.assertIn("--guardar-video", cmd)

    def test_iniciador_vial_se_mantiene_sin_cambios_de_modo(self):
        cmd = iniciar_cascos.construir_comando("prueba_motos.mp4")
        self.assertEqual(cmd[2], "modules.traffic.road_monitor")

    def test_rechaza_modo_desconocido(self):
        with self.assertRaises(ValueError):
            iniciar_cascos.construir_comando("prueba_motos.mp4", modo="invalid")
