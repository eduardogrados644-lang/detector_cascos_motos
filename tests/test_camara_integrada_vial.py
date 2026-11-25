"""Pruebas exclusivas del acceso a la camara integrada en CASCOS.

No requieren hardware real ni modelos YOLO y no abren webcams.
"""
import sys
import types
import unittest
from unittest.mock import Mock, patch

import iniciar_cascos
import iniciar_intencion
from modules.camera_sources import (
    abrir_captura_vial, elegir_fuente_menu_vial, seleccionar_fuente_vial,
    validar_fuente, validar_fuente_vial,
)


class TestCamaraIntegradaVial(unittest.TestCase):
    def test_menu_integrada_elegida_manual(self):
        with patch("builtins.input", side_effect=["1"]):
            self.assertEqual(elegir_fuente_menu_vial(), "camara:0")

    def test_menu_no_salta_a_integrada(self):
        with patch("builtins.input", side_effect=["4"]):
            self.assertEqual(elegir_fuente_menu_vial(), "prueba_motos.mp4")
        with patch("builtins.input", side_effect=["0"]):
            self.assertIsNone(elegir_fuente_menu_vial())

    def test_menu_externa_celular_archivo(self):
        for respuestas, esperado in (
            (["2", "1"], "camara:1"),
            (["3", "http://192.168.1.50:8080"], "http://192.168.1.50:8080/video"),
            (["5", "video.mp4"], "video.mp4"),
        ):
            with self.subTest(respuestas=respuestas), patch("builtins.input", side_effect=respuestas):
                self.assertEqual(elegir_fuente_menu_vial(), esperado)

    def test_cli_integrada_abre_road_monitor_sin_cascos_conductual(self):
        with patch("iniciar_cascos.subprocess.run") as run:
            run.return_value.returncode = 0
            rc = iniciar_cascos.main(["--integrada", "--sin-mostrar"])
        self.assertEqual(rc, 0)
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--fuente")+1], "camara:0")
        self.assertIn("modules.traffic.road_monitor", cmd)
        self.assertNotIn("modules.behavioral", " ".join(cmd))

    def test_cli_integrada_modo_solo_cascos(self):
        with patch("iniciar_cascos.subprocess.run") as run:
            run.return_value.returncode = 0
            rc = iniciar_cascos.main(["--integrada", "--modo", "cascos", "--sin-mostrar"])
        self.assertEqual(rc, 0)
        cmd=run.call_args.args[0]
        self.assertIn("modules.traffic.helmet_monitor", cmd)
        self.assertIn("camara:0", cmd)

    def test_integrada_nunca_se_usa_por_defecto(self):
        with patch("modules.camera_sources.elegir_fuente_menu_vial",return_value=None):
            with patch("iniciar_cascos.subprocess.run") as run:
                self.assertEqual(iniciar_cascos.main(["--sin-mostrar"]),0)
                run.assert_not_called()

    def test_coexisten_integrada_y_externa_pero_no_simultaneas(self):
        self.assertEqual(seleccionar_fuente_vial(integrada=True), "camara:0")
        self.assertEqual(seleccionar_fuente_vial(usb=1), "camara:1")
        self.assertEqual(seleccionar_fuente_vial(fuente="camara:0"), "camara:0")
        self.assertEqual(seleccionar_fuente_vial(fuente="camara:1"), "camara:1")
        with self.assertRaises(ValueError):
            seleccionar_fuente_vial(usb=1, integrada=True)

    def test_conductual_seguira_rechazando_camara_cero(self):
        for v in ("0", "camara:0"):
            with self.assertRaises(ValueError):
                validar_fuente(v)
            with self.assertRaises(ValueError):
                iniciar_intencion.construir_comando_demo(v)
        self.assertEqual(validar_fuente_vial("camara:0"), "camara:0")

    def test_apertura_pc_exclusiva_vial_sin_dispositivos_reales(self):
        mock = Mock(return_value="captura-simulada")
        mod = types.SimpleNamespace(abrir_usb=mock)
        with patch.dict(sys.modules, {"camara_zoom": mod}):
            self.assertEqual(abrir_captura_vial(0), "captura-simulada")
        mock.assert_called_once_with(0, ancho=640, alto=480, fps=30)


if __name__ == "__main__":
    unittest.main()
