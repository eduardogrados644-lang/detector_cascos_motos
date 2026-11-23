"""Pruebas de selección de video para cámaras EXTERNAS de ambos apartados."""
import unittest
from unittest.mock import patch

import iniciar_cascos
import iniciar_intencion
from modules.camera_sources import (
    elegir_fuente_menu,
    indice_usb_externo,
    normalizar_wifi,
    seleccionar_fuente,
    validar_fuente,
)


class TestFuentesExternas(unittest.TestCase):
    def test_indice_usb_requiere_eleccion_explicita(self):
        self.assertEqual(indice_usb_externo(1), "camara:1")
        self.assertEqual(indice_usb_externo(3), "camara:3")
        for indice in (0, -1, 21, True):
            with self.subTest(indice=indice), self.assertRaises(ValueError):
                indice_usb_externo(indice)

    def test_nunca_se_elije_camara_integrada_en_iniciador(self):
        for valor in ("0", "camara:0", "camara:-1", ""):
            with self.subTest(valor=valor), self.assertRaises(ValueError):
                validar_fuente(valor)
        with self.assertRaises(ValueError):
            iniciar_cascos.construir_comando("camara:0")
        with self.assertRaises(ValueError):
            iniciar_intencion.construir_comando_demo("0")

    def test_url_ip_webcam_normaliza_ruta(self):
        self.assertEqual(
            normalizar_wifi("http://192.168.1.50:8080"),
            "http://192.168.1.50:8080/video",
        )
        self.assertEqual(
            normalizar_wifi("http://192.168.1.50:8080/video"),
            "http://192.168.1.50:8080/video",
        )
        self.assertEqual(
            normalizar_wifi("https://camara.local:8080/mjpegfeed"),
            "https://camara.local:8080/mjpegfeed",
        )
        for valor in ("192.168.1.50", "ftp://ejemplo.com", "http://", ""):
            with self.subTest(valor=valor), self.assertRaises(ValueError):
                normalizar_wifi(valor)

    def test_menu_vuelve_a_preguntar_si_se_pulsa_enter_vacio(self):
        with patch("builtins.input", side_effect=["", "1"]):
            self.assertEqual(elegir_fuente_menu(), "prueba_motos.mp4")

    def test_menu_vuelve_a_preguntar_ante_opcion_desconocida(self):
        with patch("builtins.input", side_effect=["9", "0"]):
            self.assertIsNone(elegir_fuente_menu())

    def test_menu_solo_usb_externa(self):
        with patch("builtins.input", side_effect=["2", "1"]):
            self.assertEqual(elegir_fuente_menu(), "camara:1")

    def test_menu_celular_tablet_android(self):
        with patch("builtins.input", side_effect=["3", "http://192.168.1.50:8080"]):
            self.assertEqual(
                elegir_fuente_menu(), "http://192.168.1.50:8080/video"
            )

    def test_menu_video_local_y_salir(self):
        with patch("builtins.input", return_value="1"):
            self.assertEqual(elegir_fuente_menu(), "prueba_motos.mp4")
        with patch("builtins.input", return_value="0"):
            self.assertIsNone(elegir_fuente_menu())

    def test_ambos_iniciadores_respetan_usb_y_wifi(self):
        with patch("iniciar_cascos.subprocess.run") as mock:
            mock.return_value.returncode = 0
            self.assertEqual(
                iniciar_cascos.main(["--usb", "1", "--sin-mostrar"]), 0
            )
            self.assertEqual(
                mock.call_args.args[0][
                    mock.call_args.args[0].index("--fuente") + 1
                ], "camara:1"
            )
        with patch("iniciar_intencion.subprocess.run") as mock:
            mock.return_value.returncode = 0
            self.assertEqual(
                iniciar_intencion.main(
                    [
                        "--demo-posturas", "--wifi",
                        "http://192.168.1.50:8080", "--sin-mostrar",
                    ]
                ), 0
            )
            self.assertEqual(
                mock.call_args.args[0][
                    mock.call_args.args[0].index("--fuente") + 1
                ], "http://192.168.1.50:8080/video"
            )

    def test_sin_parametros_pregunta_fuente_no_asume_interna(self):
        with patch("modules.camera_sources.elegir_fuente_menu", return_value="camara:1"):
            self.assertEqual(seleccionar_fuente(), "camara:1")
        with self.assertRaises(ValueError):
            seleccionar_fuente(fuente="video.mp4", usb=1)

    def test_conductual_sin_fuente_no_enchufa_camara_0(self):
        with patch("modules.camera_sources.elegir_fuente_menu", return_value=None):
            with patch("iniciar_intencion.subprocess.run") as mock:
                resultado = iniciar_intencion.main(["--demo-posturas"])
                self.assertEqual(resultado, 0)
                mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
