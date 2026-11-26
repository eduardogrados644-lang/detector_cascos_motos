"""Regresiones de iluminación y alertas sin usar cámara ni inferencia."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from modules.behavioral.lighting import mejorar_iluminacion
from modules.behavioral.alert_sound import reproducir_aviso
import iniciar_intencion


class PruebasMejorasConductuales(unittest.TestCase):
    def test_normalizacion_preserva_formato_y_geometria(self):
        imagen = np.full((120, 160, 3), 30, dtype=np.uint8)
        mejorada, aplicada = mejorar_iluminacion(imagen)
        self.assertTrue(aplicada)
        self.assertEqual(mejorada.shape, imagen.shape)
        self.assertEqual(mejorada.dtype, imagen.dtype)
        self.assertGreaterEqual(float(mejorada.mean()), float(imagen.mean()))

    def test_no_modifica_escena_clara_ni_modo_desactivado(self):
        imagen = np.full((120, 160, 3), 200, dtype=np.uint8)
        salida, aplicada = mejorar_iluminacion(imagen)
        self.assertIs(salida, imagen)
        self.assertFalse(aplicada)
        oscura = np.full((100, 100, 3), 20, dtype=np.uint8)
        salida, aplicada = mejorar_iluminacion(oscura, habilitado=False)
        self.assertIs(salida, oscura)
        self.assertFalse(aplicada)

    def test_sonido_no_es_necesario_fuera_de_windows(self):
        with patch.dict("sys.modules", {"winsound": None}):
            self.assertFalse(reproducir_aviso())

    def test_agresiones_suena_por_defecto_en_iniciador(self):
        with patch("iniciar_intencion.subprocess.run") as ejecutar:
            ejecutar.return_value.returncode = 0
            iniciar_intencion.main(["--demo-agresiones", "--usb", "1", "--sin-mostrar"])
        self.assertIn("--alerta-sonora", ejecutar.call_args.args[0])

    def test_sonido_e_iluminacion_se_pueden_desactivar(self):
        with patch("iniciar_intencion.subprocess.run") as ejecutar:
            ejecutar.return_value.returncode = 0
            iniciar_intencion.main([
                "--demo-agresiones", "--usb", "1", "--sin-mostrar",
                "--sin-alerta-sonora", "--sin-mejorar-iluminacion",
            ])
        cmd = ejecutar.call_args.args[0]
        self.assertNotIn("--alerta-sonora", cmd)
        self.assertIn("--sin-mejorar-iluminacion", cmd)

    def test_armas_emite_opciones_de_sonido_y_luz(self):
        cmd = iniciar_intencion.construir_comando_armas(
            "camara:1", alerta_sonora=True, mejorar_luz=False,
        )
        self.assertIn("--alerta-sonora", cmd)
        self.assertIn("--sin-mejorar-iluminacion", cmd)

    def test_instalacion_de_modelo_no_es_automatica(self):
        with patch("builtins.input", side_effect=["7", "NO"]):
            with patch("iniciar_intencion.subprocess.run") as ejecutar:
                codigo = iniciar_intencion.main([])
        self.assertEqual(codigo, 0)
        ejecutar.assert_not_called()

    def test_instalacion_requiere_acepto_explicito(self):
        with patch("builtins.input", side_effect=["7", "ACEPTO"]):
            with patch("iniciar_intencion.subprocess.run") as ejecutar:
                ejecutar.return_value.returncode = 0
                codigo = iniciar_intencion.main([])
        self.assertEqual(codigo, 0)
        comando = ejecutar.call_args.args[0]
        self.assertIn("descargar_modelo_armas.py", " ".join(comando))
        self.assertIn("--acepto-modelo-tercero", comando)

    def test_luz_en_sombras_localizadas(self):
        clara = np.full((120, 160, 3), 180, dtype=np.uint8)
        clara[:, :80] = 20
        resultado, aplicado = mejorar_iluminacion(clara)
        self.assertTrue(aplicado)
        self.assertEqual(resultado.shape, clara.shape)
        self.assertGreaterEqual(float(resultado[:, :80].mean()), 20.0)


if __name__ == "__main__":
    unittest.main()
