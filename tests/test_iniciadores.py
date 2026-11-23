"""Pruebas de ambos iniciadores sin cargar modelos, video o cámara."""
import sys
import unittest
from unittest.mock import patch

import iniciar_cascos
import iniciar_intencion


class PruebasIniciadores(unittest.TestCase):
    def test_iniciador_vial_usa_solo_modulo_vial(self):
        comando = iniciar_cascos.construir_comando("prueba_motos.mp4")
        self.assertEqual(comando[:3], [sys.executable, "-m", "modules.traffic.road_monitor"])
        self.assertIn("--modelo-cascos", comando)
        self.assertIn("videos_revisados/weights/best.pt", " ".join(comando))
        self.assertNotIn("modules.behavioral", " ".join(comando))

    def test_vial_pasa_opciones_de_video(self):
        comando = iniciar_cascos.construir_comando(
            "camara:1", guardar_video=True, mostrar=False,
            stride=2, max_fotogramas=100,
        )
        self.assertIn("--guardar-video", comando)
        self.assertNotIn("--mostrar", comando)
        self.assertEqual(comando[comando.index("--stride") + 1], "2")
        self.assertEqual(comando[comando.index("--max-fotogramas") + 1], "100")

    def test_iniciador_conductual_no_usa_pesos_viales(self):
        comando = iniciar_intencion.construir_comando_demo("prueba_motos.mp4")
        self.assertEqual(
            comando[:3], [sys.executable, "-m", "modules.behavioral.pose_detector"]
        )
        self.assertNotIn("best.pt", " ".join(comando))
        self.assertNotIn("modules.traffic", " ".join(comando))

    def test_conductual_solo_estado_no_inicia_video(self):
        with patch("iniciar_intencion.subprocess.run") as ejecutor:
            resultado = iniciar_intencion.main(["--estado"])
        self.assertEqual(resultado, 0)
        ejecutor.assert_not_called()

    def test_conductual_demo_opcional(self):
        with patch("iniciar_intencion.subprocess.run") as ejecutar:
            ejecutar.return_value.returncode = 0
            resultado = iniciar_intencion.main(
                ["--demo-posturas", "--fuente", "camara:1", "--sin-mostrar"]
            )
        self.assertEqual(resultado, 0)
        comando = ejecutar.call_args.args[0]
        self.assertNotIn("--mostrar", comando)
        self.assertEqual(comando[-2:], ["--fuente", "camara:1"])

    def test_proximidad_usa_pose_sin_pesos_viales(self):
        comando = iniciar_intencion.construir_comando_demo(
            "helmet.mp4", mostrar=False, guardar_video=True,
            max_fotogramas=60, modo="proximidad",
        )
        self.assertIn("modules.behavioral.pose_detector", comando)
        self.assertEqual(comando[comando.index("--modo") + 1], "proximidad")
        self.assertIn("--guardar-video", comando)
        self.assertNotIn("--mostrar", comando)
        self.assertNotIn("best.pt", " ".join(comando))

    def test_iniciador_modo_proximidad(self):
        with patch("iniciar_intencion.subprocess.run") as ejecutor:
            ejecutor.return_value.returncode = 0
            resultado = iniciar_intencion.main([
                "--demo-proximidad", "--fuente", "helmet.mp4",
                "--max-fotogramas", "30", "--sin-mostrar",
            ])
        self.assertEqual(resultado, 0)
        self.assertIn("proximidad", ejecutor.call_args.args[0])

    def test_comando_brazos_reutiliza_pose_y_no_cascos(self):
        cmd = iniciar_intencion.construir_comando_demo(
            "prueba_motos.mp4", modo="brazos", guardar_video=True, mostrar=False,
        )
        self.assertEqual(cmd[cmd.index("--modo") + 1], "brazos")
        self.assertIn("modules.behavioral.pose_detector", cmd)
        self.assertNotIn("--modelo-cascos", cmd)

    def test_brazos_lanzador_sin_abrir_camara(self):
        with patch("iniciar_intencion.subprocess.run") as ejecutor:
            ejecutor.return_value.returncode = 0
            codigo = iniciar_intencion.main([
                "--demo-brazos", "--fuente", "helmet.mp4", "--sin-mostrar",
            ])
        self.assertEqual(codigo, 0)
        self.assertIn("brazos", ejecutor.call_args.args[0])

    def test_validaciones_argumentos(self):
        with self.assertRaises(ValueError):
            iniciar_cascos.construir_comando("", stride=3)
        with self.assertRaises(ValueError):
            iniciar_cascos.construir_comando("archivo.mp4", stride=0)
        with self.assertRaises(ValueError):
            iniciar_intencion.construir_comando_demo("", max_fotogramas=5)
        with self.assertRaises(ValueError):
            iniciar_intencion.construir_comando_demo("helmet.mp4", modo="violencia")


if __name__ == "__main__":
    unittest.main()
