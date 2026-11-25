"""Pruebas de clips de eventos, sin requerir OpenCV en el CI básico."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

from modules.behavioral.event_clips import GrabadorClips

TIENE_VIDEO = (importlib.util.find_spec("cv2") is not None
               and importlib.util.find_spec("numpy") is not None)


class PruebasClips(unittest.TestCase):
    def test_sin_activar_no_crea_archivos(self):
        with tempfile.TemporaryDirectory() as tmp:
            grabador = GrabadorClips(tmp)
            grabador.cerrar()
            self.assertEqual(grabador.cantidad, 0)
            self.assertFalse(list(Path(tmp).glob("*.mp4")))

    @unittest.skipUnless(TIENE_VIDEO, "Requiere OpenCV y NumPy opcionales")
    def test_no_guarda_si_no_hay_evento(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            grabador = GrabadorClips(tmp)
            for i in range(10):
                grabador.procesar(np.zeros((20,20,3), dtype=np.uint8), i * .2)
            grabador.cerrar()
            self.assertEqual(grabador.cantidad, 0)
            self.assertFalse(list(Path(tmp).glob("*.mp4")))

    @unittest.skipUnless(TIENE_VIDEO, "Requiere OpenCV y NumPy opcionales")
    def test_guarda_clip_con_buffer(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            grabador = GrabadorClips(tmp, max_clips=2)
            fotograma = np.zeros((240,320,3), dtype=np.uint8)
            for i in range(25):
                grabador.procesar(fotograma, i * .2, evento=(i == 9))
            grabador.cerrar()
            self.assertEqual(grabador.cantidad, 1)
            self.assertTrue(grabador.archivos[0].is_file())
            cap = cv2.VideoCapture(str(grabador.archivos[0]))
            try:
                self.assertTrue(cap.isOpened())
            finally:
                cap.release()


if __name__ == "__main__":
    unittest.main()
