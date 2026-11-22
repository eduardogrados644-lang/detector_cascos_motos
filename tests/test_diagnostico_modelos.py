"""Pruebas sin Ultralytics para el diagnóstico comparativo."""
import tempfile
import unittest
from pathlib import Path

from scripts.diagnostico_modelos import (
    MODELOS_PREDETERMINADOS,
    es_puntero_lfs,
    resolver_archivo,
)


class PruebasDiagnosticoModelos(unittest.TestCase):
    def test_tres_modelos_diferentes(self):
        self.assertEqual(len(MODELOS_PREDETERMINADOS), 3)
        self.assertEqual(len(set(MODELOS_PREDETERMINADOS)), 3)

    def test_distingue_puntero_y_pesos(self):
        with tempfile.TemporaryDirectory() as carpeta:
            archivo = Path(carpeta) / "modelo.pt"
            archivo.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:00\nsize 12345678\n",
                encoding="utf-8",
            )
            self.assertTrue(es_puntero_lfs(archivo))
            archivo.write_bytes(b"pesos-de-prueba")
            self.assertFalse(es_puntero_lfs(archivo))

    def test_resuelve_archivo_existente(self):
        with tempfile.TemporaryDirectory() as carpeta:
            archivo = Path(carpeta) / "video.mp4"
            archivo.write_bytes(b"datos-de-prueba")
            self.assertEqual(resolver_archivo(str(archivo)), archivo.resolve())


if __name__ == "__main__":
    unittest.main()
