"""Pruebas sin GPU de etiquetas gun y separacion de variantes Roboflow."""
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.preparar_dataset_armas import analizar_zip, leer_etiquetas, preparar

CLASES = "names: ['arm', 'body', 'face', 'foot', 'gun', 'hand', 'human', 'leg']\n"


class PruebasDatasetArmas(unittest.TestCase):
    def test_remapear_gun_y_descartar_otras(self):
        self.assertEqual(
            leer_etiquetas('4 0.5 0.5 0.2 0.3\n5 0.1 0.2 0.3 0.4\n'),
            ['0 0.5 0.5 0.2 0.3'])
        self.assertEqual(leer_etiquetas('5 0.1 0.2 0.3 0.4'), [])
        for linea in ('8 0.5 0.5 0.2 0.3', '4 nan 0.5 0.2 0.3', '4 0 0 0 0'):
            with self.subTest(linea=linea), self.assertRaises(ValueError):
                leer_etiquetas(linea)

    def test_variantes_no_cruzan_particiones(self):
        with tempfile.TemporaryDirectory() as tmp:
            carpeta = Path(tmp)
            origen = carpeta / 'armas.zip'
            with zipfile.ZipFile(origen, 'w') as z:
                z.writestr('data.yaml', CLASES)
                for split, nombre, clase in [
                    ('train', 'misma.rf.a.jpg', 4), ('test', 'misma.rf.b.jpg', 4),
                    ('valid', 'otra.rf.c.jpg', 4), ('train', 'tercera.rf.d.jpg', 5),
                    ('train', 'cuarta.rf.e.jpg', 4), ('test', 'quinta.rf.f.jpg', 4),
                ]:
                    z.writestr(f'{split}/images/{nombre}', b'JPEG-falso-para-prueba')
                    z.writestr(f'{split}/labels/{Path(nombre).stem}.txt',
                               f'{clase} 0.5 0.5 0.2 0.2\n')
            entradas, grupos = analizar_zip(origen)
            self.assertEqual(len(entradas), 6)
            self.assertEqual(len(grupos['misma']), 2)
            destino = carpeta / 'preparado'
            resumen = preparar(origen, destino)
            self.assertEqual(sum(resumen['imagenes'].values()), 6)
            self.assertTrue((destino / 'procedencia.json').is_file())
            self.assertIn('gun', (destino / 'data.yaml').read_text())
            divisiones = []
            for split in ('train', 'valid', 'test'):
                if (destino / split / 'images' / 'misma.rf.a.jpg').exists():
                    divisiones.append(split)
                if (destino / split / 'images' / 'misma.rf.b.jpg').exists():
                    divisiones.append(split)
            self.assertEqual(len(set(divisiones)), 1)
            self.assertEqual(len(divisiones), 2)
            with self.assertRaises(FileExistsError):
                preparar(origen, destino)

    def test_no_acepta_otras_clases(self):
        with tempfile.TemporaryDirectory() as tmp:
            origen = Path(tmp) / 'bad.zip'
            with zipfile.ZipFile(origen, 'w') as z:
                z.writestr('data.yaml', "names: ['person','gun']\n")
            with self.assertRaises(ValueError):
                analizar_zip(origen)


if __name__ == '__main__':
    unittest.main()
