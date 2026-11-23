"""Pruebas de memoria visual sin OpenCV ni modelos."""
import unittest
from modules.behavioral.tracking_boxes import CajaVisual, MemoriaCajas, iou

def caja(id, xyxy=(0,0,20,40)):
    return CajaVisual(id, xyxy, .85)

class PruebasSeguimientoVisual(unittest.TestCase):
    def test_solapamiento_conserva_dos_detectados(self):
        memoria=MemoriaCajas()
        resultado=memoria.procesar(1,[caja(1),caja(2,(5,2,25,42))])
        self.assertEqual([x.id for x in resultado],[1,2])
        self.assertEqual([x.ausente for x in resultado],[0,0])
    def test_se_muestra_ultima_caja_y_expira(self):
        memoria=MemoriaCajas(conservar=2)
        memoria.procesar(1,[caja(1)])
        self.assertEqual(memoria.procesar(2,[])[0].ausente,1)
        self.assertEqual(memoria.procesar(3,[])[0].ausente,2)
        self.assertEqual(memoria.procesar(4,[]),[])
    def test_reaparicion_no_duplica(self):
        memoria=MemoriaCajas(conservar=4)
        memoria.procesar(1,[caja(1)])
        memoria.procesar(2,[])
        obs=memoria.procesar(3,[caja(1,(2,0,22,40))])
        self.assertEqual(len(obs),1)
        self.assertEqual(obs[0].ausente,0)
    def test_cambio_id_misma_caja_no_duplica(self):
        memoria=MemoriaCajas(conservar=6)
        memoria.procesar(1,[caja(1)])
        obs=memoria.procesar(2,[caja(99)])
        self.assertEqual(len(obs),1)
        self.assertEqual(obs[0].id,99)
    def test_sin_id_no_es_recordado(self):
        memoria=MemoriaCajas()
        memoria.procesar(1,[caja(None)])
        self.assertEqual(memoria.procesar(2,[]),[])
    def test_persistencia_cero_no_inventa_persona(self):
        memoria=MemoriaCajas(conservar=0)
        memoria.procesar(1,[caja(1)])
        self.assertEqual(memoria.procesar(2,[]),[])
    def test_valida_ids_y_fotogramas(self):
        memoria=MemoriaCajas()
        memoria.procesar(1,[])
        with self.assertRaises(ValueError):
            memoria.procesar(1,[])
        with self.assertRaises(ValueError):
            MemoriaCajas(-1)
        with self.assertRaises(ValueError):
            MemoriaCajas().procesar(1,[caja(1,(10,10,0,20))])

    def test_configuracion_experimental_tiene_solo_buffer_ampliado(self):
        from pathlib import Path
        archivo=Path(__file__).resolve().parents[1]/"config"/"trackers"/"bytetrack_buffer60.yaml"
        datos=dict(line.split(":",1) for line in archivo.read_text().splitlines()
                   if ":" in line and not line.strip().startswith("#"))
        self.assertEqual(datos["track_buffer"].strip(),"60")
        self.assertEqual(datos["tracker_type"].strip(),"bytetrack")
        self.assertEqual(datos["track_high_thresh"].strip(),"0.25")
        self.assertEqual(datos["track_low_thresh"].strip(),"0.1")
        self.assertEqual(datos["new_track_thresh"].strip(),"0.25")
        self.assertEqual(datos["match_thresh"].strip(),"0.8")

    def test_iou(self):
        self.assertEqual(iou((0,0,10,10),(0,0,10,10)),1)
        self.assertEqual(iou((0,0,10,10),(20,20,30,30)),0)

if __name__=="__main__":
    unittest.main()
