"""Tests puros: contradicciones por cabeza sin mezclar ocupantes."""
import unittest

from modules.traffic.association import Objeto, Relacion
from modules.traffic.temporal_consistency import ConsistenciaTemporalCabezas


M = Objeto("moto", (100, 150, 300, 250), .9, 11)
P = Objeto("persona", (115, 35, 265, 225), .9, 21)


def relacion(estado, cabeza_id=101, box=(174, 52, 200, 81), moto=M, persona=P):
    cabeza = Objeto(estado, box, .87, cabeza_id)
    return Relacion(moto, persona, cabeza, estado)


class TestConsistenciaTemporal(unittest.TestCase):
    def test_no_modifica_clases_estables(self):
        filtro = ConsistenciaTemporalCabezas(ventana=4)
        for paso in range(1, 7):
            res = filtro.procesar(paso, paso * 3, [relacion("con_casco")])
            self.assertEqual(res[0].estado, "con_casco")
            self.assertEqual(filtro.contradicciones, [])

    def test_detecta_cambio_sobre_misma_cabeza(self):
        filtro = ConsistenciaTemporalCabezas(ventana=4)
        filtro.procesar(1, 1, [relacion("sin_casco")])
        res = filtro.procesar(2, 4, [relacion("con_casco")])
        self.assertEqual(res[0].estado, "incierto")
        self.assertEqual(res[0].casco.clase, "con_casco")
        self.assertEqual(len(filtro.contradicciones), 1)
        self.assertEqual(filtro.contradicciones[0].clase_anterior, "sin_casco")
        self.assertEqual(filtro.contradicciones[0].clase_actual, "con_casco")

    def test_cambio_id_seguimiento_misma_posicion(self):
        filtro = ConsistenciaTemporalCabezas()
        filtro.procesar(1, 1, [relacion("sin_casco", 7)])
        res = filtro.procesar(
            2, 4, [relacion("con_casco", 13, (177, 54, 203, 83))]
        )
        self.assertEqual(res[0].estado, "incierto")
        self.assertEqual(len(filtro.contradicciones), 1)

    def test_misma_id_con_salto_grande_no_se_mezcla(self):
        filtro = ConsistenciaTemporalCabezas()
        filtro.procesar(1, 1, [relacion("sin_casco")])
        res = filtro.procesar(
            2, 4, [relacion("con_casco", 101, (245, 52, 271, 81))]
        )
        self.assertEqual(res[0].estado, "con_casco")
        self.assertEqual(filtro.contradicciones, [])

    def test_dos_cabezas_simultaneas_separadas(self):
        filtro = ConsistenciaTemporalCabezas()
        a = relacion("sin_casco", 101, (155, 52, 174, 78))
        b = relacion("con_casco", 102, (213, 52, 233, 79))
        for paso in (1, 2):
            res = filtro.procesar(paso, paso * 3, [a, b])
            self.assertEqual([r.estado for r in res], ["sin_casco", "con_casco"])
        self.assertEqual(filtro.contradicciones, [])

    def test_no_mezcla_dos_motos_diferentes(self):
        otra = Objeto("moto", (100, 150, 300, 250), .9, 12)
        filtro = ConsistenciaTemporalCabezas()
        filtro.procesar(1, 1, [relacion("sin_casco", moto=M)])
        res = filtro.procesar(2, 4, [relacion("con_casco", moto=otra)])
        self.assertEqual(res[0].estado, "con_casco")

    def test_incertidumbre_caduca_tras_ventana(self):
        filtro = ConsistenciaTemporalCabezas(ventana=3)
        filtro.procesar(1, 1, [relacion("sin_casco")])
        self.assertEqual(
            filtro.procesar(2, 4, [relacion("con_casco")])[0].estado, "incierto"
        )
        self.assertEqual(
            filtro.procesar(3, 7, [relacion("con_casco")])[0].estado, "incierto"
        )
        self.assertEqual(
            filtro.procesar(4, 10, [relacion("con_casco")])[0].estado, "con_casco"
        )

    def test_sin_track_de_moto_no_infiere_estado(self):
        filtro = ConsistenciaTemporalCabezas()
        sin_moto = Objeto("moto", M.caja, .9, None)
        r = relacion("con_casco", moto=sin_moto)
        self.assertEqual(filtro.procesar(1, 1, [r]), [r])

    def test_desconocido_no_altera_historicos(self):
        filtro = ConsistenciaTemporalCabezas()
        con = relacion("con_casco")
        unknown = Relacion(M, P, None, "desconocido")
        filtro.procesar(1, 1, [con])
        self.assertEqual(filtro.procesar(2, 4, [unknown]), [unknown])
        self.assertEqual(
            filtro.procesar(3, 7, [con])[0].estado, "con_casco"
        )

    def test_estado_incierto_no_confirma_evento(self):
        from modules.traffic.association import ConfirmadorEventosViales
        filtro = ConsistenciaTemporalCabezas(ventana=4)
        detector = ConfirmadorEventosViales(observaciones_minimas=2)
        respuestas = [
            relacion("sin_casco"), relacion("con_casco"),
            relacion("sin_casco"), relacion("sin_casco"),
        ]
        for paso, original in enumerate(respuestas, 1):
            relaciones = filtro.procesar(paso, paso * 3, [original])
            self.assertEqual(detector.procesar(paso, paso * 3, relaciones), [])
        self.assertEqual(relaciones[0].estado, "incierto")

    def test_una_persona_coco_puede_tener_dos_historias_separadas(self):
        from modules.traffic.association import relacionar
        filtro = ConsistenciaTemporalCabezas()
        a = Objeto("Without Helmet", (168, 52, 186, 80), .9, 100)
        b = Objeto("With Helmet", (219, 52, 239, 80), .89, 101)
        relaciones = relacionar([M], [P], [a, b])
        self.assertEqual(len(relaciones), 2)
        for paso in (1, 2, 3):
            reales = filtro.procesar(paso, paso * 3, relaciones)
            self.assertEqual({r.estado for r in reales}, {"sin_casco", "con_casco"})
            self.assertEqual(filtro.contradicciones, [])

    def test_rechaza_valores_invalidos(self):
        with self.assertRaises(ValueError):
            ConsistenciaTemporalCabezas(ventana=1)
        filtro = ConsistenciaTemporalCabezas()
        with self.assertRaises(ValueError):
            filtro.procesar(0, 1, [])
        with self.assertRaises(ValueError):
            filtro.procesar(1, 0, [])


if __name__ == "__main__":
    unittest.main()
