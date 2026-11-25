"""Regresiones de calidad para gestos dirigidos y objetos espaciales.

No abre camaras ni descarga modelos: comprueba reglas de candidatos.
"""
import unittest

from modules.behavioral.aggression import DetectorGestosDirigidos
from modules.behavioral.events import Persona
from modules.behavioral.weapon_events import (
    ConfirmadorArmas, cajas_compatibles, nombres_armas,
)


def actor(x, desplazamiento=0, ident=1, alzado=False):
    pts = [(0., 0., 0.)] * 17
    puntos = {
        5: (55 + desplazamiento, 70),
        7: (x - 35 + desplazamiento, 70),
        9: (x + desplazamiento, 30 if alzado else 70),
        11: (55 + desplazamiento, 150),
    }
    for idx, (px, py) in puntos.items():
        pts[idx] = (float(px), float(py), .95)
    return Persona(ident, (10 + desplazamiento, 10, 110 + desplazamiento, 210), tuple(pts))


def objetivo(ident=2, caja=(120, 10, 220, 210)):
    return Persona(ident, caja, ())


def arma(x, clase="gun", confianza=.8):
    return (clase, confianza, [x, 10, x + 30, 40])


class CalidadAgresiones(unittest.TestCase):
    def test_intento_dirigido_con_movimiento_repetido(self):
        d = DetectorGestosDirigidos()
        eventos = []
        for n, x in enumerate((60, 80, 100, 115), 1):
            eventos += d.procesar(n, [actor(x), objetivo()], n / 10)
        self.assertEqual([e.tipo for e in eventos], ["posible_intento_agresion"])

    def test_solape_de_muneca_no_prueba_contacto(self):
        d = DetectorGestosDirigidos()
        eventos = []
        for n, x in enumerate((70, 90, 123, 145, 170), 1):
            eventos += d.procesar(n, [actor(x), objetivo()], n / 10)
        self.assertIn("posible_agresion", [e.tipo for e in eventos])

    def test_manos_alzadas_quietas_no_activan(self):
        d = DetectorGestosDirigidos()
        for n in range(1, 9):
            self.assertEqual(d.procesar(n, [actor(55, alzado=True), objetivo()], n / 10), [])

    def test_traslacion_global_no_es_golpe(self):
        d = DetectorGestosDirigidos()
        for n, shift in enumerate((0, 8, 16, 24, 32), 1):
            self.assertEqual(d.procesar(n, [
                actor(70, desplazamiento=shift),
                objetivo(caja=(120 + shift, 10, 220 + shift, 210))
            ], n / 10), [])

    def test_salidas_no_se_heredan_al_cambiar_pareja(self):
        d = DetectorGestosDirigidos()
        d.procesar(1, [actor(60), objetivo()], .1)
        d.procesar(2, [actor(80), objetivo()], .2)
        self.assertEqual(d.procesar(3, [actor(100), objetivo(caja=(10, 10, 110, 210))], .3), [])


class CalidadArmas(unittest.TestCase):
    def test_no_acumula_detecciones_de_zonas_distintas(self):
        d = ConfirmadorArmas(consecutivos=3)
        for n, x in enumerate((10, 300, 40, 270, 70), 1):
            self.assertEqual(d.procesar_objetos(n, [arma(x)]), [])

    def test_un_objeto_continuo_si_se_confirma(self):
        d = ConfirmadorArmas(consecutivos=3)
        self.assertEqual(d.procesar_objetos(1, [arma(10)]), [])
        self.assertEqual(d.procesar_objetos(2, [arma(12)]), [])
        eventos = d.procesar_objetos(3, [arma(14)])
        self.assertEqual(len(eventos), 1)
        self.assertEqual(eventos[0].objeto_id, 1)
        self.assertEqual(eventos[0].clase, "gun")

    def test_dos_objetos_separados_con_ids_independientes(self):
        d = ConfirmadorArmas(consecutivos=2)
        d.procesar_objetos(1, [arma(10), arma(250)])
        eventos = d.procesar_objetos(2, [arma(252), arma(13)])
        self.assertEqual({e.objeto_id for e in eventos}, {1, 2})

    def test_corte_de_fotograma_reinicia_racha(self):
        d = ConfirmadorArmas(consecutivos=2)
        d.procesar_objetos(1, [arma(10)])
        d.procesar_objetos(2, [])
        self.assertEqual(d.procesar_objetos(3, [arma(10)]), [])

    def test_datos_incorrectos_y_escala_incoherente(self):
        d = ConfirmadorArmas()
        with self.assertRaises(ValueError):
            d.procesar_objetos(1, [("person", .8, (10, 10, 40, 40))])
        self.assertFalse(cajas_compatibles((10, 10, 20, 20), (0, 0, 300, 300)))
        self.assertEqual(nombres_armas({0: "gun"}), {0: "gun"})


if __name__ == "__main__":
    unittest.main()
