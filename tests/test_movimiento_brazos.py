"""Pruebas del detector observable de brazos, sin cámaras ni modelos."""
import unittest
from modules.behavioral.events import Persona
from modules.behavioral.arm_motion import (
    DetectorMovimientosBrazos, observar_brazos,
)


def persona(
    wx: float = 55, wy: float = 20,
    ident: int | None = 1,
    conf: float = 0.95,
    caja=(10, 0, 120, 220),
) -> Persona:
    pts = [(0.0, 0.0, 0.0)] * 17
    # Hombro (55,60), codo (55,40), muñeca variable y cadera (55,140).
    for idx, xy in {5:(55,60),7:(55,40),9:(wx,wy),11:(55,140)}.items():
        pts[idx] = (*xy, conf)
    return Persona(ident, caja, tuple(pts))


class PruebasBrazos(unittest.TestCase):
    def test_reconoce_brazo_elevado(self):
        obs = observar_brazos(persona())
        self.assertTrue(obs["izquierdo"].elevado)

    def test_no_confunde_brazo_bajo_con_elevado(self):
        self.assertFalse(observar_brazos(persona(wy=90))["izquierdo"].elevado)

    def test_omite_puntos_sin_confianza(self):
        self.assertEqual(observar_brazos(persona(conf=.2)), {})

    def test_rechaza_caja_invalida(self):
        self.assertEqual(observar_brazos(persona(caja=(10,0,10,220))), {})

    def test_requiere_postura_persistente(self):
        d = DetectorMovimientosBrazos(min_elevados=3)
        self.assertEqual(d.procesar(1, [persona()], 0.0), [])
        self.assertEqual(d.procesar(2, [persona()], .1), [])
        eventos = d.procesar(3, [persona()], .2)
        self.assertEqual(len(eventos), 1)
        self.assertEqual(eventos[0].tipo, "brazo_elevado_persistente")

    def test_brazo_abajo_resetea_elevacion(self):
        d = DetectorMovimientosBrazos(min_elevados=2)
        d.procesar(1, [persona()], 0)
        d.procesar(2, [persona(wy=90)], .1)
        # El brazo vuelve a elevarse: no confirma aún elevación persistente,
        # aunque sí puede producir un evento diferente de movimiento rápido.
        eventos = d.procesar(3, [persona()], .2)
        self.assertFalse(any(e.tipo == "brazo_elevado_persistente" for e in eventos))
        eventos = d.procesar(4, [persona()], .3)
        self.assertEqual(
            [e.tipo for e in eventos].count("brazo_elevado_persistente"), 1
        )

    def test_desplazamiento_rapido_repetido(self):
        d = DetectorMovimientosBrazos(min_elevados=20, min_movimientos=2)
        d.procesar(1, [persona(wx=35,wy=80)], 0)
        self.assertEqual(d.procesar(2, [persona(wx=90,wy=80)], .1), [])
        eventos = d.procesar(3, [persona(wx=35,wy=80)], .2)
        self.assertEqual(len(eventos), 1)
        self.assertEqual(eventos[0].tipo, "desplazamiento_rapido_aparente_de_brazo")

    def test_movimiento_lento_no_alerta(self):
        d = DetectorMovimientosBrazos(min_elevados=20, min_movimientos=2)
        for i in range(6):
            self.assertEqual(
                d.procesar(i+1, [persona(wx=50+i,wy=80)], i), []
            )

    def test_no_alerta_sin_track(self):
        d = DetectorMovimientosBrazos(min_elevados=2)
        for i in range(3):
            self.assertEqual(d.procesar(i+1, [persona(ident=None)], i/10), [])

    def test_cambio_id_reinicia(self):
        d = DetectorMovimientosBrazos(min_elevados=2)
        d.procesar(1, [persona()], 0)
        self.assertEqual(d.procesar(2, [persona(ident=2)], .1), [])

    def test_salto_espacial_interrumpe(self):
        d = DetectorMovimientosBrazos(min_elevados=2)
        d.procesar(1, [persona()], 0)
        self.assertEqual(
            d.procesar(2, [persona(caja=(400,0,510,220))], .1), []
        )

    def test_perdida_fotograma_reinicia(self):
        d = DetectorMovimientosBrazos(min_elevados=2)
        d.procesar(1, [persona()], 0)
        self.assertEqual(d.procesar(3, [persona()], .2), [])

    def test_separacion_alertas(self):
        d = DetectorMovimientosBrazos(min_elevados=2,separacion_alertas=5)
        d.procesar(1, [persona()], 0)
        self.assertEqual(len(d.procesar(2, [persona()], .1)), 1)
        for i in range(3,7):
            self.assertEqual(d.procesar(i, [persona()], i/10), [])
        self.assertEqual(len(d.procesar(7, [persona()], .7)), 1)

    def test_parametros_invalidos(self):
        with self.assertRaises(ValueError):
            DetectorMovimientosBrazos(min_elevados=1)
        with self.assertRaises(ValueError):
            DetectorMovimientosBrazos(velocidad_minima=0)
        with self.assertRaises(ValueError):
            DetectorMovimientosBrazos().procesar(1, [], float("nan"))


if __name__ == "__main__":
    unittest.main()
