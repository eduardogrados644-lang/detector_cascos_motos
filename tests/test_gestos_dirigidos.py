"""Pruebas de gestos dirigidos: NO inferir intenciones por una mano alzada."""
import unittest
from modules.behavioral.aggression import DetectorGestosDirigidos
from modules.behavioral.events import Persona


def actor(x, ident=1, y=70, elevacion=False):
    pts = [(0., 0., 0.)] * 17
    hombro = (55, 70)
    codo = (x-35, y)
    muneca = (x, y if not elevacion else 30)
    for i, v in {5:hombro, 7:codo, 9:muneca, 11:(55,150)}.items():
        pts[i] = (float(v[0]), float(v[1]), 0.95)
    return Persona(ident,(10,10,110,210),tuple(pts))


def objetivo(ident=2, caja=(120,10,220,210)):
    return Persona(ident,caja,())


class PruebasGestosDirigidos(unittest.TestCase):
    def test_mano_alzada_aislada_no_es_agresion(self):
        d=DetectorGestosDirigidos()
        for i in range(1,12):
            self.assertEqual(d.procesar(i,[actor(55,elevacion=True)],i/10),[])

    def test_proximidad_estatica_no_es_agresion(self):
        d=DetectorGestosDirigidos()
        for i in range(1,12):
            self.assertEqual(d.procesar(i,[actor(70),objetivo()],i/10),[])

    def test_intento_por_mano_movida_hacia_persona(self):
        d=DetectorGestosDirigidos(min_gestos=2)
        eventos=[]
        for i,x in enumerate((60,80,100,115),1):
            eventos += d.procesar(i,[actor(x),objetivo()],i/10)
        self.assertEqual([e.tipo for e in eventos],['posible_intento_agresion'])
        self.assertEqual((eventos[0].track_id,eventos[0].objetivo_id),(1,2))

    def test_movimiento_alejandose_no_alerta(self):
        d=DetectorGestosDirigidos()
        for i,x in enumerate((100,80,60,40,20),1):
            self.assertEqual(d.procesar(i,[actor(x),objetivo()],i/10),[])

    def test_requiere_pareja_cercana(self):
        d=DetectorGestosDirigidos()
        lejano=objetivo(caja=(350,10,450,210))
        for i,x in enumerate((60,80,100,115),1):
            self.assertEqual(d.procesar(i,[actor(x),lejano],i/10),[])

    def test_posible_agresion_si_superpone_torso_varias_veces(self):
        d=DetectorGestosDirigidos(min_gestos=2,min_superposicion=2)
        salida=[]
        for i,x in enumerate((70,90,123,145,170),1):
            salida.extend(d.procesar(i,[actor(x),objetivo()],i/10))
        self.assertIn('posible_agresion',[e.tipo for e in salida])

    def test_corte_de_fotograma_y_cambio_id_resetean(self):
        d=DetectorGestosDirigidos(min_gestos=2)
        d.procesar(1,[actor(60),objetivo()],.1)
        d.procesar(2,[actor(80),objetivo()],.2)
        d.procesar(4,[actor(100),objetivo()],.4)
        self.assertEqual(d.procesar(5,[actor(115),objetivo(ident=6)],.5),[])

    def test_datos_invalidos_y_sin_keypoints(self):
        d=DetectorGestosDirigidos()
        self.assertEqual(d.procesar(1,[Persona(1,(10,10,110,210),()),objetivo()],.1),[])
        with self.assertRaises(ValueError):
            d.procesar(1,[],.2)
        with self.assertRaises(ValueError):
            DetectorGestosDirigidos(min_gestos=1)


if __name__=='__main__':
    unittest.main()
