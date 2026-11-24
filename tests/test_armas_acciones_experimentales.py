"""Pruebas de la lógica de candidatos sin cargar modelos externos."""
import unittest
from modules.behavioral.weapon_events import ConfirmadorArmas, nombres_armas
from modules.behavioral.action_video import inicios_ventanas


class PruebasArmamento(unittest.TestCase):
    def test_confirmador_exige_racha(self):
        a=ConfirmadorArmas(consecutivos=3,enfriamiento=9)
        self.assertEqual(a.procesar(1,["knife"]),[])
        self.assertEqual(a.procesar(2,["knife"]),[])
        self.assertEqual(a.procesar(3,["knife"]),["knife"])
        self.assertEqual(a.procesar(4,[]),[])
        self.assertEqual(a.procesar(5,["knife"]),[])
    def test_no_atribuye_objetos_a_personas(self):
        a=ConfirmadorArmas(consecutivos=1)
        self.assertEqual(a.procesar(1,["gun","knife","gun"]),["gun","knife"])
    def test_nombres_especializados(self):
        self.assertEqual(nombres_armas({0:"gun",1:"knife"}),{0:"gun",1:"knife"})
        with self.assertRaises(ValueError):
            nombres_armas({43:"knife", 0:"person"})
    def test_fotogramas_crecientes(self):
        a=ConfirmadorArmas()
        a.procesar(1,[])
        with self.assertRaises(ValueError):
            a.procesar(1,[])
    def test_ventanas_video(self):
        self.assertEqual(inicios_ventanas(2.,3.,3.),[0.])
        self.assertEqual(inicios_ventanas(10.,3.,3.),[0.,3.,6.,7.])
        with self.assertRaises(ValueError):
            inicios_ventanas(0.)


if __name__=="__main__":
    unittest.main()
