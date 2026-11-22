"""Pruebas unitarias de asociación vial sin cámara, YOLO ni OpenCV."""
import unittest

from modules.traffic.association import (
    ConfirmadorEventosViales,
    Objeto,
    normalizar_clase,
    relacionar,
)


MOTO = Objeto("motorcycle", (100, 150, 300, 250), 0.90, 11)
PERSONA = Objeto("person", (170, 45, 230, 210), 0.95, 21)
SIN_CASCO = Objeto("Without Helmet", (190, 53, 211, 79), 0.82)
CON_CASCO = Objeto("With Helmet", (190, 53, 211, 79), 0.90)


class TestAsociacionVial(unittest.TestCase):
    def test_normaliza_clases_del_dataset_y_coco(self):
        self.assertEqual(normalizar_clase("With Helmet"), "con_casco")
        self.assertEqual(normalizar_clase("Without Helmet"), "sin_casco")
        self.assertEqual(normalizar_clase("motorcycle"), "moto")
        self.assertEqual(normalizar_clase("person"), "persona")
        self.assertEqual(normalizar_clase("licence"), "placa")
        self.assertEqual(normalizar_clase("numberplate"), "placa")
        self.assertEqual(normalizar_clase("number plate"), "placa")
        self.assertEqual(normalizar_clase("nohelmet"), "sin_casco")

    def test_asocia_persona_moto_y_etiqueta_de_casco(self):
        relaciones = relacionar([MOTO], [PERSONA], [SIN_CASCO])
        self.assertEqual(len(relaciones), 1)
        self.assertEqual(relaciones[0].moto.track_id, 11)
        self.assertEqual(relaciones[0].persona.track_id, 21)
        self.assertEqual(relaciones[0].estado, "sin_casco")

    def test_bystander_fuera_de_la_moto_no_se_asocia(self):
        peaton = Objeto("person", (330, 20, 380, 260), 0.9, 44)
        self.assertEqual(relacionar([MOTO], [peaton], [SIN_CASCO]), [])

    def test_motos_ambiguas_no_asignan_una_persona(self):
        moto2 = Objeto("motorcycle", (110, 150, 310, 250), 0.9, 12)
        self.assertEqual(relacionar([MOTO, moto2], [PERSONA], [SIN_CASCO]), [])

    def test_cascos_contradictorios_no_dan_infraccion(self):
        relaciones = relacionar([MOTO], [PERSONA], [SIN_CASCO, CON_CASCO])
        self.assertEqual(relaciones[0].estado, "desconocido")

    def test_sin_detecion_casco_estado_desconocido(self):
        relaciones = relacionar([MOTO], [PERSONA], [])
        self.assertEqual(relaciones[0].estado, "desconocido")

    def test_no_alerta_sin_track_id(self):
        moto = Objeto("motorcycle", MOTO.caja, 0.9, None)
        detector = ConfirmadorEventosViales(3)
        relaciones = relacionar([moto], [PERSONA], [SIN_CASCO])
        for paso in range(1, 8):
            self.assertEqual(detector.procesar(paso, paso * 3, relaciones), [])

    def test_confirmation_y_no_duplicados(self):
        detector = ConfirmadorEventosViales(3)
        relaciones = relacionar([MOTO], [PERSONA], [SIN_CASCO])
        self.assertEqual(detector.procesar(1, 3, relaciones), [])
        self.assertEqual(detector.procesar(2, 6, relaciones), [])
        eventos = detector.procesar(3, 9, relaciones)
        self.assertEqual(len(eventos), 1)
        self.assertEqual(eventos[0].tipo, "posible_persona_sin_casco")
        self.assertEqual(eventos[0].moto_id, 11)
        self.assertEqual(eventos[0].persona_id, 21)
        for paso in range(4, 12):
            self.assertEqual(detector.procesar(paso, paso * 3, relaciones), [])

    def test_interrupcion_reinicia_racha(self):
        detector = ConfirmadorEventosViales(2)
        relaciones = relacionar([MOTO], [PERSONA], [SIN_CASCO])
        detector.procesar(1, 3, relaciones)
        detector.procesar(2, 6, [])
        self.assertEqual(detector.procesar(3, 9, relaciones), [])
        self.assertEqual(len(detector.procesar(4, 12, relaciones)), 1)

    def test_misma_moto_cambiando_persona_no_confirma(self):
        detector = ConfirmadorEventosViales(2)
        primera = relacionar([MOTO], [PERSONA], [SIN_CASCO])
        otra = Objeto("person", PERSONA.caja, 0.9, 88)
        segunda = relacionar([MOTO], [otra], [SIN_CASCO])
        self.assertEqual(detector.procesar(1, 3, primera), [])
        self.assertEqual(detector.procesar(2, 6, segunda), [])

    def test_diferentes_motos_pueden_emitir_eventos(self):
        detector = ConfirmadorEventosViales(2)
        moto2 = Objeto("motorcycle", (400, 150, 600, 250), 0.95, 12)
        persona2 = Objeto("person", (470, 45, 530, 210), 0.95, 22)
        sin_casco2 = Objeto("Without Helmet", (490, 53, 511, 79), 0.9)
        relaciones = relacionar([MOTO, moto2], [PERSONA, persona2], [SIN_CASCO, sin_casco2])
        self.assertEqual(len(relaciones), 2)
        detector.procesar(1, 3, relaciones)
        eventos = detector.procesar(2, 6, relaciones)
        self.assertEqual({evento.moto_id for evento in eventos}, {11, 12})


if __name__ == "__main__":
    unittest.main()
