"""Pruebas unitarias de asociación vial sin cámara, YOLO ni OpenCV."""
import unittest

from modules.traffic.association import (
    ConfirmadorEventosViales,
    Objeto,
    Relacion,
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


    def test_dos_ocupantes_misma_moto_dos_candidatos(self):
        """Ya no se limita a una sola alerta por motocicleta."""
        primera = Objeto("person", (135, 45, 190, 210), 0.93, 21)
        segunda = Objeto("person", (210, 45, 265, 210), 0.91, 22)
        cabeza_a = Objeto("Without Helmet", (148, 55, 173, 80), 0.86, 101)
        cabeza_b = Objeto("Without Helmet", (225, 55, 250, 80), 0.82, 102)
        relaciones = relacionar(
            [MOTO], [primera, segunda], [cabeza_a, cabeza_b]
        )
        self.assertEqual(len(relaciones), 2)
        detector = ConfirmadorEventosViales(2)
        self.assertEqual(detector.procesar(1, 3, relaciones), [])
        eventos = detector.procesar(2, 6, relaciones)
        self.assertEqual(len(eventos), 2)
        self.assertEqual({e.cabeza_id for e in eventos}, {101, 102})
        self.assertEqual({e.moto_id for e in eventos}, {11})

    def test_caja_persona_agrupada_permite_dos_cabezas_distintas(self):
        """Dos cabezas visibles no se reducen a una caja COCO grande."""
        primera = Objeto("Without Helmet", (190, 53, 211, 79), 0.82, 101)
        segunda = Objeto("Without Helmet", (171, 53, 186, 79), 0.88, 102)
        relaciones = relacionar([MOTO], [PERSONA], [primera, segunda])
        self.assertEqual(len(relaciones), 2)
        self.assertEqual({r.casco.track_id for r in relaciones}, {101, 102})
        detector = ConfirmadorEventosViales(2)
        detector.procesar(1, 3, relaciones)
        alertas = detector.procesar(2, 6, relaciones)
        self.assertEqual(len(alertas), 2)
        self.assertEqual({a.persona_id for a in alertas}, {21})
        self.assertEqual({a.cabeza_id for a in alertas}, {101, 102})
        self.assertEqual({a.personas_asociadas for a in alertas}, {1})

    def test_dos_cajas_sobre_misma_cabeza_no_crean_dos_alertas(self):
        primera = Objeto("Without Helmet", SIN_CASCO.caja, 0.82, 101)
        repetida = Objeto("Without Helmet", SIN_CASCO.caja, 0.67, 102)
        relaciones = relacionar([MOTO], [PERSONA], [primera, repetida])
        self.assertEqual(len(relaciones), 1)
        self.assertEqual(relaciones[0].casco.track_id, 101)
        detector = ConfirmadorEventosViales(2)
        detector.procesar(1, 3, relaciones)
        self.assertEqual(len(detector.procesar(2, 6, relaciones)), 1)

    def test_cabeza_estable_no_depende_de_id_coco_inestable(self):
        cabeza = Objeto("Without Helmet", SIN_CASCO.caja, 0.86, 105)
        detector = ConfirmadorEventosViales(2)
        primera = relacionar([MOTO], [PERSONA], [cabeza])
        persona_otra_id = Objeto("person", PERSONA.caja, 0.9, 88)
        segunda = relacionar([MOTO], [persona_otra_id], [cabeza])
        self.assertEqual(detector.procesar(1, 3, primera), [])
        eventos = detector.procesar(2, 6, segunda)
        self.assertEqual(len(eventos), 1)
        self.assertEqual(eventos[0].cabeza_id, 105)
        self.assertEqual(eventos[0].persona_id, 88)

    def test_un_mismo_id_con_salto_brutal_reinicia_confirmacion(self):
        """Previene atribuir a otro ocupante un track ID reciclado."""
        detector = ConfirmadorEventosViales(2)
        primera = relacionar([MOTO], [PERSONA], [SIN_CASCO])
        persona_saltada = Objeto("person", (235, 45, 295, 210), 0.94, 21)
        casco_saltado = Objeto("Without Helmet", (255, 53, 274, 79), 0.90)
        segunda = relacionar([MOTO], [persona_saltada], [casco_saltado])
        self.assertEqual(len(segunda), 1)
        self.assertEqual(segunda[0].estado, "sin_casco")
        detector.procesar(1, 3, primera)
        self.assertEqual(detector.procesar(2, 6, segunda), [])
        self.assertEqual(len(detector.procesar(3, 9, segunda)), 1)

    def test_no_repite_cabeza_ya_confirmada(self):
        cabeza = Objeto("Without Helmet", SIN_CASCO.caja, 0.84, 101)
        relaciones = relacionar([MOTO], [PERSONA], [cabeza])
        detector = ConfirmadorEventosViales(2)
        detector.procesar(1, 3, relaciones)
        self.assertEqual(len(detector.procesar(2, 6, relaciones)), 1)
        for paso in range(3, 15):
            self.assertEqual(detector.procesar(paso, paso * 3, relaciones), [])

    def test_reidentificacion_178_232_no_es_otro_candidato(self):
        """Regresión con cajas aproximadas de los dos avisos del video recibido."""
        moto_inicio = Objeto("moto", (755, 277, 952, 564), 0.93, 6)
        persona_inicio = Objeto("persona", (772, 160, 966, 491), 0.94, 2)
        cabeza_14 = Objeto("sin_casco", (825, 188, 878, 261), 0.78, 14)
        moto_final = Objeto("moto", (659, 131, 924, 512), 0.92, 6)
        persona_final = Objeto("persona", (684, 13, 942, 413), 0.95, 2)
        cabeza_25 = Objeto("sin_casco", (755, 25, 838, 108), 0.86, 25)
        primera = Relacion(moto_inicio, persona_inicio, cabeza_14, "sin_casco")
        segunda = Relacion(moto_final, persona_final, cabeza_25, "sin_casco")
        detector = ConfirmadorEventosViales(2)

        self.assertEqual(detector.procesar(1, 175, [primera]), [])
        aceptadas = detector.procesar(2, 178, [primera])
        self.assertEqual(len(aceptadas), 1)
        self.assertEqual(aceptadas[0].cabeza_id, 14)

        for paso in range(3, 20):
            self.assertEqual(detector.procesar(paso, 178 + 3 * (paso - 2), []), [])
        detector.procesar(20, 229, [segunda])
        self.assertEqual(detector.procesar(21, 232, [segunda]), [])
        self.assertEqual(len(detector.posibles_duplicados), 1)
        duplicado = detector.posibles_duplicados[0]
        self.assertEqual((duplicado.moto_id, duplicado.cabeza_id, duplicado.cabeza_anterior_id), (6, 25, 14))

    def test_otra_cabeza_otra_zona_no_se_oculta(self):
        """Dos ocupantes pueden confirmar por separado aunque no sean simultáneos."""
        moto = Objeto("moto", (100, 150, 300, 250), 0.92, 11)
        persona = Objeto("persona", (120, 30, 275, 210), 0.92, 21)
        cabeza_a = Objeto("sin_casco", (135, 55, 155, 75), 0.89, 101)
        cabeza_b = Objeto("sin_casco", (240, 55, 260, 75), 0.90, 102)
        primero = Relacion(moto, persona, cabeza_a, "sin_casco")
        segundo = Relacion(moto, persona, cabeza_b, "sin_casco")
        detector = ConfirmadorEventosViales(2)
        detector.procesar(1, 3, [primero])
        self.assertEqual(len(detector.procesar(2, 6, [primero])), 1)
        detector.procesar(3, 9, [])
        detector.procesar(4, 12, [segundo])
        eventos = detector.procesar(5, 15, [segundo])
        self.assertEqual(len(eventos), 1)
        self.assertEqual(eventos[0].cabeza_id, 102)
        self.assertEqual(detector.posibles_duplicados, [])

    def test_coobservadas_distintas_no_se_fusionan_pese_a_zona_cercana(self):
        """Si están a la vista juntas, los ID no se tratan como reidentificación."""
        moto = Objeto("moto", (100, 150, 300, 250), 0.9, 11)
        persona = Objeto("persona", (165, 40, 255, 220), 0.9, 21)
        cabeza_a = Objeto("sin_casco", (188, 50, 208, 76), 0.9, 101)
        cabeza_b = Objeto("sin_casco", (211, 50, 231, 76), 0.9, 102)
        a = Relacion(moto, persona, cabeza_a, "sin_casco")
        b = Relacion(moto, persona, cabeza_b, "sin_casco")
        detector = ConfirmadorEventosViales(2)
        detector.procesar(1, 3, [a, b])
        aceptadas = detector.procesar(2, 6, [a])
        self.assertEqual(len(aceptadas), 1)
        detector.procesar(3, 9, [b])
        aceptadas = detector.procesar(4, 12, [b])
        self.assertEqual(len(aceptadas), 1)
        self.assertEqual(aceptadas[0].cabeza_id, 102)
        self.assertEqual(detector.posibles_duplicados, [])

    def test_reidentificacion_no_fusiona_motocicletas_distintas(self):
        moto_1 = Objeto("moto", (100, 150, 300, 250), 0.9, 11)
        moto_2 = Objeto("moto", (100, 150, 300, 250), 0.9, 12)
        persona = Objeto("persona", (150, 40, 250, 220), 0.9, 21)
        cabeza_a = Objeto("sin_casco", (190, 50, 210, 80), 0.9, 101)
        cabeza_b = Objeto("sin_casco", (190, 50, 210, 80), 0.9, 102)
        a = Relacion(moto_1, persona, cabeza_a, "sin_casco")
        b = Relacion(moto_2, persona, cabeza_b, "sin_casco")
        detector = ConfirmadorEventosViales(2)
        detector.procesar(1, 3, [a])
        self.assertEqual(len(detector.procesar(2, 6, [a])), 1)
        detector.procesar(3, 9, [])
        detector.procesar(4, 12, [b])
        self.assertEqual(len(detector.procesar(5, 15, [b])), 1)


if __name__ == "__main__":
    unittest.main()
