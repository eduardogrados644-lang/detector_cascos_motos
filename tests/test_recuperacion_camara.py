"""Comportamiento ante perdida de camara sin abrir dispositivos ni modelos reales."""
import unittest
from modules.camera_recovery import es_flujo_en_vivo, recuperar_flujo, limpiar_rastreadores


class CapturaFalsa:
    def __init__(self, lecturas):
        self.lecturas = list(lecturas)
        self.cerrada = False
    def read(self):
        return self.lecturas.pop(0) if self.lecturas else (False, None)
    def release(self):
        self.cerrada = True


class PruebasRecuperacion(unittest.TestCase):
    def test_fuentes_permitidas_y_video_local(self):
        for valor in (1, 3, "http://camara/video", "rtsp://camara/stream"):
            self.assertTrue(es_flujo_en_vivo(valor))
        for valor in (0, -1, "prueba.mp4", "C:/prueba.mp4", True):
            self.assertFalse(es_flujo_en_vivo(valor))

    def test_usb_se_reconecta_sin_cambiar_de_indice(self):
        vieja = CapturaFalsa([(False, None)])
        intento_fallido = CapturaFalsa([(False, None)])
        nueva = CapturaFalsa([(True, "imagen-recuperada")])
        cola = [intento_fallido, nueva]
        recibidas = []
        esperas = []
        def abrir(fuente):
            recibidas.append(fuente)
            return cola.pop(0)
        result, frame = recuperar_flujo(
            vieja, 1, abrir, dormir=esperas.append, espera=1,
            max_intentos=3)
        self.assertIs(result, nueva)
        self.assertEqual(frame, "imagen-recuperada")
        self.assertEqual(recibidas, [1, 1])
        self.assertEqual(esperas, [1, 1])
        self.assertTrue(vieja.cerrada)
        self.assertTrue(intento_fallido.cerrada)
        self.assertFalse(nueva.cerrada)

    def test_error_apertura_continua_misma_fuente(self):
        vieja = CapturaFalsa([])
        llamadas = []
        nueva = CapturaFalsa([(True, b"ok")])
        def abrir(fuente):
            llamadas.append(fuente)
            if len(llamadas) == 1:
                raise OSError("corte de USB")
            return nueva
        resultado, frame = recuperar_flujo(
            vieja, 2, abrir, dormir=lambda _: None, max_intentos=2)
        self.assertIs(resultado, nueva)
        self.assertEqual(frame, b"ok")
        self.assertEqual(llamadas, [2, 2])

    def test_archivo_no_se_reabre(self):
        vieja = CapturaFalsa([])
        llamadas = []
        res = recuperar_flujo(vieja, "video.mp4", llamadas.append)
        self.assertEqual(res, (None, None))
        self.assertFalse(vieja.cerrada)
        self.assertEqual(llamadas, [])

    def test_usuario_puede_cancelar(self):
        vieja = CapturaFalsa([])
        llamados = []
        res = recuperar_flujo(
            vieja, "http://camara/video", llamados.append,
            detener=lambda: True, dormir=lambda _: None)
        self.assertEqual(res, (None, None))
        self.assertTrue(vieja.cerrada)
        self.assertEqual(llamados, [])

    def test_no_bucle_infinito_si_se_configura_limite(self):
        vieja = CapturaFalsa([])
        llamadas = []
        def abrir(fuente):
            llamadas.append(fuente)
            raise RuntimeError("desconectada")
        res = recuperar_flujo(
            vieja, 1, abrir, max_intentos=3, dormir=lambda _: None)
        self.assertEqual(res, (None, None))
        self.assertEqual(llamadas, [1, 1, 1])

    def test_limpieza_rastreadores(self):
        class T:
            def __init__(self):
                self.n = 0
            def reset(self):
                self.n += 1
        class P:
            pass
        class M:
            pass
        modelo = M()
        modelo.predictor = P()
        modelo.predictor.trackers = [T(), T()]
        limpiar_rastreadores(modelo)
        self.assertEqual([t.n for t in modelo.predictor.trackers], [1, 1])
        limpiar_rastreadores(object())


if __name__ == "__main__":
    unittest.main()
