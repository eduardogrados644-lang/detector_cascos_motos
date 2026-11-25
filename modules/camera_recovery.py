"""Recuperacion de capturas EN VIVO sin cambiar de camara ni reabrir videos.

Una lectura fallida puede ser temporal. Solo se vuelve a abrir el dispositivo
USB/IP que selecciono el usuario, nunca se prueba la webcam integrada.
"""
import time


def es_flujo_en_vivo(fuente):
    """Indice USB externo explicito o URL RTSP/MJPEG, nunca video local."""
    return ((type(fuente) is int and 1 <= fuente <= 20)
            or (isinstance(fuente, str) and fuente.startswith(
                ("http://", "https://", "rtsp://", "rtsps://"))))


def recuperar_flujo(
    camara, fuente, abrir_captura, *, detener=None, espera=1.0,
    max_intentos=0, dormir=time.sleep,
):
    """Devuelve (captura_nueva, primera_imagen) o (None, None).

    max_intentos=0 significa reintentar sin limite hasta detener la sesion.
    Los videos locales no se reabren al acabar. No hay cambio de fuente.
    """
    if not es_flujo_en_vivo(fuente):
        return None, None
    if espera < 0 or max_intentos < 0:
        raise ValueError("Parametros de recuperacion incorrectos")
    camara.release()
    intento = 0
    print("[CAMARA] Senal interrumpida; reconectando la misma fuente. Ctrl+C para salir.")
    while max_intentos == 0 or intento < max_intentos:
        if detener is not None and detener():
            return None, None
        dormir(espera)
        if detener is not None and detener():
            return None, None
        intento += 1
        nueva = None
        try:
            nueva = abrir_captura(fuente)
            ok, imagen = nueva.read()
            if ok and imagen is not None:
                print(f"[CAMARA] Senal restaurada tras {intento} intento(s).")
                return nueva, imagen
        except (OSError, RuntimeError, ValueError) as exc:
            if intento == 1 or intento % 10 == 0:
                print(f"[CAMARA] Reintento {intento}: {exc}")
        if nueva is not None:
            nueva.release()
        if intento % 10 == 0:
            print(f"[CAMARA] Esperando reconexion de {fuente}; intento {intento}.")
    print("[CAMARA] Limite de reintentos alcanzado.")
    return None, None


def limpiar_rastreadores(modelo):
    """Reinicia IDs de seguimiento tras un corte para no mezclar sesiones."""
    predictor = getattr(modelo, "predictor", None)
    for tracker in getattr(predictor, "trackers", ()) or ():
        reset = getattr(tracker, "reset", None)
        if callable(reset):
            reset()
