"""Fuentes de video compartidas por los DOS apartados, sin mezclar sus modelos.

No se selecciona automáticamente la cámara integrada. Los índices de OpenCV
varían entre equipos; el usuario debe identificar su dispositivo USB externo.
La transmisión IP Webcam funciona para un celular/tablet Android en la misma red.
"""
from urllib.parse import urlsplit, urlunsplit


def indice_usb_externo(indice: int) -> str:
    """Solicita un índice explícito, evitando el índice 0 más común en laptops."""
    if isinstance(indice, bool) or not isinstance(indice, int) or not 1 <= indice <= 20:
        raise ValueError(
            "Indica el índice de tu cámara USB externa (1 a 20). "
            "No se selecciona automáticamente la cámara integrada."
        )
    return f"camara:{indice}"


def normalizar_wifi(url: str) -> str:
    """Acepta la URL base IP Webcam o la ruta exacta de su flujo MJPEG."""
    if not isinstance(url, str):
        raise ValueError("La URL de la cámara debe ser un texto.")
    partes = urlsplit(url.strip())
    if partes.scheme not in ("http", "https") or not partes.hostname:
        raise ValueError(
            "Utiliza la URL http(s) que muestra IP Webcam en el celular/tablet, "
            "por ejemplo http://192.168.1.50:8080 (solo ejemplo)."
        )
    if partes.path in ("", "/"):
        partes = partes._replace(path="/video")
    return urlunsplit(partes)


def validar_fuente(fuente: str) -> str:
    """Evita abrir la webcam integrada por una fuente numérica implícita."""
    if not isinstance(fuente, str) or not fuente.strip():
        raise ValueError("La fuente de video es obligatoria")
    texto = fuente.strip()
    if texto.isdecimal():
        return indice_usb_externo(int(texto))
    if texto.startswith("camara:"):
        indice = texto.partition(":")[2]
        if not indice.isdecimal():
            raise ValueError("Formato inválido; usa camara:1 para una cámara USB")
        return indice_usb_externo(int(indice))
    return texto


def elegir_fuente_menu() -> str | None:
    """Pregunta la fuente sin abrir ninguna cámara al recibir opción inválida."""
    while True:
        print("\n--- ORIGEN DEL VIDEO ---")
        print("1. Video de prueba del proyecto")
        print("2. Cámara USB EXTERNA conectada a la PC")
        print("3. Cámara de CELULAR o TABLET Android (IP Webcam por Wi-Fi)")
        print("4. Archivo de video u otra URL RTSP/MJPEG")
        print("0. Volver / salir")
        eleccion = input("Selecciona la fuente: ").strip()
        if eleccion == "0":
            return None
        if eleccion == "1":
            return "prueba_motos.mp4"
        if eleccion == "2":
            print(
                "Conecta la cámara USB externa. En Windows, el índice depende "
                "de las cámaras instaladas; 1 es un ejemplo, NO una garantía."
            )
            valor = input("Índice de la cámara USB EXTERNA (1 a 20): ").strip()
            if not valor.isdecimal():
                raise ValueError("El índice USB debe ser un número entre 1 y 20")
            return indice_usb_externo(int(valor))
        if eleccion == "3":
            print(
                "En el celular/tablet Android abre IP Webcam, inicia el servidor "
                "y conecta ambos dispositivos a la misma red Wi-Fi."
            )
            print("Copia aquí la dirección que muestra la aplicación.")
            direccion = input("URL de IP Webcam: ").strip()
            return normalizar_wifi(direccion)
        if eleccion == "4":
            ruta = input("Ruta del video o URL RTSP/MJPEG: ").strip()
            return validar_fuente(ruta)
        print("Opción no válida. Escribe 1, 2, 3, 4 o 0 y pulsa Enter.")


def seleccionar_fuente(
    fuente: str | None = None,
    usb: int | None = None,
    wifi: str | None = None,
) -> str | None:
    """Selecciona la fuente exactamente pedida, sin cambiar a otra cámara."""
    opciones = sum(valor is not None for valor in (fuente, usb, wifi))
    if opciones > 1:
        raise ValueError("Selecciona sólo una opción: --fuente, --usb o --wifi")
    if usb is not None:
        return indice_usb_externo(usb)
    if wifi is not None:
        return normalizar_wifi(wifi)
    if fuente is not None:
        return validar_fuente(fuente)
    return elegir_fuente_menu()


def abrir_captura(fuente):
    """Abre una fuente solicitada SIN recurrir a otra cámara si falla.

    Aprovecha el soporte USB DirectShow/MSMF y Wi-Fi MJPEG de camara_zoom.py.
    La fuente ya debe estar explícitamente seleccionada por el iniciador.
    """
    import cv2

    if isinstance(fuente, int):
        from camara_zoom import abrir_usb

        print(
            f"Abriendo índice USB elegido: {fuente}. "
            "Verifica visualmente que sea la cámara externa."
        )
        return abrir_usb(fuente, ancho=1280, alto=720, fps=30)
    if isinstance(fuente, str) and fuente.startswith(("http://", "https://")):
        from camara_zoom import abrir_wifi

        return abrir_wifi(fuente)
    captura = cv2.VideoCapture(fuente)
    if not captura.isOpened():
        captura.release()
        raise OSError(f"No se pudo abrir la fuente solicitada: {fuente}")
    return captura
