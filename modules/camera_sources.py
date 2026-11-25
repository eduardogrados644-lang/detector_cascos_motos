"""Fuentes de video compartidas por los DOS apartados, sin mezclar sus modelos.

La integrada solo puede seleccionarse en Cascos, nunca por defecto. Los índices de OpenCV
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


def validar_fuente_vial(fuente: str) -> str:
    """Permite la webcam integrada SOLAMENTE cuando Cascos la solicita."""
    if not isinstance(fuente, str):
        raise ValueError("La fuente debe ser texto")
    texto = fuente.strip()
    if texto in ("0", "camara:0"):
        return "camara:0"
    return validar_fuente(fuente)


def elegir_fuente_menu_vial() -> str | None:
    """Menu exclusivo de Cascos: la camara integrada se elige manualmente."""
    while True:
        print("\n--- ORIGEN DEL VIDEO: DETECCION DE CASCOS ---")
        print("1. Camara INTEGRADA de la laptop/PC (indice 0 habitual)")
        print("2. Camara USB EXTERNA")
        print("3. Camara de celular/tablet Android por Wi-Fi")
        print("4. Video de prueba del proyecto")
        print("5. Otro archivo de video o URL RTSP/MJPEG")
        print("0. Salir")
        eleccion = input("Selecciona la fuente: ").strip()
        if eleccion == "0":
            return None
        if eleccion == "1":
            print("Se intentara abrir indice 0. Verifica que sea la camara integrada.")
            return "camara:0"
        if eleccion == "2":
            valor = input("Indice de la camara USB externa (1 a 20): ").strip()
            if valor.isdecimal():
                try:
                    return indice_usb_externo(int(valor))
                except ValueError as error:
                    print("Opcion invalida:", error)
            else:
                print("Escribe un indice USB entre 1 y 20.")
            continue
        if eleccion == "3":
            print("Inicia IP Webcam en el celular/tablet y usa la misma red Wi-Fi.")
            url = input("URL de IP Webcam: ").strip()
            try:
                return normalizar_wifi(url)
            except ValueError as error:
                print("URL invalida:", error)
            continue
        if eleccion == "4":
            return "prueba_motos.mp4"
        if eleccion == "5":
            ruta = input("Ruta del video o URL RTSP/MJPEG: ").strip()
            try:
                return validar_fuente_vial(ruta)
            except ValueError as error:
                print("Fuente invalida:", error)
            continue
        print("Opcion invalida. Elige 1, 2, 3, 4, 5 o 0.")


def seleccionar_fuente_vial(
    fuente: str | None = None,
    usb: int | None = None,
    wifi: str | None = None,
    integrada: bool = False,
) -> str | None:
    """Seleccion especifica del apartado CASCOS; no afecta Intenciones."""
    if type(integrada) is not bool:
        raise ValueError("--integrada debe ser verdadero o falso")
    opciones = sum((fuente is not None, usb is not None, wifi is not None, integrada))
    if opciones > 1:
        raise ValueError("Elige solo una fuente: --integrada, --usb, --wifi o --fuente")
    if integrada:
        return "camara:0"
    if usb is not None:
        return indice_usb_externo(usb)
    if wifi is not None:
        return normalizar_wifi(wifi)
    if fuente is not None:
        return validar_fuente_vial(fuente)
    return elegir_fuente_menu_vial()


def abrir_captura_vial(fuente):
    """Permite indice 0 solo al modulo VIAL, sin fallback automatico.

    En laptop ligera baja captura integrada a 640x480; la inferencia sigue
    usando el tamano indicado por cada detector YOLO.
    """
    if type(fuente) is int and fuente == 0:
        from camara_zoom import abrir_usb
        print("Abriendo camara integrada elegida (indice 0, 640x480).")
        print("Verifica en pantalla que el indice 0 sea realmente tu webcam.")
        return abrir_usb(0, ancho=640, alto=480, fps=30)
    return abrir_captura(fuente)


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
