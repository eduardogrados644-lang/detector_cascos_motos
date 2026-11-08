r"""Detector de cascos, cabezas sin casco y placas: USB externa o movil por Wi-Fi.

Requisitos instalados en el proyecto: opencv-python, ultralytics, torch.
No usa la camara integrada automaticamente ni modifica modelos/datasets.

Ejemplos desde la raiz del proyecto (CMD):
    .venv\Scripts\python.exe camara_zoom.py --wifi "http://192.168.1.50:8080" --escena motos
    .venv\Scripts\python.exe camara_zoom.py --camara 1 --escena interior

IMPORTANTE: 192.168.1.50 es solo un EJEMPLO. Sustituir por la IP real
que muestra IP Webcam. El programa anade /video si no hay una ruta.

Teclas: + y - zoom; WASD mover; 0 reset; M cambia motos/interior;
        P foto original; E ejemplo para etiquetar; Q/Esc salir.
"""

import argparse
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import cv2

RAIZ = Path(__file__).resolve().parent
PESOS = RAIZ / "resultados" / "entrenamientos" / "videos_revisados" / "weights" / "best.pt"
CAPTURAS = RAIZ / "resultados" / "capturas_camara"
EJEMPLOS = RAIZ / "resultados" / "ejemplos_para_etiquetar"
NOMBRES = {"casco": "CON CASCO", "sin_casco": "SIN CASCO", "placa": "POSIBLE PLACA"}
COLORES = {"casco": (0, 190, 0), "sin_casco": (0, 70, 245), "placa": (230, 160, 0)}


def normalizar_url(texto):
    """Acepta URL base de IP Webcam o ruta completa del flujo MJPEG."""
    url = texto.strip()
    partes = urlsplit(url)
    if partes.scheme not in ("http", "https") or not partes.hostname:
        raise ValueError("--wifi necesita una URL http(s) valida mostrada por IP Webcam")
    if partes.path in ("", "/"):
        partes = partes._replace(path="/video")
    return urlunsplit(partes)


def abrir_wifi(url):
    """Recibir el MJPEG de IP Webcam; nunca abrir webcam interna por error."""
    try:
        captura = cv2.VideoCapture(
            url,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
            ],
        )
    except (TypeError, cv2.error):
        captura = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not captura.isOpened():
        captura.release()
        captura = cv2.VideoCapture(url)
    if not captura.isOpened():
        captura.release()
        raise RuntimeError(
            "No se pudo abrir la transmision del celular. "
            "Confirma que IP Webcam sigue activo y que la URL /video funciona."
        )
    captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Puede no ser admitido por FFmpeg.
    return captura


def abrir_usb(indice, ancho, alto, fps):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        cam = cv2.VideoCapture(indice, backend)
        if cam.isOpened():
            cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
            cam.set(cv2.CAP_PROP_FPS, fps)
            cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cam
        cam.release()
    raise RuntimeError(
        f"No se pudo abrir la camara USB indice {indice}. "
        "No se cambiara automaticamente a la integrada."
    )


def categoria(nombre):
    nombre = " ".join(str(nombre).lower().replace("_", " ").replace("-", " ").split())
    if nombre in {"with helmet", "helmet"}:
        return "casco"
    if nombre in {"without helmet", "no helmet", "nohelmet"}:
        return "sin_casco"
    if nombre in {"licence", "license", "license plate", "numberplate", "number plate"}:
        return "placa"
    return None


def interseccion(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    area = max(0, x2-x1) * max(0, y2-y1)
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    ab = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    divisor = aa + ab - area
    return area / divisor if divisor else 0.0


def quitar_duplicados(detecciones):
    seleccionadas = []
    for det in sorted(detecciones, key=lambda d: d[1], reverse=True):
        grupo = "placa" if det[0] == "placa" else "cabeza"
        if any(
            ("placa" if otro[0] == "placa" else "cabeza") == grupo
            and interseccion(det[2], otro[2]) > 0.43
            for otro in seleccionadas
        ):
            continue
        seleccionadas.append(det)
    return seleccionadas


def region_zoom(ancho, alto, zoom, centro_x, centro_y):
    w, h = max(1, round(ancho / zoom)), max(1, round(alto / zoom))
    x = max(0, min(ancho - w, round(centro_x*ancho - w/2)))
    y = max(0, min(alto - h, round(centro_y*alto - h/2)))
    return x, y, x+w, y+h


def region_detalle(ancho, alto, indice, zoom, cx, cy):
    w, h = max(1, round(ancho*.60)), max(1, round(alto*.60))
    if zoom > 1.05:
        x, y = round(cx*ancho-w/2), round(cy*alto-h/2)
    else:
        x = (0, ancho-w)[indice % 2]
        y = (0, alto-h)[(indice // 2) % 2]
    x = max(0, min(ancho-w, x))
    y = max(0, min(alto-h, y))
    return x, y, x+w, y+h


class Lector(threading.Thread):
    """Leer fotogramas continuamente para evitar colas atrasadas."""

    def __init__(self, cam, url=None):
        super().__init__(daemon=True, name="lectura_camara")
        self.cam = cam
        self.url = url
        self.parar = threading.Event()
        self.lock = threading.Lock()
        self.fotograma = None
        self.numero = 0
        self.instante = 0.0
        self.fps = 0.0
        self.error = None

    def obtener(self):
        with self.lock:
            return self.numero, self.fotograma, self.instante, self.fps

    def run(self):
        cuenta = 0
        marca = time.monotonic()
        fallos = 0
        reintentos = 0
        try:
            while not self.parar.is_set():
                ok, imagen = self.cam.read()
                if not ok or imagen is None:
                    fallos += 1
                    if fallos < 8:
                        self.parar.wait(0.05)
                        continue
                    if not self.url or reintentos >= 3:
                        raise RuntimeError("La fuente dejo de transmitir imagenes")
                    reintentos += 1
                    print(f"Wi-Fi interrumpido: reconectando ({reintentos}/3)...")
                    with self.lock:
                        self.fotograma = None
                    self.cam.release()
                    self.parar.wait(0.5)
                    if self.parar.is_set():
                        break
                    self.cam = abrir_wifi(self.url)
                    fallos = 0
                    continue
                fallos = 0
                reintentos = 0
                cuenta += 1
                ahora = time.monotonic()
                tiempo = ahora - marca
                with self.lock:
                    self.fotograma = imagen
                    self.numero += 1
                    self.instante = ahora
                    if tiempo >= 1.0:
                        self.fps = cuenta / tiempo
                if tiempo >= 1.0:
                    marca, cuenta = ahora, 0
        except (cv2.error, RuntimeError, ValueError, TypeError, OSError) as exc:
            self.error = exc
            self.parar.set()


class Estabilizador:
    def __init__(self):
        self.pistas = []

    def actualizar(self, nuevas, ahora):
        self.pistas = [p for p in self.pistas if ahora - p["visto"] < 1.5]
        usadas = set()
        for clase, conf, caja in sorted(nuevas, key=lambda d: d[1], reverse=True):
            grupo = "placa" if clase == "placa" else "cabeza"
            coincidencia, mejor = -1, .25
            for i, pista in enumerate(self.pistas):
                if i in usadas or pista["grupo"] != grupo:
                    continue
                similaridad = interseccion(caja, pista["caja"])
                if similaridad > mejor:
                    coincidencia, mejor = i, similaridad
            if coincidencia < 0:
                self.pistas.append({
                    "grupo": grupo, "caja": caja, "visto": ahora,
                    "historial": deque([clase], maxlen=5), "conf": conf,
                })
                coincidencia = len(self.pistas) - 1
            else:
                pista = self.pistas[coincidencia]
                pista["caja"] = caja
                pista["visto"] = ahora
                pista["historial"].append(clase)
                pista["conf"] = conf
            usadas.add(coincidencia)
        visibles = []
        for pista in self.pistas:
            if ahora - pista["visto"] > .7:
                continue
            clase, votos = Counter(pista["historial"]).most_common(1)[0]
            if votos >= (3 if clase == "placa" else 2):
                visibles.append((clase, pista["conf"], pista["caja"]))
        return quitar_duplicados(visibles)


class Detector(threading.Thread):
    def __init__(self, modelo, lector, opciones):
        super().__init__(daemon=True, name="yolo_cpu")
        self.modelo = modelo
        self.lector = lector
        self.opciones = opciones
        self.parar = threading.Event()
        self.lock = threading.Lock()
        self.vista_lock = threading.Lock()
        self.zoom = (1.0, .5, .5)
        self.placas = opciones.escena == "motos"
        self.ultimo = (0.0, [], 0.0, "global")
        self.error = None
        self.estabilizador = Estabilizador()
        self.cuadrante = 0
        self.turnos_zona = 0

    def configurar(self, zoom, cx, cy, placas):
        with self.vista_lock:
            self.zoom = (zoom, cx, cy)
            self.placas = placas

    def resultado(self):
        with self.lock:
            return self.ultimo

    def inferir(self, imagen, origen, dimensiones, mostrar_placas):
        ancho_total, alto_total = dimensiones
        ox, oy = origen
        salida = self.modelo.predict(
            source=imagen,
            imgsz=self.opciones.imgsz,
            conf=.20,
            iou=.45,
            agnostic_nms=True,
            max_det=35,
            device="cpu",
            verbose=False,
        )[0]
        if salida.boxes is None or not len(salida.boxes):
            return []
        posiciones = salida.boxes.xyxy.cpu().tolist()
        indices = salida.boxes.cls.int().cpu().tolist()
        confianzas = salida.boxes.conf.cpu().tolist()
        detecciones = []
        for puntos, indice, conf in zip(posiciones, indices, confianzas):
            clase = categoria(salida.names[int(indice)])
            if clase is None or (clase == "placa" and not mostrar_placas):
                continue
            limite = self.opciones.conf_placa if clase == "placa" else self.opciones.conf_casco
            if conf < limite:
                continue
            x1, y1, x2, y2 = (round(v) for v in puntos)
            x1, y1 = max(0, min(ancho_total-1, ox+x1)), max(0, min(alto_total-1, oy+y1))
            x2, y2 = max(0, min(ancho_total-1, ox+x2)), max(0, min(alto_total-1, oy+y2))
            bw, bh = x2-x1, y2-y1
            if bw < 5 or bh < 5:
                continue
            if clase == "placa":
                razon = bw / bh
                area = bw*bh/(ancho_total*alto_total)
                if not (1.3 <= razon <= 4.8 and .00004 <= area <= .08):
                    continue
            detecciones.append((clase, float(conf), (x1, y1, x2, y2)))
        return detecciones

    def run(self):
        ultimo_numero = 0
        ultimo_detalle = time.monotonic() - self.opciones.detalle_cada
        siguiente = 0.0
        try:
            while not self.parar.is_set():
                ahora = time.monotonic()
                numero, imagen, instante, _ = self.lector.obtener()
                if imagen is None or numero == ultimo_numero or ahora < siguiente:
                    self.parar.wait(.012)
                    continue
                if ahora - instante > 1.0:
                    self.parar.wait(.02)
                    continue
                ultimo_numero = numero
                siguiente = ahora + 1.0 / self.opciones.yolo_fps
                inicio = time.monotonic()
                alto, ancho = imagen.shape[:2]
                with self.vista_lock:
                    zoom, cx, cy = self.zoom
                    placas = self.placas
                detecciones = self.inferir(imagen, (0, 0), (ancho, alto), placas)
                modo = "global"
                if self.opciones.detalle and inicio - ultimo_detalle >= self.opciones.detalle_cada:
                    x1, y1, x2, y2 = region_detalle(ancho, alto, self.cuadrante, zoom, cx, cy)
                    region = imagen[y1:y2, x1:x2]
                    if region.size:
                        detecciones.extend(self.inferir(region, (x1, y1), (ancho, alto), placas))
                        modo = "global+zona"
                        self.turnos_zona += 1
                        if self.turnos_zona >= 2:
                            self.cuadrante += 1
                            self.turnos_zona = 0
                        ultimo_detalle = time.monotonic()
                detecciones = self.estabilizador.actualizar(
                    quitar_duplicados(detecciones), time.monotonic()
                )
                with self.lock:
                    self.ultimo = (time.monotonic(), detecciones, time.monotonic()-inicio, modo)
        except (cv2.error, RuntimeError, ValueError, TypeError, OSError, KeyError, IndexError) as exc:
            self.error = exc
            self.parar.set()


def dibujar(imagen, detecciones, region, mostrar_placas):
    x0, y0, x3, y3 = region
    alto, ancho = imagen.shape[:2]
    sx, sy = ancho / max(1, x3-x0), alto / max(1, y3-y0)
    for clase, conf, (x1, y1, x2, y2) in detecciones:
        if clase == "placa" and not mostrar_placas:
            continue
        x1, x2 = max(x1, x0), min(x2, x3)
        y1, y2 = max(y1, y0), min(y2, y3)
        if x1 >= x2 or y1 >= y2:
            continue
        a, b, c, d = (
            round((x1-x0)*sx), round((y1-y0)*sy),
            round((x2-x0)*sx), round((y2-y0)*sy),
        )
        color = COLORES[clase]
        cv2.rectangle(imagen, (a, b), (c, d), color, 2)
        etiqueta = f"{NOMBRES[clase]} {conf:.2f}"
        tx, ty = max(3, min(a, ancho-170)), max(19, b-7)
        cv2.putText(imagen, etiqueta, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(imagen, etiqueta, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, color, 2, cv2.LINE_AA)


def panel(imagen, lineas):
    alto, ancho = imagen.shape[:2]
    banda = min(alto, 12 + 23 * len(lineas))
    cv2.rectangle(imagen, (0, 0), (ancho, banda), (17, 17, 17), -1)
    for i, texto in enumerate(lineas):
        y = 22 + i*23
        if y < alto:
            cv2.putText(imagen, texto[:max(10, ancho//9)], (9, y),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (240, 240, 240), 1, cv2.LINE_AA)


def guardar_original(imagen, carpeta):
    carpeta.mkdir(parents=True, exist_ok=True)
    nombre = carpeta / datetime.now().strftime("camara_%Y%m%d_%H%M%S_%f.png")
    if not cv2.imwrite(str(nombre), imagen):
        raise OSError(f"No se pudo guardar la imagen: {nombre}")
    return nombre


def main():
    p = argparse.ArgumentParser(description="Detector con camara USB externa o celular IP Webcam Wi-Fi")
    fuente = p.add_mutually_exclusive_group()
    fuente.add_argument("--wifi", type=str, help="URL de IP Webcam; se agrega /video automaticamente")
    fuente.add_argument("--camara", type=int, help="Indice de webcam externa USB, por ejemplo 1")
    p.add_argument("--modelo", default=str(PESOS))
    p.add_argument("--ancho", type=int, default=1920, help="Resolucion solicitada solo a USB")
    p.add_argument("--alto", type=int, default=1080, help="Resolucion solicitada solo a USB")
    p.add_argument("--fps", type=int, default=30, help="FPS solicitados solo a USB")
    p.add_argument("--imgsz", type=int, choices=(416, 512, 640), default=416)
    p.add_argument("--yolo-fps", type=float, default=8)
    p.add_argument("--detalle-cada", type=float, default=.8)
    p.add_argument("--sin-detalle", action="store_true", help="Desactiva analisis adicional por zonas")
    p.add_argument("--escena", choices=("interior", "motos"), default="motos")
    p.add_argument("--conf-casco", type=float, default=.30)
    p.add_argument("--conf-placa", type=float, default=.72)
    p.add_argument("--zoom-max", type=float, default=1.5)
    p.add_argument("--vista-ancho", type=int, default=960)
    p.add_argument("--vista-alto", type=int, default=540)
    args = p.parse_args()
    args.detalle = not args.sin_detalle

    if args.camara is not None and not 0 <= args.camara <= 20:
        p.error("--camara debe estar entre 0 y 20")
    if not 1 <= args.fps <= 120 or not 1 <= args.yolo_fps <= 30:
        p.error("FPS fuera de rango")
    if not .4 <= args.detalle_cada <= 15:
        p.error("--detalle-cada debe estar entre 0.4 y 15")
    if not 1.0 <= args.zoom_max <= 3.0:
        p.error("--zoom-max debe estar entre 1 y 3")
    if not (.05 <= args.conf_casco <= .95 and .05 <= args.conf_placa <= .95):
        p.error("Confianzas fuera de rango")
    if not (320 <= args.ancho <= 4096 and 240 <= args.alto <= 2160):
        p.error("Resolucion USB fuera de rango")
    if not (320 <= args.vista_ancho <= 1920 and 240 <= args.vista_alto <= 1080):
        p.error("Tamano de vista fuera de rango")

    ruta_modelo = Path(args.modelo).expanduser()
    if not ruta_modelo.is_absolute():
        ruta_modelo = RAIZ / ruta_modelo
    if not ruta_modelo.is_file():
        raise FileNotFoundError(f"No existe el modelo entrenado: {ruta_modelo}")

    import torch
    from ultralytics import YOLO
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    red = YOLO(str(ruta_modelo))
    print("Modelo:", ruta_modelo)
    if args.wifi:
        url = normalizar_url(args.wifi)
        print("Fuente: CELULAR POR WI-FI (IP Webcam)")
        print("Resolucion y FPS se ajustan en la aplicacion IP Webcam.")
        cam = abrir_wifi(url)
    else:
        url = None
        indice = 1 if args.camara is None else args.camara
        print(f"Fuente: USB externa (indice {indice}); no se usa integrada automaticamente")
        cam = abrir_usb(indice, args.ancho, args.alto, args.fps)

    lector = Lector(cam, url)
    detector = Detector(red, lector, args)
    zoom, cx, cy = 1.0, .5, .5
    placas = args.escena == "motos"
    ventana = "YOLO - USB/Wi-Fi | +/- zoom | WASD | 0 | M | P | E | Q"
    fps_vista, numero_vista = 0.0, 0
    reloj_vista = time.monotonic()
    ultima_muestra, ultima_vista = -1, 0.0
    try:
        lector.start()
        detector.start()
        cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
        print("Controles: + / - zoom, WASD mover, 0 reiniciar, M modo, P foto, E ejemplo, Q salir")
        while True:
            ahora = time.monotonic()
            numero, fotograma, instante, fps_entrada = lector.obtener()
            if lector.error:
                raise RuntimeError(f"Error de la camara: {lector.error}") from lector.error
            if detector.error:
                raise RuntimeError(f"Error de YOLO: {detector.error}") from detector.error
            detector.configurar(zoom, cx, cy, placas)
            if fotograma is not None and ahora-instante <= 1.5:
                if (numero != ultima_muestra or ahora-ultima_vista > .3) and ahora-ultima_vista >= 1/24:
                    alto, ancho = fotograma.shape[:2]
                    region = region_zoom(ancho, alto, zoom, cx, cy)
                    x1, y1, x2, y2 = region
                    recorte = fotograma[y1:y2, x1:x2]
                    rh, rw = recorte.shape[:2]
                    escala = min(args.vista_ancho/rw, args.vista_alto/rh)
                    vista = cv2.resize(
                        recorte, (max(1, round(rw*escala)), max(1, round(rh*escala))),
                        interpolation=cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR,
                    )
                    momento, detecciones, duracion, modo = detector.resultado()
                    if ahora-momento < 1.2:
                        dibujar(vista, detecciones, region, placas)
                    numero_vista += 1
                    delta = ahora-reloj_vista
                    if delta >= 1:
                        fps_vista = numero_vista/delta
                        numero_vista, reloj_vista = 0, ahora
                    panel(vista, [
                        f"CAMARA {ancho}x{alto} {fps_entrada:.0f} FPS | VISTA {fps_vista:.0f} FPS | ZOOM {zoom:.2f}x",
                        f"YOLO {args.imgsz} | {modo} | ultimo {duracion*1000:.0f} ms",
                        f"MODO {'MOTOS + PLACAS' if placas else 'INTERIOR SIN PLACAS'} | + - WASD 0 M P E Q",
                    ])
                    cv2.imshow(ventana, vista)
                    ultima_muestra, ultima_vista = numero, ahora
            elif lector.parar.is_set():
                raise RuntimeError("La camara ya no entrega fotogramas")
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), ord("Q"), 27):
                break
            if tecla in (ord("+"), ord("=")):
                zoom = min(args.zoom_max, round(zoom+.25, 2))
            elif tecla in (ord("-"), ord("_")):
                zoom = max(1.0, round(zoom-.25, 2))
            elif tecla == ord("0"):
                zoom, cx, cy = 1.0, .5, .5
            elif tecla in (ord("w"), ord("W")):
                cy = max(0.0, cy-.06/zoom)
            elif tecla in (ord("s"), ord("S")):
                cy = min(1.0, cy+.06/zoom)
            elif tecla in (ord("a"), ord("A")):
                cx = max(0.0, cx-.06/zoom)
            elif tecla in (ord("d"), ord("D")):
                cx = min(1.0, cx+.06/zoom)
            elif tecla in (ord("m"), ord("M")):
                placas = not placas
                print("Modo:", "motos (placas activas)" if placas else "interior (placas ocultas)")
            elif tecla in (ord("p"), ord("P")) and fotograma is not None:
                print("Foto:", guardar_original(fotograma, CAPTURAS))
            elif tecla in (ord("e"), ord("E")) and fotograma is not None:
                print("Ejemplo sin etiquetar:", guardar_original(fotograma, EJEMPLOS))
            if numero == ultima_muestra:
                time.sleep(.005)
    finally:
        detector.parar.set()
        lector.parar.set()
        if detector.is_alive():
            detector.join(timeout=2.0)
        if lector.is_alive():
            lector.join(timeout=2.0)
        lector.cam.release()
        cv2.destroyAllWindows()
        print("Finalizado. Modelos y datasets originales sin modificaciones.")


if __name__ == "__main__":
    try:
        main()
    except (cv2.error, RuntimeError, ValueError, TypeError, OSError, ImportError) as exc:
        print("ERROR:", exc)
        raise SystemExit(1) from exc
