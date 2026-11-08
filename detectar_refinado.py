"""Detector YOLO de cascos y placas para imágenes y vídeos.

No modifica los modelos entrenados ni utiliza servicios externos.
"""

import argparse
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
MODELO_PREDETERMINADO = (
    BASE / "resultados" / "entrenamientos" / "refinado_completo" / "weights" / "best.pt"
)
SALIDAS = BASE / "resultados" / "deteccion_refinada"
IMAGENES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEOS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def clase_normalizada(nombre):
    """Clasifica nombres de las dos versiones de YOLO del proyecto."""
    texto = str(nombre).lower().replace("_", " ").replace("-", " ").strip()
    texto = " ".join(texto.split())
    if texto in {"with helmet", "helmet"}:
        return "casco"
    if texto in {"without helmet", "no helmet", "nohelmet"}:
        return "sin_casco"
    if texto in {"licence", "license", "license plate", "numberplate", "number plate"}:
        return "placa"
    return None


def resolver_ruta(texto):
    if texto.startswith("camara:"):
        indice = texto.split(":", 1)[1]
        if not indice.isdecimal():
            raise ValueError("La cámara debe indicarse como camara:0 o camara:1.")
        return int(indice)
    ruta = Path(texto).expanduser()
    if not ruta.is_absolute():
        ruta = BASE / ruta
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró la entrada: {ruta}")
    return ruta


def obtener_archivos(fuente):
    if isinstance(fuente, int):
        return [fuente]
    if fuente.is_file():
        return [fuente]
    if not fuente.is_dir():
        raise ValueError("La entrada no es un archivo ni una carpeta válida.")
    archivos = sorted(
        (x for x in fuente.iterdir() if x.is_file() and x.suffix.lower() in IMAGENES | VIDEOS),
        key=lambda x: x.name.lower(),
    )
    if not archivos:
        raise ValueError("La carpeta no contiene imágenes o videos compatibles.")
    return archivos


def nueva_carpeta_salida():
    SALIDAS.mkdir(parents=True, exist_ok=True)
    for numero in range(1, 10000):
        carpeta = SALIDAS / f"analisis_{numero:03d}"
        if not carpeta.exists():
            carpeta.mkdir()
            return carpeta
    raise RuntimeError("No hay un nombre de carpeta de salida disponible.")


def extraer_detecciones(resultado, alto, ancho, conf_casco, conf_placa):
    """Filtra resultados: umbrales distintos y geometría básica de placas."""
    detecciones = []
    cajas = resultado.boxes
    if cajas is None or len(cajas) == 0:
        return detecciones
    xyxy = cajas.xyxy.cpu().tolist()
    ids_clase = cajas.cls.int().cpu().tolist()
    confianzas = cajas.conf.cpu().tolist()
    ids_track = cajas.id.int().cpu().tolist() if cajas.id is not None else [None] * len(xyxy)

    for coords, id_clase, confianza, id_track in zip(xyxy, ids_clase, confianzas, ids_track):
        clase = clase_normalizada(resultado.names[int(id_clase)])
        if clase is None:
            continue
        minimo = conf_placa if clase == "placa" else conf_casco
        if confianza < minimo:
            continue
        x1, y1, x2, y2 = (round(v) for v in coords)
        x1, y1 = max(0, min(ancho, x1)), max(0, min(alto, y1))
        x2, y2 = max(0, min(ancho, x2)), max(0, min(alto, y2))
        bw, bh = x2 - x1, y2 - y1
        if bw < 4 or bh < 4:
            continue
        if clase == "placa":
            proporcion = bw / bh
            area_relativa = bw * bh / (ancho * alto)
            # Parámetros exploratorios: no certifican que sea una matrícula real.
            if not (1.15 <= proporcion <= 5.0 and 0.0001 <= area_relativa <= 0.07):
                continue
        detecciones.append((clase, confianza, (x1, y1, x2, y2), id_track))
    return detecciones


def dibujar(frame, detecciones, historial, numero_frame, seguimiento):
    colores = {"casco": (0, 170, 0), "sin_casco": (0, 90, 240), "placa": (230, 160, 0)}
    titulos = {"casco": "CON CASCO", "sin_casco": "SIN CASCO", "placa": "POSIBLE PLACA"}
    vistos = set()

    for clase, conf, (x1, y1, x2, y2), id_track in detecciones:
        color = colores[clase]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        sufijo = f" ID:{id_track}" if id_track is not None else ""
        etiqueta = f"{titulos[clase]} {conf:.2f}{sufijo}"
        cv2.putText(frame, etiqueta, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
        if seguimiento and clase != "placa" and id_track is not None:
            clave = (clase, int(id_track))
            historial[clave] = ((x1, y1, x2, y2), numero_frame)
            vistos.add(clave)

    # Persistencia VISUAL de 2 fotogramas en la última posición conocida.
    # No constituye una detección nueva ni se aplica a las matrículas.
    if seguimiento:
        for clave, (caja, ultimo) in list(historial.items()):
            distancia = numero_frame - ultimo
            if distancia > 2:
                del historial[clave]
            elif clave not in vistos and distancia > 0:
                x1, y1, x2, y2 = caja
                cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 1)
                cv2.putText(frame, "ULTIMA POSICION", (x1, max(18, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (150, 150, 150), 1)
    return frame


def procesar_imagen(archivo, modelo, salida, configuracion, contador):
    imagen = cv2.imread(str(archivo))
    if imagen is None:
        print(f"[AVISO] No se pudo leer la imagen: {archivo}")
        return
    resultado = modelo.predict(imagen, imgsz=416, conf=min(configuracion.conf_casco, configuracion.conf_placa),
                               device="cpu", verbose=False)[0]
    alto, ancho = imagen.shape[:2]
    detecciones = extraer_detecciones(resultado, alto, ancho,
                                     configuracion.conf_casco, configuracion.conf_placa)
    contador.update(d[0] for d in detecciones)
    anotada = dibujar(imagen.copy(), detecciones, {}, 1, False)
    destino = salida / archivo.name
    if not cv2.imwrite(str(destino), anotada):
        raise RuntimeError(f"No se pudo guardar la imagen: {destino}")
    print(f"[IMAGEN] {archivo.name}: {len(detecciones)} detecciones")
    if configuracion.mostrar:
        cv2.imshow("Detector de cascos y placas", anotada)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def crear_escritor(destino, fps, tamano):
    escritor = cv2.VideoWriter(str(destino), cv2.VideoWriter_fourcc(*"mp4v"), fps, tamano)
    if not escritor.isOpened():
        escritor.release()
        destino = destino.with_suffix(".avi")
        escritor = cv2.VideoWriter(str(destino), cv2.VideoWriter_fourcc(*"XVID"), fps, tamano)
    if not escritor.isOpened():
        raise RuntimeError("No se pudo iniciar el guardado de vídeo (MP4/AVI).")
    return escritor, destino


def procesar_video(fuente, modelo_ruta, salida, configuracion, contador):
    # Cada vídeo carga un tracker nuevo: no se transfieren IDs entre grabaciones.
    modelo = YOLO(str(modelo_ruta))
    captura = cv2.VideoCapture(fuente if isinstance(fuente, int) else str(fuente))
    if not captura.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {fuente}")

    fps = captura.get(cv2.CAP_PROP_FPS)
    if not (1 <= fps <= 120):
        fps = 30.0
    numero = 0
    historial = {}
    escritor = None
    nombre = (f"camara_{fuente}" if isinstance(fuente, int)
              else f"{fuente.stem}_{fuente.suffix.lower().lstrip(chr(46))}")
    destino = salida / f"{nombre}_analizado.mp4"
    print(f"[VIDEO] Procesando: {nombre}")
    try:
        while True:
            ok, imagen = captura.read()
            if not ok:
                break
            numero += 1
            alto, ancho = imagen.shape[:2]
            resultados = modelo.track(
                imagen, persist=True, tracker="bytetrack.yaml", imgsz=416,
                conf=min(configuracion.conf_casco, configuracion.conf_placa),
                device="cpu", verbose=False,
            )
            detecciones = extraer_detecciones(resultados[0], alto, ancho,
                                             configuracion.conf_casco, configuracion.conf_placa)
            contador.update(d[0] for d in detecciones)
            anotada = dibujar(imagen.copy(), detecciones, historial, numero, True)
            if escritor is None:
                escritor, destino = crear_escritor(destino, fps, (ancho, alto))
            escritor.write(anotada)
            if configuracion.mostrar:
                cv2.imshow("Detector de cascos y placas", anotada)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Detenido desde la ventana.")
                    break
            if numero % 50 == 0:
                print(f"  Fotogramas procesados: {numero}")
    finally:
        captura.release()
        if escritor is not None:
            escritor.release()
        if configuracion.mostrar:
            cv2.destroyAllWindows()
    print(f"[VIDEO] {numero} fotogramas. Guardado en: {destino if numero else 'sin salida'}")


def main():
    parser = argparse.ArgumentParser(description="Detección refinada de cascos y posibles matrículas")
    parser.add_argument("entrada", nargs="?", default="helmet.mp4",
                        help="Ruta de imagen, vídeo, carpeta o camara:0")
    parser.add_argument("--modelo", default=str(MODELO_PREDETERMINADO), help="Ruta de pesos .pt")
    parser.add_argument("--conf-casco", type=float, default=0.30, help="Confianza mínima para cascos")
    parser.add_argument("--conf-placa", type=float, default=0.65, help="Confianza mínima para matrículas")
    parser.add_argument("--mostrar", action="store_true", help="Mostrar una ventana OpenCV")
    opciones = parser.parse_args()

    try:
        if not (0 < opciones.conf_casco <= 1 and 0 < opciones.conf_placa <= 1):
            raise ValueError("Las confianzas deben estar entre 0 (exclusivo) y 1.")
        ruta_modelo = Path(opciones.modelo).expanduser()
        if not ruta_modelo.is_absolute():
            ruta_modelo = BASE / ruta_modelo
        if not ruta_modelo.is_file():
            raise FileNotFoundError(f"No se encontró el modelo: {ruta_modelo}")
        fuentes = obtener_archivos(resolver_ruta(opciones.entrada))
        salida = nueva_carpeta_salida()
        contador = Counter()
        modelo_imagenes = None
        print(f"Modelo: {ruta_modelo}\nArchivos/fuentes: {len(fuentes)}\nSalida: {salida}")
        for fuente in fuentes:
            if isinstance(fuente, int) or fuente.suffix.lower() in VIDEOS:
                procesar_video(fuente, ruta_modelo, salida, opciones, contador)
            else:
                if modelo_imagenes is None:
                    modelo_imagenes = YOLO(str(ruta_modelo))
                procesar_imagen(fuente, modelo_imagenes, salida, opciones, contador)
        print("\nResumen de detecciones observadas (no objetos únicos):")
        for clave in ("casco", "sin_casco", "placa"):
            print(f"  {clave}: {contador[clave]}")
        print(f"Resultados: {salida}")
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1) from error
    except KeyboardInterrupt:
        print("\n[INFO] Ejecución interrumpida.")


if __name__ == "__main__":
    main()
