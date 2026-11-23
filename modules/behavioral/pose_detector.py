"""Prueba de análisis postural con un modelo YOLO Pose separado.

No detecta armas ni demuestra violencia o intención. La postura horizontal
solo crea un candidato de revisión si persiste en un track identificado.
"""
import argparse
import csv
import time
from math import isfinite
from datetime import datetime, timezone
from pathlib import Path

import cv2
from ultralytics import YOLO

from .events import DetectorEventos, Persona
from .proximity import DetectorProximidad
from .arm_motion import DetectorMovimientosBrazos
from modules.camera_sources import abrir_captura


RAIZ = Path(__file__).resolve().parents[2]


def resolver_fuente(texto: str):
    if texto.startswith("camara:"):
        indice = texto.partition(":")[2]
        if not indice.isdecimal():
            raise ValueError("Formato de cámara: camara:0")
        return int(indice)
    if texto.isdecimal():
        return int(texto)
    if texto.startswith(("rtsp://", "http://", "https://")):
        return texto
    archivo = Path(texto).expanduser()
    if not archivo.is_absolute():
        archivo = Path.cwd() / archivo
    if not archivo.is_file():
        archivo = RAIZ / texto
    if not archivo.is_file():
        raise FileNotFoundError(f"Archivo no encontrado: {texto}")
    return str(archivo.resolve())


def extraer_personas(resultado) -> list[Persona]:
    if resultado.boxes is None or resultado.keypoints is None:
        return []
    cajas = resultado.boxes.xyxy.cpu().tolist()
    ids = (
        resultado.boxes.id.int().cpu().tolist()
        if resultado.boxes.id is not None
        else [None] * len(cajas)
    )
    puntos = resultado.keypoints.xy.cpu().tolist()
    confianza = (
        resultado.keypoints.conf.cpu().tolist()
        if resultado.keypoints.conf is not None
        else [[0.0] * len(p) for p in puntos]
    )
    personas = []
    for indice in range(min(len(cajas), len(puntos), len(ids))):
        datos = tuple(
            (float(xy[0]), float(xy[1]), float(p))
            for xy, p in zip(puntos[indice], confianza[indice])
        )
        personas.append(
            Persona(
                ids[indice],
                tuple(float(v) for v in cajas[indice]),
                datos,
            )
        )
    return personas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Análisis postural experimental, independiente de cascos"
    )
    parser.add_argument("--fuente", default="prueba_motos.mp4", help="video, camara:1 (USB externa), IP Webcam o RTSP")
    parser.add_argument("--modelo", default="yolo11n-pose.pt")
    parser.add_argument("--modo", choices=("posturas", "proximidad", "brazos"), default="posturas")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--min-fotogramas", type=int, default=6)
    parser.add_argument("--separacion-alertas", type=int, default=90)
    parser.add_argument("--max-fotogramas", type=int, default=0)
    parser.add_argument("--mostrar", action="store_true")
    parser.add_argument("--guardar-video", action="store_true")
    parser.add_argument(
        "--guardar-capturas", action="store_true",
        help="Guardar sólo fotogramas con avisos (máximo 20 por sesión)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Registro de candidatos; no guarda imágenes por defecto",
    )
    args = parser.parse_args()

    if not (0 < args.conf <= 1) or args.imgsz < 64 or args.max_fotogramas < 0:
        parser.error("Parámetros de confianza, imagen o límite inválidos")
    detector = (
        DetectorProximidad(args.min_fotogramas, args.separacion_alertas)
        if args.modo == "proximidad"
        else DetectorMovimientosBrazos(
            min_elevados=max(2, args.min_fotogramas),
            separacion_alertas=args.separacion_alertas,
        ) if args.modo == "brazos"
        else DetectorEventos(args.min_fotogramas, args.separacion_alertas)
    )
    fuente = resolver_fuente(args.fuente)
    modelo = YOLO(args.modelo)
    if getattr(modelo, "task", None) != "pose":
        raise ValueError("El modelo conductual debe ser de tipo pose, no detect")
    camara = abrir_captura(fuente)

    nombre_csv = {"posturas": "eventos.csv",
                  "proximidad": "eventos_proximidad.csv",
                  "brazos": "eventos_brazos.csv"}[args.modo]
    destino = Path(args.csv if args.csv else RAIZ / "resultados" / "conductual" / nombre_csv).expanduser()
    fps_fuente = camara.get(cv2.CAP_PROP_FPS)
    if not isfinite(fps_fuente) or fps_fuente <= 0:
        fps_fuente = 20.0
    es_archivo_local = isinstance(fuente, str) and Path(fuente).is_file()
    inicio_reloj = time.monotonic()
    destino.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not destino.exists() or destino.stat().st_size == 0
    escritor_video = None
    max_capturas = 20  # Tope de privacidad/espacio por ejecución.
    capturas_guardadas = 0
    carpeta_capturas = None
    fotograma = 0
    try:
        if args.guardar_video:
            anchura = int(camara.get(cv2.CAP_PROP_FRAME_WIDTH))
            altura = int(camara.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = camara.get(cv2.CAP_PROP_FPS) or 20.0
            if anchura <= 0 or altura <= 0:
                raise ValueError("La fuente de video no informa dimensiones válidas")
            salida = destino.parent / (
                {"posturas": "resultado_posturas_",
                 "proximidad": "resultado_proximidad_",
                 "brazos": "resultado_brazos_"}[args.modo]
                + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                + ".mp4"
            )
            escritor_video = cv2.VideoWriter(
                str(salida),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (anchura, altura),
            )
            if not escritor_video.isOpened():
                raise OSError(f"No se pudo crear el video: {salida}")

        with destino.open("a", encoding="utf-8", newline="") as archivo:
            csv_writer = csv.writer(archivo)
            if nuevo:
                csv_writer.writerow(
                    ["timestamp_utc", "fuente", "fotograma", "track_ids" if args.modo == "proximidad" else "track_id", "tipo", "estado"]
                )

            while True:
                correcto, imagen = camara.read()
                if not correcto:
                    break
                fotograma += 1
                resultado = modelo.track(
                    imagen,
                    persist=True,
                    tracker="bytetrack.yaml",
                    imgsz=args.imgsz,
                    conf=args.conf,
                    device="cpu",
                    verbose=False,
                )[0]
                personas = extraer_personas(resultado)
                if args.modo == "brazos":
                    # Usar reloj de video para archivos y monotónico para cámaras.
                    if es_archivo_local:
                        pos_ms = camara.get(cv2.CAP_PROP_POS_MSEC)
                        tiempo_s = (pos_ms / 1000.0 if isfinite(pos_ms) and pos_ms > 0
                                    else fotograma / fps_fuente)
                    else:
                        tiempo_s = time.monotonic() - inicio_reloj
                    alertas = detector.procesar(fotograma, personas, tiempo_s)
                else:
                    alertas = detector.procesar(fotograma, personas)
                for alerta in alertas:
                    instante = datetime.now(timezone.utc).isoformat()
                    csv_writer.writerow(
                        [
                            instante,
                            args.fuente,
                            alerta.fotograma,
                            f"{alerta.track_ids[0]}|{alerta.track_ids[1]}" if args.modo == "proximidad" else alerta.track_id,
                            (f"{alerta.tipo}:{alerta.brazo}" if args.modo == "brazos"
                             else alerta.tipo),
                            "candidato_para_revision",
                        ]
                    )
                    archivo.flush()
                    if args.modo == "proximidad":
                        print(f"[REVISAR PROXIMIDAD] Fotograma {fotograma}; IDs {alerta.track_ids}: cercanía persistente, NO pelea confirmada")
                    elif args.modo == "brazos":
                        print(f"[REVISAR BRAZO] Fotograma {fotograma}; ID {alerta.track_id}, "
                              f"{alerta.brazo}: {alerta.tipo} (NO agresión confirmada)")
                    else:
                        print(f"[REVISAR] Fotograma {fotograma} ID {alerta.track_id}: postura horizontal persistente (no caída confirmada)")

                # Las capturas sólo se producen si el usuario las solicita y
                # realmente existen avisos. No guardamos todas las personas.
                capturar = (
                    args.guardar_capturas and bool(alertas)
                    and capturas_guardadas < max_capturas
                )
                if args.mostrar or args.guardar_video or capturar:
                    vista = resultado.plot()
                    if alertas:
                        cv2.putText(
                            vista, ({"posturas": "REVISAR POSTURA - NO CONFIRMADA",
                                   "proximidad": "PROXIMIDAD - NO VIOLENCIA CONFIRMADA",
                                   "brazos": "MOVIMIENTO DE BRAZO - REVISAR"}[args.modo]),
                            (12, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2,
                        )
                    if capturar:
                        if carpeta_capturas is None:
                            carpeta_capturas = destino.parent / (
                                f"capturas_{args.modo}_"
                                + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                            )
                            carpeta_capturas.mkdir(parents=True, exist_ok=False)
                        # Metadatos visibles: sólo IDs temporales y observaciones.
                        for i, evento in enumerate(alertas[:3]):
                            if args.modo == "brazos":
                                detalle = (
                                    f"ID {evento.track_id} {evento.brazo} {evento.tipo}"
                                )
                            elif args.modo == "proximidad":
                                detalle = (
                                    f"IDs {evento.track_ids[0]}-{evento.track_ids[1]} "
                                    "proximidad_persistente"
                                )
                            else:
                                detalle = f"ID {evento.track_id} {evento.tipo}"
                            cv2.putText(
                                vista, detalle[:90], (12, 65 + i * 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 0, 255), 2,
                            )
                        nombre = carpeta_capturas / f"fotograma_{fotograma:06d}.jpg"
                        if not cv2.imwrite(str(nombre), vista):
                            raise OSError(f"No se pudo guardar la captura: {nombre}")
                        capturas_guardadas += 1
                        print(f"[CAPTURA PARA REVISIÓN] {nombre}")
                    if escritor_video is not None:
                        escritor_video.write(vista)
                    if args.mostrar:
                        cv2.imshow("Analisis conductual experimental", vista)
                        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                            break
                if fotograma % 100 == 0:
                    print(f"Fotogramas procesados: {fotograma}")
                if args.max_fotogramas and fotograma >= args.max_fotogramas:
                    break
    except KeyboardInterrupt:
        print("Ejecución interrumpida.")
    finally:
        camara.release()
        if escritor_video is not None:
            escritor_video.release()
        if args.mostrar:
            cv2.destroyAllWindows()

    print(f"Fin. Modo {args.modo}; {fotograma} fotogramas. Archivo de eventos: {destino}")
    if args.guardar_capturas:
        if carpeta_capturas is None:
            print("No hubo eventos nuevos para guardar capturas.")
        else:
            print(f"Capturas para revisión: {capturas_guardadas} en {carpeta_capturas}")


if __name__ == "__main__":
    main()
