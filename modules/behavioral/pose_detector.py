"""Prueba de análisis postural con un modelo YOLO Pose separado.

No detecta armas ni demuestra violencia o intención. La postura horizontal
solo crea un candidato de revisión si persiste en un track identificado.
"""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import cv2
from ultralytics import YOLO

from .events import DetectorEventos, Persona
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
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--min-fotogramas", type=int, default=6)
    parser.add_argument("--separacion-alertas", type=int, default=90)
    parser.add_argument("--max-fotogramas", type=int, default=0)
    parser.add_argument("--mostrar", action="store_true")
    parser.add_argument("--guardar-video", action="store_true")
    parser.add_argument(
        "--csv",
        default=str(RAIZ / "resultados" / "conductual" / "eventos.csv"),
        help="Registro de candidatos; no guarda imágenes por defecto",
    )
    args = parser.parse_args()

    if not (0 < args.conf <= 1) or args.imgsz < 64 or args.max_fotogramas < 0:
        parser.error("Parámetros de confianza, imagen o límite inválidos")
    detector = DetectorEventos(args.min_fotogramas, args.separacion_alertas)
    fuente = resolver_fuente(args.fuente)
    modelo = YOLO(args.modelo)
    if getattr(modelo, "task", None) != "pose":
        raise ValueError("El modelo conductual debe ser de tipo pose, no detect")
    camara = abrir_captura(fuente)

    destino = Path(args.csv).expanduser()
    destino.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not destino.exists() or destino.stat().st_size == 0
    escritor_video = None
    fotograma = 0
    try:
        if args.guardar_video:
            anchura = int(camara.get(cv2.CAP_PROP_FRAME_WIDTH))
            altura = int(camara.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = camara.get(cv2.CAP_PROP_FPS) or 20.0
            if anchura <= 0 or altura <= 0:
                raise ValueError("La fuente de video no informa dimensiones válidas")
            salida = destino.parent / (
                "resultado_posturas_"
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
                    ["timestamp_utc", "fuente", "fotograma", "track_id", "tipo", "estado"]
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
                alertas = detector.procesar(fotograma, personas)
                for alerta in alertas:
                    instante = datetime.now(timezone.utc).isoformat()
                    csv_writer.writerow(
                        [
                            instante,
                            args.fuente,
                            alerta.fotograma,
                            alerta.track_id,
                            alerta.tipo,
                            "candidato_para_revision",
                        ]
                    )
                    archivo.flush()
                    print(
                        f"[REVISAR] Fotograma {fotograma} ID {alerta.track_id}: "
                        "postura horizontal persistente (no caída confirmada)"
                    )

                if args.mostrar or args.guardar_video:
                    vista = resultado.plot()
                    if alertas:
                        cv2.putText(
                            vista, "REVISAR POSTURA - NO CONFIRMADA",
                            (12, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2,
                        )
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

    print(f"Fin. {fotograma} fotogramas. Archivo de eventos: {destino}")


if __name__ == "__main__":
    main()
