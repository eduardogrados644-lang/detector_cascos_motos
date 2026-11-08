r"""Prepara fotogramas de videos para ampliar un dataset YOLO.

IMPORTANTE: Las etiquetas que genera el modelo son BORRADORES.
Nunca se deben incorporar al entrenamiento sin revisión manual.

Proyecto: detector_cascos_motos (Windows / VS Code / Python 3.13)
Uso:
  .venv\Scripts\python.exe preparar_videos_entrenamiento.py
  .venv\Scripts\python.exe preparar_videos_entrenamiento.py --intervalo 2 --max-por-video 10
  .venv\Scripts\python.exe preparar_videos_entrenamiento.py --solo-extraer
"""

import argparse
import csv
import hashlib
import re
from pathlib import Path

import cv2

BASE = Path(__file__).resolve().parent
VIDEOS = BASE / "videos_prueba"
MODELO = BASE / "resultados" / "entrenamientos" / "refinado_completo" / "weights" / "best.pt"
DESTINO = BASE / "datos_videos_pendientes"
FORMATOS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
CLASES_ESPERADAS = {0: "With Helmet", 1: "Without Helmet", 2: "licence"}
UMBRALES = {0: 0.40, 1: 0.40, 2: 0.70}
COLORES = {0: (0, 170, 0), 1: (0, 80, 255), 2: (255, 160, 0)}


def nombre_seguro(valor: str) -> str:
    limpio = re.sub(r"[^a-zA-Z0-9_-]+", "_", valor).strip("_")
    return limpio[:35] or "video"


def guardar_jpg(ruta: Path, imagen) -> None:
    valido, datos = cv2.imencode(".jpg", imagen, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not valido:
        raise RuntimeError(f"No se pudo codificar la imagen: {ruta}")
    # Evita fallos de imwrite en algunas rutas Unicode de Windows.
    datos.tofile(str(ruta))


def crear_lote() -> Path:
    DESTINO.mkdir(parents=True, exist_ok=True)
    for numero in range(1, 10000):
        carpeta = DESTINO / f"lote_{numero:03d}"
        if not carpeta.exists():
            carpeta.mkdir()
            for nombre in ("images", "etiquetas_borrador", "vistas_previas"):
                (carpeta / nombre).mkdir()
            return carpeta
    raise RuntimeError("No hay nombres libres para nuevos lotes.")


def localizar_videos() -> list[Path]:
    if not VIDEOS.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de videos: {VIDEOS}")
    archivos = sorted(
        (ruta for ruta in VIDEOS.rglob("*")
         if ruta.is_file() and ruta.suffix.lower() in FORMATOS),
        key=lambda ruta: str(ruta).casefold(),
    )
    if not archivos:
        raise FileNotFoundError(
            "No se encontraron videos compatibles dentro de videos_prueba "
            "(incluidas sus subcarpetas)."
        )
    return archivos


def obtener_modelo(solo_extraer: bool):
    if solo_extraer:
        return None
    if not MODELO.is_file():
        raise FileNotFoundError(f"No se encontró el modelo para sugerir etiquetas: {MODELO}")

    from ultralytics import YOLO

    modelo = YOLO(str(MODELO))
    clases = {int(i): str(nombre) for i, nombre in modelo.names.items()}
    if clases != CLASES_ESPERADAS:
        raise ValueError(
            f"Las clases del modelo son {clases}; "
            f"se esperaban {CLASES_ESPERADAS}. No se generaron etiquetas."
        )
    return modelo


def sugerir_etiquetas(modelo, imagen):
    """Devuelve sugerencias en formato YOLO y una imagen marcada para revisión."""
    if modelo is None:
        return [], imagen.copy()

    resultado = modelo.predict(
        source=imagen, device="cpu", imgsz=416,
        conf=min(UMBRALES.values()), verbose=False,
    )[0]
    vista = imagen.copy()
    etiquetas = []

    if resultado.boxes is None:
        return etiquetas, vista

    cajas = resultado.boxes
    clases = cajas.cls.int().cpu().tolist()
    confianzas = cajas.conf.cpu().tolist()
    posiciones = cajas.xyxy.cpu().tolist()
    coordenadas = cajas.xywhn.cpu().tolist()

    for clase, confianza, xyxy, xywh in zip(
        clases, confianzas, posiciones, coordenadas
    ):
        if clase not in UMBRALES or confianza < UMBRALES[clase]:
            continue
        if not all(0.0 <= valor <= 1.0 for valor in xywh):
            continue
        if xywh[2] <= 0 or xywh[3] <= 0:
            continue
        etiquetas.append(f"{clase} " + " ".join(f"{valor:.6f}" for valor in xywh))
        x1, y1, x2, y2 = [round(valor) for valor in xyxy]
        color = COLORES[clase]
        cv2.rectangle(vista, (x1, y1), (x2, y2), color, 2)
        texto = f"{CLASES_ESPERADAS[clase]} {confianza:.2f}"
        cv2.putText(
            vista, texto, (max(0, x1), max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
        )

    return etiquetas, vista


def extraer_video(video: Path, carpeta: Path, modelo,
                  intervalo: float, maximo: int, escritor) -> int:
    captura = cv2.VideoCapture(str(video))
    if not captura.isOpened():
        print(f"[AVISO] No se pudo abrir el video: {video}")
        return 0

    fps = captura.get(cv2.CAP_PROP_FPS)
    if not 0 < fps <= 240:
        fps = 25.0
    salto = max(1, round(fps * intervalo))
    relativo = video.relative_to(VIDEOS)
    grupo = relativo.parts[0] if len(relativo.parts) > 1 else "Sin_grupo"
    firma = hashlib.sha256(relativo.as_posix().encode("utf-8")).hexdigest()[:10]
    prefijo = f"{nombre_seguro(grupo)}_{nombre_seguro(video.stem)}_{firma}"
    extraidas = 0
    numero_frame = 0

    try:
        while extraidas < maximo:
            correcto, fotograma = captura.read()
            if not correcto:
                break
            if numero_frame % salto == 0:
                alto, ancho = fotograma.shape[:2]
                if alto > 0 and ancho > 0:
                    # Mantiene la proporción y limita tamaño de archivos.
                    lado_maximo = max(alto, ancho)
                    if lado_maximo > 1280:
                        escala = 1280 / lado_maximo
                        fotograma = cv2.resize(
                            fotograma, (round(ancho * escala), round(alto * escala)),
                            interpolation=cv2.INTER_AREA,
                        )

                    nombre = f"{prefijo}_f{numero_frame:06d}"
                    guardar_jpg(carpeta / "images" / f"{nombre}.jpg", fotograma)
                    etiquetas, vista = sugerir_etiquetas(modelo, fotograma)
                    # NO usar etiquetas_borrador como labels de entrenamiento.
                    (carpeta / "etiquetas_borrador" / f"{nombre}.txt").write_text(
                        "\n".join(etiquetas) + ("\n" if etiquetas else ""),
                        encoding="utf-8",
                    )
                    guardar_jpg(carpeta / "vistas_previas" / f"{nombre}.jpg", vista)
                    escritor.writerow([
                        str(relativo), grupo, numero_frame,
                        f"{numero_frame / fps:.2f}", f"{nombre}.jpg",
                        len(etiquetas), "PENDIENTE_DE_REVISION",
                    ])
                    extraidas += 1
            numero_frame += 1
    finally:
        captura.release()
    return extraidas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extraer fotogramas y sugerir etiquetas para revisar antes de entrenar."
    )
    parser.add_argument("--intervalo", type=float, default=2.0,
                        help="Segundos entre fotogramas (predeterminado: 2).")
    parser.add_argument("--max-por-video", type=int, default=10,
                        help="Máximo de imágenes por video (predeterminado: 10).")
    parser.add_argument("--solo-extraer", action="store_true",
                        help="Extraer sin cargar YOLO ni generar sugerencias.")
    args = parser.parse_args()
    if not 0.2 <= args.intervalo <= 60:
        parser.error("--intervalo debe estar entre 0.2 y 60 segundos.")
    if not 1 <= args.max_por_video <= 100:
        parser.error("--max-por-video debe estar entre 1 y 100.")

    videos = localizar_videos()
    modelo = obtener_modelo(args.solo_extraer)
    lote = crear_lote()
    print(f"Videos encontrados: {len(videos)}")
    print(f"Salida: {lote}")
    print("Las anotaciones son BORRADORES, no etiquetas verificadas.\n")

    total = 0
    with (lote / "indice.csv").open("w", newline="", encoding="utf-8-sig") as f:
        escritor = csv.writer(f)
        escritor.writerow([
            "video", "grupo", "numero_fotograma", "segundo",
            "imagen", "cajas_sugeridas", "estado",
        ])
        for i, video in enumerate(videos, start=1):
            try:
                cantidad = extraer_video(
                    video, lote, modelo, args.intervalo, args.max_por_video,
                    escritor,
                )
                print(f"[{i}/{len(videos)}] {video.name}: {cantidad} fotogramas")
                total += cantidad
            except (OSError, ValueError, RuntimeError, cv2.error) as exc:
                print(f"[ERROR] {video.name}: {exc}")

    print(f"\nFinalizado: {total} fotogramas extraídos.")
    print(f"Originales para revisar: {lote / 'images'}")
    print(f"Previsualizaciones: {lote / 'vistas_previas'}")
    print(f"Etiquetas NO verificadas: {lote / 'etiquetas_borrador'}")
    print("IMPORTANTE: No entrenar todavía con estas etiquetas automáticas.")
    print("Hay que revisar y corregir todos los rectángulos y las clases.")
    print("El dataset original, los videos y los pesos no se modificaron.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"\nERROR: {error}")
        raise SystemExit(1) from error
