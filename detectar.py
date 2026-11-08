
import argparse
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

RUTA_BASE = Path(__file__).resolve().parent
RUTA_MODELO = RUTA_BASE / "best.pt"
RUTA_RESULTADOS = RUTA_BASE / "resultados"


def obtener_fuente(entrada: str):
    """Identifica si la entrada es una cámara, video o imagen."""

    if entrada.startswith("camara:"):
        indice = entrada.split(":", 1)[1]

        if not indice.isdecimal():
            raise ValueError("El índice de cámara debe ser numérico.")

        return int(indice), True

    if entrada.startswith(("rtsp://", "http://", "https://")):
        return entrada, True

    ruta = Path(entrada).expanduser()

    if not ruta.is_absolute():
        ruta = RUTA_BASE / ruta

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró la entrada: {ruta}"
        )

    return str(ruta), False


def detectar(entrada: str, mostrar: bool = False):
    """Ejecuta la detección sobre imágenes, videos o cámaras."""

    if not RUTA_MODELO.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo: {RUTA_MODELO}"
        )

    fuente, es_camara = obtener_fuente(entrada)

    print("\nCargando modelo YOLO...")
    modelo = YOLO(str(RUTA_MODELO))

    print("Modelo cargado correctamente.")
    print(f"Clases disponibles: {modelo.names}")

    print("\nIniciando detección...\n")

    resultados = modelo.predict(
        source=fuente,
        device="cpu",
        imgsz=416,
        conf=0.35,
        vid_stride=3,
        stream=True,
        save=True,
        show=mostrar or es_camara,
        project=str(RUTA_RESULTADOS),
        name="detecciones",
        exist_ok=False,
        verbose=False,
    )

    contador_fotogramas = 0
    conteo_clases = Counter()

    try:
        for resultado in resultados:
            contador_fotogramas += 1

            if resultado.boxes is not None:
                for clase in resultado.boxes.cls.tolist():
                    nombre = modelo.names[int(clase)]
                    conteo_clases[nombre] += 1

            if contador_fotogramas % 20 == 0:
                print(
                    f"Fotogramas procesados: "
                    f"{contador_fotogramas}"
                )

    except KeyboardInterrupt:
        print("\nDetección interrumpida por el usuario.")

    finally:
        resultados.close()

    print("\n--- RESUMEN ---")
    print(f"Fotogramas procesados: {contador_fotogramas}")

    for clase, cantidad in sorted(conteo_clases.items()):
        print(f"{clase}: {cantidad} apariciones")

    print(
        "\nNota: las apariciones no representan "
        "objetos únicos."
    )

    print(f"\nResultados guardados en: {RUTA_RESULTADOS}")


def main():
    parser = argparse.ArgumentParser(
        description="Detector de cascos y matrículas con YOLO"
    )

    parser.add_argument(
        "entrada",
        nargs="?",
        default="helmet.mp4",
        help="Imagen, video, carpeta o camara:0",
    )

    parser.add_argument(
        "--mostrar",
        action="store_true",
        help="Mostrar la detección en una ventana",
    )

    argumentos = parser.parse_args()

    try:
        detectar(
            entrada=argumentos.entrada,
            mostrar=argumentos.mostrar,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"\nERROR: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
