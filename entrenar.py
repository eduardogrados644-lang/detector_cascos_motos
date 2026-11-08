
import argparse
from pathlib import Path

from ultralytics import YOLO

# Rutas principales del proyecto
RUTA_BASE = Path(__file__).resolve().parent
RUTA_DATASET = RUTA_BASE / "dataset" / "data.yaml"
RUTA_RESULTADOS = RUTA_BASE / "resultados" / "entrenamientos"

# Modelo pequeño para equipos con recursos limitados
MODELO_BASE = "yolo11n.pt"


def entrenar_modelo(epocas: int, fraccion: float) -> None:
    """Entrena un detector de cascos y placas con YOLO."""

    if not RUTA_DATASET.is_file():
        raise FileNotFoundError(
            "No se encontró el archivo del dataset:\n"
            f"{RUTA_DATASET}\n\n"
            "Comprueba que hayas extraído correctamente "
            "el dataset descargado."
        )

    if epocas < 1:
        raise ValueError(
            "Las épocas deben ser un número mayor que cero."
        )

    if not 0 < fraccion <= 1:
        raise ValueError(
            "La fracción debe estar entre 0 y 1."
        )

    print("\n====================================")
    print(" ENTRENAMIENTO DE CASCOS Y PLACAS")
    print("====================================")

    print(f"\nDataset: {RUTA_DATASET}")
    print(f"Modelo base: {MODELO_BASE}")
    print(f"Épocas: {epocas}")
    print(f"Porcentaje de imágenes: {fraccion * 100:.0f}%")
    print("Dispositivo: CPU")
    print("Resolución: 416")
    print("Imágenes por lote: 2")

    print("\nCargando modelo preentrenado...")

    modelo = YOLO(MODELO_BASE)

    print("Modelo cargado correctamente.")
    print("\nComenzando entrenamiento...\n")

    modelo.train(
        data=str(RUTA_DATASET),
        epochs=epocas,
        fraction=fraccion,
        imgsz=416,
        batch=2,
        device="cpu",
        workers=0,
        cache=False,
        amp=False,
        pretrained=True,
        project=str(RUTA_RESULTADOS),
        name="cascos_placas",
        exist_ok=False,
        plots=True,
        verbose=True,
    )

    print("\n====================================")
    print(" ENTRENAMIENTO FINALIZADO")
    print("====================================")

    print("\nLos resultados se guardaron en:")
    print(RUTA_RESULTADOS)

    print(
        "\nBusca la carpeta del entrenamiento realizado "
        "y abre la subcarpeta 'weights'."
    )

    print(
        "\nAllí encontrarás el archivo 'best.pt' "
        "con los pesos de tu modelo entrenado."
    )

    print(
        "\nEl modelo original del proyecto "
        "no ha sido modificado."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrenamiento de un detector de cascos y placas"
    )

    parser.add_argument(
        "--epocas",
        type=int,
        default=10,
        help="Número de épocas de entrenamiento.",
    )

    parser.add_argument(
        "--fraccion",
        type=float,
        default=0.30,
        help="Proporción del dataset que se utilizará.",
    )

    argumentos = parser.parse_args()

    try:
        entrenar_modelo(
            epocas=argumentos.epocas,
            fraccion=argumentos.fraccion,
        )

    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"\nERROR: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
