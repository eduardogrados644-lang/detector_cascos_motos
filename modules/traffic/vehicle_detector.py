"""Detector complementario de personas y motocicletas con YOLO COCO.

No interpreta cascos ni empareja automáticamente una persona con una moto.
El detector de cascos y matrículas existente se mantiene independiente.
"""
import argparse
from collections import Counter
from pathlib import Path

from ultralytics import YOLO


RAIZ = Path(__file__).resolve().parents[2]
OBJETIVOS = {"person", "motorcycle", "motorbike"}


def resolver_fuente(entrada: str):
    if entrada.startswith("camara:"):
        indice = entrada.partition(":")[2]
        if not indice.isdecimal():
            raise ValueError("Usa camara:0, camara:1, etc.")
        return int(indice)
    if entrada.startswith(("rtsp://", "http://", "https://")):
        return entrada
    ruta = Path(entrada).expanduser()
    if not ruta.is_absolute():
        ruta = Path.cwd() / ruta
    if not ruta.exists():
        ruta = RAIZ / entrada
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró: {entrada}")
    return str(ruta.resolve())


def ids_objetivo(nombres) -> list[int]:
    elementos = nombres.items() if isinstance(nombres, dict) else enumerate(nombres)
    ids = [
        int(indice)
        for indice, nombre in elementos
        if str(nombre).strip().lower() in OBJETIVOS
    ]
    if not ids:
        raise ValueError("El modelo indicado no tiene clases persona/motocicleta")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detecta personas y motocicletas con pesos independientes de cascos"
    )
    parser.add_argument("entrada", nargs="?", default="prueba_motos.mp4")
    parser.add_argument("--modelo", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--mostrar", action="store_true")
    parser.add_argument("--guardar", action="store_true")
    argumentos = parser.parse_args()

    if not 0 < argumentos.conf <= 1 or argumentos.imgsz < 64 or argumentos.stride < 1:
        parser.error("Confianza, imgsz o stride fuera de rango")
    fuente = resolver_fuente(argumentos.entrada)
    modelo = YOLO(argumentos.modelo)
    clases = ids_objetivo(modelo.names)
    print(f"Buscando clases: {[modelo.names[i] for i in clases]}")
    print("Estas cifras son apariciones por fotograma, no vehículos únicos.")

    resultados = modelo.predict(
        source=fuente,
        classes=clases,
        conf=argumentos.conf,
        imgsz=argumentos.imgsz,
        vid_stride=argumentos.stride,
        device="cpu",
        stream=True,
        show=argumentos.mostrar,
        save=argumentos.guardar,
        project=str(RAIZ / "resultados" / "vehiculos"),
        name="deteccion",
        exist_ok=False,
        verbose=False,
    )
    conteo = Counter()
    fotogramas = 0
    try:
        for resultado in resultados:
            fotogramas += 1
            if resultado.boxes is not None:
                for indice in resultado.boxes.cls.cpu().tolist():
                    conteo[str(modelo.names[int(indice)])] += 1
            if fotogramas % 100 == 0:
                print(f"Fotogramas analizados: {fotogramas}")
    except KeyboardInterrupt:
        print("Procesamiento interrumpido.")
    finally:
        resultados.close()
    print(f"Finalizado: {fotogramas} fotogramas; apariciones: {dict(conteo)}")


if __name__ == "__main__":
    main()
