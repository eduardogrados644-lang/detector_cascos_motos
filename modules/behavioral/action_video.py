"""Clasificación exploratoria de clips de posibles acciones violentas.

Modelo temporal de terceros en Transformers. Muy pesado para PC con 4GB RAM:
preferir Colab con GPU. No identifica responsables, intención ni delitos.
"""
import argparse
import csv
from datetime import datetime, timezone
from math import ceil, isfinite
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
REPOSITORIO = "HappyGook/videomae-violence-detector"


def inicios_ventanas(duracion, ventana=3.0, salto=3.0):
    if not all(isfinite(v) and v > 0 for v in (duracion, ventana, salto)):
        raise ValueError("Duraciones inválidas")
    if duracion <= ventana:
        return [0.0]
    inicios = []
    t = 0.0
    while t + ventana <= duracion + 1e-6:
        inicios.append(round(t, 3))
        t += salto
    final = round(max(0.0, duracion - ventana), 3)
    if inicios[-1] < final - 0.05:
        inicios.append(final)
    return inicios


def tomar_fotogramas(cap, fps, inicio_s, duracion_s, cantidad=16):
    import cv2
    muestras = []
    for i in range(cantidad):
        segundo = inicio_s + duracion_s * (i + 0.5) / cantidad
        indice = max(0, int(segundo * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, indice)
        bien, imagen = cap.read()
        if not bien:
            raise RuntimeError(f"No se pudo leer el fotograma {indice}")
        muestras.append(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    return muestras


def main(argv=None):
    parser = argparse.ArgumentParser(description="Posibles escenas físicas violentas, prueba experimental")
    parser.add_argument("--fuente", required=True, help="Solo video local para esta etapa")
    parser.add_argument("--modelo", default=REPOSITORIO)
    parser.add_argument("--ventana-segundos", type=float, default=3.0)
    parser.add_argument("--salto-segundos", type=float, default=3.0)
    parser.add_argument("--umbral", type=float, default=0.80)
    parser.add_argument("--max-ventanas", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if (not isfinite(args.umbral) or not 0 < args.umbral <= 1
            or args.max_ventanas < 0):
        parser.error("Umbral o límite inválidos")
    fuente = Path(args.fuente).expanduser()
    if not fuente.is_absolute():
        fuente = RAIZ / fuente
    if not fuente.is_file():
        parser.error("No se encuentra el video: " + str(fuente))
    import cv2
    import torch
    try:
        from transformers import AutoImageProcessor, AutoModelForVideoClassification
    except ImportError as exc:
        raise SystemExit(
            "Falta Transformers. En Colab: %pip install transformers safetensors"
        ) from exc
    cap = cv2.VideoCapture(str(fuente))
    if not cap.isOpened():
        raise OSError("No se pudo abrir el video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not all(isfinite(v) and v > 0 for v in (fps, total)):
        cap.release()
        raise ValueError("El video debe indicar FPS y cantidad de fotogramas")
    duracion = total / fps
    inicios = inicios_ventanas(duracion, args.ventana_segundos, args.salto_segundos)
    if args.max_ventanas:
        inicios = inicios[:args.max_ventanas]
    if args.device == "cuda" and not torch.cuda.is_available():
        cap.release()
        parser.error("No hay GPU CUDA, usa --device cpu o ejecuta en Colab GPU")
    print("Descargando/cargando modelo temporal de terceros:", args.modelo)
    print("AVISO: clasificaciones de clips, no agresiones confirmadas.")
    procesador = AutoImageProcessor.from_pretrained(args.modelo)
    modelo = AutoModelForVideoClassification.from_pretrained(
        args.modelo, use_safetensors=True
    ).to(args.device)
    modelo.eval()
    if int(modelo.config.num_labels) != 2:
        raise ValueError("Modelo sin clasificación binaria compatible")
    carpeta = RAIZ / "resultados" / "conductual"
    carpeta.mkdir(parents=True, exist_ok=True)
    csv_path = carpeta / (
        "candidatos_violencia_"
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + ".csv"
    )
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as archivo:
            writer = csv.writer(archivo)
            writer.writerow(("inicio_s", "fin_s", "p_violencia", "estado"))
            for t in inicios:
                frames = tomar_fotogramas(
                    cap, fps, t, min(args.ventana_segundos, duracion-t),
                    cantidad=16
                )
                entrada = procesador(frames, return_tensors="pt")
                entrada = {k: v.to(args.device) for k, v in entrada.items()}
                with torch.inference_mode():
                    logits = modelo(**entrada).logits[0]
                    probabilidades = torch.softmax(logits, dim=-1).cpu().tolist()
                # Índice 1 = 'Violent' según ficha del modelo, NO su config.json.
                prob = float(probabilidades[1])
                estado = "candidato_revisar" if prob >= args.umbral else "sin_candidato"
                writer.writerow((round(t, 2), round(min(duracion,t+args.ventana_segundos),2),
                                 round(prob, 4), estado))
                archivo.flush()
                print(f"{t:.1f}s | prob. clase violenta {prob:.3f} | {estado}")
    finally:
        cap.release()
    print("Informe por ventanas guardado:", csv_path)
    print("Las probabilidades no están calibradas ni validadas para BEHAVE/KTH.")


if __name__ == "__main__":
    main()
