"""Modo experimental para evaluar cascos SIN exigir detectar motocicletas.

Usa únicamente el modelo de casco seleccionado. No busca o identifica personas
ni afirma que una detección sea una infracción; no involucra el módulo conductual.
"""
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

from modules.camera_sources import abrir_captura
from .association import normalizar_clase
from .helmet_summary import resumen_cascos
from .road_monitor import crear_salida, ruta_modelo
from .vehicle_detector import resolver_fuente

RAIZ = Path(__file__).resolve().parents[2]
TIPOS = {"con_casco", "sin_casco", "placa"}


def ejecutar(args) -> None:
    if not (0 < args.conf_casco <= 1 and 0 < args.conf_sin_casco <= 1):
        raise ValueError("Confianzas fuera de rango")
    if not (0 < args.conf_placa <= 1) or args.imgsz < 64:
        raise ValueError("Confianza de placa o imgsz inválido")
    if args.stride < 1 or args.max_fotogramas < 0:
        raise ValueError("Salto o límite de fotogramas inválido")

    modelo = YOLO(ruta_modelo(args.modelo_cascos))
    nombres = modelo.names
    nombres_iterables = nombres.values() if isinstance(nombres, dict) else nombres
    if not {"con_casco", "sin_casco"} <= {
        normalizar_clase(str(nombre)) for nombre in nombres_iterables
    }:
        raise ValueError("Los pesos deben contener las clases con casco y sin casco")

    fuente = resolver_fuente(args.fuente)
    captura = abrir_captura(fuente)
    salida = crear_salida(RAIZ / "resultados" / "solo_cascos")
    fps_original = float(captura.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps_original) or fps_original <= 0:
        fps_original = 20.0
    fps_salida = max(1.0, fps_original / args.stride)
    escritor = None
    conteos = Counter()
    numero, paso = 0, 0
    inicio = time.perf_counter()
    print(f"MODO SOLO CASCOS. Fuente: {args.fuente}")
    print(f"Modelo: {args.modelo_cascos}")
    print(f"Clases: {modelo.names}")
    print("Sin asociación con motocicletas; NO se verifican infracciones.")
    try:
        while True:
            if args.max_fotogramas and numero >= args.max_fotogramas:
                break
            ok, frame = captura.read()
            if not ok:
                break
            numero += 1
            if (numero - 1) % args.stride:
                continue
            paso += 1
            resultado = modelo.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=min(args.conf_casco, args.conf_sin_casco, args.conf_placa),
                device="cpu",
                verbose=False,
            )[0]
            umbrales = {
                "con_casco": args.conf_casco,
                "sin_casco": args.conf_sin_casco,
                "placa": args.conf_placa,
            }
            vistas = []
            cajas = resultado.boxes
            if cajas is not None and len(cajas):
                for coords, clase, confianza in zip(
                    cajas.xyxy.cpu().tolist(),
                    cajas.cls.int().cpu().tolist(),
                    cajas.conf.cpu().tolist(),
                ):
                    nombre = normalizar_clase(str(resultado.names[int(clase)]))
                    if nombre not in TIPOS or confianza < umbrales[nombre]:
                        continue
                    conteos[nombre] += 1
                    vistas.append((coords, nombre, float(confianza)))
            if args.guardar_video or args.mostrar:
                anotado = frame.copy()
                for coords, nombre, confianza in vistas:
                    x1, y1, x2, y2 = [int(v) for v in coords]
                    color = (
                        (0, 170, 0) if nombre == "con_casco" else
                        (0, 0, 230) if nombre == "sin_casco" else
                        (220, 190, 0)
                    )
                    cv2.rectangle(anotado, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        anotado, f"{nombre} {confianza:.2f}",
                        (x1, max(16, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1,
                    )
                cv2.putText(
                    anotado, "SOLO CASCOS | SIN ASOCIACION VIAL",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (255, 255, 255), 2,
                )
                if args.guardar_video:
                    if escritor is None:
                        alto, ancho = anotado.shape[:2]
                        escritor = cv2.VideoWriter(
                            str(salida / "anotado_cascos.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps_salida, (ancho, alto),
                        )
                        if not escritor.isOpened():
                            raise OSError("No se pudo escribir anotado_cascos.mp4")
                    escritor.write(anotado)
                if args.mostrar:
                    cv2.imshow("Solo cascos (experimental)", anotado)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
            if paso % 50 == 0:
                print(f"Fotogramas: {paso} | con casco: {conteos['con_casco']} | sin casco: {conteos['sin_casco']}")
    except KeyboardInterrupt:
        print("Análisis interrumpido.")
    finally:
        captura.release()
        if escritor is not None:
            escritor.release()
        if args.mostrar:
            cv2.destroyAllWindows()
    informe = resumen_cascos(paso, conteos)
    with (salida / "resumen_cascos.json").open("w", encoding="utf-8") as f:
        json.dump(informe, f, ensure_ascii=False, indent=2)
    segundos = max(time.perf_counter() - inicio, 0.001)
    print(
        f"Fin: {numero} frames origen, {paso} analizados, "
        f"{paso / segundos:.2f} frames analizados/s"
    )
    print(
        f"Detecciones: con casco={conteos['con_casco']}, "
        f"sin casco={conteos['sin_casco']}, placa={conteos['placa']}"
    )
    print(f"ESTADO: {informe['estado']} | Reporte: {salida / 'resumen_cascos.json'}")
    print(informe["advertencia"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analiza cascos aunque no haya motos detectadas (sin infracciones)"
    )
    parser.add_argument("--fuente", default="prueba_motos.mp4")
    parser.add_argument("--modelo-cascos", default="resultados/entrenamientos/videos_revisados/weights/best.pt")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-fotogramas", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf-casco", type=float, default=0.40)
    parser.add_argument("--conf-sin-casco", type=float, default=0.45)
    parser.add_argument("--conf-placa", type=float, default=0.65)
    parser.add_argument("--mostrar", action="store_true")
    parser.add_argument("--guardar-video", action="store_true")
    args = parser.parse_args()
    try:
        ejecutar(args)
    except (OSError, ValueError, FileNotFoundError, RuntimeError) as e:
        parser.exit(1, f"ERROR: {e}\n")


if __name__ == "__main__":
    main()
