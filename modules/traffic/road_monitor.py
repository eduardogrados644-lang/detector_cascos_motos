"""Control vial experimental con dos modelos, una sola lectura de video y tracking.

Modelo COCO: persona y motocicleta. Modelo entrenado: casco/sin casco/placa.
No reconoce conductores, identidades, infracciones legales ni matrículas OCR.
Cada aviso CSV es un CANDIDATO de revisión humana, nunca una multa automática.
"""
import argparse
import csv
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
from ultralytics import YOLO

from .association import (
    ConfirmadorEventosViales,
    Objeto,
    normalizar_clase,
    relacionar,
)
from .vehicle_detector import ids_objetivo, resolver_fuente
from modules.camera_sources import abrir_captura


RAIZ = Path(__file__).resolve().parents[2]
CLASES_CASCO = {"con_casco", "sin_casco", "placa"}


def ruta_modelo(texto: str) -> str:
    """Prefiere pesos locales, pero permite descargar un nombre público YOLO."""
    ruta = Path(texto).expanduser()
    if ruta.is_file():
        return str(ruta.resolve())
    if (RAIZ / ruta).is_file():
        return str((RAIZ / ruta).resolve())
    if ruta.name == texto and texto.lower().endswith(".pt"):
        return texto
    raise FileNotFoundError(f"No se encontraron los pesos indicados: {texto}")


def extraer_objetos(resultado, categorias: set[str], umbrales: dict[str, float]) -> list[Objeto]:
    cajas = resultado.boxes
    if cajas is None or len(cajas) == 0:
        return []
    coordenadas = cajas.xyxy.cpu().tolist()
    clases = cajas.cls.int().cpu().tolist()
    confianzas = cajas.conf.cpu().tolist()
    track_ids = (
        cajas.id.int().cpu().tolist()
        if cajas.id is not None
        else [None] * len(coordenadas)
    )
    objetos = []
    for caja, clase, conf, track_id in zip(
        coordenadas, clases, confianzas, track_ids
    ):
        nombre = normalizar_clase(str(resultado.names[int(clase)]))
        if nombre not in categorias or conf < umbrales.get(nombre, 0.0):
            continue
        if caja[2] <= caja[0] or caja[3] <= caja[1]:
            continue
        objetos.append(
            Objeto(nombre, tuple(float(v) for v in caja), float(conf), track_id)
        )
    return objetos


def dibujar(frame, motos, personas, cascos, relaciones, eventos, fotograma):
    """Visualiza hipótesis y clases sin declararlas infracciones confirmadas."""
    colores = {
        "moto": (30, 175, 245),
        "persona": (250, 150, 35),
        "con_casco": (20, 175, 40),
        "sin_casco": (20, 20, 235),
        "placa": (140, 220, 220),
    }
    for objeto in [*motos, *personas, *cascos]:
        x1, y1, x2, y2 = [round(v) for v in objeto.caja]
        color = colores.get(objeto.clase, (150, 150, 150))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        texto = objeto.clase.upper()
        if objeto.track_id is not None:
            texto += f" ID:{objeto.track_id}"
        cv2.putText(
            frame, texto, (x1, max(17, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1,
        )
    for relacion in relaciones:
        p = relacion.persona
        m = relacion.moto
        x1, y1, x2, _ = [round(v) for v in p.caja]
        etiqueta = f"ASOCIACION TENTATIVA: {relacion.estado}"
        color = (
            (20, 20, 235) if relacion.estado == "sin_casco"
            else (20, 175, 40) if relacion.estado == "con_casco"
            else (180, 180, 180)
        )
        cv2.putText(
            frame, etiqueta, (x1, min(frame.shape[0] - 12, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1,
        )
        inicio = ((x1 + x2) // 2, round(p.caja[3]))
        fin = (round((m.caja[0] + m.caja[2]) / 2), round(m.caja[1]))
        cv2.line(frame, inicio, fin, color, 1)
    if eventos:
        cv2.putText(
            frame, "POSIBLE SIN CASCO: REVISAR MANUALMENTE",
            (10, max(25, frame.shape[0] - 18)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 0, 255), 2,
        )
    cv2.putText(
        frame, f"Fotograma {fotograma} | EXPERIMENTAL",
        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2,
    )
    return frame


def crear_salida(base: Path) -> Path:
    nombre = datetime.now(timezone.utc).strftime("sesion_%Y%m%d_%H%M%S_%f")
    destino = base / nombre
    destino.mkdir(parents=True, exist_ok=False)
    return destino


def ejecutar(args) -> None:
    if not (0 < args.conf_vehiculos <= 1):
        raise ValueError("--conf-vehiculos fuera de rango")
    if not (0 < args.conf_casco <= 1 and 0 < args.conf_sin_casco <= 1):
        raise ValueError("Confianzas de casco fuera de rango")
    if not 0 < args.conf_placa <= 1 or args.stride < 1 or args.imgsz < 64:
        raise ValueError("Configuración de detección inválida")
    if args.max_fotogramas < 0:
        raise ValueError("--max-fotogramas debe ser >= 0")

    fuente = resolver_fuente(args.fuente)
    modelo_vehiculos = YOLO(ruta_modelo(args.modelo_vehiculos))
    modelo_cascos = YOLO(ruta_modelo(args.modelo_cascos))
    clases_coco = ids_objetivo(modelo_vehiculos.names)
    clases_cascos = {normalizar_clase(n) for n in modelo_cascos.names.values()}
    if not {"con_casco", "sin_casco"} <= clases_cascos:
        raise ValueError(
            "El modelo de cascos no contiene las clases With/Without Helmet"
        )

    captura = abrir_captura(fuente)

    salida = crear_salida(RAIZ / "resultados" / "vial_integrado")
    confirmar = ConfirmadorEventosViales(args.confirmaciones)
    total = Counter()
    escritor_video = None
    fps_original = float(captura.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps_original) or fps_original <= 0:
        fps_original = 20.0
    fps_video = max(1.0, fps_original / args.stride)
    numero = paso = 0
    inicio = time.perf_counter()

    print(f"Fuente: {args.fuente}")
    print(f"Pesos: {args.modelo_vehiculos} + {args.modelo_cascos}")
    print(f"Clases viales: {modelo_vehiculos.names}")
    print(f"Clases casco/placa: {modelo_cascos.names}")
    print(
        "Seguimiento independiente de cabezas: "
        + ("NO (comparación)" if args.sin_seguimiento_cabezas else "SÍ")
    )
    print("Solo se registran candidatos a revisar; no infracciones confirmadas.")
    print(f"Resultados: {salida}")
    ruta_duplicados = salida / "posibles_duplicados.csv"
    with ruta_duplicados.open("w", encoding="utf-8", newline="") as archivo_duplicados:
        csv.writer(archivo_duplicados).writerow([
            "timestamp_utc", "fuente", "fotograma", "moto_track_id",
            "persona_track_id", "cabeza_nueva_id", "cabeza_previa_id",
            "estado",
        ])

    try:
        with (salida / "eventos.csv").open(
            "w", encoding="utf-8", newline=""
        ) as archivo:
            escritor_csv = csv.writer(archivo)
            escritor_csv.writerow([
                "timestamp_utc", "fuente", "fotograma",
                "moto_track_id", "persona_track_id",
                "confianza_sin_casco", "personas_asociadas",
                "evento", "estado", "cabeza_track_id",
            ])
            while True:
                if args.max_fotogramas and numero >= args.max_fotogramas:
                    break
                ok, frame = captura.read()
                if not ok:
                    break
                numero += 1
                if (numero - 1) % args.stride != 0:
                    continue
                paso += 1
                vehiculos = modelo_vehiculos.track(
                    frame,
                    classes=clases_coco,
                    persist=True,
                    tracker="bytetrack.yaml",
                    imgsz=args.imgsz,
                    conf=args.conf_vehiculos,
                    device="cpu",
                    verbose=False,
                )[0]
                # Seguimiento separado de cabezas: los ID de persona COCO
                # pueden cambiar cuando varios pasajeros se superponen.
                metodo_cabezas = (
                    modelo_cascos.predict
                    if args.sin_seguimiento_cabezas
                    else modelo_cascos.track
                )
                cascos_resultado = metodo_cabezas(
                    frame,
                    imgsz=args.imgsz,
                    conf=min(
                        args.conf_casco, args.conf_sin_casco, args.conf_placa
                    ),
                    **(
                        {}
                        if args.sin_seguimiento_cabezas
                        else {"persist": True, "tracker": "bytetrack.yaml"}
                    ),
                    device="cpu",
                    verbose=False,
                )[0]
                objetos_viales = extraer_objetos(
                    vehiculos,
                    {"persona", "moto"},
                    {"persona": args.conf_vehiculos, "moto": args.conf_vehiculos},
                )
                objetos_cascos = extraer_objetos(
                    cascos_resultado,
                    CLASES_CASCO,
                    {
                        "con_casco": args.conf_casco,
                        "sin_casco": args.conf_sin_casco,
                        "placa": args.conf_placa,
                    },
                )
                motos = [o for o in objetos_viales if o.clase == "moto"]
                personas = [o for o in objetos_viales if o.clase == "persona"]
                relaciones = relacionar(motos, personas, objetos_cascos)
                eventos = confirmar.procesar(paso, numero, relaciones)
                duplicados = confirmar.posibles_duplicados
                total["posibles_duplicados"] += len(duplicados)
                if duplicados:
                    with ruta_duplicados.open(
                        "a", encoding="utf-8", newline=""
                    ) as archivo_duplicados:
                        escritor_duplicados = csv.writer(archivo_duplicados)
                        for duplicado in duplicados:
                            escritor_duplicados.writerow([
                                datetime.now(timezone.utc).isoformat(),
                                args.fuente, duplicado.fotograma,
                                duplicado.moto_id, duplicado.persona_id,
                                duplicado.cabeza_id, duplicado.cabeza_anterior_id,
                                "posible_reidentificacion_para_revision",
                            ])
                            print(
                                f"[POSIBLE DUPLICADO] Moto ID:{duplicado.moto_id}, "
                                f"cabeza {duplicado.cabeza_id} podría repetir "
                                f"la alerta de cabeza {duplicado.cabeza_anterior_id} "
                                f"en frame {duplicado.fotograma}"
                            )

                total["personas"] += len(personas)
                total["motos"] += len(motos)
                total["relaciones"] += len(relaciones)
                total["candidatos"] += len(eventos)
                total.update(f"deteccion_{objeto.clase}" for objeto in objetos_cascos)
                total.update(f"asociacion_{relacion.estado}" for relacion in relaciones)

                for evento in eventos:
                    escritor_csv.writerow([
                        datetime.now(timezone.utc).isoformat(),
                        args.fuente,
                        evento.fotograma,
                        evento.moto_id,
                        evento.persona_id,
                        round(evento.confianza, 4),
                        evento.personas_asociadas,
                        evento.tipo,
                        "candidato_para_revision",
                        evento.cabeza_id if evento.cabeza_id is not None else "",
                    ])
                    archivo.flush()
                    print(
                        f"[REVISAR] Moto ID:{evento.moto_id}, "
                        f"persona ID:{evento.persona_id}, "
                        f"cabeza ID:{evento.cabeza_id} en frame {evento.fotograma}"
                    )

                if args.mostrar or args.guardar_video:
                    vista = dibujar(
                        frame.copy(), motos, personas, objetos_cascos,
                        relaciones, eventos, numero,
                    )
                    if args.guardar_video:
                        if escritor_video is None:
                            alto, ancho = vista.shape[:2]
                            escritor_video = cv2.VideoWriter(
                                str(salida / "anotado.mp4"),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                fps_video,
                                (ancho, alto),
                            )
                            if not escritor_video.isOpened():
                                raise OSError("No se pudo abrir anotado.mp4")
                        escritor_video.write(vista)
                    if args.mostrar:
                        cv2.imshow("Control vial (experimental)", vista)
                        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                            break
                if paso % 30 == 0:
                    print(
                        f"Analizados {paso} frames; asociaciones "
                        f"{total['relaciones']}, candidatos {total['candidatos']}"
                    )
    except KeyboardInterrupt:
        print("Proceso interrumpido.")
    finally:
        captura.release()
        if escritor_video is not None:
            escritor_video.release()
        if args.mostrar:
            cv2.destroyAllWindows()

    segundos = max(time.perf_counter() - inicio, 0.001)
    print(
        f"Fin: {numero} frames fuente, {paso} analizados, "
        f"{paso / segundos:.2f} frames analizados/seg. "
        f"Apariciones persona/moto: {total['personas']}/{total['motos']}; "
        f"asociaciones: {total['relaciones']}; "
        f"candidatos a revisar: {total['candidatos']}; "
        f"posibles duplicados de seguimiento: {total['posibles_duplicados']}"
    )
    print(
        "Detecciones casco/placa por clase: "
        + str({nombre: total[f"deteccion_{nombre}"] for nombre in sorted(CLASES_CASCO)})
    )
    print(
        "Estados de asociaciones: "
        + str({nombre: total[f"asociacion_{nombre}"] for nombre in ("con_casco", "sin_casco", "desconocido")})
    )
    print(f"CSV: {salida / 'eventos.csv'}")
    print(f"Revisiones de duplicados: {ruta_duplicados}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combina detección de motos/personas y casco en un video"
    )
    parser.add_argument("--fuente", default="prueba_motos.mp4")
    parser.add_argument("--modelo-vehiculos", default="yolo11n.pt")
    parser.add_argument("--modelo-cascos", default="best.pt")
    parser.add_argument("--conf-vehiculos", type=float, default=0.35)
    parser.add_argument("--conf-casco", type=float, default=0.40)
    parser.add_argument("--conf-sin-casco", type=float, default=0.45)
    parser.add_argument("--conf-placa", type=float, default=0.65)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--confirmaciones", type=int, default=4)
    parser.add_argument("--max-fotogramas", type=int, default=0)
    parser.add_argument("--mostrar", action="store_true")
    parser.add_argument("--guardar-video", action="store_true")
    parser.add_argument(
        "--sin-seguimiento-cabezas", action="store_true",
        help="Diagnóstico: usa predict sin tracking de cabezas",
    )
    args = parser.parse_args()
    try:
        ejecutar(args)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
