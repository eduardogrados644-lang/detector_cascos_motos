"""Candidatos de pistola/cuchillo con modelo especializado, módulo INTENCIONES.

Nunca infiere posesión, amenazas ni culpabilidad; requiere revisión humana.
No usa el modelo vial ni cambia eventos de proximidad.
"""
import argparse
import csv
import time
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path

from .weapon_events import ConfirmadorArmas, nombres_armas, solape
from .person_tracker import fuente_video
from modules.camera_recovery import recuperar_flujo

RAIZ = Path(__file__).resolve().parents[2]
TRADUCIR = {"gun": "POSIBLE PISTOLA", "knife": "POSIBLE CUCHILLO"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Detección preliminar de armas (no confirmación)")
    parser.add_argument("--fuente", required=True, help="MP4, camara:1, RTSP o MJPEG")
    parser.add_argument("--modelo", default="modelos_intenciones/armas/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--min-fotogramas", type=int, default=3)
    parser.add_argument("--max-fotogramas", type=int, default=0)
    parser.add_argument("--guardar-video", action="store_true")
    parser.add_argument("--guardar-capturas", action="store_true",
                        help="Guardar solo fotogramas con candidatos (hasta 20 por sesion)")
    parser.add_argument("--mostrar", action="store_true")
    args = parser.parse_args(argv)
    if (not isfinite(args.conf) or not 0 < args.conf <= 1 or args.imgsz < 64
            or args.min_fotogramas < 1 or args.max_fotogramas < 0):
        parser.error("Valores inválidos")
    ruta = Path(args.modelo).expanduser()
    if not ruta.is_absolute():
        ruta = RAIZ / ruta
    if not ruta.is_file():
        parser.error("Faltan pesos especializados de armas: " + str(ruta)
                     + ". Lee docs/DETECCION_AGRESION_ARMAS.md")
    import cv2
    from ultralytics import YOLO
    from modules.camera_sources import abrir_captura
    modelo = YOLO(str(ruta))
    if getattr(modelo, "task", None) != "detect":
        raise ValueError("Se requiere un modelo YOLO Detect de armas")
    clases = nombres_armas(modelo.names)
    print("Clases de armas disponibles:", ", ".join(sorted(set(clases.values()))))
    if "knife" not in clases.values():
        print("AVISO: estos pesos NO permiten detectar cuchillos.")
    if "gun" not in clases.values():
        print("AVISO: estos pesos NO permiten detectar pistolas.")
    fuente_captura = fuente_video(args.fuente)
    captura = abrir_captura(fuente_captura)
    fps = captura.get(cv2.CAP_PROP_FPS)
    if not isfinite(fps) or fps <= 0:
        fps = 20.0
    confirmador = ConfirmadorArmas(consecutivos=args.min_fotogramas)
    carpeta = RAIZ / "resultados" / "conductual"
    carpeta.mkdir(parents=True, exist_ok=True)
    sufijo = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    csv_path = carpeta / ("candidatos_armas_" + sufijo + ".csv")
    video_path = carpeta / ("candidatos_armas_" + sufijo + ".mp4")
    carpeta_capturas = carpeta / ("capturas_armas_" + sufijo)
    n = 0
    avisos = 0
    capturas_guardadas = 0
    inicio_reloj = time.monotonic()
    es_video = isinstance(fuente_captura, str) and Path(fuente_captura).is_file()
    writer_video = None
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            registro = csv.writer(f)
            registro.writerow((
                "timestamp_utc", "fotograma", "segundo_fuente",
                "clase", "tipo", "estado", "objeto_id",
                "confianza", "x1", "y1", "x2", "y2",
            ))
            while True:
                ok, imagen = captura.read()
                if not ok or imagen is None:
                    captura_nueva, imagen = recuperar_flujo(
                        captura, fuente_captura, abrir_captura,
                        detener=(lambda: (cv2.waitKey(1) & 0xFF)
                                 in (27, ord("q"))) if args.mostrar else None,
                    )
                    if captura_nueva is None:
                        break
                    captura = captura_nueva
                    # Las confirmaciones anteriores no sobreviven al corte.
                    confirmador = ConfirmadorArmas(consecutivos=args.min_fotogramas)
                    print("[CAMARA] Detector de armas reanudado, historial reiniciado.")
                n += 1
                pred = modelo.predict(
                    imagen, imgsz=args.imgsz, conf=args.conf,
                    device="cpu", verbose=False,
                )[0]
                objetos = []
                for xyxy, idx, score in zip(
                    pred.boxes.xyxy.cpu().tolist(),
                    pred.boxes.cls.int().cpu().tolist(),
                    pred.boxes.conf.cpu().tolist(),
                ):
                    clase = clases.get(idx)
                    if clase is None:
                        continue
                    objetos.append((clase, float(score), xyxy))
                # No acumular rachas por clase: cada caja debe ser estable.
                nuevos = confirmador.procesar_objetos(n, objetos)
                if es_video:
                    pos_ms = captura.get(cv2.CAP_PROP_POS_MSEC)
                    instante_s = (pos_ms / 1000.0 if isfinite(pos_ms) and pos_ms > 0
                                  else n / fps)
                else:
                    instante_s = time.monotonic() - inicio_reloj
                for candidato in nuevos:
                    avisos += 1
                    x1, y1, x2, y2 = candidato.caja
                    registro.writerow((
                        datetime.now(timezone.utc).isoformat(), n,
                        round(instante_s, 3), candidato.clase,
                        "objeto_potencial", "revisar_manualmente",
                        candidato.objeto_id, round(candidato.confianza, 4),
                        round(x1), round(y1), round(x2), round(y2),
                    ))
                    f.flush()
                    print(f"[REVISAR] fotograma {n}, objeto {candidato.objeto_id}: "
                          f"{TRADUCIR[candidato.clase]} (no confirmado)")
                if (args.guardar_video or args.mostrar
                        or (args.guardar_capturas and nuevos
                            and capturas_guardadas < 20)):
                    vista = imagen.copy()
                    for clase, conf, caja in objetos:
                        x1, y1, x2, y2 = map(int, caja)
                        revisable = any(
                            cand.clase == clase and solape(cand.caja, caja) > .95
                            for cand in nuevos
                        )
                        color = (40, 40, 235) if revisable else (0, 160, 255)
                        cv2.rectangle(vista, (x1, y1), (x2, y2), color, 2)
                        texto = (f"{TRADUCIR[clase]} {conf:.2f} "
                                 + ("REVISAR" if revisable else "OBSERVADO"))
                        cv2.putText(vista, texto, (max(0,x1), max(20,y1-5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    alto, ancho = vista.shape[:2]
                    cv2.putText(vista, "CANDIDATOS PARA REVISION - NO CONFIRMADO",
                                (8, max(16, alto-12)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.47, (255,255,255), 1)
                    if (args.guardar_capturas and nuevos
                            and capturas_guardadas < 20):
                        carpeta_capturas.mkdir(parents=True, exist_ok=True)
                        ruta_captura = carpeta_capturas / f"fotograma_{n:06d}.jpg"
                        if not cv2.imwrite(str(ruta_captura), vista):
                            raise OSError(f"No se pudo guardar captura: {ruta_captura}")
                        capturas_guardadas += 1
                        print("[CAPTURA PARA REVISION]", ruta_captura)
                    if args.guardar_video:
                        if writer_video is None:
                            writer_video = cv2.VideoWriter(
                                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                fps, (ancho, alto))
                            if not writer_video.isOpened():
                                raise OSError(f"No se pudo guardar {video_path}")
                        writer_video.write(vista)
                    if args.mostrar:
                        cv2.imshow("INTENCIONES - armas (experimental)", vista)
                        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                            break
                if args.max_fotogramas and n >= args.max_fotogramas:
                    break
    finally:
        captura.release()
        if writer_video is not None:
            writer_video.release()
        if args.mostrar:
            cv2.destroyAllWindows()
    print(f"Fotogramas: {n}; candidatos para revisión: {avisos}")
    print(f"Registro local: {csv_path}")
    if args.guardar_video:
        print(f"Video anotado: {video_path}")
    if args.guardar_capturas:
        print(f"Capturas de candidatos: {capturas_guardadas}, carpeta: {carpeta_capturas}")


if __name__ == "__main__":
    main()
