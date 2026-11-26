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
from .lighting import mejorar_iluminacion
from .alert_sound import reproducir_aviso
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
    parser.add_argument("--alerta-sonora", action="store_true",
                        help="Aviso audible ante un nuevo candidato")
    parser.add_argument("--sin-mejorar-iluminacion", action="store_true",
                        help="No ajustar escenas oscuras o con sombras")
    args = parser.parse_args(argv)
    if (not isfinite(args.conf) or not 0 < args.conf <= 1 or args.imgsz < 64
            or args.min_fotogramas < 1 or args.max_fotogramas < 0):
        parser.error("Valores inválidos")
    ruta = Path(args.modelo).expanduser()
    if not ruta.is_absolute():
        ruta = RAIZ / ruta
    if not ruta.is_file():
        parser.error(
            "Faltan pesos especializados de armas: " + str(ruta)
            + "\nLa camara NO es la causa. Para instalar pesos externos:"
            + "\n1) Revisa procedencia y licencia en "
              "https://huggingface.co/cosgun99/gun-knife-yolo11n"
            + "\n2) Si confias en ese origen, ejecuta:"
            + "\n   python -m pip install huggingface_hub"
            + "\n   python scripts/descargar_modelo_armas.py --acepto-modelo-tercero"
            + "\nO indica tus pesos especializados con --modelo-armas RUTA."
            + "\nNo uses pesos de cascos ni de YOLO COCO comun."
        )
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
    capturas_manuales = 0
    candidatos_recientes = []  # (fotograma, clase, caja), solo para visualización.
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
                    candidatos_recientes.clear()
                    print("[CAMARA] Detector de armas reanudado, historial reiniciado.")
                n += 1
                imagen_analisis, luz_adaptada = mejorar_iluminacion(
                    imagen, not args.sin_mejorar_iluminacion)
                pred = modelo.predict(
                    imagen_analisis, imgsz=args.imgsz, conf=args.conf,
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
                candidatos_recientes = [
                    c for c in candidatos_recientes if n - c[0] <= 45
                ]
                candidatos_recientes.extend(
                    (n, c.clase, c.caja) for c in nuevos
                )
                if args.alerta_sonora and nuevos:
                    reproducir_aviso()
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
                    vista = imagen_analisis.copy()
                    alto, ancho = vista.shape[:2]
                    hay_aviso = bool(candidatos_recientes)
                    for clase, conf, caja in objetos:
                        x1, y1, x2, y2 = map(int, caja)
                        revisable = any(
                            cl == clase and solape(ant_caja, caja) >= 0.25
                            for _, cl, ant_caja in candidatos_recientes
                        )
                        color = (0, 0, 255) if revisable else (0, 190, 255)
                        grosor = 4 if revisable else 2
                        cv2.rectangle(vista, (x1, y1), (x2, y2), (0, 0, 0), grosor + 2)
                        cv2.rectangle(vista, (x1, y1), (x2, y2), color, grosor)
                        texto = (f"{TRADUCIR[clase]} {conf:.2f} "
                                 + ("REVISAR" if revisable else "OBSERVADO"))
                        fuente_letra = cv2.FONT_HERSHEY_SIMPLEX
                        escala = 0.52
                        (w, h), base = cv2.getTextSize(
                            texto, fuente_letra, escala, 1)
                        tx = max(3, min(max(3, ancho - w - 4), x1))
                        ty = max(h + 9, min(alto - 4, y1 - 6))
                        cv2.rectangle(vista, (tx - 2, ty - h - 4),
                                      (tx + w + 3, ty + base + 3), color, -1)
                        cv2.putText(vista, texto, (tx, ty),
                                    fuente_letra, escala, (255, 255, 255),
                                    1, cv2.LINE_AA)
                    banner = ((0, 0, 205) if hay_aviso else (28, 28, 28))
                    cv2.rectangle(vista, (0, 0), (ancho - 1, 30), banner, -1)
                    aviso_texto = ("CANDIDATO DE OBJETO - REVISAR"
                                   if hay_aviso else "SIN CANDIDATOS RECIENTES")
                    cv2.putText(vista, aviso_texto, (8, 21),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.54,
                                (255, 255, 255), 2 if hay_aviso else 1,
                                cv2.LINE_AA)
                    info = (f"Objetos: {len(objetos)} | clase gun/knife"
                            + (" | luz ajustada" if luz_adaptada else ""))
                    cv2.rectangle(vista, (0, max(0, alto - 40)),
                                  (ancho - 1, alto - 1), (25, 25, 25), -1)
                    cv2.putText(vista, info, (8, max(16, alto - 23)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                                (225, 225, 225), 1, cv2.LINE_AA)
                    cv2.putText(vista, "Objeto potencial, NO amenaza confirmada",
                                (8, max(28, alto - 7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                                (255, 255, 255), 1, cv2.LINE_AA)
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
                        tecla = cv2.waitKey(1) & 0xFF
                        if tecla in (ord("f"), ord("F")):
                            if capturas_manuales >= 20:
                                print("[CAPTURA] Límite manual de 20 fotos por sesión.")
                            else:
                                carpeta_manual = carpeta / "capturas_manuales" / "armas"
                                carpeta_manual.mkdir(parents=True, exist_ok=True)
                                foto = carpeta_manual / (
                                    datetime.now(timezone.utc).strftime(
                                        "foto_%Y%m%d_%H%M%S_%f_")
                                    + f"{n:06d}.jpg")
                                if not cv2.imwrite(str(foto), vista):
                                    raise OSError(f"No se pudo guardar captura manual: {foto}")
                                capturas_manuales += 1
                                print(f"[FOTO MANUAL] {foto}")
                        if tecla in (27, ord("q")):
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
    if capturas_manuales:
        print(f"Fotos manuales: {capturas_manuales}, en {carpeta / 'capturas_manuales' / 'armas'}")
    if args.guardar_capturas:
        print(f"Capturas de candidatos: {capturas_guardadas}, carpeta: {carpeta_capturas}")


if __name__ == "__main__":
    main()
