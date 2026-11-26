"""Prueba de análisis postural con un modelo YOLO Pose separado.

No detecta armas ni demuestra violencia o intención. La postura horizontal
solo crea un candidato de revisión si persiste en un track identificado.
"""
import argparse
import csv
import time
from math import isfinite
from datetime import datetime, timezone
from pathlib import Path

import cv2
from ultralytics import YOLO

from .events import DetectorEventos, Persona
from .proximity import DetectorProximidad, estan_cerca
from .aggression import DetectorGestosDirigidos
from .event_clips import GrabadorClips
from .arm_motion import DetectorMovimientosBrazos
from .visual_layout import dimensiones_anotadas, texto_evento_resumido
from .lighting import mejorar_iluminacion
from .alert_sound import reproducir_aviso
from modules.camera_sources import abrir_captura
from modules.camera_recovery import recuperar_flujo, limpiar_rastreadores


RAIZ = Path(__file__).resolve().parents[2]


def aviso_sonoro():
    """Reproduce dos tonos breves solo ante un aviso nuevo."""
    return reproducir_aviso()


def _rotulo(vista, texto, x, y, color, tamano=0.48):
    """Etiqueta legible incluso cuando la persona está en una zona oscura."""
    alto, ancho = vista.shape[:2]
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    texto = str(texto)
    escala = tamano
    while escala > 0.30:
        (w, h), base = cv2.getTextSize(texto, fuente, escala, 1)
        if w <= ancho - 10:
            break
        escala -= 0.03
    (w, h), base = cv2.getTextSize(texto, fuente, escala, 1)
    x = max(4, min(ancho - w - 5, x))
    y = max(h + 7, min(alto - base - 5, y))
    cv2.rectangle(vista, (x - 3, y - h - 5),
                  (x + w + 3, y + base + 3), color, -1)
    cv2.putText(vista, texto, (x, y), fuente, escala,
                (255, 255, 255), 1, cv2.LINE_AA)


def renderizar_vista(resultado, personas: list[Persona], alertas: list,
                     modo: str, avisos_activos=None, resumen_reciente=None,
                     luz_adaptada=False):
    """Contraste alto entre personas observadas y avisos experimentales.

    Los recuadros muestran rastreos y candidatos, NO agresiones confirmadas.
    Si un evento ya pasó, el panel conserva un aviso breve para que sea visible.
    """
    vista = resultado.plot(labels=False, conf=False, line_width=2, kpt_radius=2)
    altura_original, ancho_original = vista.shape[:2]
    ancho, alto, panel = dimensiones_anotadas(ancho_original, altura_original)
    if (ancho, alto) != (ancho_original, altura_original):
        vista = cv2.resize(vista, (ancho, alto), interpolation=cv2.INTER_LINEAR)

    sx, sy = ancho / ancho_original, alto / altura_original

    def caja_persona(persona, color, grosor, texto):
        x1, y1, x2, y2 = persona.caja
        p1 = (max(0, min(ancho - 1, int(x1 * sx))),
              max(0, min(alto - 1, int(y1 * sy))))
        p2 = (max(0, min(ancho - 1, int(x2 * sx))),
              max(0, min(alto - 1, int(y2 * sy))))
        if p2[0] <= p1[0] or p2[1] <= p1[1]:
            return
        cv2.rectangle(vista, p1, p2, (0, 0, 0), grosor + 2)
        cv2.rectangle(vista, p1, p2, color, grosor)
        _rotulo(vista, texto, p1[0], p1[1] - 6, color)

    por_id = {p.track_id: p for p in personas if p.track_id is not None}
    for persona in personas:
        # La cantidad de keypoints con confianza ayuda a interpretar sombras
        # y oclusiones, pero NO es una medida de precisión del sistema.
        validos = sum(1 for p in persona.puntos if p[2] >= 0.4)
        nombre = (f"ID {persona.track_id}" if persona.track_id is not None
                  else "PERSONA")
        caja_persona(persona, (80, 180, 50), 2,
                     f"{nombre} | puntos {validos}/{len(persona.puntos)}")

    # Destacar el actor en rojo vivo y el objetivo en celeste: señales
    # distintas de una persona simplemente observada.
    if modo == "agresiones" and avisos_activos:
        for actor_id, (tipo, objetivo_id, ultimo) in avisos_activos.items():
            actor = por_id.get(actor_id)
            objetivo = por_id.get(objetivo_id)
            if objetivo is not None:
                caja_persona(objetivo, (255, 205, 0), 3,
                             f"ID {objetivo_id} | OTRA PERSONA")
            if actor is not None:
                caja_persona(actor, (0, 0, 255), 4,
                             f"ID {actor_id} | GESTO A REVISAR")

    # Señalar la persona o pareja que ha generado un aviso en este frame.
    if alertas and modo != "agresiones":
        for evento in alertas[:3]:
            ids = (evento.track_ids if modo == "proximidad"
                   else (evento.track_id,))
            for ident in ids:
                persona = por_id.get(ident)
                if persona is not None:
                    caja_persona(persona, (0, 150, 255), 3,
                                 f"ID {ident} | AVISO PARA REVISION")

    vista = cv2.copyMakeBorder(
        vista, 0, panel, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 24))
    titulos = {
        "posturas": "POSTURAS: observacion de posiciones",
        "proximidad": "PROXIMIDAD: NO indica pelea",
        "brazos": "BRAZOS: gestos NO confirman agresion",
        "agresiones": "INTERACCIONES: NUNCA confirma agresion",
    }
    cv2.putText(vista, titulos[modo], (10, alto + 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1,
                cv2.LINE_AA)
    info = (f"Personas: {len(personas)} | IDs: {len(por_id)}"
            + (" | luz ajustada" if luz_adaptada else " | luz original")
            + " | verde=seguimiento rojo=aviso")
    cv2.putText(vista, info, (10, alto + 39),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (215, 215, 215), 1,
                cv2.LINE_AA)

    if resumen_reciente:
        cv2.rectangle(vista, (0, alto + 44),
                      (ancho - 1, alto + 67), (12, 20, 210), -1)
        mensaje = "REVISAR | " + resumen_reciente
        escala = 0.43
        while escala > 0.30 and cv2.getTextSize(
                mensaje, cv2.FONT_HERSHEY_SIMPLEX, escala, 1)[0][0] > ancho - 20:
            escala -= 0.03
        cv2.putText(vista, mensaje, (10, alto + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), 1,
                    cv2.LINE_AA)
    else:
        cv2.putText(vista, "Sin avisos recientes (la escena sigue en analisis)",
                    (10, alto + 59), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                    (210, 210, 210), 1, cv2.LINE_AA)

    for indice, evento in enumerate(alertas[:2]):
        mensaje = texto_evento_resumido(evento, modo)
        cv2.putText(vista, mensaje, (10, alto + 80 + 19 * indice),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 220, 255), 1,
                    cv2.LINE_AA)
    cv2.putText(vista, "Candidatos experimentales | revision humana obligatoria",
                (10, alto + 103), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (190, 190, 190), 1, cv2.LINE_AA)
    return vista


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
        # Una sombra o un falso cuerpo suele carecer de puntos COCO fiables.
        # Mantener un filtro conservador de 4 puntos sin dar garantías.
        if sum(1 for punto in datos if punto[2] >= 0.35) < 4:
            continue
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
    parser.add_argument("--modo", choices=("posturas", "proximidad", "brazos", "agresiones"), default="posturas")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--min-fotogramas", type=int, default=6)
    parser.add_argument("--separacion-alertas", type=int, default=90)
    parser.add_argument("--max-fotogramas", type=int, default=0,
                        help="0 (por defecto) = sin limite; mayor a 0 = prueba corta")
    parser.add_argument(
        "--filtro-brazos", action="store_true",
        help="Filtro optativo de saltos de muñeca en modo brazos",
    )
    parser.add_argument("--mostrar", action="store_true")
    parser.add_argument("--guardar-video", action="store_true")
    parser.add_argument("--guardar-clips", action="store_true",
                        help="Clips de ~2 s antes y 3 s despues de avisos, solo agresiones")
    parser.add_argument("--alerta-sonora", action="store_true",
                        help="Dos tonos por nuevo candidato; opcional en todos los modos")
    parser.add_argument("--sin-mejorar-iluminacion", action="store_true",
                        help="No ajustar luminancia de fotogramas oscuros")
    parser.add_argument(
        "--guardar-capturas", action="store_true",
        help="Guardar sólo fotogramas con avisos (máximo 20 por sesión)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Registro de candidatos; no guarda imágenes por defecto",
    )
    args = parser.parse_args()

    if not (0 < args.conf <= 1) or args.imgsz < 64 or args.max_fotogramas < 0:
        parser.error("Parámetros de confianza, imagen o límite inválidos")
    if args.filtro_brazos and args.modo != "brazos":
        parser.error("--filtro-brazos sólo está disponible en modo brazos")
    if args.guardar_clips and args.modo != "agresiones":
        parser.error("--guardar-clips solo existe en modo agresiones")
    def nuevo_detector():
        return (
            DetectorGestosDirigidos(
                min_cercania=max(2, args.min_fotogramas // 2),
                separacion_alertas=args.separacion_alertas,
            ) if args.modo == "agresiones"
            else DetectorProximidad(args.min_fotogramas, args.separacion_alertas)
            if args.modo == "proximidad"
            else DetectorMovimientosBrazos(
                min_elevados=max(2, args.min_fotogramas),
                separacion_alertas=args.separacion_alertas,
                filtro_saltos=args.filtro_brazos,
            ) if args.modo == "brazos"
            else DetectorEventos(args.min_fotogramas, args.separacion_alertas)
        )

    detector = nuevo_detector()
    fuente = resolver_fuente(args.fuente)
    modelo = YOLO(args.modelo)
    if getattr(modelo, "task", None) != "pose":
        raise ValueError("El modelo conductual debe ser de tipo pose, no detect")
    camara = abrir_captura(fuente)

    nombre_csv = (
        "eventos_brazos_filtrados.csv"
        if args.modo == "brazos" and args.filtro_brazos
        else {"posturas": "eventos.csv",
              "proximidad": "eventos_proximidad.csv",
              "brazos": "eventos_brazos.csv",
              "agresiones": "eventos_agresiones.csv"}[args.modo]
    )
    destino = Path(args.csv if args.csv else RAIZ / "resultados" / "conductual" / nombre_csv).expanduser()
    fps_fuente = camara.get(cv2.CAP_PROP_FPS)
    if not isfinite(fps_fuente) or fps_fuente <= 0:
        fps_fuente = 20.0
    es_archivo_local = isinstance(fuente, str) and Path(fuente).is_file()
    inicio_reloj = time.monotonic()
    destino.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not destino.exists() or destino.stat().st_size == 0
    escritor_video = None
    max_capturas = 20  # Tope de privacidad/espacio por ejecución.
    capturas_guardadas = 0
    carpeta_capturas = None
    avisos_activos = {}  # actor -> (tipo, objetivo, último fotograma)
    grabador_clips = GrabadorClips(destino.parent) if args.guardar_clips else None
    fotograma = 0
    ultimo_resumen = None
    ultimo_aviso_frame = -1000000
    try:
        if args.guardar_video:
            anchura = int(camara.get(cv2.CAP_PROP_FRAME_WIDTH))
            altura = int(camara.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = camara.get(cv2.CAP_PROP_FPS) or 20.0
            if anchura <= 0 or altura <= 0:
                raise ValueError("La fuente de video no informa dimensiones válidas")
            salida = destino.parent / (
                {"posturas": "resultado_posturas_",
                 "proximidad": "resultado_proximidad_",
                 "brazos": ("resultado_brazos_filtrados_" if args.filtro_brazos
                            else "resultado_brazos_"),
                 "agresiones": "resultado_agresiones_"}[args.modo]
                + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                + ".mp4"
            )
            ancho_vista, alto_vista, panel_vista = dimensiones_anotadas(
                anchura, altura
            )
            escritor_video = cv2.VideoWriter(
                str(salida),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (ancho_vista, alto_vista + panel_vista),
            )
            if not escritor_video.isOpened():
                raise OSError(f"No se pudo crear el video: {salida}")

        with destino.open("a", encoding="utf-8", newline="") as archivo:
            csv_writer = csv.writer(archivo)
            if nuevo:
                csv_writer.writerow(
                    ["timestamp_utc", "fuente", "fotograma", "track_ids" if args.modo in ("proximidad", "agresiones") else "track_id", "tipo", "estado"]
                )

            while True:
                correcto, imagen = camara.read()
                if not correcto or imagen is None:
                    # Un MP4 debe terminar al llegar al final; la camara USB/IP no.
                    camara_nueva, imagen = recuperar_flujo(
                        camara, fuente, abrir_captura,
                        detener=(lambda: (cv2.waitKey(1) & 0xFF)
                                 in (27, ord("q"))) if args.mostrar else None,
                    )
                    if camara_nueva is None:
                        break
                    camara = camara_nueva
                    # Un corte de señal invalida las rachas y los IDs anteriores.
                    limpiar_rastreadores(modelo)
                    detector = nuevo_detector()
                    avisos_activos.clear()
                    ultimo_resumen = None
                    ultimo_aviso_frame = -1000000
                    if grabador_clips is not None:
                        grabador_clips.cerrar()
                        grabador_clips = GrabadorClips(destino.parent)
                    print("[CAMARA] Analisis conductual reanudado.")
                fotograma += 1
                imagen_analisis, luz_adaptada = mejorar_iluminacion(
                    imagen, not args.sin_mejorar_iluminacion)
                resultado = modelo.track(
                    imagen_analisis,
                    persist=True,
                    tracker="bytetrack.yaml",
                    imgsz=args.imgsz,
                    conf=args.conf,
                    device="cpu",
                    verbose=False,
                )[0]
                personas = extraer_personas(resultado)
                if args.modo in ("brazos", "agresiones"):
                    # Usar reloj de video para archivos y monotónico para cámaras.
                    if es_archivo_local:
                        pos_ms = camara.get(cv2.CAP_PROP_POS_MSEC)
                        tiempo_s = (pos_ms / 1000.0 if isfinite(pos_ms) and pos_ms > 0
                                    else fotograma / fps_fuente)
                    else:
                        tiempo_s = time.monotonic() - inicio_reloj
                    alertas = detector.procesar(fotograma, personas, tiempo_s)
                else:
                    alertas = detector.procesar(fotograma, personas)
                if alertas:
                    ultimo_resumen = texto_evento_resumido(alertas[-1], args.modo)
                    ultimo_aviso_frame = fotograma
                for alerta in alertas:
                    instante = datetime.now(timezone.utc).isoformat()
                    csv_writer.writerow(
                        [
                            instante,
                            args.fuente,
                            alerta.fotograma,
                            f"{alerta.track_ids[0]}|{alerta.track_ids[1]}" if args.modo in ("proximidad", "agresiones") else alerta.track_id,
                            (f"{alerta.tipo}:{alerta.brazo}" if args.modo in ("brazos", "agresiones")
                             else alerta.tipo),
                            "candidato_para_revision",
                        ]
                    )
                    archivo.flush()
                    if args.modo == "agresiones":
                        print(f"[REVISAR INTERACCION] Fotograma {fotograma}; IDs {alerta.track_id}-{alerta.objetivo_id}: {alerta.tipo}, movimiento de {alerta.brazo} (NO confirmado)")
                    elif args.modo == "proximidad":
                        print(f"[REVISAR PROXIMIDAD] Fotograma {fotograma}; IDs {alerta.track_ids}: cercanía persistente, NO pelea confirmada")
                    elif args.modo == "brazos":
                        print(f"[REVISAR BRAZO] Fotograma {fotograma}; ID {alerta.track_id}, "
                              f"{alerta.brazo}: {alerta.tipo} (NO agresión confirmada)")
                    else:
                        print(f"[REVISAR] Fotograma {fotograma} ID {alerta.track_id}: postura horizontal persistente (no caída confirmada)")

                if args.alerta_sonora and alertas:
                    aviso_sonoro()
                if args.modo == "agresiones":
                    presentes = {p.track_id: p for p in personas if p.track_id is not None}
                    for ident, (tipo, otro, ultimo) in list(avisos_activos.items()):
                        if fotograma - ultimo > 45 or ident not in presentes:
                            del avisos_activos[ident]
                    for evento in alertas:
                        anterior = avisos_activos.get(evento.track_id)
                        if (anterior is None or anterior[0] != "posible_agresion"
                                or evento.tipo == "posible_agresion"):
                            avisos_activos[evento.track_id] = (
                                evento.tipo, evento.objetivo_id, fotograma)

                # Las capturas sólo se producen si el usuario las solicita y
                # realmente existen avisos. No guardamos todas las personas.
                capturar = (
                    args.guardar_capturas and bool(alertas)
                    and capturas_guardadas < max_capturas
                )
                if args.mostrar or args.guardar_video or capturar or grabador_clips is not None:
                    vista = renderizar_vista(
                        resultado, personas, alertas, args.modo, avisos_activos,
                        ultimo_resumen if fotograma - ultimo_aviso_frame <= 45 else None,
                        luz_adaptada,
                    )
                    if capturar:
                        if carpeta_capturas is None:
                            carpeta_capturas = destino.parent / (
                                f"capturas_{args.modo}{'_filtradas' if args.filtro_brazos else ''}_"
                                + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                            )
                            carpeta_capturas.mkdir(parents=True, exist_ok=False)
                        nombre = carpeta_capturas / f"fotograma_{fotograma:06d}.jpg"
                        if not cv2.imwrite(str(nombre), vista):
                            raise OSError(f"No se pudo guardar la captura: {nombre}")
                        capturas_guardadas += 1
                        print(f"[CAPTURA PARA REVISIÓN] {nombre}")
                    if escritor_video is not None:
                        escritor_video.write(vista)
                    if grabador_clips is not None:
                        grabador_clips.procesar(vista, tiempo_s, evento=bool(alertas))
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
        if grabador_clips is not None:
            grabador_clips.cerrar()
        if args.mostrar:
            cv2.destroyAllWindows()

    print(f"Fin. Modo {args.modo}; {fotograma} fotogramas. Archivo de eventos: {destino}")
    if grabador_clips is not None:
        print(f"Clips para revision: {grabador_clips.cantidad}, en {destino.parent}")
    if args.guardar_capturas:
        if carpeta_capturas is None:
            print("No hubo eventos nuevos para guardar capturas.")
        else:
            print(f"Capturas para revisión: {capturas_guardadas} en {carpeta_capturas}")


if __name__ == "__main__":
    main()
