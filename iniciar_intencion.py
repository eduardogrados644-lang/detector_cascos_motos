"""Punto de entrada independiente del apartado conductual.

Inicia prototipos separados de pose, armas y analisis temporal de video.
Todos son experimentales y requieren modelos y evaluaciones independientes.
No mezcla pesos ni salidas del módulo de cascos y NO infiere intenciones.
"""
import argparse
import math
import subprocess
import sys
from pathlib import Path

from modules.camera_sources import seleccionar_fuente, validar_fuente


RAIZ = Path(__file__).resolve().parent


def mostrar_estado() -> None:
    print("\n=== APARTADO CONDUCTUAL / DETECCIÓN DE EVENTOS ===")
    print("Estado: EXPERIMENTAL, desarrollo especializado pendiente.")
    print("Disponible: prueba de posturas horizontales con YOLO Pose.")
    print("Disponible: proximidad persistente entre personas; NO es pelea.")
    print("Disponible: brazos elevados y desplazamientos aparentes (experimental).")
    print("Disponible: gestos dirigidos entre personas (posibles agresiones, no confirmadas).")
    print("Disponible: candidatos de armas con pesos especializados (gun/knife).")
    print("Disponible: clasificacion temporal VideoMAE solo sobre archivos de video.")
    print("Pendiente: validacion real, precision, y deteccion de acciones concretas.")
    print("ATENCIÓN: una postura u objeto no revela la intención de una persona.")
    print("Los resultados nunca son acusaciones ni eventos confirmados.")


def construir_comando_demo(
    fuente: str = "prueba_motos.mp4",
    mostrar: bool = True,
    guardar_video: bool = False,
    guardar_capturas: bool = False,
    filtro_brazos: bool = False,
    max_fotogramas: int = 0,
    modo: str = "posturas",
    imgsz: int | None = None,
    conf: float | None = None,
    guardar_clips: bool = False,
    alerta_sonora: bool = False,
) -> list[str]:
    """Sólo lanza la demostración conductual, nunca el detector de cascos."""
    if not fuente.strip() or max_fotogramas < 0:
        raise ValueError("Fuente o número de fotogramas inválidos")
    if modo not in ("posturas", "proximidad", "brazos", "agresiones"):
        raise ValueError("Modo conductual inválido")
    if imgsz is not None and (type(imgsz) is not int or imgsz < 64):
        raise ValueError("imgsz debe ser entero mayor o igual a 64")
    if conf is not None and (
        not isinstance(conf, (int, float)) or
        not math.isfinite(conf) or not 0 < conf <= 1
    ):
        raise ValueError("conf debe estar entre 0 (exclusivo) y 1")
    cmd = [
        sys.executable, "-m", "modules.behavioral.pose_detector",
        "--fuente", validar_fuente(fuente),
    ]
    if modo in ("proximidad", "brazos", "agresiones"):
        cmd.extend(["--modo", modo])
    if filtro_brazos:
        if modo != "brazos":
            raise ValueError("El filtro de brazos requiere modo brazos")
        cmd.append("--filtro-brazos")
    if mostrar:
        cmd.append("--mostrar")
    if guardar_video:
        cmd.append("--guardar-video")
    if guardar_capturas:
        cmd.append("--guardar-capturas")
    if guardar_clips:
        if modo != "agresiones":
            raise ValueError("Los clips de eventos requieren --demo-agresiones")
        cmd.append("--guardar-clips")
    if alerta_sonora:
        if modo != "agresiones":
            raise ValueError("La alerta sonora requiere --demo-agresiones")
        cmd.append("--alerta-sonora")
    if max_fotogramas:
        cmd.extend(["--max-fotogramas", str(max_fotogramas)])
    if imgsz is not None:
        cmd.extend(["--imgsz", str(imgsz)])
    if conf is not None:
        cmd.extend(["--conf", str(conf)])
    return cmd



def construir_comando_armas(
    fuente: str, mostrar: bool = True, guardar_video: bool = False,
    max_fotogramas: int = 0, imgsz: int | None = None,
    conf: float | None = None, modelo: str | None = None,
) -> list[str]:
    """Lanza el detector especializado de objetos, nunca los pesos viales."""
    if not fuente.strip() or max_fotogramas < 0:
        raise ValueError("Fuente o limite invalido")
    if imgsz is not None and (type(imgsz) is not int or imgsz < 64):
        raise ValueError("imgsz debe ser entero >= 64")
    if conf is not None and (not isinstance(conf, (int, float))
                             or not math.isfinite(conf) or not 0 < conf <= 1):
        raise ValueError("conf debe estar entre 0 (exclusivo) y 1")
    cmd = [sys.executable, "-m", "modules.behavioral.weapon_detector",
           "--fuente", validar_fuente(fuente)]
    if mostrar:
        cmd.append("--mostrar")
    if guardar_video:
        cmd.append("--guardar-video")
    if max_fotogramas:
        cmd.extend(["--max-fotogramas", str(max_fotogramas)])
    if imgsz is not None:
        cmd.extend(["--imgsz", str(imgsz)])
    if conf is not None:
        cmd.extend(["--conf", str(conf)])
    if modelo:
        cmd.extend(["--modelo", modelo])
    return cmd


def construir_comando_acciones(
    fuente: str, device: str = "cpu", umbral: float = 0.80,
    max_ventanas: int = 0,
) -> list[str]:
    """VideoMAE solamente admite archivos locales, no camaras."""
    if (not fuente.strip() or fuente.startswith(("camara:", "rtsp://", "http://", "https://"))
            or device not in ("cpu", "cuda") or max_ventanas < 0
            or not math.isfinite(umbral) or not 0 < umbral <= 1):
        raise ValueError("Acciones: indica video local y parametros validos")
    cmd = [sys.executable, "-m", "modules.behavioral.action_video",
           "--fuente", fuente, "--device", device, "--umbral", str(umbral)]
    if max_ventanas:
        cmd.extend(["--max-ventanas", str(max_ventanas)])
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Iniciador reservado para análisis conductual independiente"
    )
    demos = parser.add_mutually_exclusive_group()
    demos.add_argument(
        "--demo-posturas", action="store_true",
        help="Ejecutar sólo la demostración actual de poses; NO detecta armas/violencia",
    )
    demos.add_argument("--demo-proximidad", action="store_true", help="Cercanía persistente, no violencia")
    demos.add_argument("--demo-brazos", action="store_true", help="Movimientos aparentes de brazos, no agresión")
    demos.add_argument("--demo-agresiones", action="store_true", help="Gestos dirigidos a otra persona (posible agresion, NO confirmada)")
    demos.add_argument("--demo-armas", action="store_true", help="Posibles pistolas/cuchillos con modelo especializado")
    demos.add_argument("--demo-violencia", action="store_true", help="Clasificacion temporal de video local, no agresion confirmada")
    fuentes = parser.add_mutually_exclusive_group()
    fuentes.add_argument("--fuente", help="Archivo de video o URL RTSP/MJPEG")
    fuentes.add_argument(
        "--usb", type=int, help="Índice de cámara USB externa (1 a 20)"
    )
    fuentes.add_argument(
        "--wifi", help="URL IP Webcam de celular/tablet Android"
    )
    parser.add_argument("--max-fotogramas", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Tamaño de análisis para YOLO Pose (por defecto: 320)")
    parser.add_argument("--conf", type=float, default=None,
                        help="Confianza mínima de YOLO Pose (por defecto: 0.45)")
    parser.add_argument("--sin-mostrar", action="store_true")
    parser.add_argument("--guardar-video", action="store_true")
    parser.add_argument("--guardar-clips", action="store_true",
                        help="Clips cortos por aviso (solo modo agresiones)")
    parser.add_argument("--alerta-sonora", action="store_true",
                        help="Aviso de Windows por evento candidato (solo modo agresiones)")
    parser.add_argument(
        "--filtro-brazos", action="store_true",
        help="Filtro experimental de saltos de muñeca, sólo en demo-brazos",
    )
    parser.add_argument(
        "--guardar-capturas", action="store_true",
        help="Guardar capturas de avisos, opcional y sólo en resultados locales"
    )
    parser.add_argument("--modelo-armas", help="Pesos YOLO Detect gun/knife para --demo-armas")
    parser.add_argument("--umbral-violencia", type=float, default=0.80)
    parser.add_argument("--max-ventanas", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--estado", action="store_true")
    args = parser.parse_args(argv)
    mostrar_estado()
    if args.estado:
        return 0
    try:
        demo = (args.demo_posturas or args.demo_proximidad or args.demo_brazos
                or args.demo_agresiones or args.demo_armas or args.demo_violencia)
        modo = ("violencia" if args.demo_violencia else "armas" if args.demo_armas else
                "agresiones" if args.demo_agresiones else
                "brazos" if args.demo_brazos else
                "proximidad" if args.demo_proximidad else "posturas")
        if not demo:
            print("\n1. Posible agresion / intento entre personas (experimental)")
            print("2. Observar posturas corporales")
            print("3. Observar proximidad persistente (no significa pelea)")
            print("4. Observar brazos elevados y movimientos aparentes")
            print("5. Buscar posibles armas (requiere modelo de armas)")
            print("6. Clasificar videos con VideoMAE (solo archivos)")
            print("0. Salir sin iniciar análisis")
            opcion = input("Opción: ").strip()
            if opcion == "0":
                return 0
            if opcion not in ("1", "2", "3", "4", "5", "6"):
                raise ValueError("Opción no válida")
            modo = ("violencia" if opcion == "6" else "armas" if opcion == "5" else
                    "brazos" if opcion == "4" else
                    "proximidad" if opcion == "3" else
                    "posturas" if opcion == "2" else "agresiones")
            demo = True
        if demo:
            if modo == "violencia":
                if args.usb is not None or args.wifi or not args.fuente:
                    raise ValueError("VideoMAE usa solo --fuente archivo.mp4, no camaras")
                if (args.guardar_video or args.guardar_capturas or args.filtro_brazos
                        or args.guardar_clips or args.alerta_sonora):
                    raise ValueError("VideoMAE solo guarda CSV; no admite video/capturas/filtro brazos")
                cmd = construir_comando_acciones(
                    args.fuente, device=args.device,
                    umbral=args.umbral_violencia, max_ventanas=args.max_ventanas)
                print("Analizando video temporal experimental, NO agresion confirmada:", args.fuente)
                return subprocess.run(cmd, cwd=RAIZ, check=False).returncode
            if modo == "armas":
                if (args.guardar_capturas or args.filtro_brazos or args.guardar_clips
                        or args.alerta_sonora):
                    raise ValueError("Modo armas no permite capturas ni filtro de brazos")
                fuente = seleccionar_fuente(args.fuente, args.usb, args.wifi)
                if fuente is None:
                    return 0
                cmd = construir_comando_armas(
                    fuente, mostrar=not args.sin_mostrar,
                    guardar_video=args.guardar_video, max_fotogramas=args.max_fotogramas,
                    imgsz=args.imgsz, conf=args.conf, modelo=args.modelo_armas)
                print("Deteccion de posibles objetos para revision, NO amenaza confirmada:", fuente)
                return subprocess.run(cmd, cwd=RAIZ, check=False).returncode
            fuente = seleccionar_fuente(args.fuente, args.usb, args.wifi)
            if fuente is None:
                print("Se canceló la selección de fuente conductual.")
                return 0
            cmd = construir_comando_demo(
                fuente,
                mostrar=not args.sin_mostrar,
                guardar_video=args.guardar_video,
                guardar_capturas=args.guardar_capturas,
                guardar_clips=args.guardar_clips,
                alerta_sonora=args.alerta_sonora,
                filtro_brazos=args.filtro_brazos,
                max_fotogramas=args.max_fotogramas,
                modo=modo,
                imgsz=args.imgsz,
                conf=args.conf,
            )
            print({"posturas": "\nIniciando sólo la prueba de posturas.",
                   "proximidad": "\nIniciando prueba de proximidad, NO violencia.",
                   "brazos": "\nIniciando prueba de movimientos de brazos, NO agresión.",
                   "agresiones": "\nIniciando observacion de gestos dirigidos (NO agresion confirmada)."}[modo])
            print("Fuente seleccionada:", fuente)
            if fuente.startswith("camara:"):
                print("AVISO: verifica que el índice corresponda a la USB externa.")
            return subprocess.run(cmd, cwd=RAIZ, check=False).returncode
    except (EOFError, KeyboardInterrupt):
        print("\nOperación cancelada.")
        return 130
    except ValueError as exc:
        parser.error(str(exc))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
