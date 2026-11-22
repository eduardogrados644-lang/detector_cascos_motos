"""Punto de entrada independiente del apartado conductual.

Preparado para incorporar futuros modelos de armas y acciones observables.
Por ahora sólo permite ejecutar el prototipo experimental de postura YOLO Pose.
No mezcla pesos ni salidas del módulo de cascos y NO infiere intenciones.
"""
import argparse
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
    print("Pendiente: modelos validados de armas, objetos y acciones violentas.")
    print("ATENCIÓN: una postura u objeto no revela la intención de una persona.")
    print("Los resultados nunca son acusaciones ni eventos confirmados.")


def construir_comando_demo(
    fuente: str = "prueba_motos.mp4",
    mostrar: bool = True,
    guardar_video: bool = False,
    max_fotogramas: int = 0,
    modo: str = "posturas",
) -> list[str]:
    """Sólo lanza la demostración conductual, nunca el detector de cascos."""
    if not fuente.strip() or max_fotogramas < 0:
        raise ValueError("Fuente o número de fotogramas inválidos")
    if modo not in ("posturas", "proximidad"):
        raise ValueError("Modo conductual inválido")
    cmd = [
        sys.executable, "-m", "modules.behavioral.pose_detector",
        "--fuente", validar_fuente(fuente),
    ]
    if modo == "proximidad":
        cmd.extend(["--modo", "proximidad"])
    if mostrar:
        cmd.append("--mostrar")
    if guardar_video:
        cmd.append("--guardar-video")
    if max_fotogramas:
        cmd.extend(["--max-fotogramas", str(max_fotogramas)])
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
    fuentes = parser.add_mutually_exclusive_group()
    fuentes.add_argument("--fuente", help="Archivo de video o URL RTSP/MJPEG")
    fuentes.add_argument(
        "--usb", type=int, help="Índice de cámara USB externa (1 a 20)"
    )
    fuentes.add_argument(
        "--wifi", help="URL IP Webcam de celular/tablet Android"
    )
    parser.add_argument("--max-fotogramas", type=int, default=0)
    parser.add_argument("--sin-mostrar", action="store_true")
    parser.add_argument("--guardar-video", action="store_true")
    parser.add_argument("--estado", action="store_true")
    args = parser.parse_args(argv)
    mostrar_estado()
    if args.estado:
        return 0
    try:
        demo = args.demo_posturas or args.demo_proximidad
        modo = "proximidad" if args.demo_proximidad else "posturas"
        if not demo:
            print("\n1. Probar el prototipo de posturas (experimental)")
            print("2. Observar proximidad persistente (no significa pelea)")
            print("0. Salir sin iniciar análisis")
            opcion = input("Opción: ").strip()
            if opcion == "0":
                return 0
            if opcion not in ("1", "2"):
                raise ValueError("Opción no válida")
            modo = "proximidad" if opcion == "2" else "posturas"
            demo = True
        if demo:
            fuente = seleccionar_fuente(args.fuente, args.usb, args.wifi)
            if fuente is None:
                print("Se canceló la selección de fuente conductual.")
                return 0
            cmd = construir_comando_demo(
                fuente,
                mostrar=not args.sin_mostrar,
                guardar_video=args.guardar_video,
                max_fotogramas=args.max_fotogramas,
                modo=modo,
            )
            print("\nIniciando prueba de proximidad, NO violencia." if modo == "proximidad" else "\nIniciando sólo la prueba de posturas.")
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
