"""Punto de entrada independiente del apartado conductual.

Preparado para incorporar futuros modelos de armas y acciones observables.
Por ahora sólo permite ejecutar el prototipo experimental de postura YOLO Pose.
No mezcla pesos ni salidas del módulo de cascos y NO infiere intenciones.
"""
import argparse
import subprocess
import sys
from pathlib import Path


RAIZ = Path(__file__).resolve().parent


def mostrar_estado() -> None:
    print("\n=== APARTADO CONDUCTUAL / DETECCIÓN DE EVENTOS ===")
    print("Estado: EXPERIMENTAL, desarrollo especializado pendiente.")
    print("Disponible: prueba de posturas horizontales con YOLO Pose.")
    print("Pendiente: modelos validados de armas, objetos y acciones violentas.")
    print("ATENCIÓN: una postura u objeto no revela la intención de una persona.")
    print("Los resultados nunca son acusaciones ni eventos confirmados.")


def construir_comando_demo(
    fuente: str = "0",
    mostrar: bool = True,
    guardar_video: bool = False,
    max_fotogramas: int = 0,
) -> list[str]:
    """Sólo lanza la demostración conductual, nunca el detector de cascos."""
    if not fuente.strip() or max_fotogramas < 0:
        raise ValueError("Fuente o número de fotogramas inválidos")
    cmd = [
        sys.executable, "-m", "modules.behavioral.pose_detector",
        "--fuente", fuente,
    ]
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
    parser.add_argument(
        "--demo-posturas", action="store_true",
        help="Ejecutar sólo la demostración actual de poses; NO detecta armas/violencia",
    )
    parser.add_argument("--fuente", default="0")
    parser.add_argument("--max-fotogramas", type=int, default=0)
    parser.add_argument("--sin-mostrar", action="store_true")
    parser.add_argument("--guardar-video", action="store_true")
    parser.add_argument("--estado", action="store_true")
    args = parser.parse_args(argv)
    mostrar_estado()
    if args.estado:
        return 0
    try:
        demo = args.demo_posturas
        if not demo:
            print("\n1. Probar el prototipo de posturas (experimental)")
            print("0. Salir sin iniciar análisis")
            opcion = input("Opción: ").strip()
            if opcion == "0":
                return 0
            if opcion != "1":
                raise ValueError("Opción no válida")
            demo = True
        if demo:
            cmd = construir_comando_demo(
                args.fuente,
                mostrar=not args.sin_mostrar,
                guardar_video=args.guardar_video,
                max_fotogramas=args.max_fotogramas,
            )
            print("\nIniciando sólo la prueba de posturas.")
            return subprocess.run(cmd, cwd=RAIZ, check=False).returncode
    except (EOFError, KeyboardInterrupt):
        print("\nOperación cancelada.")
        return 130
    except ValueError as exc:
        parser.error(str(exc))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
