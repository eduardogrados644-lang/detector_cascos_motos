"""Punto de entrada del apartado de control vial.

Utiliza los dos modelos del apartado vial, nunca el modelo conductual.
El detector experimental requiere validación humana: no emite sanciones.
"""
import argparse
import subprocess
import sys
from pathlib import Path


RAIZ = Path(__file__).resolve().parent
MODELO_CASCOS = "resultados/entrenamientos/videos_revisados/weights/best.pt"


def elegir_fuente() -> str | None:
    print("\n=== CONTROL VIAL: CASCOS Y MOTOCICLETAS ===")
    print("1. Video de prueba incluido (prueba_motos.mp4)")
    print("2. Cámara local (camara:0)")
    print("3. Escribir ruta de video o cámara IP")
    print("0. Salir")
    opcion = input("Opción: ").strip()
    if opcion == "0":
        return None
    if opcion == "1":
        return "prueba_motos.mp4"
    if opcion == "2":
        return "camara:0"
    if opcion == "3":
        ruta = input("Ruta/URL de la fuente: ").strip()
        if not ruta:
            raise ValueError("La fuente no puede estar vacía")
        return ruta
    raise ValueError(f"Opción no válida: {opcion}")


def construir_comando(
    fuente: str,
    modelo_cascos: str = MODELO_CASCOS,
    modelo_vehiculos: str = "yolo11n.pt",
    mostrar: bool = True,
    guardar_video: bool = False,
    stride: int = 3,
    max_fotogramas: int = 0,
) -> list[str]:
    """Sólo prepara argumentos: no carga modelos ni abre cámaras."""
    if not fuente.strip() or not modelo_cascos.strip() or not modelo_vehiculos.strip():
        raise ValueError("Fuente y modelos obligatorios")
    if stride < 1 or max_fotogramas < 0:
        raise ValueError("stride o max_fotogramas inválidos")
    comando = [
        sys.executable, "-m", "modules.traffic.road_monitor",
        "--fuente", fuente,
        "--modelo-vehiculos", modelo_vehiculos,
        "--modelo-cascos", modelo_cascos,
        "--stride", str(stride),
    ]
    if max_fotogramas:
        comando.extend(["--max-fotogramas", str(max_fotogramas)])
    if mostrar:
        comando.append("--mostrar")
    if guardar_video:
        comando.append("--guardar-video")
    return comando


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Iniciador del apartado vial: cascos, personas y motocicletas"
    )
    parser.add_argument("--fuente", help="Video, camara:0 o URL")
    parser.add_argument("--modelo-cascos", default=MODELO_CASCOS)
    parser.add_argument("--modelo-vehiculos", default="yolo11n.pt")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-fotogramas", type=int, default=0)
    parser.add_argument("--guardar-video", action="store_true")
    parser.add_argument("--sin-mostrar", action="store_true")
    args = parser.parse_args(argv)
    try:
        fuente = args.fuente if args.fuente is not None else elegir_fuente()
        if fuente is None:
            print("Saliendo del apartado vial.")
            return 0
        guardar = args.guardar_video
        if args.fuente is None and not guardar:
            guardar = input("¿Guardar video anotado? [s/N]: ").strip().lower() in (
                "s", "si", "sí",
            )
        cmd = construir_comando(
            fuente,
            args.modelo_cascos,
            args.modelo_vehiculos,
            mostrar=not args.sin_mostrar,
            guardar_video=guardar,
            stride=args.stride,
            max_fotogramas=args.max_fotogramas,
        )
        print("\nIniciando apartado VIAL. Los resultados requieren revisión humana.")
        return subprocess.run(cmd, cwd=RAIZ, check=False).returncode
    except (EOFError, KeyboardInterrupt):
        print("\nOperación cancelada.")
        return 130
    except ValueError as exc:
        parser.error(str(exc))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
