"""Punto de entrada del apartado de control vial.

Utiliza los dos modelos del apartado vial, nunca el modelo conductual.
El detector experimental requiere validación humana: no emite sanciones.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from modules.camera_sources import seleccionar_fuente_vial, validar_fuente_vial


RAIZ = Path(__file__).resolve().parent
MODELO_CASCOS = "resultados/entrenamientos/videos_revisados/weights/best.pt"


def construir_comando(
    fuente: str,
    modelo_cascos: str = MODELO_CASCOS,
    modelo_vehiculos: str = "yolo11n.pt",
    mostrar: bool = True,
    guardar_video: bool = False,
    stride: int = 3,
    max_fotogramas: int = 0,
    modo: str = "vial",
) -> list[str]:
    """Sólo prepara argumentos: no carga modelos ni abre cámaras."""
    if not fuente.strip() or not modelo_cascos.strip() or not modelo_vehiculos.strip():
        raise ValueError("Fuente y modelos obligatorios")
    if stride < 1 or max_fotogramas < 0:
        raise ValueError("stride o max_fotogramas inválidos")
    if modo not in ("vial", "cascos"):
        raise ValueError("Modo inválido; utiliza vial o cascos")
    fuente_validada = validar_fuente_vial(fuente)
    modulo = (
        "modules.traffic.road_monitor" if modo == "vial"
        else "modules.traffic.helmet_monitor"
    )
    comando = [
        sys.executable, "-m", modulo,
        "--fuente", fuente_validada,
        *([] if modo == "cascos" else ["--modelo-vehiculos", modelo_vehiculos]),
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
    fuentes = parser.add_mutually_exclusive_group()
    fuentes.add_argument("--fuente", help="Video, camara:0 (integrada), camara:1 (externa) o URL")
    fuentes.add_argument("--integrada", action="store_true",
                         help="Usar explicitamente la camara integrada del PC (indice 0 habitual)")
    fuentes.add_argument(
        "--usb", type=int,
        help="Índice de cámara USB EXTERNA (1 a 20); la integrada se elige con --integrada",
    )
    fuentes.add_argument(
        "--wifi",
        help="URL de IP Webcam en celular/tablet Android, ej. http://192.168.1.50:8080",
    )
    parser.add_argument("--modelo-cascos", default=MODELO_CASCOS)
    parser.add_argument("--modelo-vehiculos", default="yolo11n.pt")
    parser.add_argument(
        "--modo", choices=("vial", "cascos"), default="vial",
        help="vial: motos/personas/cascos; cascos: sólo cascos sin exigir motos"
    )
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-fotogramas", type=int, default=0)
    parser.add_argument("--guardar-video", action="store_true")
    parser.add_argument("--sin-mostrar", action="store_true")
    args = parser.parse_args(argv)
    try:
        fuente = seleccionar_fuente_vial(args.fuente, args.usb, args.wifi, args.integrada)
        if fuente is None:
            print("Saliendo del apartado vial.")
            return 0
        guardar = args.guardar_video
        if (args.fuente is None and args.usb is None and args.wifi is None
                and not args.integrada and not guardar):
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
            modo=args.modo,
        )
        print("\nIniciando apartado VIAL. Los resultados requieren revisión humana.")
        print("Fuente solicitada:", fuente)
        print("Modo:", args.modo)
        if fuente == "camara:0":
            print("AVISO: el indice 0 suele ser la webcam integrada. Verificalo en pantalla.")
        elif fuente.startswith("camara:"):
            print("AVISO: comprueba en pantalla que el indice elegido corresponda a la USB externa.")
        return subprocess.run(cmd, cwd=RAIZ, check=False).returncode
    except (EOFError, KeyboardInterrupt):
        print("\nOperación cancelada.")
        return 130
    except ValueError as exc:
        parser.error(str(exc))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
