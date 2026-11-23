"""Comprueba carpetas, YAML y punteros Git LFS sin depender de YOLO."""
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DATOS = ("dataset", "dataset_externo", "dataset_ampliado")
SCRIPTS = (
    "detectar.py", "detectar_refinado.py", "camara_zoom.py", "entrenar.py",
    "modules/traffic/vehicle_detector.py",
    "modules/traffic/road_monitor.py",
    "modules/traffic/helmet_monitor.py",
    "iniciar_cascos.py",
    "modules/camera_sources.py",
    "iniciar_intencion.py",
    "modules/traffic/association.py",
    "modules/behavioral/pose_detector.py",
)


def revisar() -> int:
    errores = 0
    for ruta in SCRIPTS:
        if not (RAIZ / ruta).is_file():
            print(f"[ERROR] Falta: {ruta}")
            errores += 1
    for nombre in DATOS:
        yaml = RAIZ / nombre / "data.yaml"
        if not yaml.is_file():
            print(f"[ERROR] Falta: {nombre}/data.yaml")
            errores += 1
            continue
        contenido = yaml.read_text(encoding="utf-8")
        rutas = dict(re.findall(r"^(train|val|test):\s*(\S+)", contenido, re.MULTILINE))
        for clave in ("train", "val", "test"):
            relativa = rutas.get(clave)
            if not relativa or not (yaml.parent / relativa).is_dir():
                print(f"[ERROR] {nombre}: {clave}={relativa!r} no existe")
                errores += 1
        print(f"[INFO] Revisado: {nombre}")
    for nombre in ("best.pt", "yolo11n.pt"):
        archivo = RAIZ / nombre
        if not archivo.is_file():
            print(f"[AVISO] No existe: {nombre}")
        elif archivo.stat().st_size < 200 and archivo.read_bytes()[:60].startswith(
            b"version https://git-lfs.github.com/spec/v1"
        ):
            print(f"[AVISO] {nombre}: puntero Git LFS; ejecuta git lfs pull")
        else:
            print(f"[INFO] Pesos presentes: {nombre}")
    print(f"Diagnóstico terminado: {errores} errores")
    return 1 if errores else 0


if __name__ == "__main__":
    raise SystemExit(revisar())
