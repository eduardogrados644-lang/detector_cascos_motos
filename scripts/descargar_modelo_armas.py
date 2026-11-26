"""Descarga voluntaria de un modelo de terceros. Revisar procedencia y licencia.

Los .pt de PyTorch pueden contener contenido ejecutable al cargarse.
No descargar ni cargar archivos de origen desconocido.
"""
from pathlib import Path
import argparse
import shutil
import os

REPO = "cosgun99/gun-knife-yolo11n"
RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "modelos_intenciones" / "armas" / "best.pt"


def main():
    parser = argparse.ArgumentParser(description="Instalar modelo experimental externo de armas")
    parser.add_argument("--acepto-modelo-tercero", action="store_true")
    args = parser.parse_args()
    if not args.acepto_modelo_tercero:
        parser.error(
            "Debes revisar la ficha de https://huggingface.co/" + REPO
            + " y aceptar explícitamente la descarga del modelo de terceros."
        )
    if DESTINO.exists():
        print("Ya existe, no se sobrescribe:", DESTINO)
        return 0
    if os.getenv("HF_HUB_OFFLINE", "").strip().upper() in {"1", "ON", "YES", "TRUE"}:
        print("[HUGGING FACE] La variable HF_HUB_OFFLINE impide descargar los pesos.")
        print('En esta terminal CMD ejecuta: set "HF_HUB_OFFLINE="')
        print("Despues vuelve a ejecutar la opcion 7 del iniciador.")
        return 2
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit(
            "Instala dependencia con python -m pip install huggingface_hub"
        ) from e
    print("Fuente:", "https://huggingface.co/" + REPO)
    try:
        origen = Path(hf_hub_download(repo_id=REPO, filename="best.pt"))
    except Exception as exc:
        print("[HUGGING FACE] No fue posible descargar best.pt.")
        print(f"Detalle ({type(exc).__name__}): {str(exc)[:400]}")
        print("Comprueba Internet y que HF_HUB_OFFLINE no esté activo.")
        print('En CMD puedes desactivarlo con: set "HF_HUB_OFFLINE="')
        return 2
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origen, DESTINO)
    print("Pesos descargados:", DESTINO)
    print("Son pesos externos no validados con nuestras cámaras; verificar falsos positivos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
