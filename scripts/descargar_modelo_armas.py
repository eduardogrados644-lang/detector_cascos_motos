"""Descarga voluntaria de un modelo de terceros. Revisar procedencia y licencia.

Los .pt de PyTorch pueden contener contenido ejecutable al cargarse.
No descargar ni cargar archivos de origen desconocido.
"""
from pathlib import Path
import argparse
import shutil

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
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit(
            "Instala dependencia con python -m pip install huggingface_hub"
        ) from e
    print("Fuente:", "https://huggingface.co/" + REPO)
    origen = Path(hf_hub_download(repo_id=REPO, filename="best.pt"))
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origen, DESTINO)
    print("Pesos descargados:", DESTINO)
    print("Son pesos externos no validados con nuestras cámaras; verificar falsos positivos.")


if __name__ == "__main__":
    main()
