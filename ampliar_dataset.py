
r"""Preparar datos revisados de videos y ajustar YOLO11n sin modificar originales.

Proyecto detector_cascos_motos, Windows, VS Code.

REVISAR y aprobar etiquetas, con teclas A/S/Q en la ventana:
    .venv\Scripts\python.exe ampliar_dataset.py revisar --max-por-video 4

PREPARAR (solo incorpora fotos aprobadas una a una):
    .venv\Scripts\python.exe ampliar_dataset.py preparar

ENTRENAR (solo despues de preparar y leer el resumen):
    .venv\Scripts\python.exe ampliar_dataset.py entrenar --epocas 10

Las etiquetas originales de videos son sugerencias automaticas. Solo se
incorporan al entrenamiento fotogramas expresamente APROBADOS en la revision.
"""

import argparse
import csv
import json
import math
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
CLASES = {0: "With Helmet", 1: "Without Helmet", 2: "licence"}
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RESERVAR_POR_GRUPO = 2


def orden_natural(texto):
    return [
        int(p) if p.isdigit() else p.lower()
        for p in re.split(r"(\d+)", texto)
    ]


def leer_etiquetas(ruta: Path, *, admitir_vacia=True):
    if not ruta.is_file():
        raise FileNotFoundError(f"Falta la etiqueta: {ruta}")

    contenido = ruta.read_text(encoding="utf-8-sig")
    lineas = [
        l.strip()
        for l in contenido.splitlines()
        if l.strip()
    ]

    if not lineas and not admitir_vacia:
        return [], Counter()

    conteo = Counter()

    for n, linea in enumerate(lineas, start=1):
        partes = linea.split()

        if len(partes) != 5:
            raise ValueError(
                f"Formato YOLO incorrecto: {ruta}, linea {n}"
            )

        try:
            clase = int(partes[0])
            x, y, ancho, alto = [
                float(v) for v in partes[1:]
            ]

        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Etiqueta no numerica: {ruta}, linea {n}"
            ) from exc

        if clase not in CLASES:
            raise ValueError(
                f"Clase no esperada ({clase}) en {ruta}, linea {n}"
            )

        if not all(
            math.isfinite(v)
            for v in (x, y, ancho, alto)
        ):
            raise ValueError(
                f"Coordenada no finita en {ruta}, linea {n}"
            )

        if not (
            0 <= x <= 1
            and 0 <= y <= 1
            and 0 < ancho <= 1
            and 0 < alto <= 1
        ):
            raise ValueError(
                f"Coordenadas fuera de rango en {ruta}, linea {n}"
            )

        # Un pequeño margen permite el redondeo de la anotacion.
        if (
            x - ancho / 2 < -0.002
            or x + ancho / 2 > 1.002
            or y - alto / 2 < -0.002
            or y + alto / 2 > 1.002
        ):
            raise ValueError(
                f"Caja fuera de imagen en {ruta}, linea {n}"
            )

        conteo[clase] += 1

    return lineas, conteo


def comprobar_clases_yaml(data_yaml: Path):
    try:
        import yaml

    except ImportError as exc:
        raise RuntimeError(
            "Falta PyYAML. Comprueba tu instalacion de ultralytics."
        ) from exc

    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"No existe {data_yaml}"
        )

    informacion = yaml.safe_load(
        data_yaml.read_text(encoding="utf-8-sig")
    )

    if not isinstance(informacion, dict):
        raise TypeError(
            "El archivo data.yaml debe contener "
            "un mapa de configuracion"
        )

    if (
        "names" not in informacion
        or informacion["names"] is None
    ):
        raise ValueError(
            "No se encontraron clases 'names' en data.yaml"
        )

    nombres = informacion["names"]

    if isinstance(nombres, list):
        mapeo = dict(
            enumerate(str(n) for n in nombres)
        )

    elif isinstance(nombres, dict):
        try:
            mapeo = {
                int(k): str(v)
                for k, v in nombres.items()
            }

        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Los indices de las clases en "
                "data.yaml no son validos"
            ) from exc

    else:
        raise TypeError(
            "El campo 'names' de data.yaml "
            "debe ser una lista o un diccionario"
        )

    if mapeo != CLASES:
        raise ValueError(
            f"Las clases del dataset {mapeo} no coinciden "
            f"con las de los fotogramas {CLASES}. "
            "No se mezclaran etiquetas incompatibles."
        )


def localizar_lote(base: Path, numero: str):
    lote = (
        base
        / "datos_videos_pendientes"
        / numero
    )

    if not (lote / "indice.csv").is_file():
        raise FileNotFoundError(
            f"No existe {lote / 'indice.csv'}"
        )

    for carpeta in (
        "images",
        "etiquetas_borrador"
    ):
        if not (lote / carpeta).is_dir():
            raise FileNotFoundError(
                f"No existe {lote / carpeta}"
            )

    return lote


def leer_indice(lote: Path):
    with (
        lote / "indice.csv"
    ).open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        registros = list(csv.DictReader(f))

    if not registros:
        raise ValueError(
            "El indice.csv no contiene fotogramas"
        )

    requeridos = {
        "video",
        "grupo",
        "imagen"
    }

    if not requeridos.issubset(registros[0]):
        raise ValueError(
            f"Faltan columnas: "
            f"{requeridos.difference(registros[0])}"
        )

    return registros


def obtener_reservados(registros):
    grupos = defaultdict(set)

    for r in registros:
        grupos[r["grupo"]].add(r["video"])

    if set(grupos) != {
        "With Helmet",
        "Without Helmet"
    }:
        raise ValueError(
            "Se esperaban 2 grupos de videos, "
            f"encontrados: {sorted(grupos)}"
        )

    reservados = set()

    for grupo, videos in grupos.items():
        ordenados = sorted(
            videos,
            key=orden_natural
        )

        if len(ordenados) <= RESERVAR_POR_GRUPO:
            raise ValueError(
                f"No hay suficientes videos en el grupo {grupo}"
            )

        reservados.update(
            ordenados[-RESERVAR_POR_GRUPO:]
        )

    return reservados


def leer_revisiones(lote: Path):
    ruta = lote / "revisiones.csv"

    if not ruta.is_file():
        return {}

    with ruta.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        filas = list(csv.DictReader(f))

    return {
        fila["imagen"]: fila["decision"]
        for fila in filas
    }


def revisar(
    base: Path,
    numero_lote: str,
    max_por_video: int
):
    if not 1 <= max_por_video <= 30:
        raise ValueError(
            "--max-por-video debe estar entre 1 y 30"
        )

    lote = localizar_lote(
        base,
        numero_lote
    )

    registros = leer_indice(lote)
    reservados = obtener_reservados(registros)
    decisiones = leer_revisiones(lote)

    conteo_por_video = Counter(
        r["video"]
        for r in registros
        if r["imagen"] in decisiones
    )

    candidatos = []

    for r in registros:
        imagen = r["imagen"]

        if (
            r["video"] in reservados
            or imagen in decisiones
            or conteo_por_video[r["video"]] >= max_por_video
        ):
            continue

        etiqueta = (
            lote
            / "etiquetas_borrador"
            / f"{Path(imagen).stem}.txt"
        )

        lineas, _ = leer_etiquetas(etiqueta)

        if not lineas:
            # No generar falsos ejemplos negativos
            # cuando YOLO no detecto objetos.
            continue

        candidatos.append(r)
        conteo_por_video[r["video"]] += 1

    if not candidatos:
        print(
            "No quedan nuevas propuestas para revisar "
            "con este limite."
        )
        print(
            f"Decisiones guardadas: {len(decisiones)}"
        )
        return

    import cv2
    import numpy as np

    ruta_csv = lote / "revisiones.csv"
    nuevo = not ruta_csv.exists()

    print("\n=== REVISION VISUAL DE ETIQUETAS ===")
    print("A = APROBAR todas las cajas de esta imagen")
    print("S = SALTAR una imagen incorrecta o incompleta")
    print("Q o ESC = GUARDAR y salir")
    print("No se modifica ninguna etiqueta existente.")
    print(f"Candidatos nuevos a revisar: {len(candidatos)}")

    ventana = (
        "Revisar etiquetas - "
        "A aprobar / S omitir / Q salir"
    )

    try:
        cv2.namedWindow(
            ventana,
            cv2.WINDOW_NORMAL
        )

    except cv2.error as exc:
        raise RuntimeError(
            "No se pudo abrir la ventana de OpenCV en Windows"
        ) from exc

    with ruta_csv.open(
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as archivo:
        escritor = csv.writer(archivo)

        if nuevo:
            escritor.writerow([
                "imagen",
                "decision"
            ])

        try:
            for numero, r in enumerate(
                candidatos,
                start=1
            ):
                nombre = r["imagen"]

                ruta_vista = (
                    lote
                    / "vistas_previas"
                    / nombre
                )

                if not ruta_vista.is_file():
                    raise FileNotFoundError(
                        f"No existe vista previa {ruta_vista}"
                    )

                imagen = cv2.imdecode(
                    np.fromfile(
                        str(ruta_vista),
                        dtype=np.uint8
                    ),
                    cv2.IMREAD_COLOR
                )

                if imagen is None:
                    raise ValueError(
                        f"No se pudo abrir {ruta_vista}"
                    )

                h, w = imagen.shape[:2]

                escala = min(
                    1.0,
                    1100 / w,
                    720 / h
                )

                if escala < 1:
                    imagen = cv2.resize(
                        imagen,
                        (
                            round(w * escala),
                            round(h * escala)
                        )
                    )

                print(
                    f"[{numero}/{len(candidatos)}] "
                    f"{r['video']} :: {nombre}"
                )

                cv2.imshow(
                    ventana,
                    imagen
                )

                while True:
                    tecla = cv2.waitKey(0) & 0xFF

                    if tecla in (
                        ord("a"),
                        ord("A"),
                        ord("s"),
                        ord("S")
                    ):
                        decision = (
                            "APROBADO"
                            if tecla in (
                                ord("a"),
                                ord("A")
                            )
                            else "OMITIDO"
                        )

                        escritor.writerow([
                            nombre,
                            decision
                        ])

                        archivo.flush()
                        decisiones[nombre] = decision
                        break

                    if tecla in (
                        ord("q"),
                        ord("Q"),
                        27
                    ):
                        print(
                            "Revision detenida. "
                            "Decisiones guardadas: "
                            f"{len(decisiones)}"
                        )
                        return

        finally:
            cv2.destroyAllWindows()

    print(
        "Revision terminada. "
        f"Decisiones totales: {len(decisiones)}"
    )

    print(
        "Aprobadas: "
        f"{sum(v == 'APROBADO' for v in decisiones.values())}"
    )

    print(
        "Ahora puedes preparar el dataset; "
        "se incluiran solo las aprobadas."
    )


def preparar(
    base: Path,
    numero_lote: str
):
    origen = base / "dataset"

    comprobar_clases_yaml(
        origen / "data.yaml"
    )

    lote = localizar_lote(
        base,
        numero_lote
    )

    registros = leer_indice(lote)
    reservados = obtener_reservados(registros)
    decisiones = leer_revisiones(lote)

    if not any(
        valor == "APROBADO"
        for valor in decisiones.values()
    ):
        raise ValueError(
            "Ningun fotograma ha sido aprobado. "
            "Primero ejecuta ampliar_dataset.py "
            "revisar --max-por-video 4"
        )

    destino = base / "dataset_ampliado"

    if destino.exists():
        raise FileExistsError(
            f"Ya existe {destino}. "
            "No se reemplaza: conserva tus datos."
        )

    temporal = Path(
        tempfile.mkdtemp(
            prefix="_dataset_ampliado_tmp_",
            dir=str(base)
        )
    )

    cantidades = Counter()
    cajas = Counter()
    seleccionados = []
    descartados = []

    try:
        for particion in (
            "train",
            "valid",
            "test"
        ):
            imagenes_origen = (
                origen
                / particion
                / "images"
            )

            etiquetas_origen = (
                origen
                / particion
                / "labels"
            )

            if (
                not imagenes_origen.is_dir()
                or not etiquetas_origen.is_dir()
            ):
                raise FileNotFoundError(
                    "No existen las carpetas "
                    f"images/labels de {particion} "
                    f"en {origen}"
                )

            img_dest = (
                temporal
                / particion
                / "images"
            )

            lbl_dest = (
                temporal
                / particion
                / "labels"
            )

            img_dest.mkdir(
                parents=True
            )

            lbl_dest.mkdir(
                parents=True
            )

            imgs = sorted(
                p
                for p in imagenes_origen.iterdir()
                if p.is_file()
                and p.suffix.lower() in EXTENSIONES
            )

            if not imgs:
                raise ValueError(
                    f"No hay imagenes en {imagenes_origen}"
                )

            for img in imgs:
                etiqueta = (
                    etiquetas_origen
                    / f"{img.stem}.txt"
                )

                if etiqueta.exists():
                    _, conteo = leer_etiquetas(
                        etiqueta
                    )
                else:
                    # Negativos sin etiquetas son validos.
                    conteo = Counter()

                nombre = f"base_{img.name}"

                shutil.copy2(
                    img,
                    img_dest / nombre
                )

                if etiqueta.exists():
                    shutil.copy2(
                        etiqueta,
                        lbl_dest
                        / f"base_{img.stem}.txt"
                    )
                else:
                    (
                        lbl_dest
                        / f"base_{img.stem}.txt"
                    ).write_text(
                        "",
                        encoding="utf-8"
                    )

                cantidades[
                    f"original_{particion}"
                ] += 1

                cajas.update(conteo)

        vistos = set()

        for r in registros:
            nombre = r["imagen"]

            if (
                Path(nombre).name != nombre
                or Path(nombre).suffix.lower()
                not in EXTENSIONES
            ):
                raise ValueError(
                    f"Nombre de imagen no valido: {nombre}"
                )

            if nombre in vistos:
                raise ValueError(
                    f"Nombre duplicado en indice.csv: {nombre}"
                )

            vistos.add(nombre)

            if r["video"] in reservados:
                cantidades[
                    "fotogramas_reservados"
                ] += 1
                continue

            decision = decisiones.get(
                nombre
            )

            if decision != "APROBADO":
                cantidades[
                    "fotogramas_rechazados_o_no_revisados"
                ] += 1
                continue

            img = (
                lote
                / "images"
                / nombre
            )

            etiqueta = (
                lote
                / "etiquetas_borrador"
                / f"{Path(nombre).stem}.txt"
            )

            if not img.is_file():
                raise FileNotFoundError(
                    f"No existe la imagen: {img}"
                )

            lineas, conteo = leer_etiquetas(
                etiqueta
            )

            if not lineas:
                descartados.append({
                    "video": r["video"],
                    "imagen": nombre,
                    "razon": (
                        "Sin cajas revisadas: "
                        "evitar falsos negativos"
                    )
                })

                cantidades[
                    "fotogramas_sin_cajas_excluidos"
                ] += 1
                continue

            nombre_dest = "video_" + nombre

            shutil.copy2(
                img,
                temporal
                / "train"
                / "images"
                / nombre_dest
            )

            shutil.copy2(
                etiqueta,
                temporal
                / "train"
                / "labels"
                / f"video_{Path(nombre).stem}.txt"
            )

            seleccionados.append({
                "video": r["video"],
                "imagen": nombre_dest,
                "cajas": len(lineas)
            })

            cantidades[
                "fotogramas_video_entrenamiento"
            ] += 1

            cajas.update(conteo)

        if (
            cantidades[
                "fotogramas_video_entrenamiento"
            ] == 0
        ):
            raise ValueError(
                "No hay fotogramas con anotaciones "
                "para ampliar el dataset"
            )

        import yaml

        configuracion = {
            "path": str(
                destino.resolve()
            ).replace("\\", "/"),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "names": CLASES
        }

        (
            temporal / "data.yaml"
        ).write_text(
            yaml.safe_dump(
                configuracion,
                allow_unicode=True,
                sort_keys=False
            ),
            encoding="utf-8"
        )

        (
            temporal / "videos_reservados.txt"
        ).write_text(
            "\n".join(
                sorted(
                    reservados,
                    key=orden_natural
                )
            ) + "\n",
            encoding="utf-8"
        )

        with (
            temporal / "fotogramas_incluidos.csv"
        ).open(
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "video",
                    "imagen",
                    "cajas"
                ]
            )

            w.writeheader()
            w.writerows(
                seleccionados
            )

        with (
            temporal / "fotogramas_excluidos.csv"
        ).open(
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "video",
                    "imagen",
                    "razon"
                ]
            )

            w.writeheader()
            w.writerows(
                descartados
            )

        resumen = {
            "modelo_origen": (
                "resultados/entrenamientos/"
                "refinado_completo/weights/best.pt"
            ),
            "indice_origen": (
                "datos_videos_pendientes/"
                f"{numero_lote}/indice.csv"
            ),
            "fotogramas_con_revision_individual": sum(
                v == "APROBADO"
                for v in decisiones.values()
            ),
            "total_fotogramas_indice": len(
                registros
            ),
            "videos_reservados": sorted(
                reservados,
                key=orden_natural
            ),
            "cuentas": dict(cantidades),
            "cajas_por_clase": {
                CLASES[k]: v
                for k, v in sorted(
                    cajas.items()
                )
            }
        }

        (
            temporal / "resumen.json"
        ).write_text(
            json.dumps(
                resumen,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        # Publicar solo cuando todo este validado.
        temporal.rename(
            destino
        )

    except Exception:
        shutil.rmtree(
            temporal,
            ignore_errors=True
        )
        raise

    print(
        "\n=== DATASET AMPLIADO PREPARADO ==="
    )

    print(
        f"Carpeta: {destino}"
    )

    print(
        "Imagenes originales train: "
        f"{cantidades['original_train']}"
    )

    print(
        "Fotogramas nuevos incluidos: "
        f"{cantidades['fotogramas_video_entrenamiento']}"
    )

    print(
        "Imagenes de entrenamiento: "
        f"{cantidades['original_train'] + cantidades['fotogramas_video_entrenamiento']}"
    )

    print(
        "Validacion original: "
        f"{cantidades['original_valid']}"
    )

    print(
        "Prueba original: "
        f"{cantidades['original_test']}"
    )

    print(
        "Fotogramas sin cajas excluidos: "
        f"{cantidades['fotogramas_sin_cajas_excluidos']}"
    )

    print(
        "Rechazados o no revisados: "
        f"{cantidades['fotogramas_rechazados_o_no_revisados']}"
    )

    print(
        "Fotogramas reservados de video: "
        f"{cantidades['fotogramas_reservados']}"
    )

    print(
        "Videos reservados (no usados para entrenar):"
    )

    for video in sorted(
        reservados,
        key=orden_natural
    ):
        print(f"  {video}")

    print(
        "Se incluyeron solamente anotaciones "
        "aprobadas en la revision visual."
    )

    print(
        "No se modificaron dataset/, "
        "videos_prueba/ ni best.pt originales."
    )


def entrenar(
    base: Path,
    epocas: int
):
    if not 1 <= epocas <= 100:
        raise ValueError(
            "--epocas debe estar entre 1 y 100"
        )

    datos = (
        base
        / "dataset_ampliado"
        / "data.yaml"
    )

    resumen = (
        base
        / "dataset_ampliado"
        / "resumen.json"
    )

    modelo = (
        base
        / "resultados"
        / "entrenamientos"
        / "refinado_completo"
        / "weights"
        / "best.pt"
    )

    if (
        not datos.is_file()
        or not resumen.is_file()
    ):
        raise FileNotFoundError(
            "Primero ejecuta: ampliar_dataset.py preparar"
        )

    if not modelo.is_file():
        raise FileNotFoundError(
            f"Falta el modelo ya entrenado: {modelo}"
        )

    informacion = json.loads(
        resumen.read_text(
            encoding="utf-8"
        )
    )

    n_nuevas = (
        informacion
        .get("cuentas", {})
        .get(
            "fotogramas_video_entrenamiento",
            0
        )
    )

    if n_nuevas < 1:
        raise ValueError(
            "El conjunto ampliado no incluye "
            "fotogramas nuevos"
        )

    from ultralytics import YOLO

    print(
        f"\nEntrenando desde: {modelo}"
    )
    print(
        f"Dataset: {datos}"
    )
    print(
        f"Fotogramas nuevos: {n_nuevas}"
    )
    print(
        f"Epocas nuevas: {epocas}; "
        "CPU; imgsz=416; batch=2"
    )
    print(
        "Los modelos anteriores y la carpeta "
        "dataset original permaneceran intactos."
    )

    red = YOLO(
        str(modelo)
    )

    red.train(
        data=str(datos),
        epochs=epocas,
        imgsz=416,
        batch=2,
        device="cpu",
        workers=0,
        cache=False,
        amp=False,
        plots=True,
        project=str(
            base
            / "resultados"
            / "entrenamientos"
        ),
        name="videos_revisados",
        exist_ok=False,
        verbose=True
    )

    print(
        "\nEntrenamiento completado. "
        "Busca weights/best.pt en la nueva carpeta."
    )

    print(
        "Todavia hay que comparar el modelo "
        "con el anterior usando videos reservados."
    )


def main():
    p = argparse.ArgumentParser(
        description=(
            "Dataset ampliado y entrenamiento "
            "sin sobrescribir originales"
        )
    )

    sub = p.add_subparsers(
        dest="accion",
        required=True
    )

    r = sub.add_parser(
        "revisar",
        help=(
            "Aprobar u omitir fotogramas "
            "con teclas A/S/Q"
        )
    )

    r.add_argument(
        "--lote",
        default="lote_001",
        help="Lote previamente extraido"
    )

    r.add_argument(
        "--max-por-video",
        type=int,
        default=4,
        help=(
            "Numero maximo de propuestas de "
            "cada video (predeterminado 4)"
        )
    )

    a = sub.add_parser(
        "preparar",
        help=(
            "Copiar originales y fotos aprobadas"
        )
    )

    a.add_argument(
        "--lote",
        default="lote_001",
        help="Lote previamente extraido"
    )

    e = sub.add_parser(
        "entrenar",
        help=(
            "Ajustar modelo refinado "
            "con dataset ampliado"
        )
    )

    e.add_argument(
        "--epocas",
        type=int,
        default=10
    )

    aargs = p.parse_args()

    try:
        if aargs.accion == "revisar":
            revisar(
                BASE,
                aargs.lote,
                aargs.max_por_video
            )

        elif aargs.accion == "preparar":
            preparar(
                BASE,
                aargs.lote
            )

        else:
            entrenar(
                BASE,
                aargs.epocas
            )

    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError
    ) as exc:
        print(
            f"\nERROR: {exc}"
        )

        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
