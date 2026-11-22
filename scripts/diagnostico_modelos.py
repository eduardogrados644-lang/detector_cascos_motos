"""Compara modelos sobre las MISMAS muestras de un video o una imagen.

No identifica infracciones ni calcula precisión contra verdad-terreno.
Conteos son apariciones por fotograma: son útiles para localizar fallos.
Sólo usa dependencias ya presentes en requirements.txt.
"""
import argparse
import time
from collections import Counter
from pathlib import Path

from modules.traffic.association import normalizar_clase


RAIZ = Path(__file__).resolve().parents[1]
MODELOS_PREDETERMINADOS = (
    "best.pt",
    "resultados/entrenamientos/refinado_completo/weights/best.pt",
    "resultados/entrenamientos/videos_revisados/weights/best.pt",
)
SUFIJOS_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolver_archivo(texto: str) -> Path:
    ruta = Path(texto).expanduser()
    opciones = [ruta, RAIZ / ruta] if not ruta.is_absolute() else [ruta]
    for opcion in opciones:
        if opcion.is_file():
            return opcion.resolve()
    raise FileNotFoundError(
        f"No existe el archivo {texto!r}; comprueba la ruta y Git LFS"
    )


def es_puntero_lfs(ruta: Path) -> bool:
    if ruta.stat().st_size >= 200:
        return False
    with ruta.open("rb") as archivo:
        return archivo.read(60).startswith(
            b"version https://git-lfs.github.com/spec/v1"
        )


def muestras_video(ruta: Path, salto: int, max_muestras: int):
    import cv2

    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        cap.release()
        raise OSError(f"No se puede abrir: {ruta}")
    numero, muestras = 0, 0
    try:
        while muestras < max_muestras:
            disponible, frame = cap.read()
            if not disponible:
                break
            numero += 1
            if (numero - 1) % salto != 0:
                continue
            muestras += 1
            yield numero, frame
    finally:
        cap.release()


def muestras(ruta: Path, salto: int, max_muestras: int):
    import cv2

    if ruta.suffix.lower() in SUFIJOS_IMAGEN:
        imagen = cv2.imread(str(ruta))
        if imagen is None:
            raise OSError(f"No se pudo leer la imagen: {ruta}")
        yield 1, imagen
    else:
        yield from muestras_video(ruta, salto, max_muestras)


def resumir(modelo, ruta: Path, args) -> None:
    nombres = modelo.names
    clases = Counter()
    marcos_con_clase = Counter()
    mejores = {}
    bandas = Counter()
    analizados = 0
    inicio = time.perf_counter()

    for _, frame in muestras(ruta, args.salto, args.max_muestras):
        resultado = modelo.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            device="cpu",
            verbose=False,
        )[0]
        analizados += 1
        vistos = set()
        cajas = resultado.boxes
        if cajas is None or len(cajas) == 0:
            continue
        for indice, confianza in zip(
            cajas.cls.int().cpu().tolist(), cajas.conf.cpu().tolist()
        ):
            nombre = str(nombres[int(indice)])
            clases[nombre] += 1
            vistos.add(nombre)
            mejores[nombre] = max(mejores.get(nombre, 0.0), float(confianza))
            etiqueta = (
                "baja 0.10-0.24" if confianza < 0.25
                else "media 0.25-0.39" if confianza < 0.40
                else "alta >=0.40"
            )
            bandas[(nombre, etiqueta)] += 1
        marcos_con_clase.update(vistos)

    segundos = max(time.perf_counter() - inicio, 0.000001)
    print(f"  Muestras procesadas: {analizados}")
    print(f"  Tiempo: {segundos:.2f} s | {analizados / segundos:.2f} muestras/s")
    if not clases:
        print("  Ninguna clase detectada con el umbral diagnóstico.")
        return
    for nombre, apariciones in sorted(clases.items()):
        print(
            f"  {nombre} -> {normalizar_clase(nombre)}: "
            f"{apariciones} apariciones; {marcos_con_clase[nombre]} muestras; "
            f"conf. máxima {mejores[nombre]:.3f}"
        )
        for banda in ("baja 0.10-0.24", "media 0.25-0.39", "alta >=0.40"):
            if bandas[(nombre, banda)]:
                print(f"    {banda}: {bandas[(nombre, banda)]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnóstico comparativo de clases detectadas en video/imagen"
    )
    parser.add_argument("--fuente", default="prueba_motos.mp4")
    parser.add_argument(
        "--modelo", action="append", dest="modelos",
        help="Repite para comparar varios modelos; por defecto se usan 3 pesos de cascos",
    )
    parser.add_argument("--salto", type=int, default=6)
    parser.add_argument("--max-muestras", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.10)
    args = parser.parse_args()

    if args.salto < 1 or args.max_muestras < 1 or args.imgsz < 64:
        parser.error("Salto, número de muestras o tamaño inválidos")
    if not 0 < args.conf <= 1:
        parser.error("--conf debe estar entre 0 y 1")

    from ultralytics import YOLO

    ruta = resolver_archivo(args.fuente)
    if es_puntero_lfs(ruta):
        raise SystemExit(
            f"El archivo de video es un puntero LFS: {ruta}. Ejecuta git lfs pull"
        )

    modelos = args.modelos or list(MODELOS_PREDETERMINADOS)
    print(f"Fuente: {ruta}")
    print(
        f"Comparación: cada modelo analiza los primeros {args.max_muestras} "
        f"fotogramas muestreados, salto {args.salto}, confianza {args.conf}, "
        f"imgsz {args.imgsz}."
    )
    print("ADVERTENCIA: los conteos no equivalen a precisión, víctimas ni infracciones.")
    for nombre in modelos:
        pesos = resolver_archivo(nombre)
        if es_puntero_lfs(pesos):
            print(f"\n[OMITIDO] {nombre}: puntero LFS; ejecuta git lfs pull")
            continue
        print(f"\n### MODELO: {nombre}")
        try:
            modelo = YOLO(str(pesos))
            print(f"  Clases: {modelo.names}")
            resumir(modelo, ruta, args)
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            print(f"  ERROR al usar el modelo: {exc}")
        finally:
            if "modelo" in locals():
                del modelo


if __name__ == "__main__":
    main()
