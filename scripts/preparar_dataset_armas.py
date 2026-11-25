"""Importa el ZIP Roboflow armas.v2i.yolov8.zip como dataset YOLO de pistolas.

No ejecuta archivos del ZIP. Mantiene variantes con el mismo origen en un
solo grupo train/val/test para evitar filtraciones entre particiones.
"""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import tempfile
import zipfile

RAIZ = Path(__file__).resolve().parents[1]
CLASES_ESPERADAS = ['arm', 'body', 'face', 'foot', 'gun', 'hand', 'human', 'leg']
FORMATOS = {'.jpg', '.jpeg', '.png'}


def nombres_del_yaml(texto):
    coincidencia = re.search(r"(?m)^names:\s*(\[[^\n]*\])\s*$", texto)
    if not coincidencia:
        raise ValueError("data.yaml debe contener la lista names: [...]")
    nombres = ast.literal_eval(coincidencia.group(1))
    if nombres != CLASES_ESPERADAS:
        raise ValueError("Clases inesperadas. No se reinterpretaran IDs a ciegas: " + str(nombres))
    return nombres


def leer_etiquetas(texto, clase=4):
    nuevas = []
    for linea in texto.splitlines():
        partes = linea.strip().split()
        if not partes:
            continue
        if len(partes) != 5:
            raise ValueError("Anotacion YOLO incorrecta: " + linea[:70])
        try:
            indice = int(partes[0])
            valores = [float(v) for v in partes[1:]]
        except ValueError as exc:
            raise ValueError("Etiqueta YOLO no numerica") from exc
        if not 0 <= indice < len(CLASES_ESPERADAS):
            raise ValueError("Indice de clase fuera de rango")
        if (not all(0 <= v <= 1 for v in valores)
                or not all(v == v and abs(v) != float('inf') for v in valores)
                or valores[2] <= 0 or valores[3] <= 0):
            raise ValueError("Coordenadas YOLO incorrectas")
        if indice == clase:
            nuevas.append('0 ' + ' '.join(partes[1:]))
    return nuevas


def analizar_zip(ruta):
    """Verifica el ZIP y agrupa las variantes del mismo original."""
    with zipfile.ZipFile(ruta) as z:
        info = z.infolist()
        if len(info) > 10000 or sum(i.file_size for i in info) > 250_000_000:
            raise ValueError("ZIP demasiado grande para este importador")
        if 'data.yaml' not in z.namelist():
            raise ValueError("Falta data.yaml")
        nombres_del_yaml(z.read('data.yaml').decode('utf-8-sig'))
        archivos_zip = set(z.namelist())
        entradas = {}
        grupos = defaultdict(list)
        for item in info:
            p = PurePosixPath(item.filename)
            if item.is_dir() or p.suffix.lower() not in FORMATOS:
                continue
            if len(p.parts) != 3 or p.parts[0] not in ('train', 'valid', 'test') or p.parts[1] != 'images':
                continue
            if item.file_size > 12_000_000 or p.name in ('.', '..'):
                raise ValueError("Imagen no valida")
            etiqueta = str(PurePosixPath(p.parts[0], 'labels', p.stem + '.txt'))
            if etiqueta not in archivos_zip:
                raise ValueError('Etiqueta faltante para ' + item.filename)
            nuevas = leer_etiquetas(z.read(etiqueta).decode('utf-8-sig'))
            origen = p.name.split('.rf.', 1)[0]
            entradas[item.filename] = (nuevas, origen)
            grupos[origen].append(item.filename)
        if not entradas:
            raise ValueError("No hay imagenes YOLO compatibles")
        return entradas, grupos


def preparar(archivo_zip, salida, semilla=24):
    archivo_zip = Path(archivo_zip).expanduser().resolve()
    salida = Path(salida).expanduser().resolve()
    if not archivo_zip.is_file():
        raise FileNotFoundError('No se encuentra el ZIP: ' + str(archivo_zip))
    if salida.exists():
        raise FileExistsError('Destino ya existe; no se sobrescribe: ' + str(salida))
    entradas, grupos = analizar_zip(archivo_zip)
    llaves = sorted(grupos)
    random.Random(semilla).shuffle(llaves)
    total = len(entradas)
    metas = {'train': round(total * 0.75), 'valid': round(total * 0.15)}
    asignacion = {}
    cuenta = Counter()
    for k in llaves:
        destino = 'train' if cuenta['train'] < metas['train'] else (
            'valid' if cuenta['valid'] < metas['valid'] else 'test')
        asignacion[k] = destino
        cuenta[destino] += len(grupos[k])

    salida.parent.mkdir(parents=True, exist_ok=True)
    temporal = Path(tempfile.mkdtemp(prefix='preparando_armas_', dir=str(salida.parent)))
    imagenes = Counter()
    cajas = Counter()
    try:
        with zipfile.ZipFile(archivo_zip) as z:
            for nombre, (etiquetas, origen) in sorted(entradas.items()):
                particion = asignacion[origen]
                base = PurePosixPath(nombre).name
                imagen = temporal / particion / 'images' / base
                etiqueta = temporal / particion / 'labels' / (Path(base).stem + '.txt')
                imagen.parent.mkdir(parents=True, exist_ok=True)
                etiqueta.parent.mkdir(parents=True, exist_ok=True)
                imagen.write_bytes(z.read(nombre))
                etiqueta.write_text(
                    ('\n'.join(etiquetas) + '\n') if etiquetas else '', encoding='utf-8')
                imagenes[particion] += 1
                cajas[particion] += len(etiquetas)
        (temporal / 'data.yaml').write_text(
            "train: train/images\nval: valid/images\ntest: test/images\nnames:\n  0: gun\n",
            encoding='utf-8')
        inventario = {
            'fuente': archivo_zip.name,
            'sha256_zip': hashlib.sha256(archivo_zip.read_bytes()).hexdigest(),
            'fuente_roboflow': 'https://universe.roboflow.com/teste-5ider/armas-9dhgl/dataset/2',
            'licencia_declarada': 'CC BY 4.0',
            'clases_originales': CLASES_ESPERADAS,
            'clases_entrenamiento': ['gun'],
            'sin_clase_knife': True,
            'semilla': semilla,
            'criterio_separacion': 'Igual origen antes de .rf. en una misma particion',
            'grupos_originales': len(grupos),
            'imagenes': dict(imagenes),
            'cajas_gun': dict(cajas),
            'nota': 'Revision manual y evaluacion externa pendientes.',
        }
        (temporal / 'procedencia.json').write_text(
            json.dumps(inventario, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8')
        temporal.rename(salida)
    except BaseException:
        shutil.rmtree(temporal, ignore_errors=True)
        raise
    return inventario


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convierte dataset Roboflow a solo gun, sin mezclar variantes")
    parser.add_argument('--zip', required=True, help='Ruta a armas.v2i.yolov8.zip')
    parser.add_argument('--salida', default=str(
        RAIZ / 'datos_intenciones' / 'entrenamiento' / 'armas_gun'))
    parser.add_argument('--semilla', type=int, default=24)
    args = parser.parse_args(argv)
    resumen = preparar(args.zip, args.salida, args.semilla)
    print('Dataset preparado en:', args.salida)
    print('Imagenes:', resumen['imagenes'], 'Cajas gun:', resumen['cajas_gun'])
    print('No incluye cuchillos; revisar las anotaciones antes de entrenar.')


if __name__ == '__main__':
    main()
