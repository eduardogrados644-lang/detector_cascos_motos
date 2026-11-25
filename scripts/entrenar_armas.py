"""Entrenamiento optativo YOLO11n para pistolas. Colab GPU recomendado."""
import argparse
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description='Entrenamiento experimental YOLO de pistolas')
    parser.add_argument('--data', default=str(
        RAIZ / 'datos_intenciones' / 'entrenamiento' / 'armas_gun' / 'data.yaml'))
    parser.add_argument('--epocas', type=int, default=40)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='cpu', choices=('cpu', '0'))
    parser.add_argument('--modelo-base', default='yolo11n.pt')
    args = parser.parse_args(argv)
    if args.epocas < 1 or args.batch < 1 or args.imgsz < 64:
        parser.error('Epocas, batch o resolucion invalidos')
    data = Path(args.data).expanduser().resolve()
    if not data.is_file():
        parser.error('Primero prepara el dataset: no existe ' + str(data))
    import yaml
    with data.open(encoding='utf-8') as archivo:
        config = yaml.safe_load(archivo)
    if config.get('names') != {0: 'gun'}:
        parser.error('Este entrenador solo acepta el dataset gun (una clase)')
    from ultralytics import YOLO
    print('Entrenamiento experimental; para 4 GB RAM se recomienda Colab GPU.')
    print('No detecta cuchillos ni demuestra amenazas o intenciones.')
    YOLO(args.modelo_base).train(
        data=str(data), epochs=args.epocas, imgsz=args.imgsz, batch=args.batch,
        workers=0, cache=False, device=args.device, pretrained=True,
        project=str(RAIZ / 'resultados' / 'entrenamientos' / 'armas'),
        name='pistolas', exist_ok=False,
    )
    print('Revisa resultados/entrenamientos/armas/pistolas*/weights/best.pt')
    print('Tras validar, copia pesos a modelos_intenciones/armas_pistolas/best.pt')


if __name__ == '__main__':
    main()
