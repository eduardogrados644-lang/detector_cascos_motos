"""Prueba experimental de seguimiento de PERSONAS con YOLO Detect.

Aislada del análisis de posturas, proximidad y del módulo VIAL.
Los recuadros grises discontinuos son la última caja conocida, NO detección.
"""
import argparse
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path

import cv2
from ultralytics import YOLO

from modules.camera_sources import abrir_captura
from .tracking_boxes import CajaVisual, MemoriaCajas

RAIZ = Path(__file__).resolve().parents[2]

def fuente_video(valor):
    if valor.startswith("camara:"):
        num = valor.split(":", 1)[1]
        if not num.isdecimal() or not 1 <= int(num) <= 20:
            raise ValueError("Para USB externa usa camara:1 a camara:20")
        return int(num)
    if valor.isdecimal():
        raise ValueError("No se abre webcam integrada implícitamente; usa camara:1")
    if valor.startswith(("http://", "https://", "rtsp://")):
        return valor
    ruta = Path(valor).expanduser()
    if not ruta.is_absolute():
        ruta = RAIZ / ruta
    if not ruta.is_file():
        raise FileNotFoundError(f"Video inexistente: {ruta}")
    return str(ruta.resolve())

def detecciones(resultado):
    if resultado.boxes is None:
        return []
    cajas = resultado.boxes.xyxy.cpu().tolist()
    ids = (resultado.boxes.id.int().cpu().tolist() if resultado.boxes.id is not None
           else [None] * len(cajas))
    conf = resultado.boxes.conf.cpu().tolist()
    return [CajaVisual(ids[i], tuple(float(v) for v in c), float(conf[i]))
            for i, c in enumerate(cajas)]

def linea_punteada(imagen, a, b, color=(160, 160, 160)):
    for i in range(0, 12, 2):
        inicio, fin = i / 12, (i + 1) / 12
        p = (round(a[0] + (b[0]-a[0])*inicio), round(a[1] + (b[1]-a[1])*inicio))
        q = (round(a[0] + (b[0]-a[0])*fin), round(a[1] + (b[1]-a[1])*fin))
        cv2.line(imagen, p, q, color, 1)

def vista_anotada(imagen, cajas, rastreador):
    img = imagen.copy()
    h, _ = img.shape[:2]
    for obs in cajas:
        x1, y1, x2, y2 = (int(v) for v in obs.xyxy)
        if obs.ausente:
            for a, b in (((x1,y1),(x2,y1)),((x2,y1),(x2,y2)),
                         ((x2,y2),(x1,y2)),((x1,y2),(x1,y1))):
                linea_punteada(img, a, b)
            color = (160, 160, 160)
            nombre = f"ID {obs.id} - ULTIMA CAJA ({obs.ausente})"
        else:
            color = (30, 220, 50)
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            nombre = f"ID {obs.id}" if obs.id is not None else "PERSONA SIN ID"
        cv2.putText(img, nombre, (max(0,x1), max(16, min(h-4,y1-4))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    img = cv2.copyMakeBorder(img, 0, 70, 0, 0,
                             cv2.BORDER_CONSTANT, value=(25,25,25))
    reales = sum(x.ausente == 0 for x in cajas)
    ausentes = len(cajas) - reales
    cv2.putText(img, f"{rastreador} | REALES {reales} | ULTIMA POSICION {ausentes}",
                (8,h+22), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (235,235,235), 1)
    cv2.putText(img, "Verde=detectado; gris discontinuo=sin deteccion actual",
                (8,h+46), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190,190,190), 1)
    return img

def main(argv=None):
    parser=argparse.ArgumentParser(description="Seguimiento de personas, sin clasificar violencia")
    parser.add_argument("--fuente",required=True)
    parser.add_argument("--modelo",default="yolo11n.pt")
    parser.add_argument("--rastreador",choices=('botsort.yaml','bytetrack.yaml','config/trackers/bytetrack_buffer60.yaml'),
                        default="botsort.yaml")
    parser.add_argument("--imgsz",type=int,default=640)
    parser.add_argument("--conf",type=float,default=0.20)
    parser.add_argument("--persistencia-visual",type=int,default=6)
    parser.add_argument("--max-fotogramas",type=int,default=0)
    parser.add_argument("--guardar-video",action="store_true")
    parser.add_argument("--mostrar",action="store_true")
    args=parser.parse_args(argv)
    if (not isfinite(args.conf) or not 0 < args.conf <= 1
            or args.imgsz < 64 or args.max_fotogramas < 0
            or not 0 <= args.persistencia_visual <= 60):
        parser.error("Configuración inválida")
    modelo=YOLO(args.modelo)
    if getattr(modelo,"task",None)!="detect":
        raise ValueError("Se requiere YOLO Detect (no Pose)")
    fuente=fuente_video(args.fuente)
    captura=abrir_captura(fuente)
    fps=captura.get(cv2.CAP_PROP_FPS)
    if not isfinite(fps) or fps <= 0:
        fps=20.
    memoria=MemoriaCajas(args.persistencia_visual)
    salida=None
    escritor=None
    n=0
    total_reales=0
    total_ausentes=0
    if args.guardar_video:
        directorio=RAIZ/"resultados"/"conductual"
        directorio.mkdir(parents=True,exist_ok=True)
        salida=directorio/("seguimiento_personas_"+Path(args.rastreador).stem+"_"+
                          datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")+".mp4")
    try:
        while True:
            ok,imagen=captura.read()
            if not ok:
                break
            n+=1
            resultado=modelo.track(imagen, persist=True, tracker=args.rastreador,
                                   imgsz=args.imgsz, conf=args.conf, iou=0.7,
                                   classes=[0], device="cpu",verbose=False)[0]
            cajas=memoria.procesar(n,detecciones(resultado))
            total_reales+=sum(x.ausente==0 for x in cajas)
            total_ausentes+=sum(x.ausente>0 for x in cajas)
            vista=vista_anotada(imagen,cajas,args.rastreador)
            if args.guardar_video:
                if escritor is None:
                    alto,ancho=vista.shape[:2]
                    escritor=cv2.VideoWriter(str(salida),
                        cv2.VideoWriter_fourcc(*"mp4v"),fps,(ancho,alto))
                    if not escritor.isOpened():
                        raise OSError(f"No se pudo guardar: {salida}")
                escritor.write(vista)
            if args.mostrar:
                cv2.imshow("INTENCIONES - seguimiento de personas",vista)
                if cv2.waitKey(1)&0xFF in (27,ord("q")):
                    break
            if args.max_fotogramas and n>=args.max_fotogramas:
                break
    finally:
        captura.release()
        if escritor is not None:
            escritor.release()
        if args.mostrar:
            cv2.destroyAllWindows()
    print(f"Fotogramas: {n} | detecciones reales acumuladas: {total_reales}"
          f" | ultimas cajas mostradas: {total_ausentes}")
    if salida:
        print(f"Video creado: {salida}")
    print("No se clasificaron acciones ni se generaron eventos de violencia")

if __name__=="__main__":
    main()
