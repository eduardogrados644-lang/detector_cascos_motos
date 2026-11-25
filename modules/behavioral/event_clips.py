"""Clips breves opcionales de la vista anotada, con memoria comprimida acotada.

Sólo conserva imágenes si el usuario activa --guardar-clips. No graba audio.
FPS aproximados: el análisis CPU puede procesar a velocidad variable.
"""
from collections import deque
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path


class GrabadorClips:
    def __init__(self, carpeta, antes_s=2.0, despues_s=3.0,
                 max_clips=20, ancho_max=720, fps_muestreo=5.0):
        if (antes_s <= 0 or despues_s <= 0 or max_clips < 1
                or ancho_max < 160 or fps_muestreo <= 0):
            raise ValueError('Parametros invalidos para grabacion')
        self.carpeta = Path(carpeta)
        self.antes_s = antes_s
        self.despues_s = despues_s
        self.max_clips = max_clips
        self.ancho_max = ancho_max
        self.fps_muestreo = fps_muestreo
        self._previos = deque()
        self._activo = None
        self._ultimo_guardado = None
        self.cantidad = 0
        self.archivos = []

    def _comprimir(self, imagen):
        import cv2
        ancho = imagen.shape[1]
        if ancho > self.ancho_max:
            alto = max(2, int(imagen.shape[0] * self.ancho_max / ancho))
            imagen = cv2.resize(imagen, (self.ancho_max, alto))
        ok, datos = cv2.imencode('.jpg', imagen, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            raise OSError('Error al comprimir fotograma')
        return datos.tobytes()

    def procesar(self, imagen, tiempo_s: float, evento=False):
        if not isfinite(tiempo_s) or tiempo_s < 0:
            raise ValueError('Tiempo invalido')
        if (self._ultimo_guardado is not None and tiempo_s - self._ultimo_guardado
                < 1.0 / self.fps_muestreo and not evento):
            return
        self._ultimo_guardado = tiempo_s
        muestra = (tiempo_s, self._comprimir(imagen))
        self._previos.append(muestra)
        while self._previos and tiempo_s - self._previos[0][0] > self.antes_s:
            self._previos.popleft()
        if evento and self.cantidad < self.max_clips:
            if self._activo is None:
                self._activo = {
                    'inicio': tiempo_s, 'fin': tiempo_s + self.despues_s,
                    'muestras': list(self._previos),
                }
            else:
                self._activo['fin'] = min(self._activo['inicio'] + 12.0,
                                          tiempo_s + self.despues_s)
        elif self._activo is not None:
            self._activo['muestras'].append(muestra)
        if self._activo is not None and tiempo_s >= self._activo['fin']:
            self._finalizar()

    def _finalizar(self):
        import cv2
        import numpy as np
        activo, self._activo = self._activo, None
        if activo is None or not activo['muestras'] or self.cantidad >= self.max_clips:
            return
        muestras = activo['muestras']
        primero = cv2.imdecode(np.frombuffer(muestras[0][1], dtype=np.uint8), cv2.IMREAD_COLOR)
        if primero is None:
            raise OSError('Error al decodificar clip')
        alto, ancho = primero.shape[:2]
        ancho -= ancho % 2
        alto -= alto % 2
        self.carpeta.mkdir(parents=True, exist_ok=True)
        nombre = self.carpeta / ('clip_agresiones_' +
            datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '.mp4')
        duracion = max(0.2, muestras[-1][0] - muestras[0][0])
        fps = max(1.0, min(10.0, (len(muestras) - 1) / duracion))
        escritor = cv2.VideoWriter(str(nombre), cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, (ancho, alto))
        if not escritor.isOpened():
            raise OSError('No se pudo abrir video de evento: ' + str(nombre))
        try:
            for _, datos in muestras:
                imagen = cv2.imdecode(np.frombuffer(datos, dtype=np.uint8), cv2.IMREAD_COLOR)
                if imagen is not None:
                    escritor.write(imagen[:alto, :ancho])
        finally:
            escritor.release()
        self.cantidad += 1
        self.archivos.append(nombre)
        print('[CLIP PARA REVISION]', nombre)

    def cerrar(self):
        if self._activo is not None:
            self._finalizar()
        self._previos.clear()
