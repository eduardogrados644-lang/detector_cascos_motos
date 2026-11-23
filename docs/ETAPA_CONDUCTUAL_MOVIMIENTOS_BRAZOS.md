# Etapa 2 — brazos y movimiento aparente (YOLO Pose)

Este módulo es **experimental**: reconoce observaciones geométricas, NO peleas, golpes, armas ni intenciones.

- `modules/behavioral/arm_motion.py` calcula la posición de muñeca con respecto al hombro en unidades de longitud del torso, exigiendo suficiente confianza en hombros, codos, muñecas y caderas.
- Confirma `brazo_elevado_persistente` después de 6 observaciones consecutivas por defecto.
- Señala `desplazamiento_rapido_aparente_de_brazo` cuando se observan al menos 2 desplazamientos rápidos seguidos, medidos respecto a la línea de tiempo del video y no en píxeles/fotograma.
- Rechaza observaciones ambiguas (puntos faltantes, salto de ID, cambio grande de caja, interrupciones de seguimiento). Estas reglas también pueden fallar en videos reales y necesitan evaluación etiquetada.
- Usa el mismo `yolo11n-pose.pt` que posturas y proximidad, sin descargar modelos viales adicionales.

## Comandos de comprobación en Windows

```powershell
git pull --ff-only origin mejoras-modulos-vial-conductual
python -m unittest discover -s tests -v
python iniciar_intencion.py --estado
python iniciar_intencion.py --demo-brazos --fuente helmet.mp4 --guardar-video --sin-mostrar --max-fotogramas 60
```

Los resultados se guardan en `resultados/conductual/eventos_brazos.csv` y, si se solicita, `resultado_brazos_*.mp4`. El archivo `helmet.mp4` sólo comprueba que se ejecute el nuevo módulo: no sirve como conjunto de evaluación de gestos.

## Cuándo subir videos de prueba

**Después de pasar la prueba de ejecución de este módulo**, aportar grabaciones separadas de: (a) levantar un brazo y mantenerlo arriba; (b) saludar o mover el brazo rápidamente; (c) brazos quietos mientras se conversa o camina; (d) dos personas juntas sin agresión. Idealmente, etiquetar tiempos y acciones observables. Si hay personas reales, usar escenas propias con consentimiento o material con permiso.

Los videos de entrenamiento de posibles agresiones/armas se solicitan recién cuando se elijan modelos y etiquetas; no mezclar clips de entrenamiento y evaluación. Mantener videos en la PC, no subirlos al repositorio.

Entradas previstas: cámara externa USB, celular o tablet Android mediante Wi-Fi/IP Webcam, cámaras de red RTSP/MJPEG y archivos de video. No utilizar cámara integrada automáticamente ni conectar celular por USB como cámara del proyecto.
