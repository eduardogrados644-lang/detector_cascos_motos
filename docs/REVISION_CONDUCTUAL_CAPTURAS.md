# Capturas para revisión de los avisos conductuales

La nueva opción `--guardar-capturas` permite guardar **sólo los fotogramas que generaron candidatos**, sin necesitar grabar y compartir todo el video. Es **opcional** y está desactivada por defecto. Máximo: **20 imágenes por ejecución**.

## Instrucciones de prueba (Windows)

En la rama `mejoras-modulos-vial-conductual`, dentro del entorno `.venv`:

```powershell
git pull --ff-only origin mejoras-modulos-vial-conductual
python -m unittest discover -s tests -v
python iniciar_intencion.py --demo-brazos --fuente helmet.mp4 --guardar-capturas --sin-mostrar --max-fotogramas 60
```

El programa imprimirá la carpeta exacta bajo `resultados/conductual/capturas_brazos_...` y los nombres `fotograma_000006.jpg`, etc., **si los avisos ocurren de nuevo**. Los IDs temporales pueden variar entre ejecuciones.

Envía **únicamente las capturas relevantes** para revisar si fueron movimientos reales. Para validar velocidad o secuencias de golpes es mejor subir también un clip breve alrededor del evento, pues una captura inmóvil no demuestra movimiento.

### Otras fuentes admitidas

- **Cámara externa USB**: `python iniciar_intencion.py --demo-brazos --usb 1 --guardar-capturas` (el índice 1 es un ejemplo; comprobar el dispositivo).
- **Celular/tablet Android por Wi-Fi**: `python iniciar_intencion.py --demo-brazos --wifi http://DIRECCION_IP:PUERTO --guardar-capturas` (usar IP Webcam y la URL real del dispositivo).
- **Cámara IP** por RTSP/MJPEG: `--fuente <URL_RTSP_O_MJPEG>`.
- **Archivo local**: `--fuente helmet.mp4`.

No existe modo de celular o tablet por USB en este proyecto. No se activa la cámara integrada automáticamente.

### Alcance y privacidad

La proximidad, una postura o un brazo levantado no implican agresión, arma ni intención. Las imágenes son datos potencialmente personales: guardarlas únicamente con consentimiento o derechos sobre el video; no subirlas a GitHub. `resultados/conductual/` está excluido por `.gitignore`. Las capturas pueden eliminarse localmente después de la revisión. Esta opción no entrena modelos ni calcula precisión.
