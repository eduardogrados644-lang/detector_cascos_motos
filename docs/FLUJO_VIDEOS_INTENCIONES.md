# Flujo de recepción y conversión de videos — «Intenciones»

1. **Recibir** AVI, MP4, MOV o MKV. Comprobar integridad, duración, FPS, resolución y procedencia. No importar al dataset vial de cascos.
2. **Licencia**: verificar permiso de uso, y permiso **distinto y explícito** para redistribución en un repositorio público. Para KTH Actions solo verificamos uso no comercial, no redistribución de MP4; registrar URL original en Git.
3. **Convertir a MP4 (H.264)** cuando sea necesario:
   
   ```powershell
   ffmpeg -i entrada.avi -map 0:v:0 -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -movflags +faststart -an salida.mp4
   ```

   Conservar resolución y FPS originales; no ampliar artificialmente cuadros de baja resolución. Guardar hash SHA-256 y duración.
4. **Clasificar**: `datos_intenciones/evaluacion/` para videos reservados de prueba, o `datos_intenciones/entrenamiento/` para futuros videos etiquetados. Nunca usar como entrenamiento los mismos clips de evaluación. Dividir por sujetos y grabaciones completas, no por fotogramas del mismo video.
5. **Guardar**: archivos de video en local o almacenamiento privado; **en Git público solo metadatos, hashes, enlaces originales, código y resultados agregados** hasta confirmar permiso explícito y consentimiento. La raíz del repositorio ya dispone de `.gitattributes` con LFS para `*.mp4`, `*.avi`, pero LFS no sustituye los permisos legales ni la gestión de privacidad.
6. **Validar** con `python iniciar_intencion.py --demo-brazos --fuente "RUTA_AL_MP4" --guardar-capturas --sin-mostrar --max-fotogramas 120` en Windows, preferiblemente con archivos localizados en la carpeta `datos_intenciones/evaluacion/kth/`. Usar `--guardar-video` solo si hace falta y la privacidad lo permite.
7. **Registrar** avisos, falsas alarmas, ausencias de detección, tiempos y versión de modelo. El vídeo `boxing` muestra una acción de entrenamiento, no confirma agresión ni intención del sujeto.

**Fuentes de cámara**: webcam externa USB; celular/tablet Android por Wi-Fi/IP Webcam; cámara IP por RTSP/MJPEG; archivos. No usar la cámara interna automáticamente ni teléfono por USB.

Los permisos/licencias de cada futuro video se evaluarán individualmente. No se promete automatizar la subida de binarios sin autorización.
