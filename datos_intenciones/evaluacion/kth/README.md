# KTH Actions — tres muestras para evaluación

Sitio y videos originales: https://www.csc.kth.se/cvap/actions/

KTH informa que el conjunto está disponible para **uso no comercial** y pide citar a Schuldt, Laptev y Caputo (ICPR 2004) en publicaciones. La página consultada **no indica permiso explícito de redistribución de copias modificadas en repositorios públicos**, por lo que **estos MP4 no se suben al repositorio público**.

Las 3 muestras fueron proporcionadas por el usuario para conversión local de AVI a MP4/H.264, sin cambiar la resolución/fps ni agregar audio.

| Acción visible | Nombre convertido | Duración | Resolución | FPS |
| --- | --- | --- | --- | --- |
| Caminata | `person15_walking_d1_uncomp.mp4` | 29.64 s | 160×120 | 25 |
| Saludo de manos | `person15_handwaving_d1_uncomp.mp4` | 23.24 s | 160×120 | 25 |
| Boxeo de práctica | `person15_boxing_d1_uncomp.mp4` | 16.32 s | 160×120 | 25 |

Se hicieron conversiones con FFmpeg, codec `libx264`, preset `medium`, `-crf 18`, `-pix_fmt yuv420p`, `-movflags +faststart`, sin pista de audio. Véanse nombres, hashes y URL de origen en `manifest.csv`.

**No se usan como entrenamiento.** Todas las muestras tienen persona `15`, y KTH define divisiones por sujetos; mezclar muestras de una misma persona entre entrenamiento/evaluación introduciría contaminación. Además, el video original puede incluir varias sub-secuencias. Los clips a 160×120 son de baja resolución: su uso para medir detección de muñecas con YOLO Pose puede ser limitado.

Para evaluarlos localmente, coloca los archivos MP4 descargados en esta carpeta. `.gitignore` impide incluirlos accidentalmente en commits. No confundir boxeo de entrenamiento con violencia hacia otra persona.

