# BEHAVE: primer lote de videos para evaluar interacciones

**Objetivo:** comprobar que nuestro apartado `Intenciones` detecta y sigue a varias personas; evitar confundir proximidad con pelea. No entrenar todavía.

## Conversión verificable

Archivos originales aportados por el usuario:
- `1-11200.avi`: MPEG-4, **11200 fotogramas**, **448 s**.
- `47300-58400.avi`: MPEG-4, **11101 fotogramas**, **444.04 s**.

Ambos: 640×480, 25 FPS. Conversión a MP4 con H.264 (`libx264`, `-crf 22`, `-preset veryfast`, `yuv420p`, sin audio). Se verificó el número de fotogramas, resolución, frecuencia y duración de los MP4 finales con `ffprobe`. No se cambiaron pesos de YOLO ni el código de detección.

Para evitar cargar videos de ~7,5 minutos en una laptop con CPU limitada, se extrajeron cuatro ventanas cortas (8–16 segundos), en `datos_intenciones/evaluacion/behave/fragmentos/`:

| Fragmento | Fuente completa | Ventana local | Escenas de referencia según `markup.txt` |
| --- | --- | --- | --- |
| `01_aproximacion_42s_52s.mp4` | `1-11200.mp4` | 42–52 s | `Approach`, marcos globales 1080–1251 |
| `02_caminar_juntos_198s_212s.mp4` | `1-11200.mp4` | 198–212 s | `WalkTogether`, marcos 4992–5260 |
| `03_acercamiento_encuentro_128s_136s.mp4` | `47300-58400.mp4` | 128–136 s | `Approach`/`Meet`/`Split`, marcos 50522–50660 |
| `04_interaccion_fisica_simulada_136s_152s.mp4` | `47300-58400.mp4` | 136–152 s | `Fight`, marcos 50727–51050 |

La primera secuencia empieza en el marco global 1 y la segunda en 47300: hay que **restar el inicio global del archivo** para ubicar los eventos dentro de su video. Los intervalos resumidos pueden incluir escenas superpuestas con otras etiquetas; los límites de tiempo son aproximados.

## Pruebas recomendadas en Windows

Extraer `BEHAVE_intenciones_evaluacion_MP4.zip` directamente en la carpeta raíz `detector_cascos_motos`. La extracción coloca cada MP4 bajo `datos_intenciones/evaluacion/behave/` sin publicarlo en Git.

```powershell
git pull --ff-only origin mejoras-modulos-vial-conductual
python -m unittest discover -s tests -v

python iniciar_intencion.py --demo-proximidad --fuente "datos_intenciones\evaluacion\behave\fragmentos\01_aproximacion_42s_52s.mp4" --guardar-capturas --sin-mostrar --max-fotogramas 180

python iniciar_intencion.py --demo-proximidad --fuente "datos_intenciones\evaluacion\behave\fragmentos\03_acercamiento_encuentro_128s_136s.mp4" --guardar-capturas --sin-mostrar --max-fotogramas 180
```

Si no hay eventos, no significa que el clip esté vacío: el algoritmo exige IDs estables y varios fotogramas de proximidad. Guardar también el CSV y capturas para revisión de errores; **no registrar una pelea como confirmada** a partir de proximidad.

## Procedencia y derechos

- Sitio: https://groups.inf.ed.ac.uk/vision/DATASETS/BEHAVEDATA/INTERACTIONS/index.html
- Anotaciones: https://groups.inf.ed.ac.uk/vision/DATASETS/BEHAVEDATA/INTERACTIONS/markup.txt
- Uso declarado por el proyecto académico: gratuito para investigadores, con cita a Blunsden y Fisher (2010). No hay permiso de redistribución pública confirmado para los MP4 convertidos; por eso no se suben esos videos al GitHub público.
- Los AVI y MP4 mantienen roles de **evaluación**, no de entrenamiento. Conservar los datos de evaluación separados para no medir el mismo material con el que se entrena.

## Evaluación visual y CSV de la prueba BEHAVE (250 fotogramas)

Archivo de salida recibido: `resultado_proximidad_20261123_061957_067840.mp4` (640×588, 250 fotogramas a 25 FPS; franja de 108 px) y `eventos_proximidad.csv`, con **una fila de evento**:

- `fotograma=241`; `track_ids=16|18`; `tipo=proximidad_persistente`; `estado=candidato_para_revision`. Este fotograma aparece aproximadamente en el segundo **9,60** del video, según indexación 1-based.
- Antes, con los valores por defecto `imgsz=320, conf=0.45`, la muestra de 250 fotogramas no registró avisos. La siguiente prueba fue realizada con `imgsz=640, conf=0.20` y generó uno. La coincidencia no demuestra que ambos parámetros sean óptimos ni permite medir precisión.
- Revisión de fotogramas del video anotado: al inicio aparece un cuadro/persona ID 1 y otras personas quedan sin detectar; hacia el final aparecen dos cajas con IDs **16 y 18**, mientras una tercera persona cercana puede aparecer sin caja. Es evidencia de cobertura de detección incompleta y aparente fragmentación del seguimiento, pero **no una medición exacta de cambios de ID**.
- En el fotograma 241, las dos personas con IDs 16 y 18 aparecen próximas en la escena. Por lo tanto el aviso es visualmente plausible para **proximidad**, nunca confirmación de pelea ni de intención.
- No hay prueba de que el video detecte a todas las personas visibles, de continuidad de ID durante toda la secuencia ni de exactitud temporal de interacciones. Las etiquetas oficiales de BEHAVE son externas al modelo de proximidad.

### Ajuste de experiencia de pruebas

El iniciador `iniciar_intencion.py` admite ahora argumentos **opcionales** `--imgsz` y `--conf`, además de `--demo-proximidad`, `--demo-brazos` o `--demo-posturas`. Los valores por defecto siguen siendo **320** y **0.45** cuando no se proporcionan; los módulos de cascos no cambian.

```bat
git pull --ff-only origin mejoras-modulos-vial-conductual
python -m unittest discover -s tests -v
python iniciar_intencion.py --demo-proximidad --fuente "datos_intenciones\evaluacion\behave\fragmentos\03_acercamiento_encuentro_128s_136s.mp4" --imgsz 640 --conf 0.20 --guardar-video --sin-mostrar --max-fotogramas 200
```

Esta prueba en un **segundo fragmento independiente** permitirá observar cobertura y estabilidad antes de cambiar umbrales de proximidad. El valor `conf=0.20` puede aumentar falsos positivos y `imgsz=640` puede ralentizar la CPU; **no cambiar los parámetros por defecto ni evaluar sobre el mismo video usado para ajustar umbrales**.

El MP4 de evidencia y el CSV se mantienen **locales** y no se publican en GitHub.
