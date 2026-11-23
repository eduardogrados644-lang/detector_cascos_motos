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
