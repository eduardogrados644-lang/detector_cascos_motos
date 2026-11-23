# BEHAVE — evaluación de interacciones entre personas

**Procedencia:** Universidad de Edimburgo, conjunto BEHAVE Interactions (no confundir con BEHAVE de Max Planck para interacciones persona–objeto).

- Sitio oficial: https://groups.inf.ed.ac.uk/vision/DATASETS/BEHAVEDATA/INTERACTIONS/index.html
- Anotaciones de secuencias y grupos: https://groups.inf.ed.ac.uk/vision/DATASETS/BEHAVEDATA/INTERACTIONS/markup.txt
- Pistas de verdad-terreno para cajas de personas: `1-11200.xgtf` y `47300-58400F.xgtf` enlazadas en el sitio oficial; **no se descargaron ni verificaron aquí**.

Se recibieron los dos AVI directamente del usuario y se convirtieron a MP4/H.264 (libx264, `-crf 22`, `-preset veryfast`, `yuv420p`, sin pista de audio). Se mantuvieron 640×480 píxeles y 25 fotogramas/s. Se crearon cuatro fragmentos cortos de prueba para facilitar la evaluación en CPU. Cada archivo convertido tiene hash SHA-256 y tamaño en los manifiestos.

| Archivo MP4 local | Duración | Fotogramas | Tamaño |
| --- | ---: | ---: | ---: |
| `1-11200.mp4` | 448 s | 11200 | 9.38 MB |
| `47300-58400.mp4` | 444.04 s | 11101 | 8.36 MB |

Los cuatro clips de `fragmentos/` se tomaron de ventanas cercanas a las anotaciones oficiales `Approach`, `WalkTogether`, `Meet`, `Split`, `Fight`. Esas etiquetas reflejan escenas representadas por los participantes, **no la intención que pueda deducirse de un sujeto**. Un mismo fotograma puede contener otras acciones simultáneas. Para utilizar estos intervalos como verdad-terreno, revisar la anotación y segmentación originales.

**No se suben MP4/AVI al GitHub público**, porque no está confirmada la redistribución de copias; el sitio oficial permite uso libre para investigadores y solicita citar a S. Blunsden y R. B. Fisher (2010). La conversión y los clips se entregan al usuario para evaluación local. El `.gitignore` mantiene fuera del control de versiones los videos colocados en `datos_intenciones/`.

## Ejemplo de prueba local

Desde la raíz del proyecto, dentro de `.venv` y con los videos extraídos:

```powershell
python iniciar_intencion.py --demo-proximidad --fuente "datos_intenciones\evaluacion\behave\fragmentos\01_aproximacion_42s_52s.mp4" --guardar-capturas --sin-mostrar --max-fotogramas 180
```

Se recomienda comparar con un fragmento de encuentro y con una escena física **simulada**. El código actual detecta proximidad geométrica, posturas y desplazamientos aparentes: no reconoce ni confirma violencia. Los clips pertenecen al **conjunto de evaluación**, y no deben incorporarse al conjunto de entrenamiento.
