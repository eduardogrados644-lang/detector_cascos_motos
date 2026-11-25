# Modo AGRESIONES: gestos dirigidos entre personas (EXPERIMENTAL)

## Prioridad
La detección de **posibles intentos de agresión** es el foco principal del
apartado INTENCIONES. El detector de objetos (gun/knife) permanece en
`--demo-armas`, independiente y opcional. **No se entrenó un clasificador de
golpes o agresiones**: estas alertas son heurísticas visuales de revisión,
no identifican delitos, culpables, ni intenciones verdaderas.

## Qué observa
Combina YOLO11n-Pose en CPU (personas y puntos hombro/codo/muñeca),
proximidad corporal persistente, movimiento plausible de muñeca hacia
otra persona y continuidad temporal. **Manos levantadas por sí solas**,
saludos, personas quietas o cercanía no generan alerta. Si hay movimiento
dirigido repetido, se rotula **POSIBLE INTENTO DE AGRESION (REVISAR)**;
si además la muñeca se proyecta sobre el torso ajeno en varias
observaciones, rotula **POSIBLE AGRESION (REVISAR)**.
Son indicios geométricos, **no prueba de golpe o contacto**. Pueden
existir falsos positivos (juegos, baile, abrazos, ejercicios, oclusiones),
y falsos negativos (malos ángulos, pequeños cuerpos, manos ocultas).

## Iniciar con cámara USB externa, de forma continua
Desde CMD en VS Code, tras actualizar la rama:

```bat
git pull --ff-only origin mejoras-modulos-vial-conductual
Iniciar_Intencion.bat --demo-agresiones --usb 1 --guardar-video --guardar-capturas --guardar-clips
```

Cambia `--usb 1` si tu cámara tiene otro índice. Con un móvil Android
con IP Webcam:

```bat
Iniciar_Intencion.bat --demo-agresiones --wifi "http://IP_DEL_CELULAR:8080" --guardar-clips
```

No especificar `--max-fotogramas` mantiene una sesión continua, salvo
que se pulse Q/Esc/Ctrl+C o se produzca un fallo de hardware no recuperable.
Se reintenta la misma cámara cada segundo cuando se pierde señal; no se
abre la integrada automáticamente. El clasificador VideoMAE existente
`--demo-violencia` sigue trabajando **sólo con videos guardados**.

## Guardado y privacidad

Todo se almacena **localmente**, no en GitHub, en
`resultados/conductual/`:

- `eventos_agresiones.csv`: tiempo UTC, fuente, fotograma,
  IDs temporales de dos personas, tipo candidato y estado de revisión.
- `resultado_agresiones_*.mp4`: video anotado completo **sólo con
  `--guardar-video`**.
- `capturas_agresiones_*/fotograma_*.jpg`: imágenes anotadas
  de avisos (máximo 20 por sesión), **sólo con `--guardar-capturas`**.
- `clip_agresiones_*.mp4`: segmentos breves de la imagen anotada,
  ~2 segundos anteriores y ~3 posteriores al aviso, **sólo con
  `--guardar-clips`**. No incluyen audio. Se usa JPEG comprimido
  a tamaño reducido para limitar RAM, máximo 20 clips por sesión;
  la duración reproducida es aproximada porque el ritmo de CPU varía.

Una reconexión reinicia el historial de gestos y el búfer de clips para
no mezclar identidades visuales entre cortes de señal. Los IDs son
temporales y no representan identidades personales.

## Pruebas

```bat
python -m unittest tests.test_gestos_dirigidos tests.test_clips_agresion tests.test_iniciadores tests.test_visualizacion_conductual -v
```

Las pruebas no sustituyen una evaluación con videos positivos/negativos
etiquetados, ni verifican precisión en cámaras reales. Evaluar antes
de usar en vigilancia o emitir alarmas automáticas. El apartado VIAL
queda intacto y no se fusiona a `main` sin validación.
