# UT-Interaction — tres videos de muestra para evaluación

Origen del ZIP entregado por el usuario: `human_interaction.zip` de la sección de ejemplos del sitio oficial de UT-Interaction:

https://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html

El ZIP incluía únicamente tres archivos AVI (`competition_1_1_xvid.avi`, `competition_1_2_xvid.avi`, `competition_1_3_xvid.avi`). Fueron convertidos localmente a MP4 (H.264, CRF 20, `yuv420p`, audio excluido), manteniendo resolución 720×480 y frecuencia 30000/1001 FPS. Se registraron duración, tamaño y SHA-256 en `manifest.csv`.

**No se suben MP4 ni AVI al repositorio público** porque la redistribución de copias originales o modificadas no está expresamente verificada. Los tres MP4 se entregaron por separado, comprimidos con ruta `datos_intenciones/evaluacion/ut_interaction/`, para guardarlos en la PC del usuario. Los archivos locales quedan excluidos por `.gitignore`.

## Qué se puede evaluar

- La detección simultánea de múltiples personas con YOLO Pose.
- El seguimiento temporal y los cambios de IDs ante cruces y oclusión.
- Los candidatos de proximidad y movimientos aparentes de brazos.

**No debe asumirse** que cada archivo corresponde a una sola clase de acción. El conjunto oficial UT-Interaction incluye actividades como dar la mano, señalar, abrazar, empujar, patear y golpear **simulados**, pero el ZIP de muestras no contenía anotaciones temporales para clasificar segmentos concretos. Esas anotaciones deben recuperarse de las fuentes oficiales y comprobarse antes de generar etiquetas de entrenamiento o calcular precisión.

Comando local de ejemplo:

```powershell
python iniciar_intencion.py --demo-proximidad --fuente "datos_intenciones\evaluacion\ut_interaction\competition_1_1_xvid.mp4" --guardar-capturas --sin-mostrar --max-fotogramas 150
```

Las grabaciones son material de **evaluación**, no entrenamiento; no reutilizar las mismas secuencias como entrenamiento y prueba. Proximidad o movimiento rápido no equivalen a agresión confirmada.
