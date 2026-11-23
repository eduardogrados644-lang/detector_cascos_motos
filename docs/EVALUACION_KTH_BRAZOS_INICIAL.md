# Evaluación inicial del detector de brazos con tres videos KTH

**Objetivo:** registrar lo observado en la terminal del usuario, no medir todavía precisión ni clasificar violencia. Las tres pruebas utilizaron el modo `--demo-brazos` sin `--filtro-brazos`, `--max-fotogramas 120` y `--guardar-capturas`. El modelo de pose es `yolo11n-pose.pt`. Los vídeos son los del catálogo `datos_intenciones/evaluacion/kth/manifest.csv`.

## Resultados medidos

| Acción observable del archivo | Fotogramas analizados | Duración aproximada del tramo | Eventos rápidos | Eventos de brazos elevados | Total de eventos | JPG generados |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Caminata | 120 | 4,8 s | 1 | 0 | 1 | 1 |
| Boxeo de práctica | 120 | 4,8 s | 4 | 0 | 4 | 4 |
| Saludo con brazos | 120 | 4,8 s | 4 | 4 | 8 | 5 |
| **Total** | **360** | **14,4 s** | **9** | **4** | **13** | **10** |

Los **13 eventos** son filas/avisos de brazos, no 13 acciones humanas únicas. En un mismo fotograma puede haber dos eventos (uno por brazo) y una única captura. El archivo `resultados/conductual/eventos_brazos.csv` **agrega** nuevas filas entre ejecuciones; la tabla corresponde solamente a estas tres ejecuciones, extraídas de la terminal, no al CSV acumulado.

El detalle de los fotogramas e IDs temporales está en `resultados_iniciales.csv` (metadatos, sin imágenes).

## Interpretación y límites

- El programa completó el procesamiento de las tres muestras, sin errores indicados por la terminal.
- Caminata generó un candidato de desplazamiento rápido aparente en el fotograma 15. Esto **podría ser** balanceo normal de brazos o un error de estimación de pose; la terminal no permite decidir.
- Boxeo de práctica generó cuatro candidatos de movimiento rápido, pero **ninguno** acredita contacto o agresión: es una clase de ejercicio/acción del dataset.
- Saludo generó movimientos rápidos y posiciones elevadas, compatibles conceptualmente con la actividad de saludo. Es imprescindible revisar imágenes y secuencias para establecer si cada evento tiene una correspondencia observable.
- Los videos KTH tienen **160×120 píxeles a 25 FPS**: tamaño que dificulta la localización precisa de muñecas/codos.
- Solo se procesaron **los primeros 120 fotogramas (4,8 segundos)** por clip, no la totalidad de cada video.
- Los tres archivos tienen el mismo sujeto `person15`; **no** constituyen validación independiente entre personas distintas. No usar estos mismos clips para entrenar y luego reportar eficacia sobre ellos.
- No se pueden calcular precisión, sensibilidad, falsos positivos o tasa de aciertos a partir de estos logs sin anotaciones de referencia por evento y revisión de video.

## Siguientes pasos concretos

1. Revisar capturas locales: caminata `fotograma_000015.jpg`; boxeo `000004`, `000019`, `000097`, `000118`; saludo `000004`, `000016`, `000099`, `000112`, `000113`. **Una captura no prueba velocidad**: cuando exista duda, examinar el MP4 entre fotogramas antes y después.
2. Probar tramos completos, pero conservar un conjunto de evaluación sin ajustar cada umbral para esos mismos ejemplos; registrar anotaciones temporales (acción observada y momento).
3. Comparar caminata/saludo/boxeo como **movimientos observables**, no como clasificación automatizada de actividades: hoy el código solo registra posiciones y desplazamientos de brazos.
4. Reunir escenas negativas (conversación, conducir, caminar) y positivas para una acción observada (brazo levantado o movimiento rápido), idealmente con sujetos/cámaras distintos; medir los errores.
5. Antes de un detector de empujones o golpes, definir etiquetado, análisis de secuencias e interacción entre personas; antes de armas, otro modelo con pesos de objetos y datos autorizados.
6. Conservar videos de KTH localmente, no en el repositorio público; allí solo metadatos por condiciones de redistribución no verificadas. Ver `docs/FLUJO_VIDEOS_INTENCIONES.md`.

**Entradas de cámara para el proyecto:** cámara USB **externa**, celular/tablet Android por **Wi-Fi**, cámaras IP RTSP/MJPEG y videos. No usar celular por USB ni seleccionar webcam integrada automáticamente.
