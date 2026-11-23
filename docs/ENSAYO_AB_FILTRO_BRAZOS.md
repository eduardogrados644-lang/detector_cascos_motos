# Ensayo A/B de movimientos de brazos — control de puntos inestables

**Objetivo:** comparar las detecciones de movimientos aparentes de brazos de la versión anterior frente a un filtro experimental de calidad de las articulaciones. Este cambio no detecta golpes, peleas, intención ni armas.

En tres capturas del video `helmet.mp4` (fotogramas 6, 25 y 39) aparecen motociclistas con manos cerca de los manubrios y superposición de conductor/acompañante. La imagen estática no revela la velocidad ni confirma cuál punto varió. Por tanto, estas capturas justifican **una prueba comparativa**, no una afirmación de corrección demostrada.

## Qué cambia y por qué

- Nuevo parámetro optativo `--filtro-brazos` del módulo de «Intenciones» **únicamente** al ejecutar modo `brazos`.
- Conserva `yolo11n-pose.pt`, no carga un modelo vial ni clasifica una moto.
- Calcula la longitud relativa de codo a muñeca comparada entre fotogramas; descarta para la **racha de movimiento rápido** saltos con cambios de longitud de más de 1.6 veces o desplazamientos superiores a 12 longitudes de torso/segundo.
- **No afecta** al contador de brazo elevado; el perfil anterior sigue disponible sin la bandera y como referencia.
- Es un filtro geométrico heurístico: puede eliminar verdaderos gestos y seguir generando falsas alarmas. Debe validarse con clips y anotaciones, especialmente cuando hay oclusión.

## Ejecución comparativa local (Windows / .venv)

```powershell
git pull --ff-only origin mejoras-modulos-vial-conductual
python -m unittest discover -s tests -v
python iniciar_intencion.py --demo-brazos --fuente helmet.mp4 --guardar-capturas --sin-mostrar --max-fotogramas 60
python iniciar_intencion.py --demo-brazos --filtro-brazos --fuente helmet.mp4 --guardar-capturas --sin-mostrar --max-fotogramas 60
```

Se puede añadir `--guardar-video` para estudiar la secuencia. Los resultados del perfil filtrado están separados en `resultados/conductual/eventos_brazos_filtrados.csv` y `capturas_brazos_filtradas_...`. El perfil original sigue usando `eventos_brazos.csv`; los CSV se agregan entre ejecuciones, por lo que hay que comparar **la última sesión** y no el número total de filas.

## Datos para próxima evaluación

Solicitar clips **etiquetados temporalmente** de: brazo quieto mientras se sostiene un manubrio, saludos, brazo elevado, movimientos rápidos naturales, caminata y dos personas cercanas (positivos/negativos de la acción observada). Grabar con consentimiento o usar videos con permisos. Mantenerlos localmente, sin subir datos de personas a GitHub; registrar falsos positivos y negativos en una tabla de evaluación. No usar videos del conjunto de prueba como entrenamiento.

**Cámaras admitidas:** USB de cámara externa; celular/tablet Android por Wi-Fi/IP Webcam; RTSP/MJPEG de cámara IP; archivo local. Ni webcam integrada automática ni teléfono por USB.

**Estado:** unidad y flujo comprobables mediante tests; precisión en escenas reales pendiente de los videos de evaluación.


## Resultado del ensayo con `helmet.mp4` (60 fotogramas)

Comparación de **las ejecuciones mostradas por el usuario** (no del acumulado de CSV históricos):

| Fotograma | Perfil original | Perfil con `--filtro-brazos` | Observación visual |
| --- | --- | --- | --- |
| 6 | ID 1, izquierdo, desplazamiento rápido aparente | Igual | Conducción; manos próximas al manubrio |
| 25 | ID 2, derecho, desplazamiento rápido aparente | Igual | Motociclista de la moto roja |
| 39 | ID 1, derecho, desplazamiento rápido aparente | Igual | Conducción; solapamiento entre conductor y acompañante |

En **esta muestra**: 3 avisos originales y 3 con filtro, en los mismos fotogramas, IDs y lados. **Reducción observada: 0 avisos (0 %)**. No prueba si alguno de esos movimientos físicos fue rápido, ya que se revisaron fotogramas aislados, no la secuencia completa. Tampoco implica sensibilidad, precisión o especificidad del filtro en otros escenarios. En las capturas no se aprecia una agresión y la postura de conductor/acompañante puede producir ambigüedades.

**Decisión:** conservar `--filtro-brazos` sólo como opción de investigación; **no activarlo por defecto, no endurecer umbrales para acertar en un único video y no afirmar que resuelve falsos positivos**. Priorizar evaluación con videos específicos de acciones positivas y negativas antes de modificar el clasificador, entrenar un nuevo modelo o añadir una etiqueta de «pelea».

## Próxima validación solicitada

Necesitamos videos cortos de evaluación, con sujeto completo y acciones visibles (sin usar material privado sin permiso):
1. Brazo levantado y sostenido, con tiempo de inicio y fin.
2. Saludo, movimiento rápido de brazo, **sin agresión**.
3. Persona caminando/conversando con brazos quietos.
4. Dos personas cerca o interactuando pacíficamente; idealmente una toma con oclusión.
5. Si existe material de prueba con permiso, conducción normal con manos en manubrio, como caso negativo.

Para cada clip, identificar **acción visible** y el intervalo temporal; no usar categorías de «intención» inferida. Reservar estos videos para evaluación y no reutilizarlos para entrenar. La próxima revisión decidirá si las reglas geométricas justifican seguirse usando o si requieren un método temporal aprendido.
