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
