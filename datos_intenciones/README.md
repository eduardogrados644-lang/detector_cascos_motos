# Datos para el apartado «Intenciones»

Esta carpeta organiza referencias y protocolos para **analizar acciones observables**, separada del `dataset/` de cascos y motocicletas.

**Regla:** no subir automáticamente material audiovisual al repositorio público. Cada clip requiere origen, permiso de almacenamiento/redistribución en GitHub, derechos/consentimiento cuando corresponda, calidad verificada y rol de uso claro. Los archivos audiovisuales locales están excluidos por `.gitignore`.

## Estructura

- `evaluacion/kth/manifest.csv`: catálogo de 3 muestras académicas KTH (caminata, saludo, boxeo de práctica), convertidas de AVI a MP4 localmente. GitHub **solo almacena sus metadatos y enlaces a la fuente original**.
- `evaluacion/kth/README.md`: procedencia, resolución y límites de los archivos KTH.
- `evaluacion/ut_interaction/manifest.csv`: 3 muestras AVI de UT-Interaction convertidas localmente a MP4; **solo se registra catálogo** en GitHub.
- `evaluacion/ut_interaction/README.md`: características, licencias y limitaciones de los videos de muestra.
- `../docs/FUENTES_INTERACCIONES_BEHAVE.md`: BEHAVE como reemplazo de TV Human Interactions de Oxford, cuya descarga oficial fue retirada.
- `entrenamiento/README.md`: espacio para futuros conjuntos de **entrenamiento**, actualmente sin videos.
- `../docs/FLUJO_VIDEOS_INTENCIONES.md`: instrucciones de conversión, clasificación y autorización.

## Separación de datos

**Evaluación:** medir el comportamiento del código sin entrenar ni ajustar umbrales contra estos mismos clips.

**Entrenamiento:** incorporar videos etiquetados con derecho de uso y distribuir sujetos/escenas sin contaminación de evaluación. El reconocimiento de «golpes», «caídas» o armas requerirá modelos/datos específicos. `boxing` no significa agresión ni delito.

## Cámaras

Las pruebas futuras pueden realizarse con webcam externa USB; cámara Android (celular/tablet) por Wi-Fi; cámara IP RTSP/MJPEG o archivo MP4. El teléfono **no** se usa como cámara USB y la cámara interna de la laptop no se selecciona automáticamente.

