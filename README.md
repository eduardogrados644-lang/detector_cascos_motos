# Detector de cascos, motos y análisis conductual

Proyecto experimental de visión computacional con **dos módulos independientes**.

**Control vial:** los scripts existentes detectan cascos, personas sin casco y posibles matrículas de acuerdo con las clases reales del dataset. El detector complementario de personas y motocicletas usa pesos COCO por separado. Falta asociar objetos y comprobar infracciones.

**Análisis conductual:** prueba de estimación de pose con sus propios pesos YOLO11 Pose. Solo registra candidatos de postura horizontal persistente, NO reconoce armas, violencia, caídas confirmadas ni intenciones. Estos objetivos están pendientes de modelos, datos y pruebas específicos.

## Instalación con Visual Studio Code en Windows

Abre Terminal en la carpeta del proyecto. Si ya tienes un entorno virtual puedes conservarlo. Si empiezas desde cero:

    git status
    git lfs install
    git lfs pull
    py -3 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe scripts\diagnostico.py
    .\.venv\Scripts\python.exe -m unittest discover -s tests -v

Selecciona en VS Code el intérprete del entorno virtual. Si no existe el comando py, utiliza python -m venv .venv. La primera ejecución de un modelo nuevo puede descargar pesos desde Internet.

## Detectores existentes

    .\.venv\Scripts\python.exe detectar.py helmet.mp4
    .\.venv\Scripts\python.exe detectar_refinado.py helmet.mp4
    .\.venv\Scripts\python.exe camara_zoom.py --camara 1 --escena motos

No hemos eliminado ni sustituido estos scripts. Sus modelos .pt deben estar realmente descargados con Git LFS.

## Complemento de personas y motocicletas

    .\.venv\Scripts\python.exe -m modules.traffic.vehicle_detector prueba_motos.mp4 --guardar
    .\.venv\Scripts\python.exe -m modules.traffic.vehicle_detector camara:0 --mostrar

Busca las clases person y motorcycle usando pesos de detección COCO (yolo11n.pt). Los conteos son apariciones por fotograma, no personas o vehículos únicos, y todavía NO se relacionan con los cascos.

## Prototipo conductual independiente

    .\.venv\Scripts\python.exe -m modules.behavioral.pose_detector --fuente prueba_motos.mp4 --mostrar
    .\.venv\Scripts\python.exe -m modules.behavioral.pose_detector --fuente 0 --mostrar --max-fotogramas 200

El modelo yolo11n-pose.pt es distinto de best.pt usado para cascos y de los pesos COCO del complemento vial. Requiere identificadores de seguimiento para confirmar observaciones consecutivas. Registra solo eventos candidatos en resultados/conductual/eventos.csv. La opción --guardar-video produce un video anotado local.

El CSV indica candidatos para revisión humana: una postura horizontal también puede corresponder a alguien sentado, descansando o realizando una actividad normal. No deben tratarse estos avisos como prueba de violencia.

## Diagnósticos, pruebas y pendientes

El script de diagnóstico revisa las rutas del dataset y detecta punteros de pesos en Git LFS. Las pruebas unitarias de los eventos no requieren ni YOLO ni cámara. Esto no sustituye las pruebas reales de video, inferencia y hardware.

Los archivos dataset/data.yaml y dataset_externo/data.yaml se han ajustado a las rutas de sus carpetas. entrenar.py sigue usando dataset original; ampliar_dataset.py entrenar usa dataset_ampliado.

Consulta docs/VS_CODE_SINCRONIZACION.md para actualizar tu computadora y docs/PLAN_AMPLIACION.md para los pasos siguientes.


## Asociación experimental: moto + persona + casco (fase 2)

Este módulo combina **dos modelos diferentes del apartado vial** en un único flujo de video: uno COCO para personas/motocicletas y otro entrenado para casco/sin casco/placa. No utiliza el modelo conductual de pose.

    .\.venv\Scripts\python.exe -m modules.traffic.road_monitor --fuente prueba_motos.mp4 --guardar-video
    .\.venv\Scripts\python.exe -m modules.traffic.road_monitor --fuente prueba_motos.mp4 --mostrar

Los resultados se crean en una carpeta nueva por sesión bajo resultados/vial_integrado/. Allí encontrarás eventos.csv y, solo si se pide, anotado.mp4. Sin --guardar-video no se guardan imágenes de personas.

**Interpretación**: una asociación geométrica no demuestra que alguien conduzca una moto. El CSV guarda únicamente candidatos de persona sin casco observados en 4 fotogramas analizados consecutivos, vinculados al MISMO track de persona y de moto, y limita los duplicados por moto en la versión inicial. En la mejora siguiente la deduplicación pasa a ser por cabeza rastreada. Las clases de casco se atribuyen solo cuando su caja cae cerca de la zona superior del cuerpo. Si es ambiguo, queda como desconocido, no como infracción.

Si el CSV está vacío aparte de la cabecera, significa que ningún evento cumplió los filtros, no que todos lleven casco. Revisa las cajas dibujadas y comprueba si la geometría de las etiquetas del dataset coincide con esas reglas. El script NO cuenta pasajeros reales, NO lee caracteres de matrícula y NO determina culpabilidad.

El modelo yolo11n.pt detecta clases generales; usa --modelo-cascos para elegir pesos alternativos de casco. Usa --stride 3 para analizar uno de cada tres fotogramas, o --stride 1 para mayor cobertura a costa de CPU.

Pruebas unitarias del apartado vial:

    .\.venv\Scripts\python.exe -m unittest tests.test_asociacion_vial -v

Después de actualizar la rama, repite la batería completa:

    .\.venv\Scripts\python.exe -m unittest discover -s tests -v


## Tercera entrega: diagnóstico de modelos y clases (antes de entrenar de nuevo)

Si el monitor vial integrado no encuentra motos o cabezas, **no ajustes
automáticamente las asociaciones ni declares ausencia de infracciones**.
Primero identifica qué clases devuelve cada modelo y con qué confianza.

Desde la raíz de tu proyecto, con el entorno Python activado:

    python -m scripts.diagnostico_modelos --fuente prueba_motos.mp4

Este comando compara sobre los **mismos fotogramas** los pesos originales
best.pt, refinado_completo y videos_revisados. Usa confianza de diagnóstico
0.10, por debajo del umbral operativo, y muestra las apariciones por clase
y banda de confianza; las detecciones débiles NO son infracciones fiables.

Para determinar por qué el modelo COCO no encuentra motos en un video
reservado, compruébalo por separado:

    python -m scripts.diagnostico_modelos --fuente "videos_prueba\Without Helmet\p15.mp4" --modelo yolo11n.pt

Después puedes comparar el monitor integrado con los pesos refinados,
conservando el modelo de vehículos independiente:

    python -m modules.traffic.road_monitor --fuente prueba_motos.mp4 --modelo-cascos "resultados/entrenamientos/refinado_completo/weights/best.pt" --max-fotogramas 180 --guardar-video

El monitor ahora muestra las clases reales y un resumen diferenciado
de detecciones con casco, sin casco y placa, así como las asociaciones
desconocidas. El nombre numberplate se normaliza a placa.

No entrenes con videos reservados usados para evaluación. Si los pesos
refinados no están descargados, ejecuta git lfs pull y vuelve a probar.


## Cuarta entrega: seguimiento independiente de cabezas y varios ocupantes

El análisis anterior detectó un problema de los videos: ByteTrack COCO puede
asignar la misma caja/ID de persona a pasajeros distintos cuando se superponen.
Por eso esta entrega mantiene **dos trackings separados dentro del apartado vial**:
(1) personas y motos con YOLO COCO; (2) cabezas con y sin casco con el modelo
especializado. Esto no incluye el apartado conductual, que sigue independiente.

- En el video anotado se mostrará el ID de cada cabeza cuando esté disponible.
- Si una caja de persona contiene dos cabezas diferenciables, el sistema puede
  conservar dos hipótesis. **No significa que pueda contar ocupantes reales.**
- Los candidatos se deduplican por combinación de moto y cabeza rastreada.
  Si no hay ID de cabeza, usa el ID de persona y una comprobación de continuidad
  de la caja. Dos etiquetas superpuestas de una misma cabeza no dan dos eventos.
- Un salto brusco de caja reinicia la secuencia de confirmación; esto reduce
  confusiones de identidad, pero puede omitir eventos reales en cámaras móviles.
- Los resultados siguen siendo candidatos para revisión, nunca multas.

Prueba recomendada en VS Code (el modelo original no se sobrescribe):

    python -m modules.traffic.road_monitor --fuente prueba_motos.mp4 --modelo-cascos "resultados/entrenamientos/videos_revisados/weights/best.pt" --mostrar --guardar-video

Para comparar sin seguimiento de cabezas:

    python -m modules.traffic.road_monitor --fuente prueba_motos.mp4 --modelo-cascos "resultados/entrenamientos/videos_revisados/weights/best.pt" --sin-seguimiento-cabezas --guardar-video

El archivo eventos.csv incorpora cabeza_track_id (vacío cuando no se obtiene).
Aunque existan dos registros para una moto, no confirma que haya dos ocupantes
diferentes: la estabilidad de IDs se debe comprobar revisando el video.

Verifica los modelos en los videos reservados With Helmet y Without Helmet.
La aprobación del PR hacia main depende de comprobar resultados y falsos positivos
en la computadora donde se ejecuta YOLO.


## Iniciadores de los dos apartados (mismo repositorio)

En la raíz hay **dos archivos para iniciar cada tema por separado**:

- **Control vial (disponible para pruebas):** python iniciar_cascos.py
  - Menú para video de ejemplo, cámara local o fuente personalizada.
  - Por defecto inicia el monitor con dos modelos viales separados:
    YOLO COCO para personas/motos y videos_revisados para cascos.
  - Comando no interactivo:
    python iniciar_cascos.py --fuente prueba_motos.mp4 --guardar-video
- **Análisis conductual/intención (reservado y experimental):**
  python iniciar_intencion.py
  - Muestra el estado; aún NO detecta armas, violencia ni intenciones.
  - Puede ejecutar únicamente la demo existente de poses (independiente):
    python iniciar_intencion.py --demo-posturas --fuente prueba_motos.mp4
  - Para consultar el estado sin abrir cámara:
    python iniciar_intencion.py --estado

Son dos puntos de entrada en **un solo repositorio**, sin crear ramas ni
repositorios separados para cada uno. Los modelos, resultados y reglas de
cada apartado siguen siendo independientes. La infraestructura se podrá
compartir más adelante sin acoplar las inteligencias artificiales.

Para actualizar desde VS Code:
    git status
    git pull --ff-only origin mejoras-modulos-vial-conductual
    python -m unittest discover -s tests -v

El iniciador vial no está validado aún con todos los ángulos/cámaras:
las detecciones siguen siendo candidatos a revisión, no sanciones.
