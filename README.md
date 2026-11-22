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

**Interpretación**: una asociación geométrica no demuestra que alguien conduzca una moto. El CSV guarda únicamente candidatos de persona sin casco observados en 4 fotogramas analizados consecutivos, vinculados al MISMO track de persona y de moto, y limita los duplicados por moto. Las clases de casco se atribuyen solo cuando su caja cae cerca de la zona superior del cuerpo. Si es ambiguo, queda como desconocido, no como infracción.

Si el CSV está vacío aparte de la cabecera, significa que ningún evento cumplió los filtros, no que todos lleven casco. Revisa las cajas dibujadas y comprueba si la geometría de las etiquetas del dataset coincide con esas reglas. El script NO cuenta pasajeros reales, NO lee caracteres de matrícula y NO determina culpabilidad.

El modelo yolo11n.pt detecta clases generales; usa --modelo-cascos para elegir pesos alternativos de casco. Usa --stride 3 para analizar uno de cada tres fotogramas, o --stride 1 para mayor cobertura a costa de CPU.

Pruebas unitarias del apartado vial:

    .\.venv\Scripts\python.exe -m unittest tests.test_asociacion_vial -v

Después de actualizar la rama, repite la batería completa:

    .\.venv\Scripts\python.exe -m unittest discover -s tests -v
