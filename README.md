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
