# Plan de mejoras y ampliación

## Primera entrega: estructura sin romper el detector actual

- Corrección de rutas en dataset/data.yaml y dataset_externo/data.yaml.
- Conservación de los seis scripts actuales y de todos los modelos y datasets anteriores.
- Detector complementario de personas/motocicletas con pesos COCO.
- Módulo conductual autónomo con pesos YOLO11 Pose, lectura de video/cámara, seguimiento y eventos candidatos por persistencia temporal.
- Registro CSV sin imágenes personales por defecto, diagnóstico y pruebas unitarias.
- Documentación de actualización local usando Visual Studio Code y Git LFS.

## Estructura

    detector_cascos_motos/
      detectar.py
      detectar_refinado.py
      camara_zoom.py
      entrenar.py
      modules/
        traffic/
          vehicle_detector.py
        behavioral/
          pose_detector.py
          events.py
      scripts/
        diagnostico.py
      tests/
        test_eventos_conductuales.py

## Pendientes para el detector vial

1. Probar los dos modelos en secuencias reservadas, medir FPS/CPU/RAM y falsos positivos.
2. Asociar personas, cascos, motocicletas y posibles matrículas con tracking y geometría; no asumir que la clase With Helmet representa a la persona completa.
3. Confirmar eventos a lo largo de varios fotogramas antes de guardar posibles infracciones.
4. Generar evidencias y registros por vehículo con opciones de privacidad, consentimiento cuando corresponda y control de duplicados.

## Pendientes para el análisis conductual

1. Evaluar pose con grabaciones de prueba y distinguir postura horizontal de caídas reales.
2. Integrar un **modelo independiente y entrenado** de armas y objetos con clases auditadas.
3. Agregar reconocimiento temporal de acciones e interacciones a partir de video.
4. Validar tasas de errores, condiciones de iluminación, oclusiones y recursos de CPU.
5. Implementar alertas revisables por personas, nunca inferencias definitivas sobre intención.

## Condición para llevar estas mejoras a main

Pasar las pruebas unitarias, comprobar la descarga de Git LFS y ejecutar manualmente al menos una imagen/video y cámara para los dos módulos sobre el equipo destino. No se ha realizado aún esa comprobación real en la computadora del usuario.
