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


## Segunda entrega: seguimiento y asociación vial conservadora

- Se añade modules/traffic/association.py, con hipótesis geométricas independientes de YOLO y pruebas unitarias sin hardware.
- Se añade modules/traffic/road_monitor.py, que lee una única fuente de video, ejecuta COCO y casco sobre el mismo fotograma y usa ByteTrack para persona/moto.
- Una persona no se asigna a dos motocicletas cuando la escena es ambigua y una etiqueta de casco no se reutiliza para personas diferentes.
- Solo genera candidatos con track ID de moto y persona estables y 4 observaciones consecutivas sin casco, una vez por moto durante la ejecución. No son infracciones verificadas.
- El CSV se guarda localmente; el video anotado es optativo. Se mantiene la independencia del módulo conductual.

### Validaciones todavía necesarias

1. Probar el módulo integrado en videos diferentes y revisar las cajas dibujadas.
2. Medir falsos positivos de asociación; la cercanía espacial no implica conducción y las etiquetas del dataset pueden representar regiones distintas de una cabeza.
3. Ajustar umbrales según evidencias y probar vehículos con varios ocupantes y peatones cercanos.
4. Evaluar consumo de CPU y fluidez antes de añadirlo a main.


## Cuarta entrega: trazabilidad por cabeza (experimental)

- Seguimiento por separado de COCO persona/moto y cabezas con el modelo de cascos.
- Permitir candidatos independientes por cabeza rastreada dentro de una misma moto.
- Supresión de etiquetas duplicadas y contradictorias sobre una cabeza.
- Reinicio de confirmación ante saltos importantes de caja; sin identificar personas.
- El ID de cabeza es temporal y puede cambiar con oclusiones. No hay conteo
  garantizado de pasajeros, reconocimiento de identidades ni infracciones.
- No se toca la IA conductual.

### Validación necesaria para completar control vial
- Revisar el video anotado en un caso de varios ocupantes, verificar cuál cabeza
  corresponde a cada caja, y documentar cambios de ID.
- Probar p15/p16 tanto With Helmet como Without Helmet para detectar falsos
  positivos; no usar estos videos para reentrenar antes de evaluar.
- Medir velocidad y memoria real en Windows con los dos modelos y tracking.
- Sólo después de validar, considerar fusionar el PR de desarrollo a main.
