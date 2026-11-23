# Seguimiento de personas — experimento aislado de INTENCIONES

En BEHAVE el detector `yolo11n.pt` original reconoció más personas que el primer `best.pt` entrenado en Colab. Por eso se conserva el original mientras se evalúa el seguimiento. **No se modifica el módulo vial** ni las alertas del módulo conductual.

Ejecutar desde CMD en la raíz del proyecto con `(.venv)`:

```bat
python -m modules.behavioral.person_tracker --fuente "datos_intenciones\evaluacion\behave\fragmentos\04_interaccion_fisica_simulada_136s_152s.mp4" --rastreador botsort.yaml --imgsz 640 --conf 0.20 --persistencia-visual 6 --guardar-video
python -m modules.behavioral.person_tracker --fuente "datos_intenciones\evaluacion\behave\fragmentos\04_interaccion_fisica_simulada_136s_152s.mp4" --rastreador bytetrack.yaml --imgsz 640 --conf 0.20 --persistencia-visual 6 --guardar-video
```

Se guardarán archivos únicos en `resultados/conductual/seguimiento_personas_<rastreador>_<fecha>.mp4`. **Verde** significa detección en el fotograma actual; **gris discontinuo** es la última caja conocida por pocos fotogramas, no una persona detectada de nuevo. Si dos cajas son casi idénticas, se oculta la anterior para evitar duplicación visual. `--persistencia-visual 0` desactiva las cajas anteriores para comparar. No genera CSV de eventos ni inferencias de agresión.

Compara la cantidad de personas detectadas, recuadros que desaparecen, cambios de IDs y velocidad en la laptop. Los IDs de BoT-SORT/ByteTrack pueden cambiar durante oclusiones; este experimento **no garantiza un ID estable ni detecta personas totalmente ocultas**.

Pruebas unitarias: `python -m unittest tests.test_seguimiento_personas -v`. Se pueden usar cámaras USB externas (`camara:1`) e IP sin abrir implícitamente la webcam del portátil. Las grabaciones y resultados quedan locales.
