# Prioridad funcional: armas y posibles acciones físicas violentas

**Estado:** dos experimentos independientes. Aún NO están validados como sistema de seguridad; no se deben identificar automáticamente víctimas/agresores, delitos o intenciones. Los resultados requieren revisión humana.

## Armas: prueba YOLO11n Detect (CPU, Windows)

Modelo candidato de terceros: [cosgun99/gun-knife-yolo11n](https://huggingface.co/cosgun99/gun-knife-yolo11n). La ficha pública declara licencia MIT, clases gun y knife, pero sus métricas son del autor y pueden no generalizar a cámaras lejanas. Los pesos PyTorch `.pt` pueden contener código al deserializarse; **revisar/trust** el origen antes de ejecutarlos.

Desde CMD del proyecto con `(.venv)`:

```bat
python -m pip install huggingface_hub
python scripts/descargar_modelo_armas.py --acepto-modelo-tercero
python -m modules.behavioral.weapon_detector --fuente "datos_intenciones\evaluacion\behave\fragmentos\04_interaccion_fisica_simulada_136s_152s.mp4" --imgsz 640 --conf 0.35 --min-fotogramas 3 --guardar-video
```

El video BEHAVE 04 es **una escena sin armas etiquetadas** en este proyecto, así que sirve como primer control exploratorio de falsas alarmas, **no como prueba de sensibilidad**. Para verificar pistolas/cuchillos son necesarios videos independientes con objetos reales o réplicas seguras etiquetadas y permiso de uso, y negativos difíciles (móviles, herramientas, cubiertos). Se crea `resultados/conductual/candidatos_armas_*.mp4` y `candidatos_armas_*.csv`. Las cajas se marcan «POSIBLE», no como prueba de amenaza. No subas pesos ni videos a GitHub.

Para probar otra fuente: `--fuente camara:1` abre una cámara USB externa; una URL RTSP/MJPEG compatible también funciona si OpenCV la admite.

## Acciones físicas: prueba temporal VideoMAE (GPU Colab recomendada)

Modelo candidato de terceros: [HappyGook/videomae-violence-detector](https://huggingface.co/HappyGook/videomae-violence-detector), licencia MIT. Evalúa ventanas de **16 fotogramas** con clasificación binaria de clip (violento/no violento) según la ficha del creador. Pesa ~345 MB y tiene más carga computacional que YOLO11n: con 4 GB RAM puede resultar muy lento o agotarse la memoria.

No utiliza YOLO de proximidad para inferir violencia; analiza fotogramas de un segmento. No distingue aún **golpe vs. empujón vs. pelea** ni identifica a las personas implicadas. Para clase 1 «violenta», la correspondencia proviene de la ficha del modelo y debe verificarse con clips positivos y negativos conocidos; `config.json` no aporta un mapeo explícito de clases.

Dependencias adicionales (instalar preferentemente en Colab para no cargar el PC):

```bat
python -m pip install transformers safetensors
```

Prueba de un video (sobre CPU puede ser lenta):

```bat
python -m modules.behavioral.action_video --fuente "datos_intenciones\evaluacion\behave\fragmentos\04_interaccion_fisica_simulada_136s_152s.mp4" --device cpu --max-ventanas 2
```

Salida: `resultados/conductual/candidatos_violencia_*.csv`, con probabilidades **sin calibrar** por intervalo. El modelo es externo; no lo hemos probado aún en nuestras cámaras. Para llevarlo a Colab, habrá que contar allí con el archivo de video o un conjunto autorizado de prueba.

## Integración posterior

1. Verificar primero modelo de armas con positivos/negativos, **sin atribuir amenazas**.
2. Verificar clasificador temporal con actos físicos observables, también con negativos difíciles (abrazos, baile, juego, boxeo deportivo, multitudes).
3. Comparar en videos de evaluación reservados; fijar tolerancias de falsos positivos y fallos.
4. Combinar señales de persona, objeto y acción solo después de comprobar cada componente. Añadir avisos visuales/sonoros con confirmación temporal y enfriamiento.
5. Mantener el módulo VIAL sin cambios; fusionar a `main` únicamente tras pruebas locales reproducibles.

Pruebas sin pesos: `python -m unittest tests.test_armas_acciones_experimentales -v`.

## Dataset Roboflow del usuario (gun, NO knife)

Se puede preparar sin ejecutar ZIP con `scripts/preparar_dataset_armas.py` y entrenar optativamente con `scripts/entrenar_armas.py`. Los pesos `gun` de una sola clase tambien son validos para el detector de armas; este indicara que no puede detectar cuchillos. Consulta `docs/DATASET_ARMAS_ROBOFLOW.md`.

El iniciador `python iniciar_intencion.py --demo-armas` conecta el detector especializado; `--demo-violencia --fuente archivo.mp4` ejecuta la prueba temporal ya existente. Ninguno demuestra una amenaza o agresion. No se han incorporado pesos, imagenes ni videos al repositorio.
