# Dataset de pistolas y recuperacion de respaldo (fase experimental)

## Origen y decisiones

- Respaldo **auditoria_detector_cascos_drive** comparado con la rama `mejoras-modulos-vial-conductual`: `camara_zoom.py`, `entrenar.py`, `ampliar_dataset.py`, `preparar_videos_entrenamiento.py`, `detectar.py` y `detectar_refinado.py` ya existen identicos en el proyecto. Se conservan las versiones actuales; no se copian carpetas duplicadas.
- El ZIP del usuario `armas.v2i.yolov8.zip` es un dataset Roboflow distinto, con licencia declarada **CC BY 4.0**: https://universe.roboflow.com/teste-5ider/armas-9dhgl/dataset/2
- Contiene 391 imagenes y 391 anotaciones (276 train, 75 valid, 40 test). Clases: `arm, body, face, foot, gun, hand, human, leg`; **gun tiene ID 4; no hay clase knife**.
- Hay variantes del mismo archivo original con sufijo `.rf.*` en diferentes particiones. Para reducir filtracion entre entrenamiento y pruebas, el importador agrupa por nombre de origen antes de repartir los ejemplos. Sigue faltando una evaluacion externa y revisar manualmente las cajas.
- El ZIP original y las imagenes no se versionan en GitHub. Se conserva la procedencia y el hash SHA-256 generado durante la importacion.

## Importar el ZIP sin ejecutarlo (CMD de VS Code, Windows)

```bat
git pull --ff-only origin mejoras-modulos-vial-conductual
python scripts\preparar_dataset_armas.py --zip "C:\Users\EVELYN\Downloads\armas.v2i.yolov8.zip"
```

Si el ZIP esta en otra carpeta, cambia la ruta. El resultado local es `datos_intenciones/entrenamiento/armas_gun/` con `data.yaml`, `procedencia.json`, imagenes y etiquetas de **una sola clase gun** (ID 0). No se sobrescribe un destino ya existente. Con semilla 24, este ZIP genero 293/59/39 imagenes y 722/200/150 cajas en train/valid/test respectivamente.

## Entrenamiento optativo (preferir Colab GPU)

```bat
python scripts\entrenar_armas.py --data "datos_intenciones\entrenamiento\armas_gun\data.yaml" --epocas 40 --device cpu
```

En Colab, con el repositorio y dataset accesibles desde el runtime, usa `--device 0` despues de habilitar GPU. Una laptop de 4 GB RAM no es el lugar ideal para entrenar este modelo. No entrenar antes de revisar muestras de etiquetas y comprobar que el dataset tiene derecho de uso. El archivo `best.pt` se genera localmente bajo `resultados/entrenamientos/armas/`.

## Uso de un modelo gun ya entrenado

```bat
python iniciar_intencion.py --demo-armas --fuente "videos_prueba\ejemplo.mp4" --modelo-armas "modelos_intenciones\armas_pistolas\best.pt" --guardar-video --sin-mostrar
```

El detector acepta pesos YOLO especializados con una clase `gun`, `knife`, o ambas; solo anuncia las clases realmente disponibles. El dataset **no produce por si solo un modelo** y no sirve para probar deteccion de cuchillos. Para validacion con videos de pistolas o replicas seguras, conseguir pruebas independientes, anotar verdaderos positivos y falsos avisos (moviles, herramientas, objetos parecidos).

## Clasificacion temporal de posible violencia

```bat
python iniciar_intencion.py --demo-violencia --fuente "datos_intenciones\evaluacion\behave\fragmentos\04_interaccion_fisica_simulada_136s_152s.mp4" --sin-mostrar
```

Es el modelo experimental VideoMAE, aun **no validado** como detector de agresion. Solo procesa video local, no camara en vivo, y devuelve un CSV de candidatos para revision humana. No se infieren intenciones, responsables ni delitos.

## Pruebas sin pesos

```bat
python -m unittest tests.test_dataset_armas tests.test_armas_acciones_experimentales tests.test_iniciadores -v
```

El apartado VIAL y sus pesos permanecen independientes. La incorporacion a `main` requiere pruebas con videos y revision explicita.
