# Entrenamiento YOLO11n de personas con Google Colab

Objetivo: preparar un modelo **Detect** específico para el módulo experimental de **Intenciones**. Este procedimiento **no** modifica el módulo vial ni confunde proximidad con violencia.

## Cómo ejecutar

1. Abre el cuaderno de [Colab desde GitHub](https://colab.research.google.com/github/eduardogrados644-lang/detector_cascos_motos/blob/mejoras-modulos-vial-conductual/entrenamientos_colab/entrenar_personas.ipynb).
2. En Colab, elige **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU**. Si no hay GPU, la comprobación del cuaderno detendrá el entrenamiento; la disponibilidad depende de Google.
3. El cuaderno clona la rama `mejoras-modulos-vial-conductual` de GitHub para registrar el commit usado. No necesita token ni sube cambios a GitHub.
4. Autoriza el montaje de **Google Drive** desde tu propia sesión y coloca imágenes/etiquetas YOLO de personas en:

```text
Mi unidad/detector_cascos_motos/datasets/personas_v1/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
```

Las etiquetas contienen únicamente la clase `0: person` y las coordenadas normalizadas `0 x y w h`. Los `.txt` vacíos/ausentes se reservan a imágenes sin personas. El cuaderno valida estructura, etiquetas y duplicados por nombre (no garantiza por sí solo ausencia de imágenes casi idénticas).

**No entrenar con** `datos_intenciones/evaluacion/{behave,kth,ut_interaction}`. Esos materiales se reservan a la comparación final. Preparar un dataset **nuevo y con permisos**; si se extraen fotogramas de videos, separar videos completos entre train, val y test antes de extraer imágenes.

5. Ejecuta las celdas en orden. El cuaderno inicia con los pesos preentrenados `yolo11n.pt`, 25 épocas como **primera prueba**, parada anticipada, resolución 640 y lote automático sobre GPU.
6. El resultado se guarda en Drive: `Mi unidad/detector_cascos_motos/modelos/personas/personas_FECHA_utc/best.pt`, `last.pt` y `resumen.json`.
7. Descarga `best.pt` a la carpeta local sugerida `modelos_intenciones/personas/best.pt`. Esta carpeta es una propuesta: el iniciador actual de poses **no admite** pesos Detect como sustituto; la integración real requiere el módulo de detección de personas y pruebas posteriores.

## Comparación después del entrenamiento

En CMD, desde el proyecto:

```bat
yolo detect predict model="modelos_intenciones/personas/best.pt" source="datos_intenciones/evaluacion/behave/fragmentos/04_interaccion_fisica_simulada_136s_152s.mp4" imgsz=640 conf=0.20 classes=0 device=cpu save=True
```

Comparar con `yolo11n.pt` bajo los **mismos** parámetros: personas omitidas, recuadros intermitentes, falsos positivos, cambios de IDs (en prueba track), FPS y RAM. No confundir la métrica de validación del entrenamiento con desempeño en BEHAVE.

## Límites y reglas

- Una imagen o video sin etiquetas de personas no basta para el entrenamiento supervisado.
- La GPU acelera el **entrenamiento**, no vuelve más rápida por sí sola la inferencia en una laptop de 4 GB RAM.
- Un detector de personas por sí solo no reconoce golpes, agresiones, armas ni intenciones.
- La continuidad de un ID y un recuadro estimado requieren una mejora **separada** del seguimiento; no convertir estimaciones en confirmaciones.
- El cuaderno no sincroniza automáticamente pesos a GitHub. Evitar subir datos personales, videos o modelos sin revisar derechos de uso; usar Drive para pesos.
- La rama `mejoras-modulos-vial-conductual` es la rama de trabajo. `main` solamente se actualizará después de validación.
