# Capturas legibles de YOLO Pose en KTH (160 × 120)

Durante las pruebas con `person15_boxing_d1_uncomp.mp4` se observaron cuatro capturas en las que las etiquetas azul/roja ocupaban prácticamente toda la figura. La detección de personas puede funcionar, pero no es posible validar visualmente muñecas y codos con texto superpuesto tan grande.

## Cambio

Se modificó **únicamente el renderizado del apartado conductual**:

- Para la visualización se usa `resultado.plot(labels=False, conf=False, line_width=1, kpt_radius=1)` con líneas finas y sin rótulos grandes automáticos; se dibuja un ID pequeño junto a cada caja.
- Los avisos, **solo candidatos para revisión**, se escriben en una franja inferior de 108 píxeles que no tapa la figura.
- Los clips con resolución 160×120 se presentan en 640×480 con franja adicional, conservando proporciones. Otros clips con resolución >=640×480 mantienen el tamaño de la escena; la franja también se añade.
- El escritor MP4 ahora usa el tamaño de vista ampliado/con franja. Las capturas JPG usarán el mismo formato.
- **YOLO sigue recibiendo fotogramas originales**: el reescalado es visual, no mejora por sí solo detección ni precisión.
- El registro CSV, la lógica de gestos, los pesos y el módulo vial **no cambian**. Las capturas son opcionales y permanecen en `resultados/conductual/` en la computadora local.

## Comprobación local

```powershell
git pull --ff-only origin mejoras-modulos-vial-conductual
python -m unittest discover -s tests -v
python iniciar_intencion.py --demo-brazos --fuente "datos_intenciones\evaluacion\kth\person15_boxing_d1_uncomp.mp4" --guardar-capturas --sin-mostrar --max-fotogramas 120
```

Las capturas nuevas deberían verse más grandes y mostrar claramente las líneas del esqueleto sin mensajes sobre el cuerpo. Si se solicita `--guardar-video`, también cambiará la resolución del video **anotado**; el original KTH local permanece intacto.

**Precauciones**: imágenes de 160×120 conservan pocas señales de muñeca aunque se amplíen. Estas imágenes no confirman golpes, peleas ni intenciones. Para verificar velocidad se necesita una secuencia de video, y para evaluar precisión se requieren etiquetas de referencia por fotograma/acción.
