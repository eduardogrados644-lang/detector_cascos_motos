# Visualización, sombras, sonido y capturas en el apartado conductual

**Módulo exclusivo de análisis conductual.** No modifica la detección de cascos ni mezcla sus modelos. Los resultados son **candidatos de revisión**, no prueba de intención, violencia o amenaza. El procesamiento no identifica personas.

## Qué se mejoró

- **Luz y sombras:** la rutina local de mejora de iluminación analiza la luminancia. Solo en imágenes oscuras o con una porción considerable en sombra aplica CLAHE moderado sobre el canal L de LAB. Conserva tamaño y coordenadas de las cajas. No elimina sombras, no recupera detalle ausente ni garantiza precisión; puede amplificar ruido de la cámara. Se puede desactivar.
- **Posturas, proximidad y brazos:** las cajas verdes representan personas con al menos cuatro puntos de pose razonablemente confiables. Se indica ID temporal y número de puntos visibles; no es una puntuación de precisión. Las cajas naranjas señalan avisos para revisión.
- **Posibles agresiones:** el actor del gesto candidato aparece con caja **roja y más gruesa**, y la otra persona con caja **celeste**. Las cajas normales permanecen **verdes**. El panel muestra «REVISAR» durante hasta 45 fotogramas desde el aviso, aunque el evento ya no sea nuevo. Esto NO permite inferir contacto, culpa o agresión real.
- **Posibles armas:** las cajas de objetos observados son **ámbar** y las de candidatos recientes para revisión son **rojas**. Se añade un banner de estado; la marca roja puede continuar hasta 45 fotogramas si la caja nueva se solapa razonablemente con la anterior. No se atribuye el objeto a una persona.
- **Alerta sonora:** en Windows, dos tonos breves cuando aparece un **nuevo candidato** en modo agresiones o armas (por defecto desde `Iniciar_Intencion.bat`). No debe sonar continuamente por cada fotograma. Puede silenciarse; la compatibilidad depende del audio de Windows. Para posturas, proximidad y brazos se activa explícitamente con `--alerta-sonora`.
- **Fotografía manual:** cuando está abierta la ventana de análisis, pulsar **F** guarda la imagen **con anotaciones** aunque no exista aviso; límite de 20 fotos por sesión. La carpeta es `resultados/conductual/capturas_manuales/<modo>/` (el modo de armas usa `armas`). **Q** o **ESC** detienen la ventana.
- **Fotografías y clips de eventos:** `--guardar-capturas` guarda únicamente avisos nuevos (máximo 20 por sesión); `--guardar-clips` solo está disponible en agresiones. No se activan por defecto.

## Preparar la detección de armas

Los pesos especializados no se incluyen en GitHub ni se descargan sin consentimiento. El error «Faltan pesos especializados de armas» indica ausencia del archivo `modelos_intenciones/armas/best.pt`; **no** es un fallo de la cámara.

Ejecutar `Iniciar_Intencion.bat` y elegir **7. Preparar modelo externo de armas**, revisar la URL y advertencia, y escribir **ACEPTO** solo si se confía en el proveedor. También se puede hacer manualmente tras revisar la ficha del modelo externo `https://huggingface.co/cosgun99/gun-knife-yolo11n`:

```bat
python -m pip install huggingface_hub
python scripts/descargar_modelo_armas.py --acepto-modelo-tercero
```

Los archivos `.pt` externos pueden contener código ejecutable al cargarse. No uses pesos de cascos ni YOLO COCO genérico: sus clases no sustituyen un detector `gun/knife`. Tener los pesos solo permite iniciar pruebas; se requieren positivos y negativos **independientes etiquetados** para comprobar sensibilidad y falsas alarmas. También se admiten pesos propios con `--modelo-armas RUTA`.

## Comandos recomendados (CMD, con .venv)

```bat
git pull --ff-only origin mejoras-modulos-vial-conductual
python -m unittest tests.test_mejoras_conductuales tests.test_iniciadores tests.test_armas_acciones_experimentales -v
Iniciar_Intencion.bat --demo-armas --usb 1 --imgsz 480 --conf 0.35 --guardar-capturas
Iniciar_Intencion.bat --demo-agresiones --usb 1 --guardar-capturas --guardar-clips --guardar-video
Iniciar_Intencion.bat --demo-posturas --usb 1 --alerta-sonora
Iniciar_Intencion.bat --demo-proximidad --usb 1 --alerta-sonora
Iniciar_Intencion.bat --demo-brazos --usb 1 --alerta-sonora
```

- Para apagar sonido incluso en armas/agresiones: `--sin-alerta-sonora`.
- Para dejar el fotograma sin ajuste de luz: `--sin-mejorar-iluminacion`.
- Para usar imagen de entrada más grande: `--imgsz 480` o `--imgsz 640` (mayor coste de CPU/RAM y sin garantía de mejora). En laptop de 4 GB se recomienda empezar con 320 para pose, 480 para armas.
- Para grabar todo el video con las anotaciones: `--guardar-video` (el archivo ocupa espacio).
- Para tomar foto manual: con la ventana de la cámara seleccionada, pulsa **F**.
- Las cámaras externas no cambian automáticamente a la cámara integrada.

## Prueba y aceptación pendiente

1. Confirmar visualmente que la cámara 1 sea **externa** y que una persona iluminada se detecte con caja verde; usar **F** y revisar el archivo.
2. Repetir con persona parcialmente a contraluz y en sombra, manteniendo visible el cuerpo completo. Comparar con `--sin-mejorar-iluminacion`; anotar omisiones y falsos positivos.
3. Probar acercamiento, brazos elevados y gestos simulados **sin contacto violento**, tanto con positivos como negativos cotidianos. Comprobar que no se confunden sombras, ropa o movimientos normales con acciones reales.
4. Verificar cajas rojas/celestes y los dos tonos ante eventos de interacción; en ausencia de candidatos no debe sonar.
5. Para armas, usar solo ejemplos autorizados de objetos o réplicas seguras y negativos como móviles y herramientas. No hacer pruebas con armas reales ni conclusiones automáticas.
6. Anotar precisión por clase, falsos avisos y capturas. Ninguna mejora visual constituye validación del detector.

VideoMAE sigue siendo **solo para archivos de video** y necesita validación temporal aparte; no se ha transformado en un detector de cámara en vivo.
