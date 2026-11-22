# Cierre del apartado VIAL — versión experimental para revisión humana

**Fecha de corte:** 22 de noviembre de 2026  
**Rama de trabajo:** `mejoras-modulos-vial-conductual`  
**Estado:** **versión experimental demostrable con video**; NO sistema de fiscalización automático validado.  
**Modelo por defecto:** `resultados/entrenamientos/videos_revisados/weights/best.pt` (cascos) junto con `yolo11n.pt` (COCO, persona/motocicleta).

## 1. Alcance que se conserva

- Entrada de video local, cámara USB externa por índice explícito y cámara Android por URL IP Webcam. USB/IP están implementadas, pero **no consta una validación física satisfactoria con dispositivo externo** en este corte.
- `--modo cascos`: clasificación experimental de cascos, sin exigir motocicletas.
- `--modo vial`: detecta personas, motocicletas y cajas de casco; genera relaciones geométricas y candidatos que requieren revisión humana.
- Seguimiento independiente de cajas de cabeza, confirmación por varias observaciones y control de reidentificaciones/posibles duplicados.
- Consistencia temporal por zona de cabeza: `incierto` cuando se observa alternancia reciente `con_casco` / `sin_casco`. Guarda `inconsistencias_cascos.csv` y recuentos en `estado_cobertura.json`.
- Salida por sesión con `eventos.csv`, `posibles_duplicados.csv`, `estado_cobertura.json`, `inconsistencias_cascos.csv` y `anotado.mp4` si se solicita.
- El apartado conductual utiliza otros modelos y módulos; **no se fusionan modelos, clases, datasets ni registros**.

## 2. Evidencia comprobada

- 73 pruebas unitarias aprobadas en el equipo Windows del usuario (`python -m unittest discover -s tests -v`) y en CI para Python 3.11 y 3.13 con el commit `ca395e7`.
- `helmet.mp4`, primeros 120 fotogramas originales (40 analizados con salto 3): modelo por defecto, 87 apariciones de persona y 96 de moto, 87 asociaciones; 47 apariciones `con_casco`, 8 `sin_casco`; 1 candidato en fotograma fuente 10; 0 reidentificaciones duplicadas, **0 contradicciones temporales** y 0 asociaciones inciertas. El estado de cobertura fue `asociaciones_experimentales_para_revision`.
- `refinado_completo` en los mismos primeros 120 fotogramas: 28 detecciones `con_casco` y 46 `sin_casco`, 1 candidato. Un conteo mayor no equivale a precisión mayor.
- Revisión visual: dos ocupantes sin casco en motocicleta central; se detecta una situación sin casco, pero las cajas inicialmente se solapan y **no individualizan de forma fiable conductor y pasajero**. En escenas alrededor de los fotogramas 97 y 118, ambos modelos han producido falso `CON_CASCO` sobre el cabello/cabeza descubierta del conductor.
- En video `With Helmet/p16.mp4`, el modo `--modo cascos` produjo una falsa detección `sin_casco` sobre una mano levantada con confianza aproximada 0,60. En `Without Helmet/p16.mp4`, las 100 cajas revisadas correspondieron visualmente a una cabeza descubierta, no a 100 personas.
- En videos `p16` sin motocicletas visibles, la ausencia de asociaciones no demuestra ausencia de casos sin casco. El estado `no_evaluable_sin_motocicletas` evita confundir esa limitación con una evaluación satisfactoria.

**Interpretación:** conteos de **apariciones por fotograma**, no personas únicas ni infracciones confirmadas. Los identificadores de cabeza son temporales, no identidades personales.

## 3. Limitaciones explícitas de la versión experimental

1. Puede confundir cabello o manos con cascos/no cascos, incluso con confianza alta.
2. Puede fusionar en una sola caja COCO a conductor y pasajero; una alerta no implica que se haya distinguido a ambos.
3. La consistencia temporal no corrige errores constantes de una misma clase y puede no detectar alternancia si cambian mucho las cajas.
4. Se requiere evaluación con verdad-terreno por cabeza y videos adicionales para calcular precisión, recall, falsas alarmas y omisiones. **No se afirma porcentaje de precisión**.
5. **No** identifica conductor legal, identidad, matrícula leída por OCR, intención, delito ni infracción confirmada. Los candidatos son revisiones manuales.
6. Entrada de cámara USB externa y Android IP Webcam: compatibilidad de código disponible; prueba real de dispositivo pendiente de verificar.
7. Velocidad de CPU observada en los ensayos es limitada, ~2.5–4.1 fotogramas analizados por segundo, según fuente y configuración. No se garantiza tiempo real.

## 4. Criterio de corte y transición

**Se da por cerrada la fase de implementación y evaluación inicial del prototipo VIAL en video.** Se preservan los resultados y errores conocidos como backlog; no se reentrenan pesos sin datos etiquetados. Se puede comenzar el apartado `Intenciones` de manera independiente.

Este cierre **NO** autoriza desplegar fiscalización, emitir sanciones, presentar la precisión como validada ni fusionar automáticamente `main`. El PR #1 continúa en borrador; cualquier mezcla en la rama principal necesita pruebas integrales y autorización explícita del propietario.

### Validación física pendiente (no bloqueante para iniciar Intenciones)

Cuando exista una cámara externa disponible, hacer una sesión breve con `--usb <indice-verificado>` o `--wifi <URL-del-dispositivo>`, revisar el video en pantalla y comprobar que el CSV y la fuente correspondan al dispositivo elegido. No ejecutar cámaras integradas sin elección explícita ni subir videos personales a GitHub.

### Mejora posterior sugerida (fuera del cierre inicial)

Recopilar y etiquetar, con permisos y sin contaminar videos de evaluación, ejemplos de **cabezas individuales** de conductor/pasajero y negativos difíciles (cabello, manos, mascarillas). Entrenar/evaluar un detector especializado y recalibrar la asociación usando métricas por cabeza y por escena.
