# Etapa 1 — apartado de Intenciones: proximidad observable

Estado: prototipo conductual experimental. La proximidad física NO demuestra violencia, intención, amenaza ni delito.

## Diseño y separaciones

- Reutiliza el detector de posturas `yolo11n-pose.pt` para obtener cajas de personas e IDs temporales.
- Usa la nueva regla independiente `modules/behavioral/proximity.py`: distancia horizontal y solape vertical normalizados por el tamaño de cajas, sin un límite fijo de 90 píxeles.
- Exige 6 fotogramas consecutivos cercanos para emitir una observación `proximidad_persistente` y espera 90 fotogramas antes de repetirla por pareja.
- Cada par de IDs se sigue por separado. Si se interrumpe la cercanía o cambia un ID, se reinicia la racha.
- No usa clases, pesos, datos ni resultados viales. El modo original `--demo-posturas` sigue disponible.
- Los registros de proximidad se guardan en `resultados/conductual/eventos_proximidad.csv`, separados de `eventos.csv` de posturas, y el video opcional `resultado_proximidad_*.mp4`.

## Prueba local inicial (Windows)

Desde el entorno `.venv` de la rama `mejoras-modulos-vial-conductual`:

```powershell
git pull --ff-only origin mejoras-modulos-vial-conductual
python -m unittest discover -s tests -v
python iniciar_intencion.py --estado
python iniciar_intencion.py --demo-proximidad --fuente helmet.mp4 --guardar-video --sin-mostrar --max-fotogramas 60
```

La prueba evalúa que se ejecute la detección, no que haya una pelea. Un CSV sin eventos significa que no se cumplió la regla de cercanía; no demuestra ausencia de violencia. En espacios concurridos, filas y abrazos pueden salir candidatos de proximidad legítima.

## Próximas ampliaciones (sin cambiar la fase vial)

1. Inspeccionar visualmente los ejemplos de proximidad con pares cercanos y separados para ajustar la geometría según perspectiva.
2. Incorporar movimiento del cuerpo en varias observaciones (puntos COCO Pose de brazos y torso), conservando estados observables sin inferir intención.
3. Separar un futuro detector de armas/objetos con pesos específicos y clases validadas. No atribuir al modelo Pose clases de arma hipotéticas.
4. Validar falsos positivos, cobertura y uso de CPU antes de combinar resultados; la revisión seguirá siendo humana.

## Limitaciones

Reglas geométricas y seguimiento de IDs no identifican personas, peleas ni delitos. La ausencia de detecciones también puede deberse a oclusión o tracking inestable. No se ha realizado validación física con cámara en esta etapa.
