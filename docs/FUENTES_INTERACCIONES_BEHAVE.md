# Fuentes de videos para interacciones — sustitución de TV Human Interactions

## Cambio de recomendación

La descarga oficial de **TV Human Interaction Dataset (Oxford/VGG)** ya no está disponible: la universidad comunica que retiró ese conjunto mientras revisa datos personales que pueden identificar personas.

Fuente: https://www.robots.ox.ac.uk/~vgg/data/tv_human_interactions/

**No se recomienda recurrir a copias no oficiales para eludir dicha retirada.**

## Alternativa recomendada: BEHAVE (Universidad de Edimburgo)

Sitio oficial: https://groups.inf.ed.ac.uk/vision/DATASETS/BEHAVEDATA/INTERACTIONS/index.html

Tiene escenas con personas acercándose (`Approach`), caminando juntas (`WalkTogether`), reuniéndose (`Meet`), separándose (`Split`), siguiendo (`Following`) y movimientos de pelea **simulada** (`Fight`). Documenta **640×480 a 25 FPS** y disponibilidad para uso de investigación con cita a Blunsden y Fisher (2010).

Descargas sugeridas para prueba:
- Escenas de interacción normal: https://groups.inf.ed.ac.uk/vision/DATASETS/BEHAVEDATA/INTERACTIONS/1-11200.avi (18,2 MB). El sitio identifica estas clases para la secuencia: IG,A,WT,I,S,FO.
- Escenas mixtas con acercamientos, reuniones y pelea simulada: https://groups.inf.ed.ac.uk/vision/DATASETS/BEHAVEDATA/INTERACTIONS/47300-58400.avi (16 MB). Secuencia con clases A,M,RT,IG,WT,FI,S.

La web oficial proporciona datos temporales y archivos XML cuando existen. No todos los segmentos disponen de anotaciones completas. Descargar en local, convertir a MP4 sin modificar FPS/resolución y mantener como evaluación inicialmente.

**Licencia:** la página de Edimburgo permite el uso de investigación; no deducir de eso autorización automática de **redistribución** de los clips en el repositorio GitHub público.

## UT-Interaction recibido

El archivo `human_interaction.zip` enviado por el usuario contenía tres muestras AVI. Se convirtieron a MP4/H.264 en una carpeta separada `datos_intenciones/evaluacion/ut_interaction/`; en GitHub solo están su catálogo `manifest.csv` y el `README.md`, sin incluir binarios.

## Próximas comprobaciones

Comparar dos personas acercándose vs. interactuando sin contacto, y observar saltos de track ID o cambios de proximidad. Los algoritmos actuales son experimentales; no atribuir intención de causar daño. Reservar pruebas/entrenamiento en conjuntos separados y conservar grabaciones con autorización apropiada.
