# Camara externa continua en Intenciones

El lanzador `Iniciar_Intencion.bat` ejecuta `iniciar_intencion.py`. Para dejar el
detector abierto mientras la camara USB externa este activa, ejecuta:

```bat
Iniciar_Intencion.bat --demo-posturas --usb 1
```

`--max-fotogramas` vale **0** por defecto (sin limite). Si escribes
`--max-fotogramas 100`, **terminara despues de 100 fotogramas** por diseno.

```bat
Iniciar_Intencion.bat --demo-proximidad --usb 1
Iniciar_Intencion.bat --demo-brazos --usb 1
Iniciar_Intencion.bat --demo-posturas --wifi "http://IP_DEL_CELULAR:8080"
```

Si un dispositivo USB o flujo IP/RTSP deja de entregar imagenes temporalmente,
el detector de posturas/proximidad/brazos y el detector de armas intentan reabrir
**la misma fuente** cada segundo; no abren la camara integrada ni saltan a
otro indice. Los historiales de eventos se reinician al reconectar para evitar
interpretar la discontinuidad como un evento persistente. La ventana permanece
abierta durante la reconexion siempre que OpenCV pueda mantenerla; q/Esc o Ctrl+C
permiten interrumpir manualmente. Si la camara se desconecta permanentemente
deberas reconectarla o interrumpir el programa. No se puede garantizar el
comportamiento del hardware/driver en todas las computadoras.

Los videos locales terminan normalmente al llegar al final y nunca
se reproducen en bucle por error. El modo `--demo-violencia` (VideoMAE)
sigue siendo solo para videos guardados.

Para probar sin abrir camaras reales:

```bat
python -m unittest tests.test_recuperacion_camara tests.test_fuentes_externas -v
```

El programa sigue siendo experimental: no determina intencion, amenaza o agresion.
