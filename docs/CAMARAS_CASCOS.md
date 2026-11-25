# Elegir cámara integrada, externa o celular en Detección de Cascos

La cámara **integrada de la PC** ahora es una fuente opcional y explícita
**solo en el apartado CASCOS / VIAL**. El sistema no la inicia por defecto.
No se cambió el comportamiento de las cámaras de **Intenciones**.

## Menú de inicio (recomendado)

En Windows, desde la carpeta principal, abre `Iniciar_Cascos.bat`
mediante doble clic o ejecuta en CMD:

```bat
Iniciar_Cascos.bat
```

Menú de fuentes:

1. Cámara integrada de la laptop/PC (índice 0, habitual en Windows)
2. Cámara USB externa (elige índice 1 a 20)
3. Cámara del celular/tablet Android (IP Webcam por Wi-Fi)
4. Video de prueba del proyecto
5. Otro archivo de video o URL RTSP/MJPEG
0. Salir

Luego podrás decidir si deseas guardar un video anotado.

## Comandos directos

Cámara integrada de la PC (el modo `vial` reconoce cascos, motos y personas):

```bat
Iniciar_Cascos.bat --integrada
```

Cámara integrada en modo **solo cascos** sin exigir motocicleta:

```bat
Iniciar_Cascos.bat --integrada --modo cascos
```

Cámara USB externa:

```bat
Iniciar_Cascos.bat --usb 1
```

Celular o tablet Android:

```bat
Iniciar_Cascos.bat --wifi "http://IP_DEL_CELULAR:8080"
```

Para grabar video anotado añade `--guardar-video`.
Para una prueba corta añade `--max-fotogramas 100`.
Sin ese límite, los modos por cámara continúan hasta que se interrumpa
la lectura, se cierre la aplicación o se pulse Q/Esc/Ctrl+C.

## Detalles de la integrada

- El índice `0` suele indicar la webcam del equipo, pero **no está
  garantizado**. Comprueba visualmente el dispositivo abierto.
- Para la integrada se solicita **640 × 480 a 30 FPS** a fin de evitar
  capturas HD innecesarias en PCs de 4 GB RAM. El dispositivo puede
  negociar otra resolución; el sistema no garantiza 30 FPS.
- Si no existe o está ocupada, la apertura falla con un mensaje; **no
  cambia automáticamente a otra cámara**.
- Se conserva `--usb 1` para cámaras externas y el acceso por Wi-Fi.
- El apartado Intenciones sigue sin permitir `--usb 0` ni
  `--fuente camara:0` desde su iniciador.

## Pruebas unitarias sin equipo

```bat
python -m unittest tests.test_camara_integrada_vial tests.test_fuentes_externas tests.test_iniciadores -v
```

Las pruebas validan el menú y los comandos; una prueba con la cámara
física sigue siendo necesaria.
