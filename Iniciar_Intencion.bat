@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
echo =========================================
echo   APARTADO 2 - ANALISIS CONDUCTUAL
echo =========================================
echo Camaras USB/Wi-Fi: sin limite de fotogramas por defecto.
echo Para posibles agresiones: --demo-agresiones --usb 1
echo Para grabar candidatos: --guardar-capturas --guardar-clips
echo Para sesion continua NO uses --max-fotogramas 100.
echo ALERTA SONORA: activa en agresiones y armas cuando haya candidatos.
echo Para silenciar: --sin-alerta-sonora
echo En escenas oscuras se mejora luz de forma limitada; no elimina sombras.
echo Durante la vista: F guarda foto (maximo 20); Q o ESC sale.
echo Si se pierde la senal, se reintentara la misma camara.
echo Para detener: Q/ESC en la ventana, o Ctrl+C en la terminal.
"%PYTHON_EXE%" iniciar_intencion.py %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Ventana del iniciador conductual finalizada.
pause
exit /b %EXIT_CODE%
