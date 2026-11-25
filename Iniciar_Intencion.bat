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
echo Si se pierde la senal, se reintentara la misma camara.
echo Para detener: Q/ESC en la ventana, o Ctrl+C en la terminal.
"%PYTHON_EXE%" iniciar_intencion.py %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Ventana del iniciador conductual finalizada.
pause
exit /b %EXIT_CODE%
