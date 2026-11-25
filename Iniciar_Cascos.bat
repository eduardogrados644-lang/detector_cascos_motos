@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
echo =========================================
echo   APARTADO 1 - CONTROL VIAL Y CASCOS
echo Opciones: camara integrada PC, USB externa, Wi-Fi o video.
echo Con parametros: --integrada ^| --usb 1 ^| --wifi URL
echo =========================================
"%PYTHON_EXE%" iniciar_cascos.py %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Ventana del iniciador vial finalizada.
pause
exit /b %EXIT_CODE%
