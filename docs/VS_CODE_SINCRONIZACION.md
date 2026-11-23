# Sincronizar GitHub con Visual Studio Code (Windows)

Abre la terminal de VS Code dentro de la carpeta de tu copia local del repositorio.

## Paso 1: comprobar cambios locales

    git status
    git branch --show-current
    git remote -v

Si tienes archivos modificados sin guardar en commits, NO ejecutes reset, clean, switch ni un pull a ciegas. Primero preserva ese trabajo en una rama y un commit, o revisa el conflicto.

## Paso 2: bajar la rama de mejoras ANTES de su aprobación

    git fetch origin
    git switch mejoras-modulos-vial-conductual
    git pull --ff-only origin mejoras-modulos-vial-conductual
    git lfs pull

Si no existe todavía esa rama en tu computadora:

    git switch --track origin/mejoras-modulos-vial-conductual

Así puedes probar las mejoras sin modificar tu rama local main. No mezcles manualmente las ramas antes de revisar los resultados.

## Paso 3: instalar y comprobar

    py -3 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe scripts\diagnostico.py
    .\.venv\Scripts\python.exe -m unittest discover -s tests -v

Usa ese intérprete Python dentro de VS Code. Si la carpeta .venv ya existe, omite la creación del entorno.

## Paso 4: cuando el pull request se haya aprobado y fusionado en main

    git status
    git switch main
    git pull --ff-only origin main
    git lfs pull

El comando git pull descarga cambios de código y git lfs pull obtiene los archivos grandes LFS. Si sale un conflicto o un error, no fuerces operaciones: revisa git status y conserva los archivos locales.
