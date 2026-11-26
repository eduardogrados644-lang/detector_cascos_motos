"""Aviso sonoro local para nuevos candidatos; no declara emergencias.

Windows: dos tonos cortos para diferenciar una alerta del análisis ordinario.
Las alertas solo se emiten ante nuevos eventos, no en cada fotograma.
"""


def reproducir_aviso():
    try:
        import winsound
        try:
            winsound.Beep(880, 110)
            winsound.Beep(1100, 140)
            return True
        except (OSError, RuntimeError):
            return bool(winsound.MessageBeep(winsound.MB_ICONEXCLAMATION))
    except (ImportError, OSError, RuntimeError):
        print("[AVISO] No se pudo reproducir sonido; revisar audio de Windows.")
        return False
