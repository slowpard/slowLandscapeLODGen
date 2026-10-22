#shared look of the plugin selection and settings windows: (light, dark) color pairs follow the Windows theme
FONT = 'Segoe UI'

ACCENT = ('#2563c9', '#4c8dff')
ACCENT_HOVER = ('#1f54ab', '#3b7cf0')
BG = ('#f3f3f3', '#1c1c1c')
CARD = ('#ffffff', '#2b2b2b')
CARD_BORDER = ('#e1e1e1', '#333333')
TEXT = ('#1a1a1a', '#f0f0f0')
MUTED = ('#6b6b6b', '#9a9a9a')
HOVER = ('#dcdcdc', '#2d2d2d')
WARN = ('#9a6200', '#e0a030')
DANGER = ('#c42b1c', '#ff7a6b')
SEG_BG = ('#e9e9e9', '#333333')
SEG_SEL = ('#cddcfb', '#34507e')
SEG_SEL_HOVER = ('#bcd0f8', '#3d5d92')
OFF = ('#c4c4c4', '#4a4a4a')
SWITCH_KNOB = ('#3d3d3d', '#f2f2f2')
SWITCH_KNOB_HOVER = ('#2a2a2a', '#ffffff')


def update(widget, **options):
    #customtkinter redraws a widget on every configure, so only pass the options that actually change
    changed = {k: v for k, v in options.items() if widget.cget(k) != v}
    if changed:
        widget.configure(**changed)


def fit_geometry(window, width, height):
    #requested size in logical pixels, shrunk to fit small or high-DPI screens
    import customtkinter as ctk
    scale = ctk.ScalingTracker.get_window_scaling(window)
    width = min(width, int(window.winfo_screenwidth() / scale * 0.92))
    height = min(height, int(window.winfo_screenheight() / scale * 0.85))
    window.geometry(f'{width}x{height}')
