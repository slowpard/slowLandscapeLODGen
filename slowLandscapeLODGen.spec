import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import shutil

block_cipher = None

numba_data      = collect_data_files('numba')
pyffi_data      = collect_data_files('pyffi')
pyffi_modules   = collect_submodules('pyffi')
ctk_data        = collect_data_files('customtkinter')
quicktex_data   = collect_data_files('quicktex')
triangle_data   = collect_data_files('triangle')

a = Analysis(
    ['slowLandscapeLODGen.py'],
    pathex=[],
    binaries=[],
    datas=[
        *numba_data,
        *pyffi_data,
        *ctk_data,
        *quicktex_data,
        *triangle_data,
    ],
    hiddenimports=[

    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook_numba.py'],
    excludes=['tcl',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='slowLandscapeLODGen',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,         
    console=True,   
    icon='icon.ICO',      
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='slowLandscapeLODGen',
)


dist_dir = os.path.join('dist', 'slowLandscapeLODGen')
shutil.copy2('LODGen_config.toml', dist_dir)
assets_dst = os.path.join(dist_dir, 'assets')
os.makedirs(assets_dst, exist_ok=True)
shutil.copy2('assets/default_landscape.dds', assets_dst)