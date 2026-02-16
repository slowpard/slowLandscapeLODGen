import os
import time
import re
time.clock = time.time #hack as time.clock is not available since python 3.8, open-source and backward compatibility... :cry:
import sys
import math
import logging
import winreg
import struct
import math
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import random
import subprocess
import quicktex.dds
import quicktex.s3tc.bc1
import customtkinter as ctk
import dataclasses
from dataclasses import dataclass, field
import tomllib






def get_app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        #jupyter lol
        return os.path.abspath(os.getcwd())

TOOL_DIR = get_app_dir()

os.environ['NUMBA_CACHE_DIR'] = os.path.join(TOOL_DIR, '__numba_cache__')


from numba import prange, njit, types
from numba.typed import Dict
from typing import List, Optional, Literal, Union
from numba.typed import List
from numba.core import event
import numba


from parsers.ESPParser import *
from parsers.BSAParser import *
from helpers.NormalMapHelpers import *
from helpers.ColorMapHelpers import *
from helpers.MeshGenerationHelpers import *
from helpers.NifSaver import *
from slowlodgen.ui_settings import *
from slowlodgen.ui_pluginselection import *
from slowlodgen.config import *
apply_pyffi_patches()

class ElapsedFilter(logging.Filter):
    def filter(self, record):
        record.elapsed = time.time() - start_time
        return True

start_time = time.time()
logging.basicConfig(
    level=logging.DEBUG,  
    format='%(elapsed).1fs  | %(lineno)s | %(levelname)s | %(message)s',  
    handlers=[
        logging.FileHandler(os.path.join(TOOL_DIR, 'output.log'), mode='w'), 
        logging.StreamHandler()             
    ],
    datefmt='%H:%M:%S'
)

logging.getLogger().addFilter(ElapsedFilter())

try:
    def _data_folder_is_valid(path):
        return (
            os.path.isfile(os.path.join(path, "Oblivion.esm"))
            or os.path.isfile(os.path.join(path, "Nehrim.esm"))
        )


    def resolve_data_folder(folder):

        #attempt 1: use configured path
        if folder:
            if os.path.isfile(os.path.join(folder, "Oblivion.exe")):
                data_path = os.path.join(folder, "Data")
            elif _data_folder_is_valid(folder):
                data_path = folder
            else:
                data_path = os.path.join(folder, "Data")

            if not os.path.isdir(data_path) or not _data_folder_is_valid(data_path):
                logging.critical(f"Configured game_folder '{folder}' is not a valid Oblivion installation.")
                sys.exit()
            return data_path

        #attempt 2: assume that the tool is somewhere in /Oblivion
        current = os.path.abspath(os.getcwd())

        for i in range(4):  

            #check if we're directly in the game folder (Oblivion.exe present)
            if os.path.isfile(os.path.join(current, "Oblivion.exe")):
                data_path = os.path.join(current, "Data")
                if os.path.isdir(data_path) and _data_folder_is_valid(data_path):
                    return data_path
                
            #check if we're directly in a Data folder
            if _data_folder_is_valid(current):
                return current
            
            #check if current has a Data subfolder that's valid
            candidate = os.path.join(current, "Data")
            if os.path.isdir(candidate) and _data_folder_is_valid(candidate):
                return candidate
            parent = os.path.dirname(current)

            if parent == current:
                break  # reached filesystem root

            current = parent


        #attempt 3: check registry
        registry_paths = [
            r"SOFTWARE\Bethesda Softworks\Oblivion",
            r"SOFTWARE\WOW6432Node\Bethesda Softworks\Oblivion",
        ]

        for reg_path in registry_paths:
            try:
                registry_key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE, reg_path, 0, winreg.KEY_READ
                )
                oblivion_path, _ = winreg.QueryValueEx(registry_key, "Installed Path")
                winreg.CloseKey(registry_key)
                data_path = os.path.join(oblivion_path, "Data")
                if os.path.isdir(data_path) and _data_folder_is_valid(data_path):
                    logging.info(f"Found Data folder via registry: {data_path}")
                    return data_path
            except OSError:
                continue

        logging.critical("Oblivion Data folder not found. Exiting.")
        sys.exit()



    def resolve_plugins_txt(plugins_txt, data_path):

        if plugins_txt:
            if not os.path.isfile(plugins_txt):
                logging.critical(f"plugins.txt not found at configured path '{plugins_txt}'. Exiting.")
                sys.exit()

            return plugins_txt

        game_folder = os.path.dirname(data_path)
        use_appdata = True

        ini_path = os.path.join(game_folder, "Oblivion.ini")
        if os.path.isfile(ini_path):
            try:
                with open(ini_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        match = re.match(
                            r"^\s*bUseMyGamesDirectory\s*=\s*(\d+)", line, re.IGNORECASE
                        )
                        if match:
                            use_appdata = match.group(1) != "0"
            except OSError:
                pass

        if use_appdata:
            plugins_txt = os.path.join(
                os.getenv("USERPROFILE", ""), "AppData", "Local", "Oblivion", "plugins.txt"
            )
        else:
            plugins_txt = os.path.join(game_folder, "plugins.txt")

        if not os.path.isfile(plugins_txt):
            logging.critical(f"plugins.txt not found at '{plugins_txt}'. Exiting.")
            sys.exit()

        return plugins_txt


    config_path = os.path.join(TOOL_DIR, 'LODGen_config.toml')
    cfg = load_config(config_path)
    logging.getLogger().setLevel(logging._nameToLevel[cfg.debug_level.upper()])



    folder = resolve_data_folder(cfg.paths.oblivion_folder)
    plugins_txt = resolve_plugins_txt(cfg.paths.plugins_txt, folder)

    logging.info("Oblivion path: " + folder)
    logging.info("plugins.txt path: " + plugins_txt)


    def is_esm(file_path):
        try:
            with open(file_path, 'rb') as f:
                record_type = f.read(4) 
                if record_type != b'TES4':
                    logging.error(f'Warning: file {file_path} is not a valid ESP/ESM')
                    return False
                
                f.read(4)
                flags_bytes = f.read(4)
                flags = struct.unpack('<I', flags_bytes)[0]
                return (flags & 0x00000001) != 0        
        except:
            logging.error(f'Error: error reading {file_path}')
            return False



    def sort_esp_list(filenames, folder):
        file_list = []
    
        for filename in filenames:
            if filename.startswith('#'):
                continue
            full_filename = os.path.join(folder, filename)
            if not os.path.exists(full_filename):
                logging.error(f'Error: file {full_filename} is enabled but does not exist')
                continue
            _, extension = os.path.splitext(filename)
            extension = extension.lower()
            modified_time = os.path.getmtime(full_filename)  
            is_esm_file = is_esm(full_filename) 
            file_list.append((filename, is_esm_file, modified_time))


        def sorter(item):
            filename, is_esm_file, modified_time = item
            is_esp = not is_esm_file
            return(is_esp, modified_time) 

        
        sorted_list = sorted(file_list, key=sorter)

        return [n[0] for n in sorted_list]

    with open(plugins_txt, 'r', encoding='windows-1252') as file:
        enabled_plugins = [line.strip() for line in file if line.strip()]


    current_load_order = sort_esp_list(enabled_plugins, folder)
    load_order_lowercase = [x.lower() for x in current_load_order]


    def get_plugin_files(folder_path):
        extensions = ('.esp', '.esm', '.ghost')
        files = []
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(extensions):
                files.append(filename)
        
        return sorted(files, key=str.lower)


    esps_in_data = sort_esp_list(get_plugin_files(folder), folder)


    app = PluginSelectorUI()
    app.withdraw()

    load_order = app.ShowPluginsSelection(esps_in_data, current_load_order)
    app.quit()


    if load_order is None:
        logging.critical("No plugins selected. Exiting.")
        exit()

    if len(load_order) > 255:
        logging.critical("Too many plugins selected. Exiting.")
        exit()


    logging.info(f'{load_order}')

    signatures = ['LAND', 'CELL', 'WRLD', 'LTEX']
    object_dict = {}
    worldspace_esp = {}

    for plugin in load_order:
        parser = ESPParser()
        logging.info('Reading: ' + plugin)
        parser.parse(os.path.join(folder, plugin))
        plugin_lo = parser.load_order
        master_map = {}
        try:
            for entry in plugin_lo:
                index = load_order_lowercase.index(entry[1].lower())
                master_map[entry[0]] = index
            master_map[len(master_map)] = load_order_lowercase.index(plugin.lower())
        except ValueError:
            logging.critical(f'Error: masters missing for plugin: {plugin}')
            logging.critical(f'Expected masters: {plugin_lo}')
            exit()

        if len(master_map) > 0:
            if not all(key == value for key, value in master_map.items()):
                parser.renumber_formids(master_map)

        for record in parser.formid_map:
            if parser.formid_map[record].sig in signatures:
                object_dict[record] = parser.formid_map[record]
                if parser.formid_map[record].sig == "WRLD":
                    if not parser.formid_map[record].editor_id in worldspace_esp:
                        worldspace_esp[parser.formid_map[record].editor_id] = plugin


    cell_data = {}

    quad_data = {}

    worldspace_bounds = {}

    wordlspaces = {}

    for object_id in object_dict:
        if object_dict[object_id].sig == 'LAND':
            
            land_record = object_dict[object_id]
            i = land_record.parent_group.parent_group.parent_group.records.index(land_record.parent_group.parent_group)
            try:
                cell = land_record.parent_group.parent_group.parent_group.records[i-1]
            except:
                logging.error('Error: LAND record without CELL record found')
                continue
            
            if type(cell) != RecordCELL:
                logging.error('Error: NOT CELL')
                continue
            worldspace = cell.parent_worldspace.editor_id
            
            try:
                x = cell.cell_coordinates[0]
                y = cell.cell_coordinates[1]
            except:
                if worldspace not in cell_data:
                    x = 0
                    y = 0
                else:
                    if x not in cell_data[worldspace]:
                        x = 0
                        y = 0
                    else:
                        if y not in cell_data[worldspace][x]:
                            x = 0
                            y = 0
                        else:
                            continue


            if worldspace not in cell_data:
                cell_data[worldspace] = {}
                worldspace_bounds[worldspace] = [99999, 999999, -99999, -99999]
            
            if x not in cell_data[worldspace]:
                cell_data[worldspace][x] = {}
            
            if y not in cell_data[worldspace][x]:
                cell_data[worldspace][x][y] = land_record

            cell_data[worldspace][x][y] = land_record

            quad_x = x // 32
            quad_y = y // 32

            if worldspace not in quad_data:
                quad_data[worldspace] = {}
            
            if quad_x not in quad_data[worldspace]:
                quad_data[worldspace][quad_x] = {}
            
            if quad_y not in quad_data[worldspace][quad_x]:
                quad_data[worldspace][quad_x][quad_y] = True

            worldspace_bounds[worldspace][0] = min(worldspace_bounds[worldspace][0], x)
            worldspace_bounds[worldspace][2] = max(worldspace_bounds[worldspace][2], x)
            worldspace_bounds[worldspace][1] = min(worldspace_bounds[worldspace][1], y)
            worldspace_bounds[worldspace][3] = max(worldspace_bounds[worldspace][3], y)

            #land_record.parse_texture_data()
        elif object_dict[object_id].sig == 'WRLD':
            worldspace = object_dict[object_id]
            wordlspaces[worldspace.editor_id] = worldspace
            

    def build_worldspace_list(worldspaces: dict, load_order: list) -> list:

        SMALL_WORLD_FLAG = 0x01

        result = []
        for editor_id, wrld in worldspaces.items():
            if wrld.parent_worldspace is not None:
                continue

            if editor_id in cfg.excluded_worldspaces:
                continue

            plugin = worldspace_esp[editor_id]
            small = bool(wrld.worldspace_flags & SMALL_WORLD_FLAG)

            result.append({
                "editor_id": editor_id,
                "full_name": wrld.full_name,
                "plugin": plugin,
                "small_world": small,
                "form_id": wrld.form_id,
            })
        return result

    def SaveImageAsDXT1(img, output_path, level=10):
    
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        quicktex.dds.encode(img, quicktex.s3tc.bc1.BC1Encoder(level, 
                        quicktex.s3tc.bc1.BC1Encoder.ColorMode.FourColor), 'DXT1').save(output_path)


    def GetFilenameForFile(form_id, worldspace, quad):
        if (form_id >> 24) & 0xFF == 0:
            return f'{form_id}.{(quad[0]*32):02}.{(quad[1]*32):02}.32'
        if cfg.general.use_enginebugfixes_naming_scheme:
            return f'{load_order[(form_id >> 24) & 0xFF]}\\{worldspace}.{(quad[0]*32):02}.{(quad[1]*32):02}.32'
        else:
            return f'{form_id}.{(quad[0]*32):02}.{(quad[1]*32):02}.32'

    def detect_enginebugfixes_naming(setting, game_folder):
        dll_path = os.path.join(game_folder, "OBSE", "Plugins", "EngineBugFixes.dll")
        ini_path = os.path.join(game_folder, "OBSE", "Plugins", "EngineBugFixes.ini")

        dll_exists = os.path.isfile(dll_path)
        patch_enabled = False

        if dll_exists and os.path.isfile(ini_path):
            with open(ini_path, "r") as f:
                for line in f:
                    match = re.match(r"^\s*bInstallTerrainLODLoadPatch\s*=\s*(\d+)", line)
                    if match:
                        patch_enabled = match.group(1) == "1"
                        break

        ebf_active = dll_exists and patch_enabled

        if setting == "Auto":
            return ebf_active

        elif setting is True:
            if not dll_exists:
                logging.error("EBF naming forced on but EngineBugFixes.dll not found!")
            elif not patch_enabled:
                logging.error("EBF naming forced on but bInstallTerrainLODLoadPatch is not enabled!")
            return True

        else:  # False
            if ebf_active:
                logging.warning("EBF TerrainLODLoadPatch is active but EBF naming is off — vanilla naming is load-order dependent")
            return False


    ws_list = build_worldspace_list(wordlspaces, load_order)




    while True:

        app = PluginSettingsUI()
        confirmed, worldspaces_to_generate = app.show_settings(cfg, ws_list)
        app.quit()
        cfg.general.use_enginebugfixes_naming_scheme = detect_enginebugfixes_naming(cfg.general.use_enginebugfixes_naming_scheme, folder)

        if len(worldspaces_to_generate) == 0:
            logging.warning("No worldspaces selected for generation. Please select at least one.")
            continue


        if cfg.general.generate_color_maps:

            logging.info('Loading textures...')

            list_of_texture_files = []
            texture_library = {}

            for obj in object_dict.values():
                if obj.sig == 'LTEX':
                    if obj.texture_filename:
                        list_of_texture_files.append("textures\\landscape\\" + obj.texture_filename.lower())
                    else:
                        logging.error(f'Error: LTEX record without texture filename found')


            list_of_texture_files.append('textures\\landscape\\default.dds')

            bsa_files = [f for f in os.listdir(folder) if f.endswith('.bsa')]
            bsa_loadorder = [] + cfg.hardcoded_bsas


            for plugin in load_order:
                for file in bsa_files:
                    if file.lower().startswith(plugin.lower().replace('.esp', '')):
                        bsa_loadorder.append(file)
                        
            for bsa in bsa_loadorder:
                if not os.path.exists(os.path.join(folder, bsa)):
                    continue
                bsa_obj = BSAParser()
                logging.info(f'Reading BSA: {bsa}')
                bsa_obj.parse(os.path.join(folder, bsa))
                list_to_extract = []
                for file in bsa_obj.get_list_of_files():
                    if file.lower() in list_of_texture_files:
                        list_to_extract.append(file)

                if len(list_to_extract) > 0:
                    try:
                        bsa_obj.extract(list_to_extract, os.path.join(folder, 'LandscapeLODTemp\\BSATextures'))
                        #far_mesh_list += list_to_extract
                    except:
                        logging.error(f'Failed to extract LOD files from {bsa}')


            for tex in list_of_texture_files:
                texture_path = os.path.join(folder, tex)
                if not os.path.exists(texture_path):
                    texture_path = os.path.join(folder, 'LandscapeLODTemp\\BSATextures', tex)
                if not os.path.exists(texture_path):
                    logging.error(f'Error: texture file {tex} not found')
                    texture_path = os.path.join(folder, 'LandscapeLODTemp\\BSATextures', 'textures\\landscape\\default.dds')
                    if not os.path.exists(texture_path):
                        texture_path = os.path.join(folder, 'textures\\landscape\\default.dds')
                    if not os.path.exists(texture_path):
                        logging.critical(f'Error: default texture not found at {texture_path}')
                        texture_path = os.path.join(TOOL_DIR, 'assets', 'default_landscape.dds')
                
                img = Image.open(texture_path).convert("RGBA")
                orig_width, orig_height = img.size
                tiled_img = Image.new("RGBA", (orig_width * 3, orig_height * 3))
                for i in range(3):
                    for j in range(3):
                        tiled_img.paste(img, (i * orig_width, j * orig_height))
                r, g, b, a = tiled_img.split()

                rgb_img = Image.merge('RGB', (r, g, b))
                rgb_resized = rgb_img.resize(
                    (cfg.texture.sample_res * 3, cfg.texture.sample_res * 3), 
                    Image.Resampling.LANCZOS
                ).crop((cfg.texture.sample_res, cfg.texture.sample_res, cfg.texture.sample_res * 2, cfg.texture.sample_res * 2))

                alpha_resized = a.resize(
                    (cfg.texture.sample_res * 3, cfg.texture.sample_res * 3), 
                    Image.Resampling.LANCZOS  # or Image.Resampling.NEAREST for hard edges
                ).crop((cfg.texture.sample_res, cfg.texture.sample_res, cfg.texture.sample_res * 2, cfg.texture.sample_res * 2))

                # Convert alpha to numpy array
                alpha_array = np.array(alpha_resized) / 255.0

                mid_point = cfg.texture.sample_res // 2



                texture_library[tex] = (np.array(rgb_resized),
                                        np.array(rgb_resized.crop((0, 0, mid_point, mid_point)), dtype=np.int8), #top left
                                        np.array(rgb_resized.crop((mid_point, 0, cfg.texture.sample_res, mid_point))), #top right
                                        np.array(rgb_resized.crop((0, mid_point, mid_point, cfg.texture.sample_res))), #bottom left
                                        np.array(rgb_resized.crop((mid_point, mid_point, cfg.texture.sample_res, cfg.texture.sample_res))),
                                        np.mean(np.array(rgb_resized), axis=(0,1)), alpha_array
                                        ) #bottom right

                img = None
                #os.makedirs(os.path.dirname(os.path.join(folder, 'LandscapeLODTemp\\resized', tex)), exist_ok=True)
                #img_resized.save(os.path.join(folder, 'LandscapeLODTemp\\resized', tex), 'BMP')
                


            ltex_to_texture = {}

            ltex_to_texture[-1] = texture_library['textures\\landscape\\default.dds']
            

            for obj in object_dict.values():
                if obj.sig == 'LTEX':
                    if obj.texture_filename:
                        if "textures\\landscape\\" + obj.texture_filename.lower() in texture_library:
                            ltex_to_texture[obj.form_id] = texture_library["textures\\landscape\\" + obj.texture_filename.lower()]



            ValueTypeTexture = types.Array(dtype=types.uint8, ndim=3, layout='C')


            ltex_to_texture_hashed = Dict.empty(
                key_type=types.int64,
                value_type=types.int64
            )

            ValueTypeAvgColor = types.Array(dtype=types.float64, ndim=1, layout='C')

            ltex_to_texture_avg_hashed = Dict.empty(
                key_type=types.int64,
                value_type=types.int64
            )

            ltex_to_alpha_hashed = Dict.empty(
                key_type=types.int64,
                value_type=types.float64
            )


            textures_nparray = np.zeros((len(ltex_to_texture), cfg.texture.sample_res, cfg.texture.sample_res, 3))
            textures_nparray_avg = np.zeros((len(ltex_to_texture), 3))
            textures_nparray_alpha = np.zeros((len(ltex_to_texture), cfg.texture.sample_res, cfg.texture.sample_res))

            for i, (key, value) in enumerate(ltex_to_texture.items()):
                textures_nparray[i] = value[0]
                textures_nparray_avg[i] = value[5]
                textures_nparray_alpha[i] = value[6]
                ltex_to_texture_hashed[key] = i
                ltex_to_texture_avg_hashed[key] = i
                ltex_to_alpha_hashed[key] = i


            '''
            for key, value in ltex_to_texture.items():
                ltex_to_texture_hashed[key] = value[0]
                ltex_to_texture_avg_hashed[key] = value[5]
                #add_to_dict(key, value[0], ltex_to_texture_hashed)
                #add_to_dict(key, value[5], ltex_to_texture_avg_hashed)
            '''

            texture_library, ltex_to_texture  = None, None
            bsa_obj = None
            tiled_img, rgb_img, rgb_resized, alpha_resized, alpha_array = None, None, None, None, None

        if cfg.general.generate_color_maps:
            
            logging.info('=== COLOR TEXTURE GENERATION ===')


            logging.info('Pre-calculating tiling data...')

            assert cfg.texture.internal_dimension % (cfg.texture.sample_res / 2) == 0, "Triangle grid must tile evenly"


            grid, grid_hashes, grid_params = build_vert_grid(int(cfg.texture.internal_dimension), cfg.texture.sample_res/2)
            PRECALCULATED_TILING_DATA, PRECALCULATED_TRIANGLE_WEIGHTS = triangle_weight_precalc(grid, grid_hashes, cfg.texture.internal_dimension)

            tex_point_dim_2x = int(cfg.texture.chunk_dim)

            distance_matrix_central = np.zeros((tex_point_dim_2x, tex_point_dim_2x), dtype=np.float32)
            distance_matrix_N = np.zeros((tex_point_dim_2x, tex_point_dim_2x), dtype=np.float32)
            distance_matrix_S = np.zeros((tex_point_dim_2x, tex_point_dim_2x), dtype=np.float32)
            distance_matrix_E = np.zeros((tex_point_dim_2x, tex_point_dim_2x), dtype=np.float32)
            distance_matrix_W = np.zeros((tex_point_dim_2x, tex_point_dim_2x), dtype=np.float32)
            distance_matrix_NE = np.zeros((tex_point_dim_2x, tex_point_dim_2x), dtype=np.float32)
            distance_matrix_NW = np.zeros((tex_point_dim_2x, tex_point_dim_2x), dtype=np.float32)
            distance_matrix_SE = np.zeros((tex_point_dim_2x, tex_point_dim_2x), dtype=np.float32)
            distance_matrix_SW = np.zeros((tex_point_dim_2x, tex_point_dim_2x), dtype=np.float32)

            distance_matrices = [
                distance_matrix_central, distance_matrix_N, distance_matrix_S,
                distance_matrix_E, distance_matrix_W, distance_matrix_NE,
                distance_matrix_NW, distance_matrix_SE, distance_matrix_SW
            ]

        
            #i is row, j is column
            #start from top-left corner

            dim_adj = tex_point_dim_2x - 1

            def distance_function(i, j, dim_adj, tex_point_dim_2x): #mod because it seems that's how oblivion does that
                                                                    #maybe try gaussian?
                return max(1 - max(abs(i-1/2), abs(j-dim_adj/2)) / tex_point_dim_2x, 0)

            for j in range(tex_point_dim_2x):
                for i in range(tex_point_dim_2x):
                    distance_matrix_central[j][i] = max(1 - max(abs(i-dim_adj/2), abs(j-dim_adj/2)) / tex_point_dim_2x, 0)
                    distance_matrix_N[j][i] = max(1 - max(abs(i-dim_adj/2), abs(j+dim_adj/2)) / tex_point_dim_2x, 0)
                    distance_matrix_S[j][i] = max(1 - max(abs(i-dim_adj/2), abs(j-dim_adj*3/2)) / tex_point_dim_2x, 0)
                    distance_matrix_E[j][i] = max(1 - max(abs(i-dim_adj*3/2), abs(j-dim_adj/2)) / tex_point_dim_2x, 0)
                    distance_matrix_W[j][i] = max(1 - max(abs(i+dim_adj/2), abs(j-dim_adj/2)) / tex_point_dim_2x, 0)
                    distance_matrix_NE[j][i] = max(1 - max(abs(i-dim_adj*3/2), abs(j+dim_adj/2)) / tex_point_dim_2x, 0)
                    distance_matrix_NW[j][i] = max(1 - max(abs(i+dim_adj/2), abs(j+dim_adj/2)) / tex_point_dim_2x, 0)
                    distance_matrix_SE[j][i] = max(1 - max(abs(i-dim_adj*3/2), abs(j-dim_adj*3/2)) / tex_point_dim_2x, 0)
                    distance_matrix_SW[j][i] = max(1 - max(abs(i+dim_adj/2), abs(j-dim_adj*3/2)) / tex_point_dim_2x, 0)

            sigma = tex_point_dim_2x / cfg.texture.gauss_blending_falloff
            for j in range(tex_point_dim_2x):
                for i in range(tex_point_dim_2x):
                    distance_matrix_central[j][i] = math.exp(-((i-dim_adj/2)**2 + (j-dim_adj/2)**2) / (2 * sigma**2))
                    distance_matrix_N[j][i] = math.exp(-((i-dim_adj/2)**2 + (j+dim_adj/2)**2) / (2 * sigma**2))
                    distance_matrix_S[j][i] = math.exp(-((i-dim_adj/2)**2 + (j-dim_adj*3/2)**2) / (2 * sigma**2))
                    distance_matrix_E[j][i] = math.exp(-((i-dim_adj*3/2)**2 + (j-dim_adj/2)**2) / (2 * sigma**2))
                    distance_matrix_W[j][i] = math.exp(-((i+dim_adj/2)**2 + (j-dim_adj/2)**2) / (2 * sigma**2))
                    distance_matrix_NE[j][i] = math.exp(-((i-dim_adj*3/2)**2 + (j+dim_adj/2)**2) / (2 * sigma**2))
                    distance_matrix_NW[j][i] = math.exp(-((i+dim_adj/2)**2 + (j+dim_adj/2)**2) / (2 * sigma**2))
                    distance_matrix_SE[j][i] = math.exp(-((i-dim_adj*3/2)**2 + (j-dim_adj*3/2)**2) / (2 * sigma**2))
                    distance_matrix_SW[j][i] = math.exp(-((i+dim_adj/2)**2 + (j-dim_adj*3/2)**2) / (2 * sigma**2))

            for j in range(tex_point_dim_2x):
                for i in range(tex_point_dim_2x):
                    total_weight =  sum(matrix[j][i] for matrix in distance_matrices)

                    for matrix in distance_matrices:
                        matrix[j][i] /= total_weight


            dm_central = distance_matrix_central[..., np.newaxis]
            dm_N = distance_matrix_N[..., np.newaxis]
            dm_S = distance_matrix_S[..., np.newaxis]
            dm_E = distance_matrix_E[..., np.newaxis]
            dm_W = distance_matrix_W[..., np.newaxis]
            dm_NE = distance_matrix_NE[..., np.newaxis]
            dm_NW = distance_matrix_NW[..., np.newaxis]
            dm_SE = distance_matrix_SE[..., np.newaxis]
            dm_SW = distance_matrix_SW[..., np.newaxis]


                
            def GenerateTextureMaps(texture_id_map, opacity_map, vertex_color_map, file_name, folder, ao_data):
                

                logging.info('Generating color map...')
                
                output_texture = np.zeros((cfg.texture.internal_dimension, cfg.texture.internal_dimension, 3), dtype=np.float32)
                output_texture = generate_full_texture_layer(output_texture, ltex_to_texture_hashed, distance_matrix_central, (0+32, 0+32), PRECALCULATED_TILING_DATA, PRECALCULATED_TRIANGLE_WEIGHTS, texture_id_map, opacity_map, vertex_color_map, textures_nparray, textures_nparray_avg, textures_nparray_alpha, cfg.texture.chunk_dim)
                output_texture = generate_full_texture_layer(output_texture, ltex_to_texture_hashed, distance_matrix_N, (0+32, 1+32), PRECALCULATED_TILING_DATA, PRECALCULATED_TRIANGLE_WEIGHTS, texture_id_map, opacity_map, vertex_color_map, textures_nparray, textures_nparray_avg, textures_nparray_alpha, cfg.texture.chunk_dim)
                output_texture = generate_full_texture_layer(output_texture, ltex_to_texture_hashed, distance_matrix_S, (0+32, -1+32), PRECALCULATED_TILING_DATA, PRECALCULATED_TRIANGLE_WEIGHTS, texture_id_map, opacity_map, vertex_color_map, textures_nparray, textures_nparray_avg, textures_nparray_alpha, cfg.texture.chunk_dim)
                output_texture = generate_full_texture_layer(output_texture, ltex_to_texture_hashed, distance_matrix_E, (1+32, 0+32), PRECALCULATED_TILING_DATA, PRECALCULATED_TRIANGLE_WEIGHTS, texture_id_map, opacity_map, vertex_color_map, textures_nparray, textures_nparray_avg, textures_nparray_alpha, cfg.texture.chunk_dim)
                output_texture = generate_full_texture_layer(output_texture, ltex_to_texture_hashed, distance_matrix_W, (-1+32, 0+32), PRECALCULATED_TILING_DATA, PRECALCULATED_TRIANGLE_WEIGHTS, texture_id_map, opacity_map, vertex_color_map, textures_nparray, textures_nparray_avg, textures_nparray_alpha, cfg.texture.chunk_dim)
                output_texture = generate_full_texture_layer(output_texture, ltex_to_texture_hashed, distance_matrix_NE, (1+32, 1+32), PRECALCULATED_TILING_DATA, PRECALCULATED_TRIANGLE_WEIGHTS, texture_id_map, opacity_map, vertex_color_map, textures_nparray, textures_nparray_avg, textures_nparray_alpha, cfg.texture.chunk_dim)
                output_texture = generate_full_texture_layer(output_texture, ltex_to_texture_hashed, distance_matrix_NW, (-1+32, 1+32), PRECALCULATED_TILING_DATA, PRECALCULATED_TRIANGLE_WEIGHTS, texture_id_map, opacity_map, vertex_color_map, textures_nparray, textures_nparray_avg, textures_nparray_alpha, cfg.texture.chunk_dim)
                output_texture = generate_full_texture_layer(output_texture, ltex_to_texture_hashed, distance_matrix_SE, (1+32, -1+32), PRECALCULATED_TILING_DATA, PRECALCULATED_TRIANGLE_WEIGHTS, texture_id_map, opacity_map, vertex_color_map, textures_nparray, textures_nparray_avg, textures_nparray_alpha, cfg.texture.chunk_dim)
                output_texture = generate_full_texture_layer(output_texture, ltex_to_texture_hashed, distance_matrix_SW, (-1+32, -1+32), PRECALCULATED_TILING_DATA, PRECALCULATED_TRIANGLE_WEIGHTS, texture_id_map, opacity_map, vertex_color_map, textures_nparray, textures_nparray_avg, textures_nparray_alpha, cfg.texture.chunk_dim)
                
                if cfg.ao.enabled:
                    logging.info('Generating AO map...')
                    ao_map = compute_ao_map(ao_data[0], cfg.ao.seek_range, 4096/32, cfg.ao.seek_range, cfg.ao.seek_range, 32*32 + 1) #4096/32 - 4096 units per cell of 32x32 areas
                    ao_map_transform(ao_map, min_b = cfg.ao.gamma_min, c_exp = cfg.ao.gamma_exp, c_max = cfg.ao.gamma_max)
                    ao_map = ao_map[::-1, :]
                    #Image.fromarray(np.uint8(ao_map * 255), 'L').transpose(Image.FLIP_TOP_BOTTOM).save(f"ao_{file_name}.bmp")
                    apply_ao_to_texture(output_texture, ao_map, cfg.ao.strength)
                
                ao_map = None
                #Image.fromarray(output_texture.astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM).save("unblended_output.bmp") #last minute realization that it must be flipped :sadge:
                
                #command = ['texconv.exe', '-f', 'DXT1', '-o' ,'.', 'unblended_output.bmp']
                #try:
                #    subprocess.run(command, text=True, capture_output=True, check = True)
                #except subprocess.CalledProcessError as e:
                #    print("texconv failed:")
                #    print(e.stderr)
                #subprocess.run(command, text=True, capture_output=True)

                temp_image = Image.fromarray(output_texture.astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM)
                output_texture = None

                if cfg.texture.texture_dimension != cfg.texture.internal_dimension:
                    logging.info('Resizing texture...')
                    temp_image = temp_image.resize((cfg.texture.texture_dimension, cfg.texture.texture_dimension), resample=Image.LANCZOS)
                    

                try:
                    logging.info(f'Saving texture {file_name}.dds...')
                    SaveImageAsDXT1(temp_image, os.path.join(folder, r'textures\landscapelod\generated', file_name + '.dds'), level=cfg.texture.dds_encode_quality)

                except subprocess.CalledProcessError as e:
                    logging.error(f"SAVING TEXTURE FAILED {e.returncode}")
                    logging.error(f"STDOUT: {e.stdout}")
                    logging.error(f"STDERR: {e.stderr}")
                    
            


                #print('Generating normal map...')
                #extended_grid = np.uint32(NORMAL_MAP_DIMENSION / 32 * 34 + 1)
                #resampled_heightmap = build_resampled_heightmap(extended_grid, height_map, SCALE_FACTOR)
                #normals_dx, normals_dy, _, _, _ = calculate_derivative_maps(resampled_heightmap)
                #_ = None
                #normal_map_image = build_base_normal_map(normals_dx, normals_dy)
                #Image.fromarray(np.uint8(normal_map_image), 'RGB').save('normal_map_temp.bmp')

                #print('Saving texture...')
                #command = ['texconv.exe', '-f', 'DXT1', '-o' ,'.', 'normal_map_temp.bmp']
                #try:
                #    subprocess.run(command, text=True, capture_output=True, check = True)
                #except subprocess.CalledProcessError as e:
                #    print("texconv failed:")
                #    print(e.stderr)
                #print('Renaming texture...')
                #os.rename(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'normal_map_temp.dds'), file_name + '_fn.dds')
                #shutil.move(os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name + '_fn.dds'), os.path.join(folder, r'textures\landscapelod\generated', file_name + '_fn.dds'))


            for worldspace in worldspaces_to_generate:
                form_id = wordlspaces[worldspace].form_id

                texture_id_map = np.full((34*32, 34*32, 9), 0, dtype=np.uint32)
                opacity_map = np.full((34*32, 34*32, 9), 1.0, dtype=np.float32)
                vertex_color_map = np.full((34*32, 34*32, 3), 255, dtype=np.uint8)
                #normal_map = np.full((34*32, 34*32, 3), 0, dtype=np.uint8)
                #height_map = np.full((34*32 + 1, 34*32 + 1), 0, dtype=np.float32)


                quad = (0, 0)

                x_size = worldspace_bounds[worldspace][2] - worldspace_bounds[worldspace][0] + 1
                y_size = worldspace_bounds[worldspace][3] - worldspace_bounds[worldspace][1] + 1
                x_low = worldspace_bounds[worldspace][0]
                y_low = worldspace_bounds[worldspace][1]
                x_high = worldspace_bounds[worldspace][2]
                y_high = worldspace_bounds[worldspace][3]


                if cfg.ao.enabled:
                    #note that heightmap is 33x33 vertex grid, and everything else is 32x32 rects inside these vertices

                    worldspace_heightmap = np.full((32 * y_size + 1, 32 * x_size + 1), -65000.0, dtype=np.float32)

                    for x in range(x_low, x_high + 1):
                        if x not in cell_data[worldspace]:
                            continue
                        for y in range(y_low, y_high + 1):

                            if y not in cell_data[worldspace][x]:
                                continue
                            land_record = cell_data[worldspace][x][y]
                            height_map_cell = land_record.parse_land_data_VGHT_only()
                            worldspace_heightmap[32*(y - y_low):32*(y-y_low)+32 + 1, 32*(x - x_low):32*(x - x_low)+32 + 1] = height_map_cell




                for x_quad in quad_data[worldspace]:
                    for y_quad in quad_data[worldspace][x_quad]:

                        quad = (x_quad, y_quad)
                        
                        logging.info(f'Processing worldspace {worldspace} quad {quad}...')
                        file_name = GetFilenameForFile(form_id, worldspace, quad)


                        for x in range(0, 32 + 2):
                            
                            x_cell = (x - 1) + quad[0] * 32
                            if x_cell not in cell_data[worldspace]:
                                texture_id_map[:,  x*32:(x+1)*32, :] = 0
                                opacity_map[:,  x*32:(x+1)*32, :] = 1.0
                                vertex_color_map[:,  x*32:(x+1)*32, :] = 255
                                #height_map[:,  x*32:(x+1)*32+1] = -65000.0
                                continue

                            for y in range(0, 32 + 2):
                                
                                y_cell = (y - 1) + quad[1] * 32
                                if y_cell not in cell_data[worldspace][x_cell]:
                                    texture_id_map[y*32:(y+1)*32, x*32:(x+1)*32, :] = 0
                                    opacity_map[y*32:(y+1)*32, x*32:(x+1)*32, :] = 1.0
                                    vertex_color_map[y*32:(y+1)*32, x*32:(x+1)*32, :] = 255
                                    #height_map[y*32:(y+1)*32 + 1, x*32:(x+1)*32 + 1] = -65000.0
                                    continue

                                land_record = cell_data[worldspace][x_cell][y_cell]
                                #height_map_cell = land_record.parse_land_data_VGHT_only()
                                texture_id_points, opacity_points, vertex_colors, normals, height_map_cell = land_record.parse_land_data()
                                texture_id_map[y*32:(y+1)*32, x*32:(x+1)*32] = texture_id_points
                                opacity_map[y*32:(y+1)*32, x*32:(x+1)*32] = opacity_points
                                vertex_color_map[y*32:(y+1)*32, x*32:(x+1)*32] = vertex_colors
                                #normal_map[y*32:(y+1)*32, x*32:(x+1)*32] = normals
                                #height_map[y*32:(y+1)*32 + 1, x*32:(x+1)*32 + 1] = height_map_cell
                        

                        if cfg.ao.enabled:
                            y_min = 32*(quad[1]*32 - y_low) - cfg.ao.seek_range
                            y_max = 32*(quad[1]*32 - y_low) + 32*32 + 1  + cfg.ao.seek_range
                            x_min = 32*(quad[0]*32 - x_low) - cfg.ao.seek_range
                            x_max = 32*(quad[0]*32 - x_low) + 32*32 + 1  + cfg.ao.seek_range
                            y_min_clamped = max(0, y_min)
                            y_max_clamped = min(worldspace_heightmap.shape[0], y_max)
                            x_min_clamped = max(0, x_min)
                            x_max_clamped = min(worldspace_heightmap.shape[1], x_max)

                            x_quad_coord = cfg.ao.seek_range - x_min + x_min_clamped
                            y_quad_coord = cfg.ao.seek_range - y_min + y_min_clamped
                            ao_heightmap = np.full((32*32 + 1  + 2*cfg.ao.seek_range, 32*32 + 1  + 2*cfg.ao.seek_range),
                                                -65000.0, dtype=np.float32)
                            ao_heightmap[y_min_clamped - y_min:y_max_clamped - y_min,x_min_clamped - x_min:x_max_clamped - x_min] = \
                                worldspace_heightmap[y_min_clamped:y_max_clamped, x_min_clamped:x_max_clamped]
                            ao_data = (ao_heightmap, x_quad_coord, y_quad_coord)
                        else:
                            ao_data = (None, None, None)

                        #print(ao_heightmap.shape, x_quad_coord, y_quad_coord)
                        GenerateTextureMaps(texture_id_map, opacity_map, vertex_color_map, file_name, folder, ao_data)

            PRECALCULATED_TRIANGLE_WEIGHTS, PRECALCULATED_TILING_DATA, worldspace_heightmap, texture_id_map = None, None, None, None 
            opacity_map, grid, grid_hashes, ao_heightmap = None, None, None, None  
            vertex_color_map = None
            textures_nparray, textures_nparray_avg, textures_nparray_alpha = None, None, None
            ltex_to_texture_hashed, ltex_to_texture_avg_hashed, ltex_to_alpha_hashed = None, None, None


        if cfg.general.generate_normal_maps:

            logging.info('=== NORMAL MAP GENERATION  ===')
            for worldspace in worldspaces_to_generate:
                    
                form_id = wordlspaces[worldspace].form_id

                


                x_size = worldspace_bounds[worldspace][2] - worldspace_bounds[worldspace][0] + 1
                y_size = worldspace_bounds[worldspace][3] - worldspace_bounds[worldspace][1] + 1
                x_low = worldspace_bounds[worldspace][0]
                y_low = worldspace_bounds[worldspace][1]
                x_high = worldspace_bounds[worldspace][2]
                y_high = worldspace_bounds[worldspace][3]

                y_low = (y_low // 32) * 32
                x_low = (x_low // 32) * 32
                y_high = (y_high // 32) * 32 + 31
                x_high = (x_high // 32) * 32 + 31

                y_size = y_high - y_low + 1
                x_size = x_high - x_low + 1


                #note that heightmap is 33x33 vertex grid, and everything else is 32x32 rects inside these vertices

                worldspace_heightmap = np.full((32 * y_size + 1, 32 * x_size + 1), -65000.0, dtype=np.float32)
                

                for x in range(x_low, x_high + 1):
                    if x not in cell_data[worldspace]:
                        continue
                    for y in range(y_low, y_high + 1):

                        if y not in cell_data[worldspace][x]:
                            continue
                        land_record = cell_data[worldspace][x][y]
                        height_map_cell = land_record.parse_land_data_VGHT_only()
                        worldspace_heightmap[32*(y - y_low):32*(y-y_low)+32 + 1, 32*(x - x_low):32*(x - x_low)+32 + 1] = height_map_cell


                westslope, eastslope, northslope = compute_directional_slopes(worldspace_heightmap, 1, 1024)
                

                for x_quad in quad_data[worldspace]:
                    for y_quad in quad_data[worldspace][x_quad]:

                        height_map = np.full((34*32 + 1, 34*32 + 1), 0, dtype=np.float32)
                        quad = (x_quad, y_quad)
                        
                        logging.info(f'Processing worldspace {worldspace} quad {quad}...')
                        file_name = GetFilenameForFile(form_id, worldspace, quad) + '_fn'


                        for x in range(0, 32 + 2):
                            
                            x_cell = (x - 1) + quad[0] * 32
                            if x_cell not in cell_data[worldspace]:
                                height_map[:,  x*32:(x+1)*32+1] = -65000.0
                                continue

                            for y in range(0, 32 + 2):
                                
                                y_cell = (y - 1) + quad[1] * 32
                                if y_cell not in cell_data[worldspace][x_cell]:
                                    height_map[y*32:(y+1)*32 + 1, x*32:(x+1)*32 + 1] = -65000.0
                                    continue

                                land_record = cell_data[worldspace][x_cell][y_cell]
                                height_map_cell = land_record.parse_land_data_VGHT_only()
                                height_map[y*32:(y+1)*32 + 1, x*32:(x+1)*32 + 1] = height_map_cell

                                texture_id_points, opacity_points, vertex_colors, normals, height_map_cell = land_record.parse_land_data()
                                #texture_id_map[y*32:(y+1)*32, x*32:(x+1)*32] = texture_id_points
                                #opacity_map[y*32:(y+1)*32, x*32:(x+1)*32] = opacity_points
                                #vertex_color_map[y*32:(y+1)*32, x*32:(x+1)*32] = vertex_colors
                                #normal_map[y*32:(y+1)*32, x*32:(x+1)*32] = normals
                                height_map[y*32:(y+1)*32 + 1, x*32:(x+1)*32 + 1] = height_map_cell
                                #or height_map = worldspace_heightmap[32*(quad[1] - 1 - y_low):32*(quad[1] - 1 - y_low)+34*32 + 1, 32*(quad[0] - 1 - x_low):32*(quad[0]  - 1 - x_low)+34*32 + 1]

                        if cfg.normal_map.shadows_enabled:
                            eastslope_quad = eastslope[32*(quad[1]*32  - y_low):32*(quad[1]*32 - y_low)+32*32 + 1, 32*(quad[0]*32 - x_low):32*(quad[0]*32   - x_low)+32*32 + 1]
                            northslope_quad = northslope[32*(quad[1]*32  - y_low):32*(quad[1]*32 - y_low)+32*32 + 1, 32*(quad[0]*32  - x_low):32*(quad[0]*32   - x_low)+32*32 + 1]
                            westslope_quad = westslope[32*(quad[1]*32  - y_low):32*(quad[1]*32  - y_low)+32*32 + 1, 32*(quad[0]*32  - x_low):32*(quad[0]*32   - x_low)+32*32 + 1]
                        
                        verts, tris = create_vertices_from_heightmap_simple(height_map)
                        face_normals = calc_face_normals_non_normalized(tris, verts)
                        vertex_normals = calc_vertex_normals_angle_weighted(tris, verts, face_normals)
                        vertex_normals = np.clip(vertex_normals, -1.0, 1.0).reshape(int((vertex_normals.shape[0]) ** (1/2)) , int((vertex_normals.shape[0]) ** (1/2)), 3)

                        center = vertex_normals.shape[0] // 2
                        half_dim = int(((vertex_normals.shape[0] - 1) / 34 * 32) // 2)
                        vertex_normals = vertex_normals[
                            center - half_dim : center + half_dim + 1,
                            center - half_dim : center + half_dim + 1,
                            :
                        ]

                        if cfg.normal_map.shadows_enabled:
                            vertex_normals = apply_shadows_to_vertex_normals(
                                vertex_normals,
                                westslope_quad,
                                eastslope_quad,
                                northslope_quad,
                                northboost=1.0,
                                shadow_strength=cfg.normal_map.shadow_strength,
                            )
                        
                        westslope_quad, eastslope_quad, northslope_quad = None, None, None
                        
                        normal_map = build_base_normal_map_vertexbaking(vertex_normals, normal_map_dimension=cfg.normal_map.internal_dimension, z_boost = cfg.normal_map.z_boost)
                        if cfg.normal_map_noise.noise_enabled :
                            noise_height_map = noise_generate_height_map_vectorized(
                                normal_map.shape[0], normal_map.shape[1], 
                                scale=cfg.normal_map_noise.noise_scale, 
                                octaves=cfg.normal_map_noise.noise_octaves, 
                                persistence=cfg.normal_map_noise.noise_persistence,
                                lacunarity=cfg.normal_map_noise.noise_lacunarity
                            )
                            noise_height_map_float = noise_height_to_normal_map_fast(noise_height_map, strength=cfg.normal_map_noise.noise_intensity)
                            normal_map = blend_normal_maps(normal_map, noise_height_map_float)
                            noise_height_map = None
                            noise_height_map_float = None

                        
                        combined_normal_img = Image.fromarray(np.round(np.clip((normal_map*127.5 + 127.5 ), 0, 255)).astype(np.uint8))

                        normal_map, vertex_normals, height_map, verts, tris, face_normals = None, None, None, None, None, None

                        if cfg.normal_map.texture_dimension != cfg.normal_map.internal_dimension:
                            temp_image = temp_image.resize((cfg.normal_map.texture_dimension, cfg.normal_map.texture_dimension), resample=Image.BILINEAR)

                        if cfg.normal_map.compression:

                            try:
                                logging.info(f'Saving texture {file_name}.dds...')
                                #temp_image = Image.fromarray(combined_normal_img.astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM)
                                SaveImageAsDXT1(combined_normal_img, os.path.join(folder, r'textures\landscapelod\generated', file_name + '.dds'), cfg.texture.dds_encode_quality)

                            except subprocess.CalledProcessError as e:
                                logging.error(f"SAVING TEXTURE FAILED {e.returncode}")
                                logging.error(f"STDOUT: {e.stdout}")
                                logging.error(f"STDERR: {e.stderr}")
                        else:

                            combined_normal_img.save(os.path.join(folder, r'textures\landscapelod\generated', file_name + '.dds'))
                                
                        combined_normal_img = None

        westslope, eastslope, northslope, worldspace_heightmap = None, None, None, None

        if cfg.general.generate_meshes:
            logging.info('=== MESH GENERATION  ===')
            ########MAIN LOOP########

            for worldspace in worldspaces_to_generate:

                
                form_id = wordlspaces[worldspace].form_id

                height_map = np.full((34*32 + 1, 34*32 + 1), 0, dtype=np.float32)


                quad = (0, 0)


                x_low = math.floor(worldspace_bounds[worldspace][0]/32)*32 - 2
                y_low = math.floor(worldspace_bounds[worldspace][1]/32)*32 - 2
                x_high = math.ceil(worldspace_bounds[worldspace][2]/32)*32 + 2
                y_high = math.ceil(worldspace_bounds[worldspace][3]/32)*32 + 2
                x_size = x_high - x_low + 1 
                y_size = y_high - y_low + 1

                mesh_data = {}

                #note that heightmap is 33x33 vertex grid, and everything else is 32x32 rects inside these vertices

                worldspace_heightmap = np.full((32 * y_size + 1, 32 * x_size + 1), -65000.0, dtype=np.float32)

                for x in range(x_low, x_high + 1):
                    if x not in cell_data[worldspace]:
                        continue
                    for y in range(y_low, y_high + 1):

                        if y not in cell_data[worldspace][x]:
                            continue
                        land_record = cell_data[worldspace][x][y]
                        height_map_cell = land_record.parse_land_data_VGHT_only()
                        worldspace_heightmap[32*(y - y_low):32*(y-y_low)+32 + 1, 32*(x - x_low):32*(x - x_low)+32 + 1] = height_map_cell

                #########################MESH GEOMETRY CALCULATION####################
                counter = 0
                quads = np.full((cfg.mesh.threads, 2), -1, dtype=np.int32)
                
                for x_quad in quad_data[worldspace]:
                    if x_quad not in mesh_data: mesh_data[x_quad] = {}
                    for y_quad in quad_data[worldspace][x_quad]:

                        #if x_quad > 2 or x_quad < -2 or y_quad != 0:
                        #    continue
                        if y_quad not in mesh_data[x_quad]: mesh_data[x_quad][y_quad] = {}
                        quad = (x_quad, y_quad)
                        
                        #print(f'Processing worldspace {worldspace} quad {quad}...')
                        logging.info(f'Processing worldspace {worldspace} quad {quad}...')
                        file_name = GetFilenameForFile(form_id, worldspace, quad)
                        #mesh_data[x_quad][y_quad] = _GenerateLODMeshData(worldspace_heightmap, quad, x_low, y_low)
                        quads[counter] = (x_quad, y_quad)

                        counter += 1

                        if counter >= cfg.mesh.threads:
                            if cfg.mesh.multithreaded:
                                results = GenerateLODMeshDataWrapper(worldspace_heightmap, quads, x_low, y_low, counter, 
                                                                    target_verts=cfg.mesh.target_vertices, min_error=cfg.mesh.mesh_min_error, 
                                                                    vertices_batch=cfg.mesh.vertices_batch)
                            else:
                                results = GenerateLODMeshDataWrapperST(worldspace_heightmap, quads, x_low, y_low, counter, 
                                                                    target_verts=cfg.mesh.target_vertices, min_error=cfg.mesh.mesh_min_error, 
                                                                    vertices_batch=cfg.mesh.vertices_batch)
                            for thread in range(counter):
                                mesh_data[quads[thread, 0]][quads[thread, 1]] = results[thread]
                            quads = np.full((counter, 2), -1, dtype=np.int32)
                            counter = 0
                            

                if counter > 0:
                    if cfg.mesh.multithreaded:
                        results = GenerateLODMeshDataWrapper(worldspace_heightmap, quads, x_low, y_low, counter, 
                                                            target_verts=cfg.mesh.target_vertices, min_error=cfg.mesh.mesh_min_error, 
                                                            vertices_batch=cfg.mesh.vertices_batch)
                    else:
                        results = GenerateLODMeshDataWrapperST(worldspace_heightmap, quads, x_low, y_low, counter, 
                                                            target_verts=cfg.mesh.target_vertices, min_error=cfg.mesh.mesh_min_error, 
                                                            vertices_batch=cfg.mesh.vertices_batch)
                    for thread in range(counter):
                        mesh_data[quads[thread, 0]][quads[thread, 1]] = results[thread]

                ##############BORDER MATCHING##########################
                UpdateCellBordersWrapper(mesh_data)

                ###################NIF GENERATION#####################
                GenerateNifs(mesh_data, worldspace, worldspace_heightmap, x_low, y_low, folder, form_id)

                mesh_data = None


            height_map, worldspace_heightmap, results = None, None, None

except Exception as e:
    logging.exception("slowLODGen crashed:")
    logging.error(f"Error details: {str(e)}")
    input("Press Enter to exit...")