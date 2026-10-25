import customtkinter as ctk
import ctypes
import os
import logging
from tkinter import filedialog
from slowlodgen.ui_style import *

__all__ = ['LODSettingsWindow', 'PluginSettingsUI']

RESOLUTIONS = [512, 1024, 2048, 4096, 8192, 16384]
RESOLUTION_LABELS = ['512', '1K', '2K', '4K', '8K', '16K']
INTERNAL = [2048, 4096, 8192, 16384]
INTERNAL_LABELS = ['2K', '4K', '8K', '16K']


def short(res):
    return f'{res // 1024}K' if res >= 1024 else str(res)


def nearest(value, options):
    return min(options, key=lambda o: abs(o - value))


def total_ram_gb():
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
    try:
        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / 2**30
    except Exception:
        return 0.0


def quad_count(ws):
    if 'quads' in ws:
        return ws['quads']
    b = ws['bounds'] #upper bound if the exact count wasn't passed
    return (b[2] // 32 - b[0] // 32 + 1) * (b[3] // 32 - b[1] // 32 + 1)


def estimate(v, worldspaces):
    #measured on Tamriel: DXT1 with mipmaps = res^2 * 2/3 bytes, uncompressed normals = res^2 * 3,
    #meshes ~11 bytes per target triangle; RAM: color stage ~40 B per internal pixel, normals ~60 B per pixel,
    #mesh stage ~0.45 GB per thread
    quads = sum(quad_count(w) for w in worldspaces)
    per_quad, ram = 0.0, 0.5
    if v['color_on']:
        per_quad += v['color_res'] ** 2 * 2 / 3
        ram = max(ram, 0.6 + v['color_internal'] ** 2 * 40 / 2**30)
    if v['normal_on']:
        per_quad += v['normal_res'] ** 2 * (2 / 3 if v['normal_dxt1'] else 3)
        ram = max(ram, 0.6 + max(v['normal_res'], 1024) ** 2 * 60 / 2**30)
    if v['mesh_on']:
        per_quad += v['triangles'] * 11
        ram = max(ram, 0.6 + 0.45 * v['threads'])
    return quads, per_quad * quads / 2**20, ram


def fmt_size(mb):
    return f'{mb / 1024:.1f} GB' if mb >= 1024 else f'{mb:.0f} MB'


class LODSettingsWindow(ctk.CTkToplevel):

    DESC_WRAP = 300

    def __init__(self, parent, cfg, worldspace_list):
        super().__init__(parent, fg_color=BG)
        self.title("slowLandscapeLODGen Settings")
        fit_geometry(self, 1280, 800)
        self.minsize(1000, 560)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cfg, self.ws_list = cfg, worldspace_list
        self.confirmed, self.selected_worldspaces = False, []
        self.v = self._values_from_cfg(cfg)
        self.internal_choice = self.v['color_internal'] #what the user picked; the effective value follows the resolution
        self.checked = {ws['editor_id']: False for ws in worldspace_list}
        self.refreshers = []
        self.total_ram = total_ram_gb()

        head = ctk.CTkFrame(self, fg_color='transparent')
        head.pack(fill='x', padx=26, pady=(20, 4))
        ctk.CTkLabel(head, text='slowLandscapeLODGen', font=(FONT + ' Semibold', 22), text_color=TEXT).pack(side='left')
        ctk.CTkLabel(head, text='Landscape LOD for Oblivion', font=(FONT, 12), text_color=MUTED).pack(side='left', padx=(12, 0), pady=(6, 0))

        self._footer()

        body = ctk.CTkFrame(self, fg_color='transparent')
        body.pack(fill='both', expand=True, padx=26, pady=(10, 0))
        body.grid_columnconfigure(0, weight=5, uniform='col')
        body.grid_columnconfigure(1, weight=6, uniform='col')
        body.grid_rowconfigure(0, weight=1)

        self._worldspace_pane(body)

        right = ctk.CTkScrollableFrame(body, fg_color='transparent')
        right.grid(row=0, column=1, sticky='nsew', pady=(0, 14))
        self.cards = {}
        self._output_card(right, 'color', 'Color map', 'color_on', self._color_controls, self._color_summary)
        self._output_card(right, 'normal', 'Normal map', 'normal_on', self._normal_controls, self._normal_summary)
        self._output_card(right, 'mesh', 'Mesh', 'mesh_on', self._mesh_controls, self._mesh_summary)
        self._output_options(right)

        self._fill_ws()
        self._refresh()
        self.grab_set()

    # --- Settings <-> config ---------------------------------------------------
    @staticmethod
    def _values_from_cfg(cfg):
        ebf = cfg.general.use_enginebugfixes_naming_scheme
        texture_dimension = nearest(cfg.texture.texture_dimension, RESOLUTIONS)
        return {
            'color_on': cfg.general.generate_color_maps,
            'color_res': texture_dimension,
            'color_internal': nearest(max(cfg.texture.internal_dimension, texture_dimension, 2048), INTERNAL),
            'falloff': cfg.texture.gauss_blending_falloff,
            'ao_on': cfg.ao.enabled, 'ao_strength': cfg.ao.strength,
            'normal_on': cfg.general.generate_normal_maps,
            'normal_res': nearest(cfg.normal_map.texture_dimension, RESOLUTIONS),
            'normal_dxt1': cfg.normal_map.compression, 'z_boost': cfg.normal_map.z_boost,
            'shadows_on': cfg.normal_map.shadows_enabled, 'shadow_strength': cfg.normal_map.shadow_strength,
            'noise_on': cfg.normal_map_noise.noise_enabled,
            'noise_intensity': cfg.normal_map_noise.noise_intensity, 'noise_scale': cfg.normal_map_noise.noise_scale,
            'mesh_on': cfg.general.generate_meshes, 'triangles': cfg.mesh.target_triangles,
            'threads': cfg.mesh.threads if cfg.mesh.multithreaded else 1,
            'ebf': 'On' if ebf is True else 'Off' if ebf is False else 'Auto',
            'color_dds_level': cfg.texture.dds_encode_quality,
            'normal_dds_level': cfg.normal_map.dds_encode_quality,
            'output_folder': cfg.paths.output_folder,
        }

    def submit(self):
        cfg, v = self.cfg, self.v
        self.selected_worldspaces = [ws for ws, on in self.checked.items() if on]

        cfg.general.use_enginebugfixes_naming_scheme = {'Auto': 'Auto', 'On': True, 'Off': False}[v['ebf']]
        cfg.general.generate_color_maps = v['color_on']
        cfg.general.generate_normal_maps = v['normal_on']
        cfg.general.generate_meshes = v['mesh_on']
        cfg.paths.output_folder = v['output_folder']
        cfg.texture.dds_encode_quality = int(v['color_dds_level'])
        cfg.normal_map.dds_encode_quality = int(v['normal_dds_level'])

        cfg.texture.texture_dimension = int(v['color_res'])
        cfg.texture.internal_dimension = int(v['color_internal'])
        cfg.texture.gauss_blending_falloff = round(float(v['falloff']), 1)
        cfg.ao.enabled = v['ao_on']
        cfg.ao.strength = round(float(v['ao_strength']), 2)

        cfg.normal_map.texture_dimension = int(v['normal_res'])
        cfg.normal_map.compression = v['normal_dxt1']
        cfg.normal_map.z_boost = round(float(v['z_boost']), 2)
        cfg.normal_map.shadows_enabled = v['shadows_on']
        cfg.normal_map.shadow_strength = round(float(v['shadow_strength']), 2)
        cfg.normal_map_noise.noise_enabled = v['noise_on']
        cfg.normal_map_noise.noise_intensity = int(v['noise_intensity'])
        cfg.normal_map_noise.noise_scale = int(v['noise_scale'])

        cfg.mesh.target_triangles = int(v['triangles'])
        cfg.mesh.threads = int(v['threads'])
        cfg.mesh.multithreaded = cfg.mesh.threads > 1

        cfg.apply_overrides()
        self.confirmed = True
        self.destroy()

    def on_close(self):
        self.confirmed = False
        self.destroy()
        logging.info("Tool closed without confirmation, exiting.")
        os._exit(0)

    # --- Worldspaces -----------------------------------------------------------
    def _worldspace_pane(self, parent):
        pane = ctk.CTkFrame(parent, fg_color=CARD, border_color=CARD_BORDER, border_width=1, corner_radius=8)
        pane.grid(row=0, column=0, sticky='nsew', padx=(0, 14), pady=(0, 14))

        top = ctk.CTkFrame(pane, fg_color='transparent')
        top.pack(fill='x', padx=16, pady=(14, 8))
        ctk.CTkLabel(top, text='Worldspaces', font=(FONT + ' Semibold', 14), text_color=TEXT).pack(side='left')
        ctk.CTkLabel(top, text=f'{len(self.ws_list)} found', font=(FONT, 12), text_color=MUTED).pack(side='left', padx=(10, 0))

        tools = ctk.CTkFrame(pane, fg_color='transparent')
        tools.pack(fill='x', padx=16, pady=(0, 8))
        self.search = ctk.CTkEntry(tools, placeholder_text='Search name, EditorID or plugin', height=30, font=(FONT, 12))
        self.search.pack(side='left', fill='x', expand=True)
        self.search.bind('<KeyRelease>', lambda e: self._fill_ws())
        for text, mode in (('None', 'none'), ('All', 'all'), ('Non-small', 'non_small')):
            ctk.CTkButton(tools, text=text, width=76, height=30, font=(FONT, 12), fg_color=CARD, text_color=TEXT,
                          border_color=CARD_BORDER, border_width=1, hover_color=HOVER,
                          command=lambda m=mode: self._select(m)).pack(side='right', padx=(6, 0))
        ctk.CTkLabel(tools, text='Select', font=(FONT, 12), text_color=MUTED).pack(side='right', padx=(10, 0))

        ctk.CTkFrame(pane, height=1, fg_color=CARD_BORDER).pack(fill='x', padx=16)
        self.ws_box = ctk.CTkScrollableFrame(pane, fg_color='transparent')
        self.ws_box.pack(fill='both', expand=True, padx=6, pady=(4, 10))

        #rows are built once and only shown/hidden by the search, rebuilding them flickers
        self.ws_rows = {}
        for ws in self.ws_list:
            eid = ws['editor_id']
            title = eid + (f'  ·  {ws["full_name"]}' if ws['full_name'] and ws['full_name'] != eid else '')
            b = ws['bounds']
            subtitle_parts = [ws['plugin']]
            if ws['small_world']:
                subtitle_parts.append('· small world')
            subtitle_parts.append(f'· {ws["cell_num"]} cells')
            subtitle_parts.append(f'· [{b[0]:.0f}:{b[2]:.0f} , {b[1]:.0f}:{b[3]:.0f}]')

            row = ctk.CTkFrame(self.ws_box, fg_color='transparent')
            var = ctk.BooleanVar(value=False)
            ctk.CTkCheckBox(row, text=title, variable=var, font=(FONT, 13), text_color=TEXT,
                            checkbox_width=18, checkbox_height=18, corner_radius=4, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                            command=lambda e=eid, v=var: (self.checked.__setitem__(e, v.get()), self._refresh())).pack(anchor='w')
            ctk.CTkLabel(row, text='  '.join(subtitle_parts), font=(FONT, 11), text_color=MUTED, height=14,
                         anchor='w').pack(anchor='w', padx=(31, 0))
            self.ws_rows[eid] = (row, var, (title + ws['plugin']).lower())
        self.no_match = ctk.CTkLabel(self.ws_box, text='Nothing matches.', font=(FONT, 12), text_color=MUTED)
        self.ws_shown = None

    def _select(self, mode):
        for ws in self.ws_list:
            on = mode == 'all' or (mode == 'non_small' and not ws['small_world'])
            self.checked[ws['editor_id']] = on
            var = self.ws_rows[ws['editor_id']][1]
            if var.get() != on:
                var.set(on)
        self._refresh()

    def _fill_ws(self):
        q = self.search.get().lower()
        shown = [ws['editor_id'] for ws in self.ws_list if not q or q in self.ws_rows[ws['editor_id']][2]]
        if shown == self.ws_shown:
            return
        self.ws_shown = shown
        for row, _, _ in self.ws_rows.values():
            row.pack_forget()
        self.no_match.pack_forget()
        for eid in shown:
            self.ws_rows[eid][0].pack(fill='x', padx=8, pady=(5, 3))
        if not shown:
            self.no_match.pack(pady=16)

    # --- Output cards ----------------------------------------------------------
    def _output_card(self, parent, key, title, on_key, controls_fn, summary_fn):
        card = ctk.CTkFrame(parent, fg_color=CARD, border_color=CARD_BORDER, border_width=1, corner_radius=8)
        card.pack(fill='x', pady=(0, 10), padx=(0, 4))
        head = ctk.CTkFrame(card, fg_color='transparent')
        head.pack(fill='x', padx=16, pady=12)
        self._switch(on_key)(head).pack(side='left')
        ctk.CTkLabel(head, text=title, font=(FONT + ' Semibold', 14), text_color=TEXT).pack(side='left', padx=(4, 0))
        summary = ctk.CTkLabel(head, text='', font=(FONT, 12), text_color=MUTED)
        summary.pack(side='left', padx=(14, 0))
        button = ctk.CTkButton(head, text='Adjust  ▾', width=90, height=28, font=(FONT, 12), fg_color='transparent',
                               text_color=TEXT, border_color=CARD_BORDER, border_width=1, hover_color=HOVER)
        button.pack(side='right')
        body = ctk.CTkFrame(card, fg_color='transparent')
        controls_fn(body)

        def toggle():
            if body.winfo_manager():
                body.pack_forget()
                button.configure(text='Adjust  ▾')
            else:
                body.pack(fill='x', pady=(0, 6))
                button.configure(text='Done  ▴')
        button.configure(command=toggle)
        self.cards[key] = toggle
        self.refreshers.append(lambda: update(summary, text=summary_fn() if self.v[on_key] else 'off'))

    def _color_controls(self, c):
        self._setting(c, 'Resolution', 'Each step is 4× the disk space and VRAM.',
                      self._seg('color_res', RESOLUTIONS, RESOLUTION_LABELS), enabled_key='color_on')
        self._setting(c, 'Render at', 'Supersampling: render larger, then downscale.',
                      self._seg('color_internal', INTERNAL, INTERNAL_LABELS), enabled_key='color_on')
        self._setting(c, 'Blending falloff', 'How much neighbouring texture samples blend. Lower is smooth but blurry; '
                      'higher is sharp but blocky, as each sample shows as a square.',
                      self._slider('falloff', 1, 10, 0.5, '{:.1f}'), enabled_key='color_on')
        self._setting(c, 'DXT1 quality', 'Compression effort, same file size. Higher is slower and slightly less blocky; '
                      'ground texture detail hides most of the difference.',
                      self._slider('color_dds_level', 0, 18, 1), enabled_key='color_on')
        self._setting(c, 'Ambient occlusion', 'Darkens valleys and crevices.', self._switch('ao_on'),
                      enabled_key='color_on', note='Experimental.')
        self._setting(c, 'AO strength', None, self._slider('ao_strength', 0, 1, 0.05, '{:.2f}'), enabled_key='ao_on')

    def _normal_controls(self, c):
        self._setting(c, 'Resolution', 'Above 1K: smoother shading, no new details.',
                      self._seg('normal_res', RESOLUTIONS, RESOLUTION_LABELS), enabled_key='normal_on')
        self._setting(c, 'Format', 'DXT1 is 6× smaller but blockier in lighting.',
                      self._seg('normal_dxt1', [True, False], ['DXT1', 'Uncompressed']), enabled_key='normal_on')
        self._setting(c, 'DXT1 quality', 'Compression effort, same file size. Higher is less blocky on smooth slopes; '
                      '14 is a good balance, 18 is 4× slower for little gain.',
                      self._slider('normal_dds_level', 0, 18, 1), enabled_key='normal_dxt1')
        self._setting(c, 'Z-boost', 'Higher makes distant hills and mountains stand out more, lower makes them look flatter. '
                      '1.0 matches the real terrain; other values look less natural.',
                      self._slider('z_boost', 0.1, 5, 0.05, '{:.2f}'), enabled_key='normal_on')
        self._setting(c, 'Fake shadows', None, self._switch('shadows_on'), enabled_key='normal_on')
        self._setting(c, 'Shadow strength', None, self._slider('shadow_strength', 0, 3, 0.05, '{:.2f}'), enabled_key='shadows_on')
        self._setting(c, 'Detail noise', None, self._switch('noise_on'), enabled_key='normal_on', note=self._noise_note)
        self._setting(c, 'Noise intensity', None, self._slider('noise_intensity', 1, 64, 1), enabled_key='noise_on')
        self._setting(c, 'Noise scale', None, self._slider('noise_scale', 5, 100, 1), enabled_key='noise_on')

    def _noise_note(self):
        #noise features have a fixed size on the ground (noise_scale * res / 8192 px for the coarsest octave),
        #below ~5 px the coarse octaves are lost and what is left reads as pixel grain
        v = self.v
        need = 1024
        while v['noise_scale'] * need / 8192 < 5 and need < 16384:
            need *= 2
        if v['noise_on'] and v['normal_on'] and v['normal_res'] < need:
            return (f'Normal map {short(v["normal_res"])} is too small for scale {v["noise_scale"]:.0f}: '
                    f'the noise turns into grain. Use {short(need)} or more.'), True
        return f'Needs a normal map of {short(need)} or more at scale {v["noise_scale"]:.0f}.', False

    def _mesh_controls(self, c):
        self._setting(c, 'Triangles per quad', None, self._slider('triangles', 4000, 62000, 1000, '{:,.0f}'), enabled_key='mesh_on')
        self._setting(c, 'Threads', None, self._slider('threads', 1, os.cpu_count() or 8, 1), enabled_key='mesh_on')

    def _color_summary(self):
        v = self.v
        parts = [short(v['color_res'])]
        if v['color_internal'] > v['color_res']:
            parts.append(f'rendered at {short(v["color_internal"])}')
        parts.append(f'falloff {v["falloff"]:g}')
        parts.append(f'AO {v["ao_strength"]:.2f}' if v['ao_on'] else 'no AO')
        return '  ·  '.join(parts)

    def _normal_summary(self):
        v = self.v
        parts = [short(v['normal_res']), 'DXT1' if v['normal_dxt1'] else 'uncompressed']
        if v['z_boost'] != 1:
            parts.append(f'z-boost {v["z_boost"]:.2f}')
        if v['shadows_on']:
            parts.append('shadows')
        if v['noise_on']:
            parts.append('noise')
        return '  ·  '.join(parts)

    def _mesh_summary(self):
        return f'{int(self.v["triangles"]):,} triangles  ·  {int(self.v["threads"])} threads'

    def _output_options(self, parent):
        card = ctk.CTkFrame(parent, fg_color=CARD, border_color=CARD_BORDER, border_width=1, corner_radius=8)
        card.pack(fill='x', pady=(0, 10), padx=(0, 4))
        self._setting(card, 'Output folder', 'With MO2, choose an empty folder in your mods folder.',
                      self._folder_control, first=True)
        self._setting(card, 'Engine Bug Fixes naming', 'Load-order independent names for mod worldspaces.',
                      self._seg('ebf', ['Auto', 'On', 'Off']))

    def _folder_control(self, parent):
        f = ctk.CTkFrame(parent, fg_color='transparent')
        label = ctk.CTkLabel(f, text='', font=(FONT, 12), text_color=TEXT, anchor='e', width=200)
        label.pack(side='left')
        ctk.CTkButton(f, text='Browse…', width=84, height=30, font=(FONT, 12), fg_color=CARD, text_color=TEXT,
                      border_color=CARD_BORDER, border_width=1, hover_color=HOVER,
                      command=self._browse_output_folder).pack(side='left', padx=(10, 0))
        ctk.CTkButton(f, text='Reset', width=60, height=30, font=(FONT, 12), fg_color=CARD, text_color=TEXT,
                      border_color=CARD_BORDER, border_width=1, hover_color=HOVER,
                      command=lambda: self._set('output_folder', self.cfg.paths.oblivion_folder)).pack(side='left', padx=(6, 0))
        self.refreshers.append(lambda: update(
            label, text='Game Data folder' if self.v['output_folder'] == self.cfg.paths.oblivion_folder else self.v['output_folder']))
        return f

    def _browse_output_folder(self):
        folder = filedialog.askdirectory(parent=self, title="Select Output Folder", initialdir=self.v['output_folder'])
        if folder:
            self._set('output_folder', os.path.normpath(folder))

    # --- Footer ----------------------------------------------------------------
    def _footer(self):
        bar = ctk.CTkFrame(self, fg_color=CARD, corner_radius=0, height=68)
        bar.pack(fill='x', side='bottom')
        ctk.CTkFrame(bar, height=1, fg_color=CARD_BORDER).pack(fill='x', side='top')
        inner = ctk.CTkFrame(bar, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=26, pady=12)
        self.go = ctk.CTkButton(inner, text='Generate LOD', width=170, height=40, corner_radius=6,
                                font=(FONT + ' Semibold', 14), fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self.submit)
        self.go.pack(side='right')
        left = ctk.CTkFrame(inner, fg_color='transparent')
        left.pack(side='left', fill='x', expand=True)
        self.summary = ctk.CTkLabel(left, text='', font=(FONT, 13), text_color=TEXT, anchor='w')
        self.summary.pack(anchor='w')
        self.detail = ctk.CTkLabel(left, text='', font=(FONT, 11), text_color=MUTED, anchor='w')
        self.detail.pack(anchor='w')

    def _refresh(self, *_):
        v = self.v
        v['color_internal'] = max(self.internal_choice, v['color_res'], 2048)
        for f in self.refreshers:
            f()

        chosen = [ws for ws in self.ws_list if self.checked[ws['editor_id']]]
        if not chosen:
            update(self.summary, text='No worldspaces selected')
            update(self.detail, text='Pick at least one worldspace.', text_color=MUTED)
            update(self.go, state='disabled')
            return
        update(self.go, state='normal')
        quads, disk_mb, ram_gb = estimate(v, chosen)
        names = ', '.join(ws['editor_id'] for ws in chosen[:3]) + (f' and {len(chosen) - 3} more' if len(chosen) > 3 else '')
        update(self.summary, text=names)
        too_much = self.total_ram > 0 and ram_gb > 0.7 * self.total_ram
        update(self.detail, text=f'{quads} quads  ·  ~{fmt_size(disk_mb)} on disk  ·  peak RAM ~{ram_gb:.1f} GB'
                                 + (f' of {self.total_ram:.0f} GB' if self.total_ram > 0 else ''),
               text_color=DANGER if too_much else MUTED)

    # --- Rows and controls -----------------------------------------------------
    def _setting(self, card, title, desc, control_fn, first=False, enabled_key=None, note=None):
        if not first:
            ctk.CTkFrame(card, height=1, fg_color=CARD_BORDER).pack(fill='x', padx=16)
        row = ctk.CTkFrame(card, fg_color='transparent')
        row.pack(fill='x', padx=18, pady=12)
        text = ctk.CTkFrame(row, fg_color='transparent')
        text.pack(side='left', fill='x', expand=True)
        title_label = ctk.CTkLabel(text, text=title, font=(FONT, 13), text_color=TEXT, anchor='w')
        title_label.pack(anchor='w')
        if desc:
            ctk.CTkLabel(text, text=desc, font=(FONT, 11), text_color=MUTED, anchor='w', justify='left',
                         wraplength=self.DESC_WRAP).pack(anchor='w')
        if note:
            note_label = ctk.CTkLabel(text, text='' if callable(note) else note, font=(FONT, 11), text_color=WARN,
                                      anchor='w', justify='left', wraplength=self.DESC_WRAP, height=16)
            note_label.pack(anchor='w')
            if callable(note): #note() -> (text, warning)
                def update_note():
                    message, warning = note()
                    update(note_label, text=message, text_color=WARN if warning else MUTED)
                self.refreshers.append(update_note)
        control = control_fn(row)
        control.pack(side='right', padx=(16, 0))
        if enabled_key:
            last = {'on': None}
            def update_enabled():
                on = bool(self.v[enabled_key])
                if on == last['on']:
                    return
                last['on'] = on
                title_label.configure(text_color=TEXT if on else MUTED)
                for w in [control] + list(control.winfo_children()):
                    try:
                        w.configure(state='normal' if on else 'disabled')
                    except Exception:
                        pass
                    if isinstance(w, ctk.CTkSlider):
                        w.configure(button_color=ACCENT if on else OFF, progress_color=ACCENT if on else OFF)
            self.refreshers.append(update_enabled)
        return control

    def _set(self, key, value):
        self.v[key] = value
        if key == 'color_internal':
            self.internal_choice = value
        self._refresh()

    def _switch(self, key):
        def make(parent):
            var = ctk.BooleanVar(value=self.v[key])
            self.refreshers.append(lambda: var.get() != self.v[key] and var.set(self.v[key]))
            return ctk.CTkSwitch(parent, text='', variable=var, width=46, progress_color=ACCENT, fg_color=OFF,
                                 button_color=SWITCH_KNOB, button_hover_color=SWITCH_KNOB_HOVER,
                                 command=lambda: self._set(key, var.get()))
        return make

    def _seg(self, key, options, labels=None):
        labels = labels or [str(o) for o in options]
        def make(parent):
            seg = ctk.CTkSegmentedButton(parent, values=labels, font=(FONT, 12), height=30, text_color=TEXT,
                                         fg_color=SEG_BG, unselected_color=SEG_BG, unselected_hover_color=HOVER,
                                         selected_color=SEG_SEL, selected_hover_color=SEG_SEL_HOVER,
                                         command=lambda label: self._set(key, options[labels.index(label)]))
            seg.set(labels[options.index(self.v[key])])
            def sync():
                wanted = labels[options.index(self.v[key])]
                if seg.get() != wanted:
                    seg.set(wanted)
            self.refreshers.append(sync)
            return seg
        return make

    def _slider(self, key, lo, hi, step, fmt='{:.0f}'):
        def make(parent):
            f = ctk.CTkFrame(parent, fg_color='transparent')
            entry = ctk.CTkEntry(f, width=64, height=28, font=(FONT, 12), justify='right')
            slider = ctk.CTkSlider(f, from_=lo, to=hi, number_of_steps=int(round((hi - lo) / step)), width=220,
                                   button_color=ACCENT, button_hover_color=ACCENT_HOVER, progress_color=ACCENT)
            def show_value():
                entry.delete(0, 'end')
                entry.insert(0, fmt.format(self.v[key]))
            def from_slider(x):
                self.v[key] = round(round(x / step) * step, 4)
                show_value()
                self._refresh()
            def from_entry(_event=None):
                try:
                    x = float(entry.get().replace(',', ''))
                except ValueError:
                    x = self.v[key]
                x = min(max(round(x / step) * step, lo), hi)
                self.v[key] = x
                slider.set(x)
                show_value()
                self._refresh()
            slider.configure(command=from_slider)
            entry.bind('<Return>', from_entry)
            entry.bind('<FocusOut>', from_entry)
            slider.set(self.v[key])
            show_value()
            slider.pack(side='left')
            entry.pack(side='left', padx=(10, 0))
            return f
        return make


class PluginSettingsUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.withdraw()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.destroy()
        self.quit()
        os._exit(0)

    def show_settings(self, cfg, worldspace_list) -> tuple:
        window = LODSettingsWindow(self, cfg, worldspace_list)
        self.wait_window(window)
        if window.confirmed:
            return True, window.selected_worldspaces
        return False, []
