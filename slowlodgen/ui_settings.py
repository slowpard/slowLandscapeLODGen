
import customtkinter as ctk


class ToolTip:
    """Minimal hover tooltip for any widget."""

    def __init__(self, widget, text: str, delay: int = 400):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._tip_window = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)

    def _schedule(self, _event=None):
        self._after_id = self.widget.after(self.delay, self._show)

    def _show(self):
        if self._tip_window:
            return
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        tw = self._tip_window = ctk.CTkToplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)
        ctk.CTkLabel(tw, text=self.text, corner_radius=6, padx=10, pady=4,
                     font=("Segoe UI", 11)).pack()

    def _hide(self, _event=None):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        if self._tip_window:
            self._tip_window.destroy()
            self._tip_window = None


def section_card(parent, title: str, *, expand=False, description: str = ""):
    """Create a card frame with a header. Returns (card_frame, content_frame)."""
    card = ctk.CTkFrame(parent, corner_radius=10)
    if expand:
        card.pack(fill="both", expand=True, pady=(0, 8))
    else:
        card.pack(fill="x", pady=(0, 8))

    header = ctk.CTkFrame(card, fg_color="transparent", height=36)
    header.pack(fill="x", padx=14, pady=(12, 0))

    ctk.CTkLabel(header, text=title, font=("Segoe UI Semibold", 14)).pack(side="left")
    if description:
        ctk.CTkLabel(header, text=description, font=("Segoe UI", 11)).pack(side="right")

    content = ctk.CTkFrame(card, fg_color="transparent")
    content.pack(fill="both", expand=True, padx=14, pady=(10, 14))

    return card, content


class LODSettingsWindow(ctk.CTkToplevel):
    RESOLUTION_VALUES = ["512", "1024", "2048", "4096", "8192", "16384"]

    def __init__(self, parent, cfg, worldspace_list):
        super().__init__(parent)
        self.title("slowLODGen � Settings")
        self.geometry("1160x740")
        self.minsize(960, 500)
        self.cfg = cfg
        self.confirmed = False

        root = ctk.CTkFrame(self, fg_color="transparent")
        root.pack(fill="both", expand=True, padx=16, pady=(10, 16))

        ctk.CTkLabel(root, text="slowLODGen",
                     font=("Segoe UI Black", 22)).pack(anchor="w", pady=(0, 10))

        body_scroll = ctk.CTkScrollableFrame(root, fg_color="transparent")
        body_scroll.pack(fill="both", expand=True)

        body = ctk.CTkFrame(body_scroll, fg_color="transparent")
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1, uniform="col")
        body.columnconfigure(1, weight=1, uniform="col")
        body.columnconfigure(2, weight=1, uniform="col")
        body.rowconfigure(0, weight=1)

        col1 = ctk.CTkFrame(body, fg_color="transparent")
        col1.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        col2 = ctk.CTkFrame(body, fg_color="transparent")
        col2.grid(row=0, column=1, sticky="nsew", padx=6)
        col3 = ctk.CTkFrame(body, fg_color="transparent")
        col3.grid(row=0, column=2, sticky="nsew", padx=(6, 0))

        # ================= COLUMN 1: WORLDSPACES =============================
        _, ws_content = section_card(col1, "Worldspaces", expand=True,
                                     description=f"{len(worldspace_list)} found")

        ws_scroll = ctk.CTkScrollableFrame(ws_content, corner_radius=6)
        ws_scroll.pack(fill="both", expand=True)

        self.worldspace_vars = {}
        for ws in worldspace_list:
            eid = ws["editor_id"]
            var = ctk.BooleanVar(value=False)
            self.worldspace_vars[eid] = var

            row = ctk.CTkFrame(ws_scroll, fg_color="transparent")
            row.pack(fill="x", pady=0, padx=2)

            # Main line: checkbox with EditorID
            main_font = ("Segoe UI Italic", 12) if ws["small_world"] else ("Segoe UI", 12)
            title_text = eid
            if ws["full_name"] and ws["full_name"] != eid:
                title_text = f"{eid} ({ws['full_name']})"

            ctk.CTkCheckBox(row, text=title_text, variable=var,
                            font=main_font, checkbox_height=16, 
                            checkbox_width=16, height = 16).pack(anchor="w", pady=(0, 0))

            # Subtitle: plugin name + small world tag
            subtitle_parts = [ws["plugin"]]
            if ws["small_world"]:
                subtitle_parts.append("� small world")
            subtitle = "    " + "  ".join(subtitle_parts)

            ctk.CTkLabel(row, text=subtitle,
                         font=("Segoe UI", 10),
                         text_color="gray", pady = 0, height=12).pack(anchor="w", padx=(15, 0), pady=(0, 5))

        # ================= COLUMN 2: GENERAL + TEXTURE =======================

        _, gen_content = section_card(col2, "General")

        ctk.CTkLabel(gen_content, text="EngineBugFixes naming:",
                     font=("Segoe UI", 12)).pack(anchor="w")
        ebf_val = cfg.general.use_enginebugfixes_naming_scheme
        ebf_init = "Auto" if ebf_val == "Auto" else ("Yes" if ebf_val else "No")
        self.use_ebf_naming = ctk.CTkComboBox(gen_content, values=["Auto", "Yes", "No"],
                                              state="readonly", width=110,
                                              font=("Segoe UI", 12))
        self.use_ebf_naming.set(ebf_init)
        self.use_ebf_naming.pack(anchor="w", pady=(2, 0))
        ToolTip(self.use_ebf_naming,
                '"Auto" detects EngineBugFixes automatically.')

        # -- LOD Texture ------------------------------------------------------
        _, tex_outer = section_card(col2, "LOD Texture", expand=True)

        self.texture_enabled = ctk.BooleanVar(value=cfg.general.generate_color_maps)
        ctk.CTkSwitch(tex_outer, text="Enabled", variable=self.texture_enabled,
                      command=self._toggle_texture,
                      font=("Segoe UI", 12)).pack(anchor="w", pady=(0, 8))

        self.texture_options = ctk.CTkFrame(tex_outer, fg_color="transparent")
        self.texture_options.pack(fill="x")

        self.texture_resolution = self._labeled_combo(
            self.texture_options, "Resolution", self.RESOLUTION_VALUES,
            str(cfg.texture.texture_dimension))

        self.texture_internal_resolution = self._labeled_combo(
            self.texture_options, "Internal resolution", ["2048", "4096", "8192", "16384"],
            str(cfg.texture.internal_dimension))
        ToolTip(self.texture_internal_resolution,
                "Higher = better sampling quality. Forced to be >= output resolution.")

        _, self.gauss_falloff = self._labeled_slider(
            self.texture_options, "Texture Blending Falloff", 0.1, 20,
            cfg.texture.gauss_blending_falloff, decimals=1)
        ToolTip(self.gauss_falloff,
                "Lower = smoother blending between adjacent local texture samples, but 'blurrier' resulting texture. " \
                "Default of 2.5 is good for high-res (8k-ish) textures" \
                "Consider pushing this up for low-res")

        self.ao_enabled = ctk.BooleanVar(value=cfg.ao.enabled)
        
        self.ao_switch = ctk.CTkSwitch(self.texture_options, text="Ambient Occlusion",
                      variable=self.ao_enabled, command=self._toggle_ao,
                      font=("Segoe UI", 12))
        self.ao_switch.pack(anchor="w", pady=(8, 4))
        ToolTip(self.ao_switch,
                "Darkens areas of the LOD texture that are supposed to be more occluded")

        self.ao_options = ctk.CTkFrame(self.texture_options, fg_color="transparent")
        self.ao_options.pack(fill="x", padx=(16, 0))

        _, self.ao_strength = self._labeled_slider(
            self.ao_options, "Strength", 0.0, 3.0, cfg.ao.strength, decimals=2)
        #_, self.ao_seek_range = self._labeled_slider(
        #    self.ao_options, "Seek Range", 64, 2048, cfg.ao.seek_range)

        if not self.texture_enabled.get():
            self.texture_options.pack_forget()
        if not self.ao_enabled.get():
            self.ao_options.pack_forget()

        # ================= COLUMN 3: NORMAL MAP + MESH =======================

        _, norm_outer = section_card(col3, "LOD Normal Map")

        self.normal_enabled = ctk.BooleanVar(value=cfg.general.generate_normal_maps)
        ctk.CTkSwitch(norm_outer, text="Enabled", variable=self.normal_enabled,
                      command=self._toggle_normal,
                      font=("Segoe UI", 12)).pack(anchor="w", pady=(0, 8))

        self.normal_options = ctk.CTkFrame(norm_outer, fg_color="transparent")
        self.normal_options.pack(fill="x")

        self.normal_resolution = self._labeled_combo(
            self.normal_options, "Resolution", self.RESOLUTION_VALUES,
            str(cfg.normal_map.texture_dimension))

        ctk.CTkLabel(self.normal_options, text="Compression:",
                     font=("Segoe UI", 12)).pack(anchor="w", pady=(6, 2))
        comp_row = ctk.CTkFrame(self.normal_options, fg_color="transparent")
        comp_row.pack(anchor="w", padx=(8, 0))

        self.normal_compression = ctk.StringVar(
            value="DXT1" if cfg.normal_map.compression else "Uncompressed")
        for val in ("DXT1", "Uncompressed"):
            rb = ctk.CTkRadioButton(comp_row, text=val, variable=self.normal_compression,
                               value=val, font=("Segoe UI", 12))
            rb.pack(side="left", padx=(0, 16), pady=2)
            ToolTip(rb,
                "DXT1 compression results in a 6x smaller VRAM footprint, but causes blocky artifacts, that are much more detrimential for normals than for color maps. \n" \
                "Note, that going one step below in resolution (e.g. 2048 -> 1024) results in a 4x smaller VRAM footprint. \n" \
                "No recommeneded value for now, needs testing")
            

        _, self.z_boost = self._labeled_slider(
            self.normal_options, "Z-Boost", 0.0, 5.0, cfg.normal_map.z_boost, decimals=2)
        ToolTip(self.z_boost,
            "Higher values make terrain features appear deeper and more defined but may look unnatural. 1.0 = no boost")

        self.fake_shadows = ctk.BooleanVar(value=cfg.normal_map.shadows_enabled)
        ctk.CTkSwitch(self.normal_options, text="Fake shadows",
                      variable=self.fake_shadows, command=self._toggle_shadow,
                      font=("Segoe UI", 12)).pack(anchor="w", pady=(6, 2))

        self.shadow_options = ctk.CTkFrame(self.normal_options, fg_color="transparent")
        _, self.shadow_strength = self._labeled_slider(
            self.shadow_options, "Shadow Strength", 0.0, 3.0,
            cfg.normal_map.shadow_strength, decimals=2)
        if self.fake_shadows.get():
            self.shadow_options.pack(fill="x", padx=(16, 0))

        self.noise_enabled = ctk.BooleanVar(value=cfg.normal_map_noise.noise_enabled)
        self._noise_switch = ctk.CTkSwitch(
            self.normal_options, text="Noise",
            variable=self.noise_enabled, command=self._toggle_noise,
            font=("Segoe UI", 12))
        self._noise_switch.pack(anchor="w", pady=(6, 2))

        self.noise_options = ctk.CTkFrame(self.normal_options, fg_color="transparent")
        _, self.noise_intensity = self._labeled_slider(
            self.noise_options, "Intensity", 0, 100,
            cfg.normal_map_noise.noise_intensity)
        _, self.noise_scale = self._labeled_slider(
            self.noise_options, "Scale", 1, 100,
            cfg.normal_map_noise.noise_scale)
        if self.noise_enabled.get():
            self.noise_options.pack(fill="x", padx=(16, 0))

        if not self.normal_enabled.get():
            self.normal_options.pack_forget()

        # -- Mesh -------------------------------------------------------------
        _, mesh_outer = section_card(col3, "LOD Mesh", expand=True)

        self.mesh_enabled = ctk.BooleanVar(value=cfg.general.generate_meshes)
        ctk.CTkSwitch(mesh_outer, text="Enabled", variable=self.mesh_enabled,
                      command=self._toggle_mesh,
                      font=("Segoe UI", 12)).pack(anchor="w", pady=(0, 8))

        self.mesh_options = ctk.CTkFrame(mesh_outer, fg_color="transparent")
        self.mesh_options.pack(fill="x")

        _, self.target_triangles = self._labeled_slider(
            self.mesh_options, "Target Triangles", 1000, 64000,
            cfg.mesh.target_triangles)

        self.multithreading = ctk.BooleanVar(value=cfg.mesh.multithreaded)
        ctk.CTkSwitch(self.mesh_options, text="Multithreading",
                      variable=self.multithreading, command=self._toggle_threads,
                      font=("Segoe UI", 12)).pack(anchor="w", pady=(6, 2))

        self.thread_options = ctk.CTkFrame(self.mesh_options, fg_color="transparent")
        self.thread_options.pack(fill="x", padx=(16, 0))
        _, self.num_threads = self._labeled_slider(
            self.thread_options, "Threads", 1, 32, cfg.mesh.threads)

        if not self.mesh_enabled.get():
            self.mesh_options.pack_forget()

        # ================= GENERATE BUTTON ===================================
        ctk.CTkButton(
            root, text="?   Generate LODs", command=self.submit,
            height=46, corner_radius=8, font=("Segoe UI Semibold", 15),
        ).pack(fill="x", pady=(8, 0))

        self.grab_set()

    # --- Widget Factories -----------------------------------------------------
    def _labeled_combo(self, parent, label, values, default, width=130):
        ctk.CTkLabel(parent, text=f"{label}:", font=("Segoe UI", 12)).pack(
            anchor="w", pady=(6, 2))
        combo = ctk.CTkComboBox(parent, values=values, state="readonly",
                                width=width, font=("Segoe UI", 12))
        combo.set(default)
        combo.pack(anchor="w")
        return combo

    def _labeled_slider(self, parent, label, from_, to, default, decimals=0):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(6, 0))

        ctk.CTkLabel(row, text=f"{label}:", font=("Segoe UI", 12)).pack(anchor="w")

        slider_row = ctk.CTkFrame(row, fg_color="transparent")
        slider_row.pack(fill="x")

        val_text = str(int(default)) if decimals == 0 else f"{default:.{decimals}f}"
        val_label = ctk.CTkLabel(slider_row, text=val_text, width=60,
                                 font=("Segoe UI Semibold", 12))
        val_label.pack(side="right", padx=(8, 0))

        def on_change(v):
            if decimals == 0:
                val_label.configure(text=str(int(v)))
            else:
                val_label.configure(text=f"{v:.{decimals}f}")

        slider = ctk.CTkSlider(slider_row, from_=from_, to=to, command=on_change)
        slider.set(default)
        slider.pack(side="left", fill="x", expand=True)

        return row, slider

    # --- Toggles --------------------------------------------------------------
    def _toggle_texture(self):
        if self.texture_enabled.get():
            self.texture_options.pack(fill="x")
        else:
            self.texture_options.pack_forget()

    def _toggle_ao(self):
        if self.ao_enabled.get():
            self.ao_options.pack(fill="x", padx=(16, 0))
        else:
            self.ao_options.pack_forget()

    def _toggle_normal(self):
        if self.normal_enabled.get():
            self.normal_options.pack(fill="x")
        else:
            self.normal_options.pack_forget()

    def _toggle_shadow(self):
        if self.fake_shadows.get():
            self.shadow_options.pack(fill="x", padx=(16, 0),
                                     before=self._noise_switch)
        else:
            self.shadow_options.pack_forget()

    def _toggle_noise(self):
        if self.noise_enabled.get():
            self.noise_options.pack(fill="x", padx=(16, 0))
        else:
            self.noise_options.pack_forget()

    def _toggle_mesh(self):
        if self.mesh_enabled.get():
            self.mesh_options.pack(fill="x")
        else:
            self.mesh_options.pack_forget()

    def _toggle_threads(self):
        if self.multithreading.get():
            self.thread_options.pack(fill="x", padx=(16, 0))
        else:
            self.thread_options.pack_forget()

    # --- Submit ---------------------------------------------------------------
    def submit(self):
        cfg = self.cfg

        self.selected_worldspaces = [
            ws for ws, var in self.worldspace_vars.items() if var.get()
        ]

        ebf = self.use_ebf_naming.get()
        if ebf == "Auto":
            cfg.general.use_enginebugfixes_naming_scheme = "Auto"
        else:
            cfg.general.use_enginebugfixes_naming_scheme = (ebf == "Yes")

        cfg.general.generate_color_maps = self.texture_enabled.get()
        cfg.general.generate_normal_maps = self.normal_enabled.get()
        cfg.general.generate_meshes = self.mesh_enabled.get()

        cfg.texture.texture_dimension = int(self.texture_resolution.get())
        cfg.texture.internal_dimension = int(self.texture_internal_resolution.get())
        cfg.texture.gauss_blending_falloff = round(self.gauss_falloff.get(), 1)

        cfg.ao.enabled = self.ao_enabled.get()
        cfg.ao.strength = round(self.ao_strength.get(), 2)
        #cfg.ao.seek_range = int(self.ao_seek_range.get())

        cfg.normal_map.texture_dimension = int(self.normal_resolution.get())
        cfg.normal_map.compression = (self.normal_compression.get() == "DXT1")
        cfg.normal_map.z_boost = round(self.z_boost.get(), 2)
        cfg.normal_map.shadows_enabled = self.fake_shadows.get()
        cfg.normal_map.shadow_strength = round(self.shadow_strength.get(), 2)

        cfg.normal_map_noise.noise_enabled = self.noise_enabled.get()
        cfg.normal_map_noise.noise_intensity = int(self.noise_intensity.get())
        cfg.normal_map_noise.noise_scale = int(self.noise_scale.get())

        cfg.mesh.target_triangles = int(self.target_triangles.get())
        cfg.mesh.multithreaded = self.multithreading.get()
        cfg.mesh.threads = int(self.num_threads.get())

        cfg.apply_overrides()
        self.confirmed = True
        self.destroy()

class PluginSettingsUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.withdraw()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.destroy()
        self.quit()

    def show_settings(self, cfg, worldspace_list) -> tuple:
        window = LODSettingsWindow(self, cfg, worldspace_list)
        self.wait_window(window)
        if window.confirmed:
            return True, window.selected_worldspaces
        return False, []