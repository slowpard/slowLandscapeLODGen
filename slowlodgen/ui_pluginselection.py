import os
import customtkinter as ctk

class PluginSelectionWindow(ctk.CTkToplevel):
    def __init__(self, parent, plugin_list, selected_plugins=None):
        super().__init__(parent)
        self.title("slowLandscapeLODGen")
        self.geometry("450x700")
        self.resizable(True, True)
        self.result = None
        selected_plugins = selected_plugins or []
        selected_lower = {x.lower() for x in selected_plugins}
        self.checkbox_vars = {}
        self.checkboxes = {}

        # Header
        ctk.CTkLabel(self, text="Select ESP/ESM Plugins",
                      font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(15, 5))



        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15)

        # Plugin count label
        self.count_label = ctk.CTkLabel(btn_frame, text="", font=ctk.CTkFont(size=12))
        self.count_label.pack(side="right")

        # Scrollable plugin list
        self.scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll.pack(fill="both", expand=True, padx=15, pady=(5, 10))

        for item in plugin_list:
            var = ctk.BooleanVar(value=item.lower() in selected_lower)
            var.trace_add("write", lambda *_: self._update_count())
            self.checkbox_vars[item] = var
            cb = ctk.CTkCheckBox(self.scroll, text=item, variable=var,
                                  checkbox_height=18, checkbox_width=18,
                                  height=24, font=ctk.CTkFont(size=13))
            cb.pack(anchor="w", pady=1, padx=5)
            self.checkboxes[item] = cb

        self._update_count()

        # Submit button
        ctk.CTkButton(self, text="Next", height=36,
                       font=ctk.CTkFont(size=14, weight="bold"),
                       command=self.submit).pack(pady=(0, 15), padx=15, fill="x")
    def _update_count(self):
        n = sum(1 for v in self.checkbox_vars.values() if v.get())
        self.count_label.configure(text=f"{n}/{len(self.checkbox_vars)} selected")

    def submit(self):
        self.result = [name for name, var in self.checkbox_vars.items() if var.get()]
        self.destroy()




class PluginSelectorUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("slowLandscapeLODGen")
        self.geometry("1x1")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.destroy()
        self.quit()
        os._exit(0)
        
    def ShowPluginsSelection(self, plugin_list, selected_plugins=None):
        window = PluginSelectionWindow(self, plugin_list, selected_plugins)
        self.wait_window(window)
        return window.result
