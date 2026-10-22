import os
import customtkinter as ctk
from slowlodgen.ui_style import *

__all__ = ['PluginSelectionWindow', 'PluginSelectorUI']


class PluginSelectionWindow(ctk.CTkToplevel):
    def __init__(self, parent, plugin_list, selected_plugins=None):
        super().__init__(parent, fg_color=BG)
        self.title("slowLandscapeLODGen")
        fit_geometry(self, 520, 760)
        self.minsize(420, 400)
        self.resizable(True, True)
        self.result = None
        selected_lower = {x.lower() for x in selected_plugins or []}
        self.checkbox_vars = {}
        self.rows = {}

        head = ctk.CTkFrame(self, fg_color='transparent')
        head.pack(fill='x', padx=20, pady=(18, 2))
        ctk.CTkLabel(head, text='Plugins', font=(FONT + ' Semibold', 20), text_color=TEXT).pack(side='left')
        self.count_label = ctk.CTkLabel(head, text='', font=(FONT, 12), text_color=MUTED)
        self.count_label.pack(side='left', padx=(10, 0), pady=(4, 0))
        ctk.CTkLabel(self, text='Pre-selected from your plugins.txt.', font=(FONT, 12), text_color=MUTED,
                     anchor='w').pack(fill='x', padx=20, pady=(0, 10))

        footer = ctk.CTkFrame(self, fg_color=CARD, corner_radius=0)
        footer.pack(fill='x', side='bottom')
        ctk.CTkFrame(footer, height=1, fg_color=CARD_BORDER).pack(fill='x', side='top')
        ctk.CTkButton(footer, text='Next', height=40, corner_radius=6, font=(FONT + ' Semibold', 14),
                      fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self.submit).pack(fill='x', padx=20, pady=12)

        card = ctk.CTkFrame(self, fg_color=CARD, border_color=CARD_BORDER, border_width=1, corner_radius=8)
        card.pack(fill='both', expand=True, padx=20, pady=(0, 14))
        self.search = ctk.CTkEntry(card, placeholder_text='Search plugins', height=30, font=(FONT, 12))
        self.search.pack(fill='x', padx=14, pady=(14, 8))
        self.search.bind('<KeyRelease>', lambda e: self._filter())
        ctk.CTkFrame(card, height=1, fg_color=CARD_BORDER).pack(fill='x', padx=14)
        self.scroll = ctk.CTkScrollableFrame(card, fg_color='transparent')
        self.scroll.pack(fill='both', expand=True, padx=6, pady=(4, 10))

        for item in plugin_list:
            var = ctk.BooleanVar(value=item.lower() in selected_lower)
            var.trace_add("write", lambda *_: self._update_count())
            self.checkbox_vars[item] = var
            row = ctk.CTkCheckBox(self.scroll, text=item, variable=var, font=(FONT, 13), text_color=TEXT,
                                  checkbox_width=18, checkbox_height=18, corner_radius=4, height=26,
                                  fg_color=ACCENT, hover_color=ACCENT_HOVER)
            row.pack(anchor='w', pady=1, padx=8)
            self.rows[item] = row
        self.no_match = ctk.CTkLabel(self.scroll, text='Nothing matches.', font=(FONT, 12), text_color=MUTED)
        self.shown = list(plugin_list)

        self._update_count()

    def _filter(self):
        q = self.search.get().lower()
        shown = [item for item in self.rows if q in item.lower()]
        if shown == self.shown:
            return
        self.shown = shown
        for row in self.rows.values():
            row.pack_forget()
        self.no_match.pack_forget()
        for item in shown:
            self.rows[item].pack(anchor='w', pady=1, padx=8)
        if not shown:
            self.no_match.pack(pady=16)

    def _update_count(self):
        n = sum(1 for v in self.checkbox_vars.values() if v.get())
        update(self.count_label, text=f"{n}/{len(self.checkbox_vars)} selected")

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
