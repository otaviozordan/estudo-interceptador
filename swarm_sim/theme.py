"""Tema escuro compartilhado (construtor e replay)."""
from tkinter import ttk

BG, PANEL, PANEL2, LINE = "#12161d", "#1a202a", "#232b38", "#2f3947"
FG, MUTED, ACCENT, WARN, OK, GOLD = "#e6eaf0", "#8b97a8", "#4aa3ff", "#ff5d5d", "#4cd98a", "#ffd166"


def apply_dark(r):
    """Aplica o tema escuro (ttk 'clam') a uma janela Tk."""
    r.configure(bg=BG)
    st = ttk.Style(r)
    st.theme_use("clam")
    st.configure(".", background=PANEL, foreground=FG, fieldbackground=PANEL2, bordercolor=LINE,
                 lightcolor=PANEL, darkcolor=PANEL, troughcolor=PANEL2, font=("Segoe UI", 9))
    st.configure("TFrame", background=PANEL)
    st.configure("Bar.TFrame", background=BG)
    st.configure("TLabel", background=PANEL, foreground=FG)
    st.configure("Muted.TLabel", foreground=MUTED)
    st.configure("Head.TLabel", foreground=ACCENT, font=("Segoe UI", 9, "bold"))
    st.configure("Bar.TLabel", background=BG, foreground=MUTED)
    st.configure("Alert.TLabel", background=BG, foreground=WARN, font=("Segoe UI", 10, "bold"))
    st.configure("Good.TLabel", background=BG, foreground=OK, font=("Segoe UI", 10, "bold"))
    st.configure("TButton", background=PANEL2, foreground=FG, padding=(8, 4), borderwidth=1, focuscolor=PANEL2)
    st.map("TButton", background=[("active", LINE), ("pressed", ACCENT)])
    st.configure("Accent.TButton", background=ACCENT, foreground="#06101c", font=("Segoe UI", 9, "bold"))
    st.map("Accent.TButton", background=[("active", "#7bbcff")])
    st.configure("Tool.TRadiobutton", background=BG, foreground=FG, padding=(10, 5), indicatorsize=0,
                 borderwidth=0)
    st.map("Tool.TRadiobutton", background=[("selected", ACCENT), ("active", LINE)],
           foreground=[("selected", "#06101c")])
    st.configure("TRadiobutton", background=PANEL, foreground=FG)
    st.configure("TCheckbutton", background=PANEL, foreground=FG)
    st.configure("Bar.TCheckbutton", background=BG, foreground=FG)
    st.map("Bar.TCheckbutton", background=[("active", BG)])
    st.map("TCheckbutton", background=[("active", PANEL)])
    st.configure("TEntry", fieldbackground=PANEL2, foreground=FG, insertcolor=FG, padding=3)
    st.configure("TCombobox", fieldbackground=PANEL2, foreground=FG, arrowcolor=FG, background=PANEL2)
    st.map("TCombobox", fieldbackground=[("readonly", PANEL2)], foreground=[("readonly", FG)])
    st.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(2, 4, 2, 0))
    st.configure("TNotebook.Tab", background=PANEL2, foreground=MUTED, padding=(12, 5))
    st.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", ACCENT)])
    st.configure("Treeview", background=PANEL2, fieldbackground=PANEL2, foreground=FG, rowheight=22,
                 borderwidth=0)
    st.configure("Treeview.Heading", background=LINE, foreground=FG, relief="flat")
    st.map("Treeview", background=[("selected", "#2c4a70")], foreground=[("selected", "#ffffff")])
    st.configure("Horizontal.TScale", background=PANEL)
    st.configure("TSeparator", background=LINE)
    r.option_add("*TCombobox*Listbox.background", PANEL2)
    r.option_add("*TCombobox*Listbox.foreground", FG)
    r.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
