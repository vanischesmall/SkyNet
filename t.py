import customtkinter as ctk

root = ctk.CTk()
root.title("Custom SkyNet")

root.geometry("500x400")
root.resizable(True, True)
root.configure(fg_color="#1A1A1A")

text_color = "#ebdbb2"
fg = "#689d6a"
bg = "#1A1A1A"
hover = "#8EC07C"

loadBtn = ctk.CTkButton(
    master=root,
    text="Load\nStart",
    width=40,
    height=40,
    corner_radius=10,
    fg_color=fg,
    hover_color=hover,
    text_color=text_color,
)
loadBtn.place(relx=0.05, rely=0.05)

startBtn = ctk.CTkButton(
    master=root,
    text="Start",
    width=40,
    height=40,
    corner_radius=10,
    fg_color=fg,
    hover_color=hover,
    text_color=text_color,
)
startBtn.place(relx=0.2, rely=0.05)

stopBtn = ctk.CTkButton(
    master=root,
    text="Stop",
    width=40,
    height=40,
    corner_radius=10,
    fg_color=fg,
    hover_color=hover,
    text_color=text_color,
)
stopBtn.place(relx=0.35, rely=0.05)

videoBtn = ctk.CTkButton(
    master=root,
    text="Video",
    width=40,
    height=40,
    corner_radius=10,
    fg_color=fg,
    hover_color=hover,
    text_color=text_color,
)
videoBtn.place(relx=0.5, rely=0.05)

combobox = ctk.CTkComboBox(
    master=root,
    values=["MEGA", "Bogdus", "VAIO", "ROZEN"],
    width=150,
    height=40,
    corner_radius=10,
    fg_color=fg,
    border_color=fg,
    button_color=fg,
    text_color=text_color,
    dropdown_fg_color=bg,
    dropdown_hover_color=hover,
    dropdown_text_color=text_color,
)
combobox.place(relx=0.66, rely=0.05)


root.mainloop()
