from tkinter import Tk
from tkinter import filedialog


def get_file(title: str=None, filetypes=None):
    root = Tk()
    root.withdraw()
    if filetypes is None:
        return filedialog.askopenfilename(
            title=title,
        )
    return filedialog.askopenfilename(
        title=title,
        filetypes=filetypes
    )
