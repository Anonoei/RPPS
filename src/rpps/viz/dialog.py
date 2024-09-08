"""Dialog wrappers"""
from tkinter import Tk
from tkinter import filedialog


def get_file(title: str=None, filetypes=None):
    """Prompts for file selection"""
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
