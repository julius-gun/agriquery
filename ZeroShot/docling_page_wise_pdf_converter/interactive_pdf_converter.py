import logging
import time
from pathlib import Path

## GUI for file selection with thinker
import tkinter as tk
from tkinter import filedialog, messagebox  # Added messagebox import

from pdf_converter import convert_pdf

## filetypes for thinker dialog box
filetypes = (("PDF files", "*.PDF"),)

# open-file dialog
root = tk.Tk()
tk.messagebox.showinfo("Information", "Select files (pdf)..")
filenames = tk.filedialog.askopenfilenames(
    title="Select files (pdf)..",
    filetypes=filetypes,
)
# message box
tk.messagebox.showinfo("Information", "Select output directory..")

output_dir = tk.filedialog.askdirectory(title="Select output directory..")


if filenames:
    print("Selected files:")
    for filename in filenames:
        print(filename)
else:
    print("No files selected.")
    quit()


root.destroy()

_log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)


for filename in filenames:
    input_doc_path = filename

    start_time = time.time()
    convert_pdf(input_doc_path, str(output_dir), output_format="all")
    end_time = time.time() - start_time

    _log.info(f"Document converted in {end_time:.2f} seconds.")

if __name__ == "__main__":
    main()