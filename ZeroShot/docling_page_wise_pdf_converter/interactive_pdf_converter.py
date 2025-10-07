    #  run this command from ZeroShot directory to run this script
    # cd zeroshot
# python -m docling_page_wise_pdf_converter.interactive_pdf_converter

  
import logging
import time
from pathlib import Path
import sys

# --- Dependency Check ---
# Check for required packages and provide a helpful message if they are missing.
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    # This import is now RELATIVE to the current package.
    # It tells Python to look for pdf_converter in the same folder.
    from .pdf_converter import convert_pdf
except ModuleNotFoundError as e:
    # If tkinter is available, show a GUI error. Otherwise, print to console.
    error_message = (
        f"Error: A required module is missing: {e}\n\n"
        "Please install the required packages by running the following command in your terminal:\n\n"
        "pip install -r requirements.txt"
    )
    try:
        # This will only work if tkinter was successfully imported
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Missing Dependencies", error_message)
    except (NameError, tk.TclError): # Handles case where tkinter is missing or fails to init
        print("="*80, file=sys.stderr)
        print(error_message, file=sys.stderr)
        print("="*80, file=sys.stderr)
    sys.exit(1)


# --- Main Application ---
_log = logging.getLogger(__name__)


def main():
    """Main function to run the PDF converter GUI and processing."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    try:
        # 1. Get input files
        messagebox.showinfo("Information", "Please select one or more PDF files to convert.")
        filenames = filedialog.askopenfilenames(
            title="Select PDF files",
            filetypes=(("PDF files", "*.pdf"), ("All files", "*.*")),
        )

        if not filenames:
            _log.warning("No files were selected. Exiting.")
            messagebox.showwarning("Cancelled", "No files were selected. The program will now exit.")
            return

        # 2. Get output directory
        messagebox.showinfo("Information", "Please select the directory where converted files will be saved.")
        output_dir = filedialog.askdirectory(title="Select Output Directory")

        if not output_dir:
            _log.warning("No output directory was selected. Exiting.")
            messagebox.showwarning("Cancelled", "No output directory was selected. The program will now exit.")
            return

        _log.info("--- Starting Batch Conversion ---")
        _log.info(f"Output directory: {output_dir}")
        _log.info("Selected files:")
        for f in filenames:
            _log.info(f"  - {f}")
        
        failed_files = []

        # 3. Process each file
        for filename in filenames:
            _log.info(f"Processing: {Path(filename).name}...")
            start_time = time.time()
            try:
                # The core conversion call
                convert_pdf(filename, str(output_dir), output_format="all")
                end_time = time.time() - start_time
                _log.info(f"Successfully converted {Path(filename).name} in {end_time:.2f} seconds.")
            except Exception as e:
                failed_files.append(Path(filename).name)
                _log.error(f"Failed to convert {Path(filename).name}. Error: {e}", exc_info=True)
                messagebox.showerror("Conversion Error", f"An error occurred while converting {Path(filename).name}:\n\n{e}")

        # 4. Final report
        _log.info("--- Batch Conversion Finished ---")
        if not failed_files:
            messagebox.showinfo("Success", "All selected files have been processed successfully.")
        else:
            message = "Processing finished, but the following files failed to convert:\n\n" + "\n".join(failed_files)
            messagebox.showwarning("Completed with Errors", message)

    except Exception as e:
        _log.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
        messagebox.showerror("Critical Error", f"A critical error occurred and the program must exit:\n\n{e}")
    finally:
        if 'root' in locals() and root.winfo_exists():
            root.destroy()


if __name__ == "__main__":
    main()