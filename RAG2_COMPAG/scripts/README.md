# How to Use the Ollama Model Test Script

This guide explains how to run a quick test to see which of your AI models are working correctly with Ollama and which are causing errors. This helps diagnose problems without running the full, time-consuming analysis.

### Prerequisites

Before you start, make sure the **Ollama application is running** on your computer. You should see its icon in your system tray.

### Step-by-Step Instructions

**Step 1: Open a Command Line Tool**

*   On Windows, click the Start Menu and type `Command Prompt` or `PowerShell`, then press Enter.

**Step 2: Navigate to the Project Folder**

*   In the command line window, you need to go to the main project folder.
*   Type `cd` followed by a space, and then the full path to your `RAG2_COMPAG` directory. The path will be unique to your computer.
*   For example, it might look something like this (replace with your actual path):
    ```sh
    cd C:\Users\YourUsername\Documents\p_llm_manual-1\RAG2_COMPAG
    ```
*   Press Enter. The command prompt should now show that you are inside that folder.

**Step 3: Run the Test Script**

*   Now that you are in the correct folder, copy and paste the following command into the window:
    ```sh
    python scripts/test_ollama_models.py
    ```
*   Press Enter.

### What to Expect

*   The script will begin testing each Ollama model listed in your `config.json` file, one by one.
*   For each model, you will see a status update:
    *   **✅ SUCCESS!** means the model is working correctly.
    *   **❌ FAILURE!** means the model ran into an error.
*   At the very end, the script will print a clear **TEST SUMMARY** that lists all the successful and all the failed models.

### Next Steps

Once the script is finished, please **copy the entire output** from the command line window, especially the final summary list of failed models. This information is crucial for fixing the configuration.
