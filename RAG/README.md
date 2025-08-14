# Agri-Query RAG Testing Framework



To convert a PDF file into txt for this framework, please use the docling_page_wise_pdf_converter inside the folder zeroshot\docling_page_wise_pdf_converter


More Information
*   **RAG Plots:** [RAG Plots](RAG/visualization/plots/) 
*   **RAG Results Files:** [RAG Results Folder](RAG/results/) 



This project provides a framework for building, testing, and evaluating Retrieval-Augmented Generation (RAG) pipelines. It allows testing different language manuals against a common set of English questions, using configurable LLMs, retrieval parameters, and evaluation metrics.

## Features

*   **Multi-Language Support:** Process and query documents in different languages (configured in `config.json`).
*   **Configurable Components:** Easily configure LLMs (via Ollama), vector database parameters (chunk size, overlap), retrieval settings (number of documents), and file paths through `config.json`.
*   **Vector Database Integration:** Uses ChromaDB for persistent vector storage. Includes scripts to create databases based on specific chunking strategies.
*   **Automated Testing:** Runs English question datasets against specified language document collections.
*   **LLM-Based Evaluation:** Employs a separate LLM to evaluate the correctness of the generated answers against expected answers.
*   **Performance Metrics:** Calculates Accuracy, Precision, Recall, Specificity, and F1-Score based on evaluation results.
*   **Detailed Results:** Saves comprehensive test results, including parameters, timings, metrics, and individual question/answer details, to JSON files.
*   **Docker Support:** Includes a Dockerfile for containerized setup and execution, managing Ollama service and dependencies.

## Prerequisites

*   Python 3.x (Tested with 3.12)
*   Docker (Recommended for managing Ollama and dependencies)
*   Git
*   Ollama installed and running (either locally or via the provided Docker setup).
*   **Required Ollama Models:** Ensure the LLMs specified in `config.json` (`question_model_name`, `evaluator_model_name`) are pulled in your Ollama instance (e.g., `ollama pull llama3.2:3b`, `ollama pull gemma3:12b`). The Dockerfile attempts to pull several models during the build process.
*   **Embedding Model:** The system relies on the `Alibaba-NLP/gte-Qwen2-7B-instruct` sentence transformer model for embeddings. `pip install -r requirements.txt` handles installing the necessary library (`sentence-transformers`).

## Docker Installation

*Note: Instructions for setting up and running with Docker can be found in the [DOCKER_README.md](Docker-README.md). The provided `Dockerfile` aims to set up the environment and pull necessary models.*

## Manual Installation

For manual installation, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd p_llm_manual/RAG
    ```

2.  **Create a virtual environment:**

    It is recommended to use a virtual environment to isolate project dependencies.

    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**

    *   **On Windows:**

        ```bash
        .venv\Scripts\activate
        ```

    *   **On macOS and Linux:**

        ```bash
        source .venv/bin/activate
        ```

    **Setting up a Virtual Environment in VS Code:**

    Visual Studio Code (VS Code) can automatically detect and use virtual environments. Here’s how to ensure your virtual environment is correctly set up in VS Code:

    1.  **Open the Project in VS Code:** Open the project folder (e.g., `RAG-Framework`) in VS Code.

    2.  **VS Code should detect the virtual environment:** When you open the project, VS Code should automatically detect the `.venv` virtual environment in your project directory. You might see a notification in the bottom right corner of VS Code suggesting that it has found a virtual environment.

    3.  **Select the Python Interpreter:**
        *   If VS Code doesn't automatically select the virtual environment, or if you want to verify or change the Python interpreter, press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) to open the Command Palette.
        *   Type `Python: Select Interpreter` and press Enter.
        *   VS Code will display a list of available Python interpreters. Choose the one that is within your `.venv` virtual environment. It will typically be something like: `./.venv/bin/python` (on macOS/Linux) or `.\.venv\Scripts\python.exe` (on Windows).

    4.  **Install packages within the virtual environment:** Open the terminal in VS Code (`Ctrl+\`` or `Cmd+\``). Ensure that the virtual environment is activated in the terminal. You should see `(.venv)` at the beginning of your terminal prompt. If it's activated, you can install the project dependencies:

        ```bash
        pip install -r requirements.txt
        ```

    **Helpful Tips for First-Time Users:**

    *   **Ensure Python Extension is Installed in VS Code:** Make sure you have the official Python extension installed in VS Code. This extension provides excellent support for Python development, including virtual environments.

    *   **Restart VS Code:** If you are having trouble getting VS Code to recognize your virtual environment, try restarting VS Code after creating and activating the `.venv`.

    *   **Check `.venv` Location:** Verify that the `.venv` folder is created in the root directory of your RAG project. VS Code usually looks for `.venv` in the project root.

    *   **Using the VS Code Terminal:** When you open the terminal in VS Code (using `Ctrl+\`` or `Cmd+\``), it should automatically activate the selected virtual environment for you. If it doesn't, you might need to manually activate it once in the terminal using the `source .venv/bin/activate` or `.venv\Scripts\activate` commands as mentioned in step 3 above.

    *   **Verify with `pip list`:** After activating the virtual environment and installing requirements, you can verify that packages are installed in your virtual environment by running `pip list` in the VS Code terminal. This will show the packages installed in the active virtual environment.

4.  **Install dependencies:**

    Install the required Python packages using pip:

    ```bash
    pip install -r requirements.txt
    ```
## Try it out

To quickly test the RAG pipeline with a single question, use the `ask_question_demo.ipynb` Jupyter notebook located in the `RAG` directory.

**`ask_question_demo.ipynb` - Interactive RAG Query:**

This notebook allows you to:

1.  **Input a Question:** Interactively provide a question you want to ask.
2.  **Configure Parameters:** Easily set:
    *   The LLM to use for answering (or skip LLM answering to only see retrieved context).
    *   Whether to evaluate the LLM's answer against an expected answer you provide.
    *   The `config.json` and ChromaDB directory to use (defaults are usually fine).
3.  **Automatic Context Retrieval:** The notebook automatically identifies the relevant ChromaDB collection based on the *first* language, chunk size, and overlap size specified in your `config.json`.
4.  **View Retrieved Context:** See the text chunks retrieved from the database that are most relevant to your question.
5.  **Get LLM Answer (Optional):** If an LLM is specified, it generates an answer based on your question and the retrieved context.
6.  **Evaluate Answer (Optional):** If enabled, the LLM's answer is evaluated against an "expected answer" you provide, using another LLM for the judgment.

**How to Run:**

*   Ensure you have followed the installation steps (Python environment, dependencies, Ollama models).
*   Navigate to the `p_llm_manual/RAG/` directory.
*   Open and run the `ask_question_demo.ipynb` notebook cell by cell.
*   Modify the "User Configuration" cell as needed, especially `your_question`.

This demo notebook is a great way to understand the core components of the RAG system (retrieval, LLM augmentation, evaluation) in action with minimal setup for a single query.

## Configuration

1.  **Review and modify `config.json`:** This is the central configuration file.
    *   Set `llm_models` with your desired Ollama models and their parameters (ensure the `name` matches your `ollama list` output).
    *   Update `question_model_name` and `evaluator_model_name` to point to keys within `llm_models.ollama`.
    *   Configure `language_configs` with paths to your language-specific manuals and desired base collection names.
    *   Verify `question_dataset_paths` point to your JSON QA datasets (e.g., general, table-based, unanswerable).
    *   Adjust `rag_parameters` (`chunk_size`, `overlap_size`, `num_retrieved_docs`). The `chunk_size` and `overlap_size` here determine *which* pre-existing database `rag_tester.py` will use.
    *   Set the `output_dir` for saving results.
2.  **Prepare Data:**
    *   Place language manuals in the paths specified in `config.json` (e.g., create a `manuals/` directory and place `english_manual.txt`, `french_manual.txt` etc. inside).
    *   Place question/answer JSON datasets in the paths specified (e.g., create `question_datasets/` and place `question_answers_pairs.json`, etc. inside).
    *   Ensure prompt template files exist at the paths specified in `prompt_paths` (e.g., `prompt_templates/question_prompt.txt`).

## Usage Workflow

1.  **Create Vector Databases:** You *must* create the ChromaDB vector collections before running the main test. These collections store the embedded document chunks. Choose one method:

    *   **Method A (Batch Creation):** Create databases for *multiple* chunk/overlap combinations defined *within the script* (`CHUNK_SIZES_TO_CREATE`, `OVERLAP_SIZES_TO_CREATE`). This is useful for preparing databases for various parameter tests upfront.
        ```bash
        python utils/create_databases.py
        ```
    *   **Method B (Specific Creation):** Create or update the database specifically for the *single* `chunk_size` and `overlap_size` combination currently specified in `config.json`. This script also runs a basic retrieval evaluation (source hit rate) on the created/updated database.
        ```bash
        python rag_pipeline.py
        ```
    *   **Important:** The database name includes the chunk and overlap size (e.g., `english_manual_cs2000_os50`). `rag_tester.py` will look for a database matching the parameters currently set in `config.json`. Ensure the database you intend to test exists.

2.  **Run the RAG Test:** Executes the full Question-Answering and Evaluation pipeline. It uses the ChromaDB collection corresponding to the `chunk_size` and `overlap_size` currently set in `config.json`. It tests all languages configured in `language_configs` against the English question sets.
    ```bash
    python rag_tester.py
    ```

## Output

Test results are saved as JSON files in the directory specified by `output_dir` in `config.json` (default: `results/`).

Filenames follow the pattern:
`{retrieval_algorithm}_{language}_{sanitized_question_model_name}_{chunk_size}_overlap_{overlap_size}_topk_{num_retrieved_docs}.json`

Each file contains:
*   `test_run_parameters`: Configuration used for that specific test run (language, models, RAG params, collection name).
*   `overall_metrics`: Calculated performance metrics (Accuracy, Precision, Recall, Specificity, F1-Score, TP/TN/FP/FN counts).
*   `timing`: Duration of the QA and Evaluation phases, and the overall duration.
*   `per_dataset_details`: Raw results broken down by the input English question dataset type. Includes the original question, expected answer, model's generated answer, evaluation judgment ("yes"/"no"/"error"), and timing for that dataset.
