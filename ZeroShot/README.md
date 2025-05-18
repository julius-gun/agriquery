# ZeroShot MultiFormatLLMTester

{Long Context Testing}
Please pay attention: Zeroshot means "Long Context" in all the used code. Inside the folder "Zeroshot" you find code to run results for the long context result. E.g. the visualizations of such results look like these:
![Long context Accuracy comparison for English manual](visualization/plots/zeroshot_accuracy_vs_noise_english.png)

We tested Large Language Models (LLMs) on their ability to answer questions with a given context. This was done without prior fine-tuning, assessing the models' inherent understanding across different context sizes. The target page containing the correct answer was always included in the context provided to the LLM in this scenario. Additional appending pages, or "noise", was added in varying amount, such as 10k tokens, to simulate different information levels. We also tested the LLMs using the entire document, approximately 59k tokens, as context. Performance is measured in standard metrics like accuracy, precision, recall and F1 score.

**Evaluate LLMs on Diverse Document Formats**

This project assesses the performance of Large Language Models (LLMs) in understanding and answering questions based on documents in various formats: PDF, Markdown, CSV, XML, YAML, and JSON. It utilizes **docling** for page-wise conversion of documents, ensuring accurate context extraction. The evaluation is performed in a zero-shot setting, without fine-tuning the models.

**Key Features:**

*   **Multi-Format Evaluation:** Tests LLMs on PDF, Markdown, CSV, XML, YAML, and JSON formats.
*   **Zero-Shot Learning:** Evaluates models without format-specific training.
*   **Page-Wise Context Handling:** Uses `docling` for precise, page-level document conversion.
*   **Flexible Context Retrieval:** Supports page-wise and token-wise context retrieval with adjustable noise levels.
*   **Comprehensive Metrics:** Calculates accuracy, precision, recall, and F1 score for detailed performance analysis.
*   **Modular Design:** Easily extendable with new LLM connectors and document formats.

**Dataset:**

The project uses a dataset designed to test both answerable and unanswerable questions, ensuring a balanced evaluation:

*   **General Question-Answer Pairs:** [44 pairs](question_datasets/question_answers_pairs.json)
*   **Table-Based Question-Answer Pairs:** [10 pairs](question_datasets/question_answers_tables.json)
*   **Unanswerable Question-Answer Pairs:** [54 pairs](question_datasets/question_answers_unanswerable.json) (designed to test negative constraints)

# Installation

## Docker Installation
[Docker installation instructions for Ubuntu can be found here](Docker-README.md)

## Manual Installation

For manual installation, follow these steps:

0. **Install VS Code**

    Install Python 3.12, VS Code and git
    Extensions:
    - Python

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd p_llm_manual/ZeroShot
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

    1.  **Open the Project in VS Code:** Open the `ZeroShot` folder in VS Code.

    2.  **VS Code should detect the virtual environment:** When you open the project, VS Code should automatically detect the `.venv` virtual environment in your project directory. You might see a notification in the bottom right corner of VS Code suggesting that it has found a virtual environment.

    3.  **Select the Python Interpreter:**
        *   If VS Code doesn't automatically select the virtual environment, or if you want to verify or change the Python interpreter, press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) to open the Command Palette.
        *   Type `Python: Select Interpreter` and press Enter.
        *   VS Code will display a list of available Python interpreters. Choose the one that is within your `.venv` virtual environment. It will typically be something like:  `./.venv/bin/python` (on macOS/Linux) or `.\.venv\Scripts\python.exe` (on Windows).

    4.  **Install packages within the virtual environment:** Open the terminal in VS Code (`Ctrl+\`` or `Cmd+\``). Ensure that the virtual environment is activated in the terminal. You should see `(.venv)` at the beginning of your terminal prompt. If it's activated, you can install the project dependencies:

        ```bash
        pip install -r requirements.txt
        ```

    **Helpful Tips for First-Time Users:**

    *   **Ensure Python Extension is Installed in VS Code:**  Make sure you have the official Python extension installed in VS Code. This extension provides excellent support for Python development, including virtual environments.

    *   **Restart VS Code:** If you are having trouble getting VS Code to recognize your virtual environment, try restarting VS Code after creating and activating the `.venv`.

    *   **Check `.venv` Location:**  Verify that the `.venv` folder is created in the root directory of your `ZeroShot` project. VS Code usually looks for `.venv` in the project root.

    *   **Using the VS Code Terminal:** When you open the terminal in VS Code (using `Ctrl+\`` or `Cmd+\``), it should automatically activate the selected virtual environment for you. If it doesn't, you might need to manually activate it once in the terminal using the `source .venv/bin/activate` or `.venv\Scripts\activate` commands as mentioned in step 3 above.

    *   **Verify with `pip list`:** After activating the virtual environment and installing requirements, you can verify that packages are installed in your virtual environment by running `pip list` in the VS Code terminal. This will show the packages installed in the active virtual environment.

4.  **Install dependencies:**

    Install the required Python packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **Configuration File (`config.json`):**

    The `config.json` file in the `ZeroShot` directory is used to configure the LLMs, dataset paths, output directories, and other settings.

    *   **LLM Models:**
        *   The `llm_models` section defines the configuration for different LLMs, including models from Ollama and Gemini.
        *   You can specify model names, temperature, context window, and other parameters.
        *   For cloud-based models like Gemini, ensure you have set up the necessary API keys as environment variables or directly in your code if necessary (not recommended for security reasons).
    *   **Evaluator Model:**
        *   `evaluator_model` specifies the model used for evaluation. By default, it is set to `gemma2:latest`.
    *   **Prompt Paths:**
        *   `prompt_paths` defines the paths to the prompt templates used for question answering and evaluation.
    *   **Question Dataset Paths:**
        *   `question_dataset_paths` lists the paths to the JSON files containing question-answer pairs.
    *   **Document Path:**
        *   `document_path` specifies the URL or path to the document to be tested.
    *   **Output Directory:**
        *   `output_dir` defines the directory where the test results and visualizations will be saved.
    *   **Tokenizer Name:**
        *   `tokenizer_name` specifies the tokenizer to be used.
    *   **Visualization Directory:**
        *   `visualization_dir` defines the directory for saving visualization plots.
    *   **File Extensions to Test:**
        *   `file_extensions_to_test` is a list of file extensions to be tested (e.g., `txt`, `csv`, `json`).

    **Example Configuration Snippet (`config.json`):**
    ```json
    {
        "llm_models": {
            "ollama": {
                "deepseek-r1_70B-128k": {
                    "name": "deepseek-r1:70b",
                    "temperature": 0.0,
                    "num_predict": 1024,
                    "context_window": 128000
                },
                "llama3.2_1B-128k": {
                    "name": "llama3.2:1b",
                    "temperature": 0.0,
                    "num_predict": 1024,
                    "context_window": 128000
                }
            },
            "gemini": {
                "gemini-pro": {
                    "name": "gemini-2.0-flash-thinking-exp-01-21",
                    "temperature": 0.0,
                    "num_predict": 1024
                }
            }
        },
        "evaluator_model": "gemma2:latest",
        "prompt_paths": {
            "question_prompt": "prompt_templates/question_prompt.txt",
            "evaluation_prompt": "prompt_templates/evaluation_prompt.txt"
        },
        "question_dataset_paths": [
            "question_datasets/question_answers_pairs.json",
            "question_datasets/question_answers_tables.json",
            "question_datasets/question_answers_unanswerable.json"
        ],
        "document_path": "https://www.kvgportal.com/W_global/Media/lexcom/VN/A14870/A148703540-2.pdf",
        "output_dir": "results",
        "question_tracker_path": "utils/question_tracker_data.json",
        "tokenizer_name": "pcuenq/Llama-3.2-1B-Instruct-tokenizer",
        "visualization_dir": "results/plots",
        "file_extensions_to_test": [
            "txt",
            "csv",
            "xml",
            "yaml",
            "html",
            "json"
        ]
    }
    ```

+## Running the Tests and Evaluations
+
+To run the tests and evaluations, navigate to the `ZeroShot` directory in your terminal. Ensure that you have added the `ZeroShot` folder itself to your workspace. From within the `ZeroShot` directory, you can execute `main.py` with the standard settings or using the following command:

```bash
python main.py --mode test --config config.json
```

or to run evaluations directly:

```bash
python main.py --mode evaluate --config config.json
```

**Explanation of arguments:**

*   `python main.py`:  This executes the `main.py` script using your Python interpreter.
*   `--mode`:  Specifies the mode of operation.
    *   `test`: Runs the tests to generate predictions.
    *   `evaluate`: Runs the evaluations to calculate metrics based on the generated predictions.
*   `--config config.json`:  Specifies the configuration file to be used. `config.json` contains settings for models, datasets, and output directories.

**Important:**
*   **Ollama Models:** If you are using Ollama models, ensure that you have pulled the required models before running the tests or evaluations. You can easily pull all necessary models by running the `ollama pull models.bat` script located in the `ZeroShot\trash\` directory. This script will pull all the Ollama models listed in your `config.json` file.

    ```bash
    ZeroShot\trash\ollama pull models.bat
    ```
     **Note:** Ensure that the paths in `config.json` are correct relative to your project directory. You can modify this file to adjust the models being tested, dataset locations, and output settings.