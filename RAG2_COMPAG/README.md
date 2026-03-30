# Semantic Synchronization of Operator Manuals with Embedded Large Language Models over the ISO 11783 Network

<!-- > **📝 Citation:** [Authors]. "". [Journal/Conference Name], [Year]. [DOI/Link] (To be published) -->

## 🌾 Research Context
Integrating Large Language Models (LLMs) into agricultural machinery requires a robust solution for offline data access. Modern agricultural machinery features multi-brand systems where implements connect to tractors via ISO 11783 (ISOBUS). This project explores how embedded LLMs access textual data (digital operator manuals) stored in an implement's Electronic Control Unit (ECU) and accessed by a Universal Terminal in the tractor cabin.

Access to this data is crucial in rural areas where internet connections are unreliable. This repository contains the code and framework for building, testing, and evaluating Retrieval-Augmented Generation (RAG) pipelines directly applicable to these resource-constrained vehicle networks.

### 🔑 Key Findings
* **Minimum Viable Intelligence (MVI):** Our benchmarks establish that 8B-parameter models (e.g., Qwen 2.5 7B, DeepSeek R1 8B) provide the necessary semantic reasoning for cross-lingual queries, making them the MVI for edge deployment. Models smaller than 7B struggle heavily with the "cross-lingual tax".
* **First-Time Pairing Protocol:** To mitigate ISOBUS bandwidth contention, the system transfers the manual only once per implement revision using version hash matching.
* **Search Navigator UI:** To combat AI hallucination, we propose a *Closed-Loop Human Verification* architecture where the LLM functions as a navigational aid, highlighting the verbatim source text for the operator rather than acting as a standalone oracle.

---

## 📈 Visualizations & Results

**Quick Links:**
*   📂 **RAG Results (JSON):**[results/](results/)
*   📊 **Visualizations (Plots):** [visualization/plots/](visualization/plots/)
*   📋 **LaTeX Tables:** [visualization/latex_tables/](visualization/latex_tables/)

## 🛠️ Project Overview & Features

To convert a PDF file into `txt` for this framework, please use the `docling_page_wise_pdf_converter` inside the folder `zeroshot\docling_page_wise_pdf_converter`.

This project provides a complete framework for testing different language manuals against a common set of English questions, using configurable LLMs, retrieval parameters, and evaluation metrics.

*   **Multi-Language & Cross-Lingual Support:** Process and query documents in different languages (English, German, French, Dutch, Italian, Spanish) against English queries to measure the "cross-lingual tax".
*   **Format Testing:** Evaluate the token efficiency and parsing overhead of Markdown vs. JSON vs. XML.
*   **Hybrid RAG Architecture:** Combines BM25 keyword search with semantic vector retrieval (via ChromaDB), merged using Reciprocal Rank Fusion (RRF).
*   **Automated LLM-as-a-Judge Evaluation:** Employs a separate, larger LLM (e.g., GPT-OSS 20B) to evaluate the correctness of the generated answers against ground-truth expected answers.
*   **Docker Support:** Includes a Dockerfile for containerized setup and execution, managing the Ollama service and dependencies.

---

## ⚙️ Prerequisites

*   Python 3.x (Tested with 3.12)
*   Docker (Recommended for managing Ollama and dependencies)
*   Git
*   Ollama installed and running (either locally or via the provided Docker setup).
*   **Required Ollama Models:** Ensure the LLMs specified in `config.json` (`question_models_to_test`, `evaluator_model_name`) are pulled in your Ollama instance (e.g., `ollama pull llama3.2:3b`, `ollama pull qwen2.5:7b`).
*   **Embedding Model:** The system relies on the embedding model specified in `config.json` (e.g., `Qwen/Qwen3-Embedding-8B`). `pip install -r requirements.txt` handles installing the necessary libraries.

---

## 🐳 Docker Installation

*Note: Detailed instructions for setting up and running with Docker can be found in the [Docker-README.md](Docker-README.md). The provided `Dockerfile` aims to set up the environment and pull necessary models.*

---

## 💻 Manual Installation

For manual installation, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd RAG2_COMPAG/
    ```

2.  **Create a virtual environment:**
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

    *(Optional: Setting up in VS Code)*
    1. Open the `RAG2_COMPAG` folder in VS Code.
    2. VS Code should automatically detect the `.venv` virtual environment. 
    3. Press `Ctrl+Shift+P` (or `Cmd+Shift+P`), type `Python: Select Interpreter`, and choose the one within your `.venv`.

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🎛️ Configuration

1.  **Review and modify `config.json`:** This is the central configuration file.
    *   Set `llm_models` with your desired Ollama models and their parameters (ensure the `name` matches your `ollama list` output).
    *   Update `question_models_to_test` and `evaluator_model_name`.
    *   Configure `files_to_test` and `file_extensions_to_test` with the manuals you want to process (e.g., `md`, `json`, `xml`).
    *   Verify `question_dataset_paths` point to your JSON QA datasets.
    *   Adjust `rag_parameters` (`chunk_sizes_to_test`, `overlap_sizes_to_test`, `num_retrieved_docs`, `retrieval_algorithms_to_test`).
    *   Set the `output_dir` for saving results.
2.  **Prepare Data:**
    *   Place manuals in the `manuals/` directory.
    *   Place question/answer JSON datasets in the `question_datasets/` directory.
    *   Ensure prompt template files exist at the paths specified in `prompt_paths`.

---

## 🚀 Usage Workflow

The entire testing pipeline is orchestrated by `main.py`. This single script handles creating the vector databases and then running the tests according to the settings in `config.json`.

1.  **Activate the Virtual Environment** (if running manually).
2.  **Run the Main Script:**
    ```bash
    python main.py
    ```
    The script will log its progress to the console, initialize the embedding model, build/update ChromaDB vector databases, and run the full QA & Evaluation pipeline.

**Output:** Test results are saved as JSON files in the `results/` directory. Each file contains run parameters, overall metrics (Accuracy, F1-Score, etc.), timing, and per-question evaluation details.

---

## 🧪 Interactive Demo

To quickly test the RAG pipeline with a single question, use the `ask_question_demo.ipynb` Jupyter notebook.

1. Ensure your environment is active and Ollama is running.
2. Open `ask_question_demo.ipynb`.
3. The notebook allows you to:
   * **Input a custom question.**
   * **Configure Parameters:** Choose the LLM, evaluate the answer, etc.
   * **Automatic Context Retrieval:** Identifies the correct ChromaDB collection based on your config.
   * **View Retrieved Context & Generate Answer:** See the text chunks retrieved from the DB and the final generated response.
