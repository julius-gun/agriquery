# MultiFormatLLMTester

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

*   **General Question-Answer Pairs:** [44 pairs](/question_datasets/question_answers_pairs.json)
*   **Table-Based Question-Answer Pairs:** [10 pairs](/question_datasets/question_answers_tables.json)
*   **Unanswerable Question-Answer Pairs:** [54 pairs](/question_datasets/question_answers_unanswerable.json) (designed to test negative constraints)

# Installation
[Docker installation instructions can be found here](Docker-README.md)

Manual installation
python -m venv .venv
.venv\Scripts\activate
# Preliminary Results

## Visualization Plots
For a more comprehensive view, explore all generated plots in the [results/visualization/plots](results/visualization/plots) directory.
### Heatmap Plots

#### F1 Score Heatmaps

##### F1 Score vs Language Noise (1000 tokens)
![Heatmap F1 Score LLM vs Language Noise 1000](results/visualization/plots/heatmap/heatmap_f1_score_llm_vs_language_noise_1000.png)

##### F1 Score vs Language Noise (10000 tokens)
![Heatmap F1 Score LLM vs Language Noise 10000](results/visualization/plots/heatmap/heatmap_f1_score_llm_vs_language_noise_10000.png)

##### F1 Score vs Language Noise (30000 tokens)
![Heatmap F1 Score LLM vs Language Noise 30000](results/visualization/plots/heatmap/heatmap_f1_score_llm_vs_language_noise_30000.png)

#### Precision and Recall Heatmaps

##### Precision Heatmap for Files
![Heatmap Precision Files](results/visualization/plots/heatmap/heatmap_precision_files.png)

##### Recall Heatmap for Files
![Heatmap Recall Files](results/visualization/plots/heatmap/heatmap_recall_files.png)

### Bar Chart Plots

#### Accuracy Comparison Bar Charts

##### Accuracy Comparison (1000 tokens, Text Files)
![Accuracy Comparison 1000 Text](results/visualization/plots/bar_chart/accuracy_comparison_1000_txt.png)

##### Accuracy Comparison (10000 tokens, Text Files)
![Accuracy Comparison 10000 Text](results/visualization/plots/bar_chart/accuracy_comparison_10000_txt.png)

##### Accuracy Comparison (30000 tokens, Text Files)
![Accuracy Comparison 30000 Text](results/visualization/plots/bar_chart/accuracy_comparison_30000_txt.png)

##### Accuracy Comparison (1000 tokens, File Formats)
![Accuracy Comparison File Formats 1000](results/visualization/plots/bar_chart/accuracy_comparison_file_formats_1000.png)

##### Accuracy Comparison (10000 tokens, File Formats)
![Accuracy Comparison File Formats 10000](results/visualization/plots/bar_chart/accuracy_comparison_file_formats_10000.png)

##### Accuracy Comparison (30000 tokens, File Formats)
![Accuracy Comparison File Formats 30000](results/visualization/plots/bar_chart/accuracy_comparison_file_formats_30000.png)

#### F1 Score Comparison Bar Charts

##### F1 Score Comparison (1000 tokens, Text Files)
![F1 Score Comparison 1000 Text](results/visualization/plots/bar_chart/f1_score_comparison_1000_txt.png)

##### F1 Score Comparison (10000 tokens, Text Files)
![F1 Score Comparison 10000 Text](results/visualization/plots/bar_chart/f1_score_comparison_10000_txt.png)

##### F1 Score Comparison (30000 tokens, Text Files)
![F1 Score Comparison 30000 Text](results/visualization/plots/bar_chart/f1_score_comparison_30000_txt.png)

##### F1 Score Comparison (1000 tokens, File Formats)
![F1 Score Comparison File Formats 1000](results/visualization/plots/bar_chart/f1_score_comparison_file_formats_1000.png)

##### F1 Score Comparison (10000 tokens, File Formats)
![F1 Score Comparison File Formats 10000](results/visualization/plots/bar_chart/f1_score_comparison_file_formats_10000.png)

##### F1 Score Comparison (30000 tokens, File Formats)
![F1 Score Comparison File Formats 30000](results/visualization/plots/bar_chart/f1_score_comparison_file_formats_30000.png)

#### Precision Comparison Bar Charts

##### Precision Comparison (1000 tokens, Text Files)
![Precision Comparison 1000 Text](results/visualization/plots/bar_chart/precision_comparison_1000_txt.png)

##### Precision Comparison (10000 tokens, Text Files)
![Precision Comparison 10000 Text](results/visualization/plots/bar_chart/precision_comparison_10000_txt.png)

##### Precision Comparison (30000 tokens, Text Files)
![Precision Comparison 30000 Text](results/visualization/plots/bar_chart/precision_comparison_30000_txt.png)

#### Recall Comparison Bar Charts

##### Recall Comparison (1000 tokens, Text Files)
![Recall Comparison 1000 Text](results/visualization/plots/bar_chart/recall_comparison_1000_txt.png)

##### Recall Comparison (10000 tokens, Text Files)
![Recall Comparison 10000 Text](results/visualization/plots/bar_chart/recall_comparison_10000_txt.png)

##### Recall Comparison (30000 tokens, Text Files)
![Recall Comparison 30000 Text](results/visualization/plots/bar_chart/recall_comparison_30000_txt.png)

##### Recall Comparison (1000 tokens, File Formats)
![Recall Comparison File Formats 1000](results/visualization/plots/bar_chart/recall_comparison_file_formats_1000.png)

##### Recall Comparison (10000 tokens, File Formats)
![Recall Comparison File Formats 10000](results/visualization/plots/bar_chart/recall_comparison_file_formats_10000.png)

##### Recall Comparison (30000 tokens, File Formats)
![Recall Comparison File Formats 30000](results/visualization/plots/bar_chart/recall_comparison_file_formats_30000.png)


### Line Chart Plots

#### Non-Language Aggregated Line Charts

##### F1 Score vs Noise (XML, Average Language, Average Dataset)
![F1 Score vs Noise XML Avg Lang Avg Dataset](results/visualization/plots/line_chart/non_language_aggregated/f1_score_vs_noise_xml_avg_lang_avg_dataset.png)

##### Precision vs Noise (TXT, Average Language, Average Dataset)
![Precision vs Noise TXT Avg Lang Avg Dataset](results/visualization/plots/line_chart/non_language_aggregated/precision_vs_noise_txt_avg_lang_avg_dataset.png)

##### Recall vs Noise (TXT, Average Language, Average Dataset)
![Recall vs Noise TXT Avg Lang Avg Dataset](results/visualization/plots/line_chart/non_language_aggregated/recall_vs_noise_txt_avg_lang_avg_dataset.png)
# LLM noise levels that were tested
| Model | Extension | Language | Noise Level |
|---|---|---|---|
| deepseek-r1-1.5B-128k | csv | english | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| deepseek-r1-1.5B-128k | csv | french | 1000, 10000, 30000 |
| deepseek-r1-1.5B-128k | csv | german | 1000, 10000, 30000 |
| deepseek-r1-1.5B-128k | html | english | 1000, 10000, 30000 |
| deepseek-r1-1.5B-128k | html | french | 1000, 10000, 30000 |
| deepseek-r1-1.5B-128k | html | german | 1000, 10000, 30000 |
| deepseek-r1-1.5B-128k | json | english | 1000, 10000, 30000 |
| deepseek-r1-1.5B-128k | json | french | 1000, 10000, 30000 |
| deepseek-r1-1.5B-128k | json | german | 1000, 10000, 30000 |
| deepseek-r1-1.5B-128k | txt | english | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| deepseek-r1-1.5B-128k | txt | french | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| deepseek-r1-1.5B-128k | txt | german | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| deepseek-r1-1.5B-128k | xml | english | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| deepseek-r1-1.5B-128k | xml | french | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| deepseek-r1-1.5B-128k | xml | german | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| deepseek-r1-1.5B-128k | yaml | english | 1000, 10000, 30000 |
| deepseek-r1-1.5B-128k | yaml | french | 1000, 10000, 30000 |
| deepseek-r1-1.5B-128k | yaml | german | 1000, 10000, 30000 |
| deepseek-r1-8B-128k | csv | english | 1000, 2000, 5000, 10000, 20000, 30000 |
| deepseek-r1-8B-128k | csv | french | 1000, 10000, 30000 |
| deepseek-r1-8B-128k | csv | german | 1000, 10000, 30000 |
| deepseek-r1-8B-128k | html | english | 1000, 10000, 30000 |
| deepseek-r1-8B-128k | html | french | 1000, 10000, 30000 |
| deepseek-r1-8B-128k | html | german | 1000, 10000, 30000 |
| deepseek-r1-8B-128k | json | english | 1000, 10000, 30000 |
| deepseek-r1-8B-128k | json | french | 1000, 10000, 30000 |
| deepseek-r1-8B-128k | json | german | 1000, 10000, 30000 |
| deepseek-r1-8B-128k | txt | english | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| deepseek-r1-8B-128k | txt | french | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| deepseek-r1-8B-128k | txt | german | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| deepseek-r1-8B-128k | xml | english | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| deepseek-r1-8B-128k | xml | french | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| deepseek-r1-8B-128k | xml | german | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| deepseek-r1-8B-128k | yaml | english | 1000, 10000, 30000 |
| deepseek-r1-8B-128k | yaml | french | 1000, 10000, 30000 |
| deepseek-r1-8B-128k | yaml | german | 1000, 10000, 30000 |
| llama3.1-8B-128k | csv | english | 1000, 10000, 30000 |
| llama3.1-8B-128k | csv | french | 1000, 10000, 30000 |
| llama3.1-8B-128k | csv | german | 1000, 10000, 30000 |
| llama3.1-8B-128k | html | english | 1000, 10000, 30000 |
| llama3.1-8B-128k | html | french | 1000, 10000, 30000 |
| llama3.1-8B-128k | html | german | 1000, 10000, 30000 |
| llama3.1-8B-128k | json | english | 1000, 10000, 30000 |
| llama3.1-8B-128k | json | french | 1000, 10000, 30000 |
| llama3.1-8B-128k | json | german | 1000, 10000, 30000 |
| llama3.1-8B-128k | txt | english | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| llama3.1-8B-128k | txt | french | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| llama3.1-8B-128k | txt | german | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| llama3.1-8B-128k | xml | english | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| llama3.1-8B-128k | xml | french | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| llama3.1-8B-128k | xml | german | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| llama3.1-8B-128k | yaml | english | 1000, 10000, 30000 |
| llama3.1-8B-128k | yaml | french | 1000, 10000, 30000 |
| llama3.1-8B-128k | yaml | german | 1000, 10000, 30000 |
| llama3.2-1B-128k | csv | english | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| llama3.2-1B-128k | csv | french | 1000, 10000, 30000 |
| llama3.2-1B-128k | csv | german | 1000, 10000, 30000 |
| llama3.2-1B-128k | html | english | 1000, 10000, 30000 |
| llama3.2-1B-128k | html | french | 1000, 10000, 30000 |
| llama3.2-1B-128k | html | german | 1000, 10000, 30000 |
| llama3.2-1B-128k | json | english | 1000, 10000, 30000 |
| llama3.2-1B-128k | json | french | 1000, 10000, 30000 |
| llama3.2-1B-128k | json | german | 1000, 10000, 30000 |
| llama3.2-1B-128k | txt | english | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| llama3.2-1B-128k | txt | french | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| llama3.2-1B-128k | txt | german | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| llama3.2-1B-128k | xml | english | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| llama3.2-1B-128k | xml | french | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| llama3.2-1B-128k | xml | german | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| llama3.2-1B-128k | yaml | english | 1000, 10000, 30000 |
| llama3.2-1B-128k | yaml | french | 1000, 10000, 30000 |
| llama3.2-1B-128k | yaml | german | 1000, 10000, 30000 |
| llama3.2-3B-128k | csv | english | 1000, 10000, 30000 |
| llama3.2-3B-128k | csv | french | 1000, 10000, 30000 |
| llama3.2-3B-128k | csv | german | 1000, 10000, 30000 |
| llama3.2-3B-128k | html | english | 1000, 10000, 30000 |
| llama3.2-3B-128k | html | french | 1000, 10000, 30000 |
| llama3.2-3B-128k | html | german | 1000, 10000, 30000 |
| llama3.2-3B-128k | json | english | 1000, 10000, 30000 |
| llama3.2-3B-128k | json | french | 1000, 10000, 30000 |
| llama3.2-3B-128k | json | german | 1000, 10000, 30000 |
| llama3.2-3B-128k | txt | english | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| llama3.2-3B-128k | txt | french | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| llama3.2-3B-128k | txt | german | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| llama3.2-3B-128k | xml | english | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| llama3.2-3B-128k | xml | french | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| llama3.2-3B-128k | xml | german | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| llama3.2-3B-128k | yaml | english | 1000, 10000, 30000 |
| llama3.2-3B-128k | yaml | french | 1000, 10000, 30000 |
| llama3.2-3B-128k | yaml | german | 1000, 10000, 30000 |
| phi3-14B-q4-medium-128k | csv | english | 1000, 10000 |
| phi3-14B-q4-medium-128k | txt | english | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| phi3-14B-q4-medium-128k | txt | french | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| phi3-14B-q4-medium-128k | txt | german | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| phi3-14B-q4-medium-128k | xml | english | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| phi3-14B-q4-medium-128k | xml | french | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| phi3-14B-q4-medium-128k | xml | german | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| qwen2.5-7B-128k | csv | english | 1000, 10000, 30000 |
| qwen2.5-7B-128k | csv | french | 1000, 10000, 30000 |
| qwen2.5-7B-128k | csv | german | 1000, 10000, 30000 |
| qwen2.5-7B-128k | html | english | 1000, 10000, 30000 |
| qwen2.5-7B-128k | html | french | 1000, 10000, 30000 |
| qwen2.5-7B-128k | html | german | 1000, 10000, 30000 |
| qwen2.5-7B-128k | json | english | 1000, 10000, 30000 |
| qwen2.5-7B-128k | json | french | 1000, 10000, 30000 |
| qwen2.5-7B-128k | json | german | 1000, 10000, 30000 |
| qwen2.5-7B-128k | txt | english | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| qwen2.5-7B-128k | txt | french | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| qwen2.5-7B-128k | txt | german | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| qwen2.5-7B-128k | xml | english | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| qwen2.5-7B-128k | xml | french | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| qwen2.5-7B-128k | xml | german | 1000, 2000, 5000, 10000, 20000, 30000, 59000 |
| qwen2.5-7B-128k | yaml | english | 1000, 10000, 30000 |
| qwen2.5-7B-128k | yaml | french | 1000, 10000, 30000 |
| qwen2.5-7B-128k | yaml | german | 1000, 10000, 30000 |