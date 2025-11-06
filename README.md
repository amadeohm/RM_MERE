# 🧠 AI-Driven Data Analysis Benchmark (Gemini, ChatGPT, DeepSeek vs. Human)

This repository contains the code, data, and prompts used in the comparative study evaluating **AI-generated data analyses** versus **human-led analyses** across *tabular* and *text* datasets.  
The project investigates the **quality, methodological bias, and correctness** of analyses produced by three large language models (Gemini, ChatGPT, and DeepSeek) compared to a human benchmark.

---

## 📁 Repository Structure

```
.
├── Codes
│   ├── Tabular
│   │   ├── 01_introducing_missing_values.ipynb     # Notebook introducing missing values into the dataset
│   │   ├── 02_human_analysis.ipynb                 # Human analyst’s step-by-step tabular data analysis
│   │   ├── DeepSeek/                               # DeepSeek-generated scripts (5 independent runs)
│   │   ├── Gemini/                                 # Gemini-generated scripts (5 independent runs)
│   │   └── GPT/                                    # ChatGPT-generated scripts (5 independent runs)
│   └── Text
│       ├── Human_Analysis.ipynb                    # Human analyst’s sentiment analysis workflow
│       ├── DeepSeek/                               # DeepSeek-generated text analysis scripts
│       ├── Gemini/                                 # Gemini-generated text analysis scripts
│       └── GPT/                                    # ChatGPT-generated text analysis scripts
├── Data
│   ├── diabetes_dataset.csv                        # Clean dataset used for tabular experiments
│   ├── diabetes_dataset_nan.csv                    # Version with missing values for imputation tests
│   └── SemEval2017-task4-dev.subtask-A.english.INPUT.csv  # Dataset for sentiment analysis (text experiments)
└── Prompts
    ├── Prompt_TabularData.txt                      # Prompt template used for the tabular data experiment
    └── Prompt_TextData.txt                         # Prompt template used for the text-based experiment
```

---

## 🎯 Objectives

1. **Quality of Outcomes (RQ2)**  
   Evaluate the clarity, relevance, completeness, and correctness of AI-generated analyses.  
   → Quantitative and qualitative scores were assigned based on reproducibility and methodological soundness.

2. **Methodological Bias (RQ3)**  
   Examine how different AI tools approach critical analytic decisions (e.g., imputation strategy, variable encoding, visualization design).

3. **Cross-Domain Generalization (RQ1 & RQ4)**  
   Compare model performance between *tabular* and *text* datasets to assess robustness and domain transfer.

---

## 🧩 How to Reproduce

### 1. Requirements
Make sure you have the following installed:
```bash
python >= 3.10
jupyter
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### 2. Run the Human Baselines
```bash
jupyter notebook Codes/Tabular/02_human_analysis.ipynb
jupyter notebook Codes/Text/Human_Analysis.ipynb
```

### 3. Run AI-Generated Analyses
Each AI model has 3–5 independent runs per domain:
```bash
python Codes/Tabular/Gemini/Gemini1.py
python Codes/Text/GPT/ChatGPT3.py
```

---

## 📊 Datasets

| Dataset | Type | Description |
|----------|------|-------------|
| `diabetes_dataset.csv` | Tabular | Medical dataset with health indicators (e.g., BMI, glucose) used for predictive modeling. |
| `diabetes_dataset_nan.csv` | Tabular | Version with synthetic missing values for imputation evaluation. |
| `SemEval2017-task4-dev.subtask-A.english.INPUT.csv` | Text | Dataset for sentiment classification (positive/neutral/negative). |

---

## 🧠 Models Compared

| Model | Provider | Runs | Domains Tested |
|--------|-----------|-------|----------------|
| **Gemini** | Google DeepMind | 5 (Tabular), 3 (Text) | Tabular + Text |
| **ChatGPT** | OpenAI | 5 (Tabular), 3 (Text) | Tabular + Text |
| **DeepSeek** | DeepSeek AI | 5 (Tabular), 3 (Text) | Tabular + Text |
| **Human Analyst** | Baseline | 1 | Tabular + Text |

---

## 🧾 Evaluation Criteria

Each generated analysis was evaluated on:
- **Relevance** — Are plots and metrics appropriate for the task?  
- **Clarity** — Is the reasoning clear and interpretable?  
- **Completeness** — Are all necessary analytical steps covered?  
- **Correctness** — Is the code runnable and logically sound?

---

## 🖼️ Results Overview

Key findings (see Figures and Tables in the report):
- Gemini achieved the highest overall clarity and relevance.
- ChatGPT produced complete pipelines but suffered from frequent code errors.
- DeepSeek was generally runnable but conceptually inconsistent.
- All AI tools failed to detect hidden data issues (e.g., invalid BMI values = 0), defaulting to overly simple imputation strategies.

---

## 📚 Citation

If you use this repository in your research, please cite it as:

```text
Huerta Moncho, A. (2025). AI-Driven Data Analysis Benchmark: 
Comparing Gemini, ChatGPT, and DeepSeek to Human Analysts.
```

---

## 🧑‍💻 Author


**Amadeo Huerta Moncho**  
MSc Intelligent Interactive Systems — Universitat Pompeu Fabra  
 • [GitHub](https://github.com/amadeohuerta)

**Sandra Jiménez Vargas**
MSc Intelligent Interactive Systems — Universitat Pompeu Fabra  
 • [GitHub](https://github.com/sandrajivar)

**Jone Rivas Azpiazu**
MSc Intelligent Interactive Systems — Universitat Pompeu Fabra  
 • [GitHub](https://github.com/sta05)


