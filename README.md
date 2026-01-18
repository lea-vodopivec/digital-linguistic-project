# Translation of idioms in large language models (LLMs) and machine translation (MT) tools from English to Slovene

# Digital Linguistic Project

A comprehensive research project analyzing idiom translation quality across Machine Translation (MT) systems and Large Language Models (LLMs). This study evaluates how well different translation technologies preserve the semantic meaning and cultural nuances of idiomatic expressions when translating from English to Slovenian.

## 📋 Project Overview

This project investigates the performance of **4 translation systems** on a corpus of **40 English idioms**, each appearing in **10 different sentence contexts** (400 total sentences, 1600 translations evaluated):

| System Type | Systems |
|-------------|---------|
| **Machine Translation (MT)** | DeepL, Google Translate |
| **Large Language Models (LLM)** | Gemini, ChatGPT |

### Key Research Questions

1. **Meaning Conveyance**: Does the translation preserve the idiom's intended meaning?
2. **Translation Strategy**: What approach does each system use (literal translation, paraphrase, cultural equivalence)?
3. **System Comparison**: How do LLMs compare to traditional MT systems for idiom translation?

## 📊 Key Findings

- **Best performing system**: Gemini (LLM) with **87.2% accuracy**
- **LLMs outperform MT systems**: 77.9% vs 59.8% average accuracy
- **Dominant strategy**: Paraphrasing is the most common approach across all systems
- **Cultural equivalence**: Highest accuracy rates (95-100%) when systems find cultural equivalents

## 🗂️ Project Structure

```
digital_linguistic_project/
├── README.md                          # This file
├── DLP-Research.pdf                   # Research paper/documentation
│
├── corpus_grouping/                   # Corpus preparation
│   ├── corpus_grouping.py             # Script to clean and regroup idiom sentences
│   ├── Corpus_of_idioms.xlsx          # Original idiom corpus
│   ├── Cleaned_Corpus_of_idioms.xlsx  # Processed corpus
│   └── ReGrouped_Corpus_of_idioms.txt # Regrouped sentences for annotation
│
├── corpus_enumeration/                # Sentence enumeration
│   ├── enumerate.py                   # Script to number sentences
│   ├── ReGrouped_Corpus_of_idioms.txt # Input file
│   └── ReGrouped_Corpus_of_idioms_numbered.txt  # Numbered output
│
└── translation-analysis/              # Analysis pipeline
    ├── data_analysis.py               # Main analysis script
    ├── explore_data.py                # Data exploration utilities
    ├── Translations.xlsx.ods          # Translation data with annotations
    ├── analysis_report.md             # Generated analysis report
    └── *.png                          # Generated visualizations
```

## 🔬 Methodology

### Translation Type Classification

| Type | Description | Example |
|------|-------------|---------|
| **Type 1** | Literal translation | Word-for-word translation |
| **Type 2** | Semantic paraphrase | Meaning preserved through rephrasing |
| **Type 3** | Cultural equivalence | Replaced with equivalent target-language idiom |
| **Type 4** | Other | Alternative approaches |

### Evaluation Layers

1. **Layer 1 (Meaning)**: Binary evaluation (Yes/No) of whether the idiom's meaning is conveyed
2. **Layer 2 (Type)**: Classification of translation strategy used
3. **Combined Analysis**: Cross-analysis of meaning accuracy by translation type

## 📈 Visualizations

The analysis produces several visualizations:

| Visualization | Description |
|---------------|-------------|
| Overall Accuracy | Bar chart comparing system accuracy |
| MT vs LLM Comparison | Side-by-side category comparison |
| Type Distribution | Stacked bar chart of translation strategies |
| Per-Idiom Heatmap | Detailed accuracy by idiom and system |
| Combined Analysis | Accuracy rates by translation type |
| Individual Systems | Pie charts for each system |

## 📚 Data Format

### Input Data (`Translations.xlsx.ods`)

The translation data file contains:
- **Idiom**: The English idiom being analyzed
- **Sentence**: Context sentence containing the idiom
- **System columns**: Translation output from each system
- **Meaning evaluation**: Yes/No annotation for meaning preservation

- **Type annotation**: 1-4 classification of translation strategy

