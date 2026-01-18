# Translation Data Analysis Report

*Generated on: 2026-01-18 18:23*

## Executive Summary

- **Total idioms analyzed**: 40
- **Sentences per idiom**: 10
- **Total translations evaluated**: 1600
- **Systems analyzed**: DeepL, Google Translate, Gemini, ChatGPT
  - Machine Translation (MT): DeepL, Google Translate
  - Large Language Models (LLM): Gemini, ChatGPT

---
## Layer 1: Meaning Conveyance (Yes/No)

*Does the translation convey the meaning of the original idiom?*

### Overall Accuracy by System

| System | Category | Accuracy (%) | Yes Count | No Count |
|--------|----------|--------------|-----------|----------|
| DeepL | MT | 67.1% | 267 | 131 |
| Google Translate | MT | 52.5% | 210 | 190 |
| Gemini | LLM | 87.2% | 349 | 51 |
| ChatGPT | LLM | 68.5% | 274 | 126 |

### MT vs LLM Comparison

| Category | Accuracy (%) |
|----------|--------------|
| MT | 59.8% |
| LLM | 77.9% |

### System Ranking (by accuracy)

1. **Gemini** (LLM): 87.2%
2. **ChatGPT** (LLM): 68.5%
3. **DeepL** (MT): 67.1%
4. **Google Translate** (MT): 52.5%

---
## Layer 2: Translation Type Distribution

*Translation strategy classification:*
- **Type 1**: Literal translation
- **Type 2**: Semantically accurate (paraphrasing)
- **Type 3**: Cultural equivalence
- **Type 4**: Other

### Type Distribution by System (%)

| System | Type 1 | Type 2 | Type 3 | Type 4 |
|--------|--------|--------|--------|--------|
| DeepL | 32.6% | 46.0% | 15.2% | 6.3% |
| Google Translate | 37.8% | 39.8% | 11.3% | 11.0% |
| Gemini | 12.5% | 59.9% | 23.3% | 4.3% |
| ChatGPT | 27.6% | 49.9% | 13.8% | 8.8% |

### Type Counts by System

| System | Type 1 | Type 2 | Type 3 | Type 4 | Total |
|--------|--------|--------|--------|--------|--------|
| DeepL | 129 | 182 | 60 | 25 | 396 |
| Google Translate | 151 | 159 | 45 | 44 | 399 |
| Gemini | 50 | 239 | 93 | 17 | 399 |
| ChatGPT | 110 | 199 | 55 | 35 | 399 |

---
## Combined Analysis: Meaning × Type

*Success rate of meaning conveyance by translation type*

### Accuracy by Type per System (%)

| System | Type 1 | Type 2 | Type 3 | Type 4 |
|--------|--------|--------|--------|--------|
| DeepL | 25.6% | 95.1% | 95.0% | 8.3% |
| Google Translate | 24.5% | 78.0% | 97.8% | 11.4% |
| Gemini | 48.0% | 96.7% | 100.0% | 0.0% |
| ChatGPT | 40.0% | 88.4% | 94.5% | 5.7% |

---
## Individual System Analysis

### DeepL (MT)

- **Overall accuracy**: 67.1%
- **Best performing idioms**:
  - a new lease of life: 100%
  - do the trick: 100%
  - drive a hard bargain: 100%
- **Worst performing idioms**:
  - drink like a fish: 0%
  - keep [pron] head above water: 0%
  - out of this world: 0%
- **Dominant translation strategy**: Type 2 (2-Paraphrase) - 46.0%

### Google Translate (MT)

- **Overall accuracy**: 52.5%
- **Best performing idioms**:
  - a new lease of life: 100%
  - do [pron] dirty work: 100%
  - fall on deaf ears: 100%
- **Worst performing idioms**:
  - dark horse: 0%
  - draw in [pron] horns: 0%
  - draw the shortest straw: 0%
- **Dominant translation strategy**: Type 2 (2-Paraphrase) - 39.8%

### Gemini (LLM)

- **Overall accuracy**: 87.2%
- **Best performing idioms**:
  - a new lease of life: 100%
  - bear fruit: 100%
  - dead loss: 100%
- **Worst performing idioms**:
  - play safe: 10%
  - keep [pron] head above water: 20%
  - once in a blue moon: 30%
- **Dominant translation strategy**: Type 2 (2-Paraphrase) - 59.9%

### ChatGPT (LLM)

- **Overall accuracy**: 68.5%
- **Best performing idioms**:
  - bear fruit: 100%
  - do [pron] dirty work: 100%
  - keep [pron] head above water: 100%
- **Worst performing idioms**:
  - draw fire: 20%
  - not bat an eyelid: 20%
  - play safe: 20%
- **Dominant translation strategy**: Type 2 (2-Paraphrase) - 49.9%

---
## Key Findings

1. **Best performing system**: Gemini with 87.2% accuracy
2. **Lowest performing system**: Google Translate with 52.5% accuracy
3. **LLMs outperform MT systems** on average (77.9% vs 59.8%)
4. **Gemini** (LLM) primarily uses paraphrasing (Type 2: 59.9%)

---
## Visualizations

The following visualization files have been generated:

- `overall_accuracy.png` - Bar chart of overall accuracy per system
- `mt_vs_llm_comparison.png` - Side-by-side MT vs LLM comparison
- `type_distribution.png` - Stacked bar chart of translation types
- `per_idiom_heatmap.png` - Heatmap of accuracy by idiom and system
- `combined_analysis.png` - Accuracy by translation type
- `individual_systems.png` - Pie charts for each system
- `type_distribution_pies.png` - Type distribution pie charts