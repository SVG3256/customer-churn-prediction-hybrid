***

# Customer Complaint Triage and Escalation Modeling: A Dual-Signal Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NLP: VADER](https://img.shields.io/badge/NLP-VADER%20%7C%20LDA-green.svg)](https://github.com/cjhutto/vaderSentiment)

## Project Overview
This project focuses on analyzing customer churn and presents an **Intelligent Triage Framework** for processing anonymized consumer complaints from the CFPB database (Feb 2024 – Feb 2026). Due to the absence of unique customer identifiers, this system utilizes a hybrid architecture to predict resolution complexity and discover latent escalation pathways.

The core idea is the **Dual-Signal Framework**:
* **NLP Signals:** A static classifier that evaluates immediate content risk (Topic, Sentiment, and Keywords).
* **HMM Signals:** A behavioral engine that identifies "procedural loops" and latent frustration states through synthetic sequence simulation.

---

## Technical Architecture

### 1. NLP & Feature Engineering
* **Sentiment Analysis:** Multi-dimensional polarity scoring using **VADER**.
* **Topic Modeling:** **LDA (Latent Dirichlet Allocation)** with 50 topics, optimized via perplexity and bigram validation.
* **Feature Set:** Engineered features including urgency scores, loyalty markers, and churn intent heuristics.

### 2. Supervised Classification (Static Engine)
* **Target:** 3-Tier Resolution Hierarchy (Tier 0: Explanation, Tier 1: Service Recovery, Tier 2: Monetary Relief).
* **Validation:** Strict **Temporal Out-of-Time (OOT)** split to ensure robustness against concept drift.
* **Optimization:** Focused on baseline XGBoost parameters to prioritize generalization over in-sample grid-search overfitting.

### 3. Hidden Markov Model (Behavioral Engine)
* **Sequence Generation:** Synthetic linking based on exact company/ZIP/topic matches within 7–60 day windows.
* **Model:** 8-state **CategoricalHMM** trained via Expectation-Maximization (EM).
* **State Analysis:** Identifies "Escalation Gateways" (States 2, 7) and a "Deterministic Loop" (States 5 ↔ 7).

---

## Repository Structure

| File | Description |
| :--- | :--- |
| `1_customer_churn_preprocessing.ipynb` | Data cleaning and initial EDA of 1.4M complaints. |
| `2_customer_churn_text_processing.ipynb` | Tokenization, lemmatization, and stopword removal. |
| `3_complaints_nlp.ipynb` | Comparing NLP approaches and validation of VADER vs. Transformer-based sentiment tools. |
| `4a/b_perplexity.ipynb` | Empirical determination of optimal LDA topic count ($K=50$). |
| `5_LDA_Topic_Modeling.ipynb` | Full-scale topic extraction and bigram analysis. |
| `6_Multi_Class_Evaluation_and_Target_Creation.ipynb` | Consolidation of 5 classes into the 3-tier complexity target. |
| `7_Supervised_Classification.ipynb` | Training and OOT validation of Logistic Regression, Random Forest and XGBoost (static engine). |
| `8a/b_Hyperparameter_Tuning.ipynb` | Hyperparamter tuning of XGBoost using grid search. |
| `9a/b_Synthetic_Link_Creation.ipynb` | Linking sequences in dataset to simulate customer behavior. |
| `10_HMM_Training_Categorical.ipynb` | Simulation of customer journeys and HMM state discovery. |
| `11_Final_Integration.ipynb` | Final Dual-Signal analysis and Risk vs. Resolution cross-tabulation. |

---

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shubhamghode/customer-churn-prediction-hybrid.git
   cd customer-churn-prediction-hybrid
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn xgboost hmmlearn vaderSentiment nltk tqdm
   ```

3. **Execution Order:**
   Run the notebooks in numerical order (`1_` through `11_`) to rebuild the pipeline from raw data to final integration.

---

## Dataset
Raw data can be obtained from [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/#get-the-data)

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.