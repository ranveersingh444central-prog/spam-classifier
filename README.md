# SMS Spam Classifier

An end-to-end Machine Learning pipeline and web application for classifying SMS text messages as **SPAM** or **NOT SPAM (HAM)** using Natural Language Processing (NLP), TF-IDF feature extraction, and Linear Support Vector Classification.

---

## 📌 Problem Statement

Short Message Service (SMS) remains a critical communication channel for personal messaging, business alerts, and two-factor authentication. However, consumers face a growing deluge of spam, unsolicited marketing, phishing attacks, and financial scams. The goal of this project is to build an automated, lightweight, and highly accurate text classification pipeline that filters out fraudulent and spam messages while strictly minimizing false positives (legitimate messages misclassified as spam).

---

## 📊 Dataset

- **Source**: [SMS Spam Collection Dataset](data/spam.csv)
- **Total Initial Messages**: 5,572 messages (`v1` = label, `v2` = text)
- **Duplicates Removed**: 403 duplicate rows identified and removed
- **Cleaned Dataset**: 5,169 unique messages
  - **Ham (Legitimate)**: 4,516 messages (87.37%)
  - **Spam**: 653 messages (12.63%)

---

## 🛠️ Approach

1. **Exploratory Data Analysis (EDA)**:
   - Analyzed class distribution, revealing significant class imbalance (~87% ham vs. ~13% spam).
   - Computed character length distributions, observing that spam messages peak heavily around 130–160 characters, while legitimate messages are predominantly shorter (median < 60 characters).
   - Visualized high-frequency vocabulary using dual word clouds for spam and ham classes.

2. **Text Preprocessing (`clean_text`)**:
   - **Lowercasing**: Standardized all characters to lowercase.
   - **Punctuation Removal**: Stripped punctuation and special symbols.
   - **Stopword Removal**: Filtered out standard English stopwords using NLTK.
   - **Stemming**: Applied the Porter Stemmer (`PorterStemmer`) to reduce words to their base linguistic roots (e.g., `winning` $\rightarrow$ `win`, `freebies` $\rightarrow$ `free`).

3. **Stratified Train/Test Split**:
   - Split dataset into 80% training (4,135 samples) and 20% testing (1,034 samples).
   - Used stratified sampling (`stratify=y`, `random_state=42`) to preserve the 87.3% / 12.7% class ratio across both splits.

4. **Feature Extraction**:
   - Extracted n-gram features using `TfidfVectorizer`.
   - **Fitted strictly on the training set** to prevent data leakage, yielding 6,959 unique vocabulary features.

5. **Model Exploration**:
   - Trained and compared three classic NLP text classification algorithms:
     1. **Multinomial Naive Bayes (`MultinomialNB`)**
     2. **Linear Support Vector Machine (`LinearSVC`)**
     3. **Logistic Regression (`LogisticRegression`)**

---

## 📈 Results

Evaluated on the 1,034 test samples (metrics reported for the positive **Spam** class):

| Model | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score (Spam) |
|---|:---:|:---:|:---:|:---:|
| **Linear SVM (`LinearSVC`)** | **98.26%** | **97.48%** | **88.55%** | **0.9280** |
| **Multinomial Naive Bayes** | 96.32% | 98.95% | 71.76% | 0.8319 |
| **Logistic Regression** | 96.23% | 100.00% | 70.23% | 0.8251 |

---

## 🏆 Why Linear SVM Was Chosen

**Linear SVM (`LinearSVC`)** was selected as the final production model for the following reasons:

1. **Superior Overall Accuracy & F1-Score**: Achieved the highest accuracy (**98.26%**) and F1-score (**0.9280**), outperforming both Naive Bayes (0.8319) and Logistic Regression (0.8251).
2. **Substantially Higher Recall**: Correctly caught **88.55%** (116 of 131) of spam messages in the test set. In comparison, Logistic Regression and Naive Bayes missed approximately 28–30% of incoming spam.
3. **Low False Positive Rate**: Maintained a very high precision of **97.48%**, misclassifying only 3 out of 903 legitimate ham messages.
4. **Computational Efficiency**: Extremely fast inference speed and tiny memory footprint (56 KB model weight), making it ideal for low-latency web deployment.

---

## 🚀 How to Run

### 1. Clone or Open the Repository
```bash
git clone <repo-url>
cd spam-classifier
```

### 2. Create and Activate a Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook
Open and execute all cells in `notebook.ipynb` to reproduce data analysis, preprocessing, model training, and artifact serialization:
```bash
jupyter notebook notebook.ipynb
```

### 5. Launch the Streamlit Demo App
Run the interactive web application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test custom SMS messages.

---

## 📁 Folder Structure

```text
spam-classifier/
├── data/
│   └── spam.csv              # Raw SMS Spam Collection dataset
├── model/
│   ├── model.joblib          # Serialized trained LinearSVC model
│   └── tfidf.joblib          # Serialized fitted TfidfVectorizer
├── app.py                    # Streamlit web application
├── notebook.ipynb            # End-to-end data ingestion, EDA, modeling, and evaluation
├── requirements.txt          # Project package dependencies
├── README.md                 # Project documentation
└── .gitignore                # Git ignore configuration
```

---

## ⚠️ Limitations & Future Work

- **Historical UK Data**: The SMS Spam Collection dataset consists primarily of messages collected in the mid-2000s from UK mobile phone users. Contemporary spam (e.g., modern URL phishing, cryptocurrency scams, MFA spoofing) exhibits different vocabulary, phrasing, and obfuscation techniques.
- **English-Only Scope**: The preprocessing and stopword lists are tailored specifically to English. Multilingual SMS or non-English spam requires localized tokenization and corpus support.
- **Potential Model Upgrades**: Future iterations could evaluate modern deep learning architectures and transformer representations (e.g., DistilBERT, RoBERTa, or DeBERTa) and implement real-time REST API endpoints (e.g., using FastAPI) for automated gateway integration.
