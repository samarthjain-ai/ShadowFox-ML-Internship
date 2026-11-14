# ReviewMind AI — Universal Customer Feedback Intelligence Platform

An AI-powered customer feedback analysis platform that transforms unstructured text reviews into actionable insights using Machine Learning and Natural Language Processing (NLP).

ReviewMind AI helps users understand customer opinions, identify recurring feedback patterns, explore emotions, discover topics, and visualize results through an interactive Streamlit application.

---

## 🚀 Project Overview

Businesses receive large amounts of customer feedback through reviews, surveys, and other channels. Manually examining this feedback can be time-consuming and makes it difficult to identify common problems and customer preferences.

**ReviewMind AI** addresses this challenge by combining traditional Machine Learning, Transformer-based NLP models, and interactive data analytics in one platform.

Users can upload compatible feedback datasets, select the relevant text column, perform automated or customized analysis, and explore the results through visualizations and the Ask ReviewMind assistant.

## ✨ Key Features

### 1. Dataset Upload and Profiling
- Upload customer feedback datasets.
- Supports CSV and Excel files.
- Preview uploaded data.
- Inspect dataset dimensions and data quality.
- Identify potential text and label columns.
- Handle missing and blank feedback records.

### 2. Sentiment Analysis
- Classifies feedback into positive and negative sentiment.
- Uses a TF-IDF and Logistic Regression model.
- Also includes a fine-tuned DistilBERT sentiment model.
- Displays sentiment predictions and confidence scores.

### 3. Emotion Detection
- Uses a pretrained DistilRoBERTa emotion classification model.
- Identifies emotional signals in customer feedback.
- Helps explore emotions beyond simple positive and negative sentiment.

### 4. Topic Modeling
- Uses TF-IDF and Non-negative Matrix Factorization (NMF).
- Discovers recurring themes in feedback.
- Helps identify frequently discussed subjects.

### 5. Ask ReviewMind
- Provides a local, rule-based question-answering assistant.
- Answers supported questions using the uploaded dataset and available analysis results.
- Supports questions about sentiment, emotions, topics, ratings, numerical information, EDA, and business insights.
- Avoids relying on an external paid LLM API.

### 6. Data Visualization
- Displays analysis results through interactive charts.
- Helps explore sentiment, emotion, and topic distributions.
- Supports additional dataset exploration through the visualization functionality.

### 7. Export and Analysis Results
- Provides processed analysis results.
- Supports exporting available results for further examination.

## 🧠 Machine Learning and NLP Architecture

ReviewMind AI combines multiple approaches to customer feedback intelligence.

| Component | Technique | Purpose |
|---|---|---|
| Sentiment Model 1 | TF-IDF + Logistic Regression | Positive/negative sentiment classification |
| Sentiment Model 2 | Fine-tuned DistilBERT | Transformer-based sentiment classification |
| Emotion Detection | DistilRoBERTa | Emotion classification |
| Topic Modeling | TF-IDF + NMF | Discover recurring feedback topics |
| Question Answering | Local rule-based engine | Answer supported questions using dataset evidence |

### Processing Workflow

1. User uploads a dataset.
2. The application profiles the dataset.
3. The user selects the feedback text column.
4. The selected analysis is performed.
5. Predictions and discovered topics are generated.
6. Results are visualized.
7. Ask ReviewMind answers supported questions using the available dataset and analysis results.

## 📊 Model Evaluation Results

The sentiment models were evaluated using a held-out test dataset.

### Logistic Regression

| Metric | Result |
|---|---:|
| Accuracy | 90.52% |
| Precision | 89.45% |
| Recall | 91.96% |
| F1-score | 90.69% |
| ROC-AUC | 0.9663 |

Confusion Matrix:

| | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 4400 | 540 |
| Actual Positive | 400 | 4577 |

### DistilBERT

| Metric | Result |
|---|---:|
| Accuracy | 87.56% |
| Precision | 89.47% |
| Recall | 85.23% |
| F1-score | 87.30% |
| ROC-AUC | 0.9501 |

Confusion Matrix:

| | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 4441 | 499 |
| Actual Positive | 735 | 4242 |

### Evaluation Discussion

In this experiment, Logistic Regression achieved stronger overall test performance than DistilBERT.

However, the models were trained under different data conditions:

- Logistic Regression used the full training split.
- DistilBERT was fine-tuned using a balanced subset of 4,000 training reviews.
- DistilBERT was trained for two epochs using CPU-based training.

Therefore, these results should not be interpreted as a universal comparison of the two model architectures.

## 🗂️ Dataset

The primary sentiment-analysis dataset used during development is the IMDb Movie Reviews dataset.

- Total reviews: 50,000
- Positive reviews: 25,000
- Negative reviews: 25,000
- Text column: `review`
- Sentiment column: `sentiment`

Dataset source:

[IMDb Dataset — Zenodo](https://zenodo.org/records/7928582)

The application is designed to accept other compatible feedback datasets, provided they contain suitable text data and any required analysis columns.

## 🛠️ Technology Stack

- **Programming Language:** Python
- **Application Framework:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Deep Learning:** PyTorch
- **NLP:** Hugging Face Transformers
- **Visualization:** Matplotlib, Seaborn, Streamlit charts
- **Model Storage:** Joblib and local model artifacts
- **Development:** Jupyter Notebook and Visual Studio Code
- **Version Control:** Git and GitHub

## 📁 Project Structure

```text
ReviewMind-AI/
│
├── .streamlit/
│   └── config.toml
│
├── app/
│   ├── app.py
│   └── style.css
│
├── backups/
│
├── data/
│   └── raw/
│       └── IMDB Dataset.csv
│
├── models/
│   ├── distilbert_imdb_sentiment/
│   └── tfidf_logistic_regression.pkl
│
├── notebooks/
│   ├── ReviewMind_AI.ipynb
│   └── ReviewMind_Backend_Testing.ipynb
│
├── outputs/
│   └── product_feedback_analysis.csv
│
├── src/
│   ├── analysis_engine.py
│   ├── assistant_engine.py
│   ├── data_loader.py
│   ├── data_profiler.py
│   ├── emotion.py
│   ├── evaluation.py
│   ├── exporter.py
│   ├── preprocessing.py
│   ├── sentiment.py
│   ├── topics.py
│   └── visualization.py
│
├── tests/
│   ├── test_assistant_engine.py
│   ├── test_data_quality.py
│   └── test_visualization_and_export.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/samarthjain-ai/ShadowFox-ML-Internship.git
```

### 2. Navigate to the Project

```bash
cd "ShadowFox-ML-Internship/Advanced level/ReviewMind-AI"
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app/app.py
```

The application should open in your browser at the local Streamlit address.

**Note:** The DistilBERT model weights are large and are not included in the GitHub repository. They must be made available through a configured model-hosting or download process before using that model. Other features should be used according to their installed model requirements.

## ⚠️ Limitations

- Sentiment predictions may be less reliable on datasets whose language or subject matter differs significantly from the IMDb training data.
- The fine-tuned DistilBERT model was trained on a smaller subset because of CPU training constraints.
- Emotion classification can occasionally misinterpret ambiguous or context-dependent text.
- Topic modeling depends on the quality and quantity of the uploaded feedback.
- Ask ReviewMind is a local rule-based assistant, not an unrestricted conversational LLM. It supports defined question categories and depends on available dataset information and analysis results.
- Some visualizations require particular column types, such as dates or numerical ratings.

## 🔐 Ethical Considerations

- Model predictions should be treated as analytical estimates, not verified facts about individual customers.
- Customer feedback may contain sensitive or personally identifiable information. Users should avoid uploading information they are not authorized to process.
- Predictions may reflect biases or limitations in the training dataset.
- Business decisions should not rely exclusively on automated sentiment or emotion predictions.
- Uploaded data should be handled responsibly.

## 🔮 Future Improvements

- Support for additional languages.
- More advanced topic discovery.
- Improved domain adaptation for different industries.
- Expanded conversational capabilities.
- Additional visualization and reporting options.
- Model optimization for efficient cloud deployment.
- More comprehensive testing across different customer feedback datasets.

## 📚 References

1. [IMDb Dataset — Zenodo](https://zenodo.org/records/7928582)
2. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
3. [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
4. [PyTorch Documentation](https://pytorch.org/docs/stable/)
5. [Streamlit Documentation](https://docs.streamlit.io/)
6. [DistilBERT Research Paper](https://arxiv.org/abs/1910.01108)

## 👨‍💻 Project Information

**Project:** ReviewMind AI — Universal Customer Feedback Intelligence Platform

**Internship:** ShadowFox — Machine Learning Engineer Internship

**Repository:** [ShadowFox-ML-Internship](https://github.com/samarthjain-ai/ShadowFox-ML-Internship)

**Application Demo:** To be added after deployment.

---

*Built as part of the ShadowFox Machine Learning Engineer Internship, exploring practical applications of Machine Learning, NLP, and interactive data analytics.*