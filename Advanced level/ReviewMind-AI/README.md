# ReviewMind AI — Universal Customer Feedback Intelligence Platform

ReviewMind AI is an interactive machine learning application designed to analyze customer feedback and transform text reviews into meaningful insights.

The project was developed as part of my **ShadowFox Machine Learning Engineer Internship — Advanced Level**.

The application combines traditional machine learning, transformer-based NLP models, topic modeling, and interactive data visualization in a Streamlit interface.

---

## 👨‍💻 Project Information

**Project Name:** ReviewMind AI  
**Intern:** Samarth Jain  
**Organization:** ShadowFox  
**Internship Role:** Machine Learning Engineer Intern  
**Programme:** B.Tech AI-ML (IAC)  
**University:** Sage University, Indore

---

## 🎯 Project Objective

The main objective is to build a reusable customer feedback intelligence platform that can process uploaded review datasets, analyze customer opinions, identify emotions and topics, and present the results through interactive visualizations.

---

## ✨ Key Features

### 1. Dataset Upload and Profiling

- Upload customer feedback datasets in CSV and Excel formats.
- Preview uploaded data.
- Identify text and label columns.
- Perform dataset profiling and quality checks.

### 2. Sentiment Analysis

- Classify customer reviews into positive and negative sentiment.
- Use TF-IDF with Logistic Regression.
- Support transformer-based sentiment analysis using DistilBERT.

### 3. Emotion Detection

- Analyze emotional patterns in customer feedback.
- Use the DistilRoBERTa emotion classification model.
- Present emotion predictions and related insights.

### 4. Topic Modeling

- Discover common themes and topics in customer reviews.
- Use TF-IDF and Non-negative Matrix Factorization (NMF).
- Summarize recurring discussion patterns.

### 5. Ask ReviewMind Assistant

- Provides a local, rule-based question-answering interface.
- Answers supported questions using available dataset information and analysis results.
- Does not depend on a paid external LLM API.

### 6. Interactive Visualizations

- Explore sentiment and emotion distributions.
- View topic analysis.
- Create custom visualizations using the graph builder.

### 7. Export Results

- Export supported analysis results as CSV files.

---

## 🧠 Machine Learning Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas and NumPy | Data processing |
| Scikit-learn | Machine learning and preprocessing |
| TF-IDF | Text feature extraction |
| Logistic Regression | Sentiment classification |
| DistilBERT | Transformer-based sentiment analysis |
| DistilRoBERTa | Emotion classification |
| NMF | Topic modeling |
| PyTorch | Deep learning framework |
| Transformers | Transformer model implementation |
| Streamlit | Interactive application |
| Matplotlib and Seaborn | Data visualization |

---

## 📊 Sentiment Model Evaluation

The project evaluates two sentiment classification approaches.

| Metric | Logistic Regression | DistilBERT |
|---|---:|---:|
| Accuracy | 90.52% | 87.56% |
| Precision | 89.45% | 89.47% |
| Recall | 91.96% | 85.23% |
| F1 Score | 90.69% | 87.30% |
| ROC-AUC | 0.9663 | 0.9501 |

**Important Evaluation Note:** Logistic Regression and DistilBERT were trained using different training-set sizes. Logistic Regression used the full training split, while DistilBERT was trained on a balanced subset of 4,000 reviews for two epochs. Therefore, the results are not a perfectly controlled, like-for-like comparison.

---

## 🏗️ Application Workflow

1. Upload a customer feedback dataset.
2. Preview and profile the dataset.
3. Select or identify the relevant text and label columns.
4. Run the available analyses.
5. Generate sentiment, emotion, and topic insights.
6. Explore the results through visualizations.
7. Ask supported questions using Ask ReviewMind.
8. Export available analysis results.

---

## ⚙️ Technical Notes and Limitations

- The application uses trained models for inference rather than retraining every model for every uploaded dataset.
- DistilBERT was trained using IMDb reviews, so its performance may vary on other domains.
- Emotion classification predictions may sometimes have low confidence or be inaccurate.
- Ask ReviewMind is a local rule-based assistant with supported question types, not an unrestricted conversational LLM.
- Results depend on the quality, structure, and domain of the uploaded dataset.
- Large trained model files may not be included in GitHub because of repository file-size limitations.

---

## 🚀 Live Application

**Streamlit Application:**  
https://shadowfox-ml-internship-reviewmind-ai.streamlit.app/

---

## 📁 Project Structure

```text
ReviewMind-AI/
│
├── data/
├── documentation/
├── models/
├── notebooks/
├── outputs/
├── src/
├── tests/
└── README.md
```

The folders contain the project's datasets, documentation, model resources, notebooks, outputs, source code, and tests.

---

## 🎓 Learning Outcomes

Through this project, I gained practical experience in:

- Natural Language Processing
- Text preprocessing and feature extraction
- Traditional machine learning classification
- Transformer-based NLP models
- Sentiment and emotion analysis
- Topic modeling
- Model evaluation and comparison
- Interactive dashboard development
- Data visualization
- Building an end-to-end machine learning application

---

## 🔗 Project Links

**GitHub Repository:**  
https://github.com/samarthjain-ai/ShadowFox-ML-Internship

**Live Application:**  
https://shadowfox-ml-internship-reviewmind-ai.streamlit.app/

---

**Developed by Samarth Jain**  
*Advanced Level Project — ShadowFox Machine Learning Engineer Internship*