## ReviewMind AI

ReviewMind AI is a local Streamlit customer-feedback intelligence platform.
It preserves the existing TF-IDF + Logistic Regression, DistilBERT, emotion,
topic-modelling, and evaluation pipeline while providing a compact workspace
for data validation, analysis, grounded questions, visualizations, and exports.

### Run the application

From the `ReviewMind-AI` directory:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

Upload a CSV or Excel workbook from the sidebar, or choose the small demo
dataset. Original uploaded data is kept separately from filtered prediction
results. Ask ReviewMind answers only supported questions using current data
and completed analyses; it does not execute arbitrary code or use paid APIs.