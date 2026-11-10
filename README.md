# Comparative Analysis of Machine Learning Models Using Accuracy and Error Metrics

A ready-to-run B.Tech CSE micro-project using **Python, Scikit-learn, Streamlit, Pandas, and Plotly**.

## Features
- Choose Iris, Wine, Breast Cancer or Diabetes example datasets.
- Or upload your own **CSV**, choose its target column, and select classification or regression.
- Compare **six algorithms** on the same train/test split.
- Preprocess missing values, numeric columns and categorical columns inside training pipelines to prevent test leakage.
- Classification: Accuracy, Error Rate, weighted Precision, Recall, F1; confusion matrix.
- Regression: MAE, MSE, RMSE and R²; actual-vs-predicted plot.
- Ranked comparison table, charts, model selection and downloadable CSV/printable HTML report.

## Run in VS Code
1. Open the `ml_model_comparison_project` folder in VS Code.
2. Open Terminal → New Terminal.
3. Run:

   **macOS / Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   streamlit run app.py
   ```

   **Windows PowerShell**
   ```powershell
   ṣ
   python -m pip install -r requirements.txt
   streamlit run app.py
   ```

4. Open the local address shown in the terminal, usually http://localhost:8501.
5. Select Iris → click **Train & compare models**.

## Test the engine
```bash
python -m unittest discover -s tests -v
```

## Academic abstract
This project systematically compares machine learning algorithms on a common evaluation dataset. It implements preprocessing pipelines, consistent training and testing partitions, and performance analysis for classification and regression tasks. Classification algorithms are evaluated using accuracy, error rate, precision, recall and F1-score, while regression algorithms are measured using MAE, MSE, RMSE and R². An interactive dashboard displays comparison charts, confusion matrices or prediction plots, and identifies the best-performing algorithm for the selected dataset and metric. The system provides downloadable results to support reproducible experiments and informed model selection.

## Methodology and limitations
- Models are ranked by **test accuracy** for classification and **test RMSE** for regression. The top-ranked algorithm on one split is not universally best.
- This is a **single holdout split**, not k-fold cross-validation. Do not claim cross-validated or production-level generalization results.
- Classification precision, recall, and F1 are **support-weighted**, suitable for multiclass as well as binary labels. Accuracy alone may be misleading on imbalanced data.
- CSV: at least 20 complete-target rows, two target classes each with two or more examples for classification, at least one feature column; high-cardinality categories can be slow.
- Data is processed in the locally running Python process. Avoid uploading sensitive personal records to third-party hosted copies.

## Viva questions
**Why use multiple models?** No algorithm performs best on every dataset; evaluating several helps determine the best fit for this experiment.

**What is accuracy?** (Correct predictions) / (Total predictions).

**What is error rate?** 1 − Accuracy.

**Why train/test split?** To evaluate a model on data it did not see during fitting.

**Why use a pipeline?** To fit imputation, scaling, and encoding only on training data, reducing data leakage.

**What is RMSE?** Square root of mean squared prediction error, in the same units as the regression target.

**Why can F1 be important?** It balances precision and recall, especially when classes are imbalanced.

**Future scope:** k-fold cross-validation, hyperparameter tuning, significance tests, and model explainability.
