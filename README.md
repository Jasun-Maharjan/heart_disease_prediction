
# ❤️ Heart Disease Risk Predictor

https://heartdiseaseprediction00.streamlit.app/
 
A machine learning project that predicts whether a patient is likely to have heart disease from 11 clinical measurements. Two models are trained, tested, and compared in the backend, and the results are shown in an interactive [Streamlit](https://streamlit.io) web app.
 
> **Disclaimer:** This is an educational project. It is not a medical device and must not be used to diagnose or treat any condition. Always consult a qualified doctor.
 
## Features
 
- **Predict:** enter a patient's details and see each model's estimated risk side by side, with a vote summary.
- **Model comparison:** results table, bar chart, and the winning model for each measure.
- **Details:** confusion matrix for each model and a feature-importance chart.
- **Adjustable test size:** a sidebar slider retrains every model on a new train/test split.
## Dataset
 
The data comes from the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease), which combines patient records from four sources: Cleveland, Hungary, Switzerland, and the VA Long Beach. The combined file has **920 patients**.
 
### Features used
 
| Column | Description |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (1-4) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = yes, 0 = no) |
| `restecg` | Resting ECG result (0-2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment (1-3) |
| `target` | **What we predict:** 0 = no heart disease, 1 = heart disease |
 
### Data preparation
 
The raw data was cleaned before training:
 
- The original `target` (severity 0-4) was converted to yes/no: 0 stays 0, and 1-4 become 1.
- Cholesterol and blood pressure values of `0` are impossible, so they were treated as missing.
- The columns `ca` and `thal` were dropped because most of their values are missing (66% and 53%).
- The `source` column was dropped because it only records which hospital file a row came from.
- Remaining blank cells were filled (median for numeric columns, most common value for category columns).
## Models
 
| Model | Library |
|---|---|
| Logistic Regression | scikit-learn |
| Random Forest | scikit-learn |
| XGBoost | xgboost |
 
Each model is trained on 80% of the data (by default) and tested on the remaining 20%. The split is stratified, so the healthy/disease ratio is the same in both sets.
 
### How models are compared
 
| Measure | Meaning |
|---|---|
| **CV_AUC** | How well the model ranks sick patients above healthy ones (1.0 = perfect, 0.5 = coin flip), averaged over 5-fold cross-validation on the training data |
| **Accuracy** | Share of all test predictions that were correct |
| **Recall** | Share of truly sick patients the model caught (the most important measure in a medical setting) |
| **Precision** | Share of patients called sick who really were |
| **Missed patients** | Sick patients the model called healthy |
 
## Results
 
Results on the default 80/20 split (your numbers may differ slightly depending on how the data was cleaned):
 
| Model | CV_AUC | Accuracy | Recall | Precision |
|---|---|---|---|---|
| Logistic Regression | 0.867 | 0.793 | 0.843 | 0.796 |
| Random Forest | 0.857 | 0.788 | 0.843 | 0.789 |
| XGBoost | 0.854 | 0.815 | 0.863 | 0.815 |
 
The two models perform very similarly. With a test set of only 184 patients, a difference of one or two patients is mostly noise, so neither model should be called clearly better.
 
## Project structure
 
```
heart_disease/
├── app.py               # Streamlit frontend
├── train.py             # Backend: loads data, trains and tests the models
├── heart_disease.csv    # Cleaned dataset
├── requirements.txt     # Python dependencies
└── README.md
```
 
- `train.py` does all the machine learning and displays nothing. Running it directly prints the results table in the terminal.
- `app.py` calls `train.train_and_compare()` once (the result is cached) and draws everything with Streamlit.
## Getting started
 
### 1. Clone the repository
 
```
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```
 
### 2. (Optional) Create a virtual environment
 
```
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```
 
### 3. Install the dependencies
 
```
pip install -r requirements.txt
```
 
### 4. Run the app
 
```
streamlit run app.py
```
 
The app opens in your browser, usually at `http://localhost:8501`. Models are trained when the app starts, which takes a few seconds.
 
To see only the results table without the web app:
 
```
python train.py
```

The app adapts automatically to however many models are returned.
 
## Limitations
 
- The dataset is small (920 patients) and combines data from different hospitals and decades, so the models may not generalize to other populations.
- Two features (`ca` and `thal`) were dropped because of heavy missing data, which may cost some predictive power.
- The models predict from a handful of measurements and cannot replace a clinical examination.
## Acknowledgements
 
Dataset: Janosi, A., Steinbrunn, W., Pfisterer, M., and Detrano, R. *Heart Disease*. UCI Machine Learning Repository.
 
## License
 
Add a license of your choice (for example MIT) in a `LICENSE` file.
