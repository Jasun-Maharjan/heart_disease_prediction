import pandas as pd
import streamlit as st
import train 
 
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

CHOICES = {
    "sex": ("Sex", {1: "Male", 0: "Female"}),
    "cp": ("Chest pain type", {1: "Typical angina", 2: "Atypical angina",
                               3: "Non-anginal pain", 4: "Asymptomatic"}),
    "fbs": ("Fasting blood sugar > 120 mg/dl", {0: "No", 1: "Yes"}),
    "restecg": ("Resting ECG", {0: "Normal", 1: "ST-T abnormality",
                                2: "Left ventricular hypertrophy"}),
    "exang": ("Exercise-induced angina", {0: "No", 1: "Yes"}),
    "slope": ("ST slope", {1: "Upsloping", 2: "Flat", 3: "Downsloping"}),
}

NUMBERS = {
    "age": ("Age", 20, 100, 55, 1),
    "trestbps": ("Resting blood pressure (mm Hg)", 80, 220, 130, 1),
    "chol": ("Cholesterol (mg/dl)", 100, 600, 240, 1),
    "thalach": ("Max heart rate achieved", 60, 220, 150, 1),
    "oldpeak": ("ST depression (oldpeak)", -3.0, 7.0, 1.0, 0.1),
}
 
def ask(feature):
    if feature in CHOICES:
        label, options = CHOICES[feature]
        return st.selectbox(label, list(options), format_func=lambda v: options[v])
    if feature in NUMBERS:
        label, low, high, default, step = NUMBERS[feature]
        return st.number_input(label, low, high, default, step=step)
    return st.number_input(feature, value=0.0)   # any feature we don't know about
 
 
st.title("❤️ Heart Disease Risk Predictor")
st.caption("Educational project using the UCI Heart Disease dataset. "
           "Not a medical device and not a substitute for a doctor.")
 
st.sidebar.header("Training settings")
test_size = st.sidebar.slider("Share of data used for testing", 0.10, 0.40, 0.20, step=0.05)
st.sidebar.caption("Changing this retrains every model on a new split.")
 

@st.cache_resource(show_spinner="Training and testing the models...", max_entries=1)
def run_backend(test_size):
    return train.train_and_compare(test_size=test_size)
 
 
backend = run_backend(0.2)
models = backend["models"]
results = backend["results"]
confusion = backend["confusion"]
features = backend["features"]
 
st.sidebar.write(f"Trained on **{backend['n_train']}** patients")
st.sidebar.write(f"Tested on **{backend['n_test']}** patients")
st.sidebar.write(f"Models: **{len(models)}**")
 
# Sick patients each model missed 
missed = pd.Series({name: int(matrix.iloc[1, 0]) for name, matrix in confusion.items()})
 
predict_tab, compare_tab, detail_tab = st.tabs(["Predict", "Model comparison", "Details"])
 
# ======================= TAB 1: PREDICT =======================
with predict_tab:
    st.subheader("Enter patient details")
 
    values = {}
    form_cols = st.columns(3)
    for i, feature in enumerate(features):
        with form_cols[i % 3]:
            values[feature] = ask(feature)
 
    if st.button("Predict", type="primary"):
        # One row, columns in the same order the models were trained on
        patient = pd.DataFrame([values])[features]
 
        st.subheader("Results")
        result_cols = st.columns(len(models))
        probabilities = []
 
        for box, (name, model) in zip(result_cols, models.items()):
            prob = float(model.predict_proba(patient)[0, 1])
            probabilities.append(prob)
            with box:
                st.metric(name, f"{prob:.0%} risk")
                st.progress(prob)
                if prob >= 0.5:
                    st.error("Higher risk")
                else:
                    st.success("Lower risk")
 
        votes = sum(p >= 0.5 for p in probabilities)
        st.info(f"{votes} of {len(models)} models classify this patient as higher risk. "
                f"Average risk: {sum(probabilities) / len(probabilities):.0%}.")
 
# ======================= TAB 2: COMPARISON =======================
with compare_tab:
    st.subheader("How the models performed on the test patients")
 
    table = results.copy()
    table["Missed patients"] = missed
    st.dataframe(table, width="stretch")
 
    score_cols = list(results.columns)
    st.bar_chart(results[score_cols])
 
    st.subheader("Winner for each measure")
    winner_cols = st.columns(len(score_cols) + 1)
    for box, metric in zip(winner_cols, score_cols):
        box.metric(f"Best {metric}", results[metric].idxmax())
    winner_cols[-1].metric("Fewest missed", missed.idxmin())
 
    st.markdown(
        "- **CV_AUC**: how well the model ranks sick patients above healthy ones "
        "(1.0 = perfect, 0.5 = coin flip), averaged over 5 rounds of cross-validation.\n"
        "- **Accuracy**: share of all predictions that were correct.\n"
        "- **Recall**: share of truly sick patients the model caught.\n"
        "- **Precision**: share of patients called sick who really were.\n"
        "- **Missed patients**: sick patients the model called healthy."
    )
    st.caption("Scores that are close together are mostly noise. With a small test set, "
               "one or two patients can change the ranking.")
 
# ======================= TAB 3: DETAILS =======================
with detail_tab:
    st.subheader("Confusion matrices")
    st.caption("Counts of test patients. The bottom-left box is the dangerous one: "
               "sick patients the model missed.")
    matrix_cols = st.columns(len(confusion))
    for box, (name, matrix) in zip(matrix_cols, confusion.items()):
        with box:
            st.markdown(f"**{name}**")
            st.dataframe(matrix, width="stretch")
 
    # Feature importance for any model that provides it (Random Forest, XGBoost, ...)
    importance = {name: model.feature_importances_
                  for name, model in models.items()
                  if hasattr(model, "feature_importances_")}
    if importance:
        st.subheader("Which features mattered most?")
        st.bar_chart(pd.DataFrame(importance, index=features))
        st.caption("A higher bar means the model relied on that measurement more.")