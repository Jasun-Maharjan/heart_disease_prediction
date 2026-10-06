import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
 
 
# data loaded from .csv file and split into x and y
def load_data():
    df = pd.read_csv("heart_disease.csv")
    x = df.drop(columns="target")
    y = df["target"]
    return x, y
 
 
# models to compare
def models():
    return {
        "Logistic regression": LogisticRegression(max_iter=5000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }
 
 
def train_and_compare(test_size=0.2):
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=42)
 
    fitted_models = models()
    rows = []
    confusion = {}
 
    for name, model in fitted_models.items():
        crossValidation_auc = cross_val_score(
            model, x_train, y_train, cv=5, scoring="roc_auc")
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        accuracy = (pred == y_test).mean()
 
        true_positive = int(((pred == 1) & (y_test == 1)).sum())
        false_negative = int(((pred == 0) & (y_test == 1)).sum())
        false_positive = int(((pred == 1) & (y_test == 0)).sum())
        true_negative = int(((pred == 0) & (y_test == 0)).sum())
 
        rows.append({
            "Model": name,
            "CV_AUC": round(crossValidation_auc.mean(), 3),
            "Accuracy": round(accuracy, 3),
            "Recall": round(true_positive / (true_positive + false_negative), 3),
            "Precision": round(true_positive / (true_positive + false_positive), 3),
        })
 
        confusion[name] = pd.DataFrame(
            [[true_negative, false_positive], [false_negative, true_positive]],
            index=["Actually healthy", "Actually sick"],
            columns=["Predicted healthy", "Predicted sick"],
        )
 
    return {
        "models": fitted_models,
        "results": pd.DataFrame(rows).set_index("Model"),
        "confusion": confusion,
        "features": list(x.columns),
        "n_train": len(x_train),
        "n_test": len(x_test),
    }
 
 
if __name__ == "__main__":
    out = train_and_compare()
    print(f"Trained on {out['n_train']} patients, tested on {out['n_test']}\n")
    print(out["results"])