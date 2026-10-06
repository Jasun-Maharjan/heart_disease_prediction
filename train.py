import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

df=pd.read_csv("heart_disease.csv")
df["target"] = (df["target"] > 0).astype(int)

x = df.drop("target")
y = df["target"]
x_train, x_test, y_train, y_test = train_test_split(x, y, 
    test_size=0.2, stratify=y, random_state=42)

model = {
    "Logistic Regression" : LogisticRegression(max_iter=1000)
    }
crossValidation_auc = cross_val_score(model, x_train, y_train, cv=5, scoring="roc_auc")
model.fit(x_train,y_train)
pred = model.predict(x_test)


