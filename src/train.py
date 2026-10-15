import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

models = {
    "Logistic Regression": Pipeline([("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ]),
    "Random Forest": RandomForestClassifier(n_estimators=100,class_weight="balanced",random_state=42,n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.1,eval_metric="logloss",random_state=42,n_jobs=-1)
}

best_model = None
best_score = 0
best_name = ""

for name, model in models.items():

    print("\nTraining:", name)

    grid = GridSearchCV(
        estimator=model,
        param_grid={},
        cv=3,
        scoring="average_precision",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    score = grid.best_score_

    print("Cross-validation PR-AUC:", round(score, 4))

    if score > best_score:
        best_score = score
        best_model = grid.best_estimator_
        best_name = name

# Save the selected model
joblib.dump(best_model, "models/model_pipeline.pkl")

print("\nBest model:", best_name)
print("Best cross-validation PR-AUC:", round(best_score, 4))
print("Model saved successfully!")