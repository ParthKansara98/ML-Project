import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from preprocessing import load_and_clean_data, split_and_scale_data
import os

def train_and_save_model(data_path, target_column, model_path):
    df = load_and_clean_data(data_path)
    X_train, X_test, y_train, y_test = split_and_scale_data(df, target_column)

    clf = RandomForestClassifier(random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:\n", report)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model("./data/diabetes.csv", "Outcome", "./models/diabetes_model.pkl")
