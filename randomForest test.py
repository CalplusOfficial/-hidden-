import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load and clean data
df = pd.read_csv("Cardiotocography Data.csv")
df = df.drop_duplicates().dropna()
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
print("Data loaded and cleaned.")

# Features and target
X = df.drop('nsp', axis=1)
y = df['nsp']
print("Features and target variable separated.")

# Split data
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Train RandomForest
print("Training RandomForestClassifier...")
clf = RandomForestClassifier(n_estimators=300, random_state=42)
clf.fit(X_train, y_train)
print("Model training complete.")

# Save the model
joblib.dump(clf, "randomforest_model.joblib")
print("Model saved to randomforest_model.joblib.")

# Predict and evaluate
print("Making predictions and evaluating the model...")
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Sample predictions:")

# Predict and evaluate on all samples
print("Testing model on all samples...")
y_all_pred = clf.predict(X)
print("Accuracy (all samples):", accuracy_score(y, y_all_pred))
print("Classification Report (all samples):\n", classification_report(y, y_all_pred))