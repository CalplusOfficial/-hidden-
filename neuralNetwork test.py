import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load and clean data
print("Loading and cleaning data...")
df = pd.read_csv("Cardiotocography Data.csv")
df = df.drop_duplicates().dropna()
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
print("Data loaded and cleaned.")

# Features and target
X = df.drop('nsp', axis=1)
y = df['nsp']
print("Features and target variable separated.")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Try different hidden layer sizes
max_layers = 3  # Maximum number of hidden layers to try
layer_size_range = range(1, 31, 1)  # Try sizes from 1 to 30 in steps of 1

overall_best_accuracy = -1
overall_best_config = None
overall_best_clf = None

for num_layers in range(1, max_layers + 1):
    print(f"\nSearching configurations with {num_layers} hidden layer(s)...")
    best_accuracy = -1
    best_config = None
    best_clf = None

    # Generate all possible combinations for this number of layers
    from itertools import product
    for config in product(layer_size_range, repeat=num_layers):
        print(f"  Training MLPClassifier with hidden_layer_sizes={config}...")
        clf = MLPClassifier(hidden_layer_sizes=config, max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"    Accuracy: {acc:.4f}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_config = config
            best_clf = clf

    print(f"Best configuration for {num_layers} layer(s): {best_config} with accuracy {best_accuracy:.4f}")

    # Update overall best if this is better
    if best_accuracy > overall_best_accuracy:
        overall_best_accuracy = best_accuracy
        overall_best_config = best_config
        overall_best_clf = best_clf


print("\nOverall best hidden layer configuration:", overall_best_config)
print("Overall best accuracy:", overall_best_accuracy)
print("Classification Report:\n", classification_report(y_test, overall_best_clf.predict(X_test)))

# Save the best model
joblib.dump(overall_best_clf, "mlp_best_model.joblib")
print("Best model saved to mlp_best_model.joblib.")
best_clf = joblib.load("mlp_best_model.joblib")

# Predict and evaluate on all samples using the best model
print("Testing best model on all samples...")
y_all_pred = best_clf.predict(X)
print("Accuracy (all samples):", accuracy_score(y, y_all_pred))
print("Classification Report (all samples):\n", classification_report(y, y_all_pred))