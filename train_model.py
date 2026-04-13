import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

print("========================================")
print("--- Phase 2: Neural Network Training ---")
print("========================================")

# 1. Load the Data
CSV_FILE = 'dataset.csv'
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Dataset {CSV_FILE} not found. Run Phase 1 first.")

print("Loading dataset...")
df = pd.read_csv(CSV_FILE)

# 2. Extract Features (X) and Labels (y)
X = df.drop('label', axis=1).values
y = df['label'].values
y = y.astype(int)

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data Split Success: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.\n")

# 4. Build and Train a Random Forest Classifier
# Random Forests perform exceptionally well on tabular landmark data 
# and handle dataset imbalance much better than simple Neural Networks.
print("Building and training the Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=150,     # Number of decision trees
    max_depth=None,       # Allow trees to grow completely
    random_state=42,
    n_jobs=-1             # Use all CPU cores for faster training
)
  
model.fit(X_train, y_train)

# 5. Evaluate Performance
print("\nEvaluating model on the held-out test data...")
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"✅ Final Evaluation! Test Accuracy on unseen data: {test_acc*100:.2f}%")

# 6. Save the compiled model
model_path = 'model.pkl'
joblib.dump(model, model_path)
print(f"\nModel successfully saved to '{model_path}'!")
print("Phase 2 Complete! We are fully ready to build the real-time Web App Phase 3!")
