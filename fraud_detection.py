import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Step 1: Load Data
df = pd.read_csv('fraud_data.csv')

# Step 2: Explore Data
print("Columns:", df.columns)
print("\nTarget distribution:\n", df['is_fraud'].value_counts())

# Step 3: Drop Unnecessary Columns
df = df.drop(columns=[
    'Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'first', 'last',
    'street', 'city', 'state', 'zip', 'job', 'dob', 'trans_num'
], errors='ignore')  # ignore errors if any column doesn't exist

# Step 4: Encode Categorical Columns
categorical_cols = ['merchant', 'category', 'gender']
for col in categorical_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col])

# Step 5: Split Data
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Step 6: Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Step 8: Train Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"{name} AUC Score: {auc:.4f}")

# Optional: Confusion Matrix for Random Forest
rf = models["Random Forest"]
conf_matrix = confusion_matrix(y_test, rf.predict(X_test))

sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 9: Predict on a New Transaction

# Define the new transaction
new_transaction = {
    'merchant': 'Amazon',
    'category': 'shopping',
    'amt': 9200.50,
    'gender': 'M',
    'lat': 33.74,
    'long': -84.39,
    'city_pop': 15000,
    'unix_time': 1325376000,
    'merch_lat': 33.74,
    'merch_long': -84.39
}

# Convert to DataFrame
new_df = pd.DataFrame([new_transaction])



# Rebuild the label encoders with the original training data
label_encoders = {}
for col in ['merchant', 'category', 'gender']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    new_df[col] = le.transform(new_df[col])

# --- SCALING FIX ---

# Ensure feature columns match
X_columns = X.columns.tolist()
new_df = new_df[X_columns]  # keep only model features

# Scale the new input
new_scaled = scaler.transform(new_df)

# Predict using the trained Random Forest model
prediction = rf.predict(new_scaled)[0]
probability = rf.predict_proba(new_scaled)[0][1]

# Show result
print("\n--- Fraud Detection Result ---")
print("Predicted Class:", "FRAUD" if prediction == 1 else "LEGITIMATE")
print(f"Probability of Fraud: {probability:.2%}")

