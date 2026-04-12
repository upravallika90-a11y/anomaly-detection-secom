import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# =========================
# STEP 1: LOAD DATA
# =========================
print("Loading dataset...")

data = pd.read_csv("data/secom.data", sep=r"\s+", header=None)
labels = pd.read_csv("data/secom_labels.data", sep=r"\s+", header=None)


# Fix fragmentation
data = data.copy()
# Add target column safely
data = pd.concat([data, labels[0]], axis=1)
data.columns = list(range(data.shape[1] - 1)) + ['target']

print("Shape of dataset:", data.shape)
print(data.head())


# =========================
# STEP 2: HANDLE MISSING VALUES
# =========================
print("\nHandling missing values...")

# Remove columns with >50% missing values
threshold = len(data) * 0.5
data = data.dropna(axis=1, thresh=threshold)

# Fill remaining missing values with mean
data = data.fillna(data.mean())

print("After cleaning shape:", data.shape)


# =========================
# STEP 3: SPLIT FEATURES & LABELS
# =========================
print("\nPreparing data...")

X = data.drop("target", axis=1)
y = data["target"]

# Convert labels: 1 → 0 (normal), -1 → 1 (anomaly)
y = y.apply(lambda x: 0 if x == 1 else 1)

print("Target distribution:\n", y.value_counts())


# =========================
# STEP 4: FEATURE SCALING
# =========================
print("\nScaling features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# =========================
# STEP 5: TRAIN MODEL
# =========================
print("\nTraining Isolation Forest model...")

model = IsolationForest(contamination=0.07, random_state=42)
model.fit(X_scaled)

# Predictions
pred = model.predict(X_scaled)

# Convert: -1 → 1 (anomaly), 1 → 0 (normal)
pred = [1 if x == -1 else 0 for x in pred]

print("Model training completed!")


# =========================
# STEP 6: EVALUATION
# =========================
print("\nClassification Report:\n")

print(classification_report(y, pred))


# =========================
# STEP 7: SAVE MODEL
# =========================
print("\nSaving model...")

import os
os.makedirs("model", exist_ok=True)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")


# =========================
# STEP 8: PCA VISUALIZATION
# =========================
print("\nGenerating PCA visualization...")

os.makedirs("outputs", exist_ok=True)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=pred)
plt.title("Anomaly Detection using PCA (SECOM Dataset)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="0 = Normal, 1 = Anomaly")

plt.savefig("outputs/pca_plot.png")
plt.show()

print("PCA plot saved in outputs folder")

print(X.iloc[0].tolist())