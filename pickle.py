import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ===============================
# Paths
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# ===============================
# Sample Dataset (replace with CSV if needed)
# ===============================
data = pd.DataFrame({
    "Year_Built": [2010, 2012, 2015, 2018, 2020],
    "Area_sqft": [900, 1100, 1200, 1500, 1800],
    "Bedrooms": [2, 2, 3, 3, 4],
    "Price_in_Lakhs": [40, 50, 65, 80, 95]
})

X = data.drop("Price_in_Lakhs", axis=1)
y = data["Price_in_Lakhs"]

# ===============================
# Train Model
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

# ===============================
# Save Pickle Files
# ===============================
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("✅ Pickle files created successfully")
print("Saved at:", MODEL_DIR)
