import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # Example model

# Sample training data (replace with actual dataset)
X_train = np.array([
    [80, 40, 40, 5.5, 30],  # Rice
    [80, 40, 20, 5.5, 50],  # Maize
    [40, 60, 80, 5.5, 60],  # Chickpea
    [20, 60, 20, 5.5, 45],  # Kidney Beans
])  
y_train = ["rice", "maize", "chickpea", "kidneybeans"]  # Example labels

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
model_path = "D:/Projects/Harvestify/harvestify/app/model.pkl"
with open(model_path, "wb") as file:
    pickle.dump(model, file)

print(f"✅ Model saved at {model_path}")
