import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Path to the directory where you saved your data
DATA_PATH = "ISL_data"

# Lists to hold the data and labels
data = []
labels = []

# Loop through each sign's folder
for sign_folder in os.listdir(DATA_PATH):
    sign_path = os.path.join(DATA_PATH, sign_folder)
    
    # Check if it's a directory
    if not os.path.isdir(sign_path):
        continue

    # Loop through each data file (.npy) in the sign's folder
    for file_name in os.listdir(sign_path):
        # Load the landmark data from the .npy file
        landmarks = np.load(os.path.join(sign_path, file_name))
        
        # Add the landmark data to our data list
        data.append(landmarks)
        
        # Add the corresponding label (the folder name) to our labels list
        labels.append(sign_folder)

# Convert lists to numpy arrays for scikit-learn
X = np.array(data)
y = np.array(labels)

# Split the data into training and testing sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_predict = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_predict)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model to a file
with open('isl_model.p', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to isl_model.p")