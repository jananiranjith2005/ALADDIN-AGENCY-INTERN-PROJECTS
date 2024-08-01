import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
dataset_path = ('/content/colors.csv')
data = pd.read_csv(dataset_path)

# Print the first few rows of the dataset
print(data.head())

# Preprocess the data
X = data[['R', 'G', 'B']]  # Features: RGB values
y = data['Color Name']  # Labels: Color names

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a k-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Function to predict color name based on RGB values
def predict_color_name(rgb_value):
    return knn.predict([rgb_value])[0]

# Test the model with a sample RGB value
sample_rgb = [60, 120, 200]
predicted_color = predict_color_name(sample_rgb)
print(f"Predicted Color for RGB {sample_rgb}: {predicted_color}")

# Visualize the predicted color
plt.figure(figsize=(2, 2))
plt.imshow([[sample_rgb]])
plt.title(f'Predicted Color: {predicted_color}')
plt.axis('off')
plt.show()
