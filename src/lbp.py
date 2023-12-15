import pandas as pd
import cv2
from skimage.feature import local_binary_pattern
import numpy as np
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage.feature import hog

# Uses LBP

# Feature extraction function using Local Binary Patterns (LBP)
def extract_lbp_features(images):
    features = []
    for img in images:
        img = cv2.imread(img)
        img = cv2.resize(img, (256, 256))
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        features.append(hist)
    return features

# Load the dataset and preprocess the data
data = pd.read_csv("../products.csv")
data = data.dropna()
data["price"] = (data["price"] - data["price"].mean()) / data["price"].std()

# Extract features using LBP from the images
X = extract_lbp_features(data["image"])
y = data["label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
clf = SVC()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred, labels=['counterfeit', 'genuine'], zero_division=1))

# Use the model to classify new images
def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

new_image = load_image("../test/new_product.jpg")
new_image = cv2.resize(new_image, (256, 256))
radius = 3
n_points = 8 * radius
lbp = local_binary_pattern(new_image, n_points, radius, method='uniform')
hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
hist = hist.astype("float")
hist /= (hist.sum() + 1e-7)

prediction = clf.predict([hist])
print("Prediction:", prediction)