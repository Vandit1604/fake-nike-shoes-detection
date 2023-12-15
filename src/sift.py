import pandas as pd
import cv2
from skimage.feature import local_binary_pattern
import numpy as np
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

def extract_sift_features(images):
    sift = cv2.SIFT_create()  # Initialize SIFT object
    sift_descriptors = []
    for img_path in images:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        # Store descriptors
        if descriptors is not None:
            sift_descriptors.extend(descriptors)
    # Convert descriptors to a numpy array
    sift_descriptors = np.array(sift_descriptors)
    k = 100  # You can adjust this value based on your dataset and requirements
    kmeans = KMeans(n_clusters=5, n_init=10)
    kmeans.fit(sift_descriptors)
    # Construct histograms of visual words for each image
    features = []
    for img_path in images:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        # Initialize histogram for this image
        hist = np.zeros(k)
        if descriptors is not None:
            # Assign each descriptor to the nearest cluster center
            labels = kmeans.predict(descriptors)
            # Update histogram counts
            for label in labels:
                hist[label] += 1
        # Normalize the histogram
        hist /= np.sum(hist)

        features.append(hist)

    return features


# Load the dataset
data = pd.read_csv("../products.csv")

# Preprocess the data
data = data.dropna()
data["price"] = (data["price"] - data["price"].mean()) / data["price"].std()

X = extract_sift_features(data["image"])
y = data["label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model (using a classifier, e.g., SVC)
clf = SVC()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model (using appropriate metrics)
print(classification_report(y_test, y_pred, labels=['counterfeit', 'genuine'], zero_division=1))

# Use the model to classify new images (extract SIFT features for new images and predict)
def extract_and_predict_sift(image_path, model):
    sift_features = extract_sift_features([image_path])
    prediction = model.predict(sift_features)
    return prediction

new_image_path = "/home/vandit/codes/fake-nike-shoes-detection/test/new_product.jpg"  # Path to the new image
new_image_prediction = extract_and_predict_sift(new_image_path, clf)
print("Prediction for new image:", new_image_prediction)