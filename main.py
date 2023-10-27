# Import necessary libraries
import pandas as pd
import cv2

from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage.feature import hog


# Feature extraction function from openCV. This function takes in the dataset images, loads them using OpenCV,
# resizes them to 256x256, converts them to grayscale and then uses Histograms of Oriented Gradients (HOG) to extract
# features from the images

def extract_features(images):
    features = []
    for img in images:
        # Load the image
        img = cv2.imread(img)
        # Resize the image to a consistent size
        img = cv2.resize(img, (256, 256))
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Extract features using histograms of oriented gradients (HOG)
        hog_features = hog(gray_image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        features.append(hog_features)
    return features


# Load the dataset
data = pd.read_csv("products.csv")

# Preprocess the data
data = data.dropna()
data["price"] = (data["price"] - data["price"].mean()) / data["price"].std()

# Extract features from the images
X = extract_features(data["image"])
y = data["label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
clf = SVC()             # classify the images parameters are ~> genuine and counterfeit
clf.fit(X_train, y_train)  # fitting the testing and training test / fit means using the dataset in the model

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred, labels=['counterfeit', 'genuine']))


# Use the model to classify new images
def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


new_image = load_image("new_product3.jpg")
new_image = cv2.resize(new_image, (256, 256))
features = hog(new_image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
prediction = clf.predict([features])
print("Prediction:", prediction)
