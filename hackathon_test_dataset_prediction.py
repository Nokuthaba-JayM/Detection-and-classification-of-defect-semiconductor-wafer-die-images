import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd


# CONFIG

MODEL_PATH = "wafer_defect_cnn.h5"
TEST_DIR = "hackathon_test_dataset"
IMG_SIZE = 128
LOG_FILE = "prediction_log.txt"

# Hackathon folder order 
model_classes = [
    "Bridge",
    "CMP",
    "Clean",
    "Contamination",
    "LER",
    "Opens",
    "Other",
    "Residue",
    "Surface damage",
    "Via"
]
test_classes = [
    "bridge", "clean", "CMP", "crack",
    "LER", "open", "other", "particle", "via"
]

#model training label names 
train_class_map = {
    "bridge": "Bridge",
    "clean": "Clean",
    "CMP": "CMP",
    "crack": "Surface damage",
    "particle": "Residue",
    "open": "Opens",
    "LER": "LER",
    "via": "Via",
    "other": "Other"
}

# Final class list for confusion matrix
final_classes = list(set(train_class_map.values()))


# LOAD MODEL

print("Loading trained model...")
model = load_model("C:\\Users\\Claret\\Documents\\Jackie's files\\deeptech\\wafer_defect_cnn.h5")


# IMAGE PREPROCESSING

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    return img


# PREDICTION 

y_true = []
y_pred = []
log_lines = []

print("Running inference on test dataset...")

for folder in test_classes:
    folder_path = os.path.join("C:\\Users\\Claret\\Documents\\Jackie's files\\deeptech\\hackathon_test_dataset", folder)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img = preprocess_image(img_path)
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img, verbose=0)
        pred_index = np.argmax(preds)
        confidence = np.max(preds)

        predicted_train_label = model_classes[pred_index]
        true_train_label = train_class_map[folder]

        y_true.append(true_train_label)
        y_pred.append(predicted_train_label)

        log_lines.append(
            f"{img_name} | true: {true_train_label} | predicted: {predicted_train_label} | confidence: {confidence:.4f}"
        )




# METRICS

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
cm = confusion_matrix(y_true, y_pred, labels=final_classes)

print("\n--- Evaluation Results ---")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("\nConfusion Matrix:")
print(pd.DataFrame(cm, index=final_classes, columns=final_classes))


cm_df = pd.DataFrame(cm, index=final_classes, columns=final_classes)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_FILE = os.path.join(BASE_DIR, "prediction_log.txt")
CM_FILE = os.path.join(BASE_DIR, "confusion_matrix.csv")

with open(LOG_FILE, "w") as f:
    ...
cm_df.to_csv(CM_FILE, index=False)

with open(LOG_FILE, "w") as f:
    for line in log_lines:
        f.write(line + "\n")
cm_df.to_csv(CM_FILE, index=False)

print("\nConfusion matrix saved as confusion_matrix.csv")
