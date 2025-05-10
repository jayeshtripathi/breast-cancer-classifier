import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys

# Path to your extracted BreakHis dataset
dataset_path = '/home/jayesh/work/BreaKHis_v1'

# Set seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
random.seed(0)
np.random.seed(0)
tf.random.set_seed(1)

# Define image size
IMG_SIZE = 224

# Recreate the model architecture
def create_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create model and load weights
print("Creating model and loading weights...")
model = create_model()
try:
    model.load_weights('model_weights.h5')
    print("Successfully loaded weights.")
except Exception as e:
    print(f"Error loading weights: {e}")
    print("Using default ImageNet weights.")

# Function to find all image paths and labels
def get_all_image_paths_and_labels(dataset_path):
    benign_subtypes = ['adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma']
    malignant_subtypes = ['ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']

    benign_base = os.path.join(dataset_path, 'histology_slides', 'breast', 'benign', 'SOB')
    malignant_base = os.path.join(dataset_path, 'histology_slides', 'breast', 'malignant', 'SOB')
    
    # Check if "malignant" is spelled as "malign" in the dataset
    if not os.path.exists(malignant_base):
        malignant_base = os.path.join(dataset_path, 'histology_slides', 'breast', 'malign', 'SOB')

    image_paths = []
    labels = []

    # Collect benign images
    for subtype in benign_subtypes:
        subtype_path = os.path.join(benign_base, subtype)
        if not os.path.exists(subtype_path):
            continue
        for patient in os.listdir(subtype_path):
            patient_path = os.path.join(subtype_path, patient)
            if not os.path.isdir(patient_path):
                continue
            for mag in ['40X', '100X', '200X', '400X']:
                mag_path = os.path.join(patient_path, mag)
                if not os.path.exists(mag_path):
                    continue
                for img_file in os.listdir(mag_path):
                    if img_file.endswith('.png'):
                        image_paths.append(os.path.join(mag_path, img_file))
                        labels.append(0)  # benign

    # Collect malignant images
    for subtype in malignant_subtypes:
        subtype_path = os.path.join(malignant_base, subtype)
        if not os.path.exists(subtype_path):
            continue
        for patient in os.listdir(subtype_path):
            patient_path = os.path.join(subtype_path, patient)
            if not os.path.isdir(patient_path):
                continue
            for mag in ['40X', '100X', '200X', '400X']:
                mag_path = os.path.join(patient_path, mag)
                if not os.path.exists(mag_path):
                    continue
                for img_file in os.listdir(mag_path):
                    if img_file.endswith('.png'):
                        image_paths.append(os.path.join(mag_path, img_file))
                        labels.append(1)  # malignant

    return image_paths, labels

# Process images in batches
def process_images_batch(image_paths, batch_size=32):
    predictions = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img)
            img_array /= 255.0
            batch_images.append(img_array)
        
        batch_array = np.array(batch_images)
        batch_predictions = model.predict(batch_array, verbose=0)
        predictions.extend(batch_predictions.flatten().tolist())
    
    return predictions

# Main evaluation function
def evaluate_sample(dataset_path, sample_size=100):
    print(f"Collecting image paths and labels from {dataset_path}...")
    all_image_paths, all_labels = get_all_image_paths_and_labels(dataset_path)
    
    print(f"Found {len(all_image_paths)} total images")
    
    # Sample images
    if len(all_image_paths) > sample_size:
        print(f"Randomly sampling {sample_size} images for evaluation")
        sampled_indices = random.sample(range(len(all_image_paths)), sample_size)
    else:
        print(f"Using all {len(all_image_paths)} images for evaluation")
        sampled_indices = list(range(len(all_image_paths)))
    
    sampled_image_paths = [all_image_paths[i] for i in sampled_indices]
    sampled_labels = [all_labels[i] for i in sampled_indices]
    
    # Get class distribution in sample
    benign_count = sampled_labels.count(0)
    malignant_count = sampled_labels.count(1)
    print(f"Sample distribution: {benign_count} benign, {malignant_count} malignant")
    
    # Get predictions
    print("Making predictions...")
    prediction_scores = process_images_batch(sampled_image_paths)
    predicted_labels = [1 if score > 0.5 else 0 for score in prediction_scores]
    
    return np.array(sampled_labels), np.array(predicted_labels), sampled_image_paths, np.array(prediction_scores)

# Run evaluation on sample
print("Starting evaluation on sample...")
true_labels, predicted_labels, image_paths, prediction_scores = evaluate_sample(dataset_path, sample_size=100)

# Calculate metrics
print("Calculating metrics...")
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Print results
print("\n===== Model Performance on Sample =====")
print(f"Sample size: {len(true_labels)} images")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (100 Sample Images)')
plt.savefig('confusion_matrix_sample.png')
print("Saved confusion matrix to confusion_matrix_sample.png")

# Plot ROC curve
fpr, tpr, _ = roc_curve(true_labels, prediction_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (100 Sample Images)')
plt.legend(loc="lower right")
plt.savefig('roc_curve_sample.png')
print("Saved ROC curve to roc_curve_sample.png")

print("\nEvaluation complete!")
