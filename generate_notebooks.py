import nbformat
import os
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def create_notebook_template(member_id):
    """Create a notebook template for malaria CNN classification."""
    nb = new_notebook()
    
    # Notebook metadata
    nb.metadata = {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.12.0'
        }
    }
    
    # Cells
    cells = []
    
    # Title
    cells.append(new_markdown_cell(f'# Malaria CNN Classification - Member {member_id}\\n\\nThis notebook implements a CNN for malaria cell image classification.'))
    
    # Section 1: Import and Setup
    cells.append(new_markdown_cell('## 1. Import and Setup'))
    cells.append(new_code_cell('''# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Import project modules
from src.common import *

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
'''))
    
    # Section 2: Data Loading and Exploration
    cells.append(new_markdown_cell('## 2. Data Loading and Exploration'))
    cells.append(new_code_cell('''# Define data directories
data_dirs = resolve_dirs()
print("Data directories:", data_dirs)

# Download and extract data (if needed)
# Uncomment the following lines if you need to download the dataset
# download_cell_images(data_dirs["raw"])
# extract_cell_images(data_dirs["raw"])

# Load dataset
print("Loading dataset...")
train_ds, val_ds, test_ds = build_datasets(data_dirs["processed"], validation_split=0.2, seed=SEED)

# Get class names
class_names = [data_dirs["processed"] / "Parasitized", data_dirs["processed"] / "Uninfected"]
class_names = [path.name for path in class_names]
print("Class names:", class_names)

# Explore dataset
print("\\nDataset info:")
print(f"Training batches: {len(train_ds)}")
print(f"Validation batches: {len(val_ds)}")
print(f"Test batches: {len(test_ds)}")
'''))
    
    # Section 3: Data Preprocessing
    cells.append(new_markdown_cell('## 3. Data Preprocessing'))
    cells.append(new_code_cell('''# Visualize some samples
def show_sample_images(dataset, num_images=9, class_names=None):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(num_images):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            if class_names:
                plt.title(class_names[np.argmax(labels[i])])
            plt.axis("off")
    plt.show()

print("Sample training images:")
show_sample_images(train_ds, class_names=class_names)

# Check class distribution
def get_class_distribution(dataset):
    class_counts = {}
    for _, labels in dataset.unbatch():
        label = np.argmax(labels.numpy())
        class_counts[label] = class_counts.get(label, 0) + 1
    return class_counts

train_dist = get_class_distribution(train_ds)
val_dist = get_class_distribution(val_ds)
test_dist = get_class_distribution(test_ds)

print("\\nClass distribution:")
print("Train:", {class_names[k]: v for k, v in train_dist.items()})
print("Validation:", {class_names[k]: v for k, v in val_dist.items()})
print("Test:", {class_names[k]: v for k, v in test_dist.items()})
'''))
    
    # Section 4: Model Building and Training
    cells.append(new_markdown_cell('## 4. Model Building and Training'))
    cells.append(new_code_cell('''# Define model architecture
def create_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=2):
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=input_shape),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Create and compile model
model = create_model()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
    ]
)
'''))
    
    # Section 5: Model Evaluation
    cells.append(new_markdown_cell('## 5. Model Evaluation'))
    cells.append(new_code_cell('''# Evaluate on test set
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Get predictions and true values
y_pred, y_true = collect_predictions(model, test_ds)
y_pred_labels = np.argmax(y_pred, axis=1)

# Classification report
print("\\nClassification Report:")
print(classification_report(y_true, y_pred_labels, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_labels)
print("\\nConfusion Matrix:")
print(cm)
'''))
    
    # Section 6: Visualization and Explainability
    cells.append(new_markdown_cell('## 6. Visualization and Explainability'))
    cells.append(new_code_cell('''# Plot training history
plot_training_curves(history)

# Plot confusion matrix
plot_confusion_matrix(cm, class_names=class_names)

# ROC curve
from sklearn.metrics import roc_curve, auc
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(class_names)):
    fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
plt.plot(fpr[0], tpr[0], label=f'ROC curve (area = {roc_auc[0]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Grad-CAM explanations
print("\\nGenerating Grad-CAM explanations...")
# Get a batch of test images for explanation
for images, labels in test_ds.take(1):
    test_images = images.numpy()
    test_labels = labels.numpy()
    break

# Generate heatmaps for a few misclassified examples
misclassified_idx = np.where(y_pred_labels != y_true)[0]
if len(misclassified_idx) > 0:
    print(f"Found {len(misclassified_idx)} misclassified examples")
    # Select a few examples to explain
    num_explanations = min(3, len(misclassified_idx))
    selected_idx = misclassified_idx[:num_explanations]
    
    # Generate Grad-CAM heatmaps
    heatmaps = []
    for idx in selected_idx:
        img = test_images[idx]
        heatmap = make_gradcam_heatmap(img, model, last_conv_layer_name='conv2d_3')
        heatmaps.append(heatmap)
    
    # Plot original images with heatmaps
    plt.figure(figsize=(15, 5))
    for i, (idx, heatmap) in enumerate(zip(selected_idx, heatmaps)):
        plt.subplot(1, num_explanations, i+1)
        # Original image
        plt.imshow(test_images[idx].astype("uint8"))
        # Heatmap
        plt.imshow(heatmap, alpha=0.5, cmap='jet')
        plt.title(f'True: {class_names[np.argmax(test_labels[idx])]}\\nPred: {class_names[y_pred_labels[idx]]}')
        plt.axis('off')
    plt.suptitle('Grad-CAM Explanations for Misclassified Examples')
    plt.show()
else:
    print("No misclassified examples found in this batch!")
'''))
    
    # Section 7: Conclusion and Next Steps
    cells.append(new_markdown_cell('## 7. Conclusion and Next Steps'))
    cells.append(new_code_cell('''# Save model
model_save_path = f"models/malaria_cnn_member{member_id}.h5"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Final thoughts
print("\\n=== Summary ===")
print(f"Test Accuracy: {test_acc:.4f}")
print("Next steps:")
print("- Try different architectures (ResNet, EfficientNet)")
print("- Experiment with data augmentation")
print("- Fine-tune hyperparameters")
print("- Ensemble multiple models")
'''))
    
    # Add all cells to notebook
    nb.cells = cells
    return nb

# Generate notebooks for members 1-4
for member_id in range(1, 5):
    nb = create_notebook_template(member_id)
    # Write notebook to file
    notebook_path = f"member_{member_id}/model_training.ipynb"
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Created {notebook_path}")

print("\\nAll notebook templates created successfully!")