import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing import image
import cv2

# --- Create results folder ---
os.makedirs("results", exist_ok=True)

# --- Load dataset ---
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # normalize

# --- Build the model ---
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# --- Compile ---
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- Train ---
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# --- Evaluate ---
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"✅ Test accuracy: {test_acc:.4f}")

# --- Save model ---
model.save("results/digit_recognition_model.h5")

# --- Training curves ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("results/training_curves.png")
plt.close()

# --- Confusion Matrix ---
y_pred = model.predict(x_test).argmax(axis=1)
y_true = y_test
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")
plt.close()

# --- Misclassified Samples ---
misclassified_idx = np.where(y_true != y_pred)[0][:25]
plt.figure(figsize=(10, 10))
for i, idx in enumerate(misclassified_idx):
    plt.subplot(5, 5, i+1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"T:{y_true[idx]} P:{y_pred[idx]}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("results/misclassified_samples.png")
plt.close()

# --- Classification Report ---
report = classification_report(y_true, y_pred, digits=4)
with open("results/classification_report.txt", "w") as f:
    f.write(report)

# --- Predict custom handwritten image ---
def predict_custom_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = 255 - img  # invert colors (white bg, black digit)
    img = img / 255.0
    img = img.reshape(1, 28, 28)

    pred = model.predict(img).argmax(axis=1)[0]

    plt.imshow(img.reshape(28, 28), cmap="gray")
    plt.title(f"Predicted Digit: {pred}")
    plt.axis("off")
    plt.savefig("results/custom_prediction.png")
    plt.close()

    print(f"✅ Custom image '{img_path}' predicted as: {pred}")

# Example usage:
# predict_custom_image("41355694-4329-4a33-87fd-8d58f1fccaf5.png")

print("🎉 All results saved inside 'results/' folder")
