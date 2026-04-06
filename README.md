# Pneumonia-Detection-using-Deep-Learning
##This Ai model has an success rate of 88.46%.
## How to Download and Use

Follow these steps to run the project:

1. Download both files:

   * Pneumonia Model (PM)
   * AI Summoning Code

2. Upload the model file (`pneumonia_model.h5`) to your Google Drive:

   ```
   MyDrive/pneumonia_model.h5
   ```

3. Open Google Colab.

4. Paste the AI summoning code into a new notebook.

5. Run all cells.

6. When prompted, allow access to Google Drive.

7. Make sure the file path in the code matches the model location in your Google Drive:

   ```python
   model = load_model("/content/drive/MyDrive/pneumonia_model.h5")
   ```

8. After setup, upload a chest X-ray image and the model will generate a prediction.
   (I have added two chest x rays for you to try out!)

---

### Running Predictions Again

To analyze another chest X-ray image without restarting everything, run the following part of the code again:

```python
print("Upload X-ray image")
uploaded = files.upload()

image_path = list(uploaded.keys())[0]

predict_image(image_path)
```

This allows you to upload and analyze new images instantly.

## How It Works (Detailed Explanation)

This project uses a deep learning model based on transfer learning with MobileNetV2 to analyze chest X-ray images.

---

### 1. Image Input and Preprocessing

* Input images are resized to 224 × 224 pixels
* Pixel values are normalized to a range between 0 and 1
* Images are converted into numerical arrays for model processing

```python
img = cv2.resize(img, (224, 224))
img = img / 255.0
```

---

### 2. Feature Extraction

The model uses MobileNetV2 as a pretrained backbone:

* Originally trained on ImageNet
* Learns general visual features such as edges, shapes, and textures
* These features are reused for medical image analysis

```python
base_model = MobileNetV2(weights='imagenet', include_top=False)
```

---

### 3. Custom Classification Layers

Additional layers are added on top of the base model:

* GlobalAveragePooling2D reduces spatial dimensions
* Dense layer learns task-specific patterns
* Dropout helps reduce overfitting
* Final sigmoid layer outputs a probability score

```python
Dense(1, activation='sigmoid')
```

---

### 4. Model Output

The model produces a probability value between 0 and 1:

* Values closer to 0 indicate Normal
* Values closer to 1 indicate Pneumonia

Example outputs:

* 0.30 → likely Normal
* 0.85 → strong Pneumonia indication

---

### 5. Severity Estimation

The model is trained only for binary classification (Normal vs Pneumonia), so it does not directly predict severity levels.

Severity is estimated by interpreting the model’s output probability.

* The model outputs a value between 0 and 1, which is converted into a percentage (0–100%)
* Higher percentages indicate stronger pneumonia-related patterns in the image

Based on this percentage, severity levels are defined as:

| Probability (%) | Interpretation   |
| --------------- | ---------------- |
| < 50%           | Normal           |
| 50% – 65%       | Low Pneumonia    |
| 65% – 80%       | Medium Pneumonia |
| > 80%           | High Pneumonia   |

This means:

* Lower percentages correspond to weaker or absent pneumonia patterns
* Higher percentages correspond to stronger pneumonia patterns

This method provides an estimated severity level based on the model’s output, rather than a medically labeled severity classification.


This is a heuristic interpretation based on model output.

---

### 6. Training Process

During training:

1. Images are passed through the model
2. Predictions are compared with true labels
3. Loss is computed
4. Model weights are updated using backpropagation
5. This process repeats across multiple epochs

---

### 7. Fine-Tuning

After initial training:

* Selected layers of MobileNetV2 are unfrozen
* Training continues with a lower learning rate

This allows the model to adapt more specifically to chest X-ray patterns.

---

### 8. Summary

* Input: Chest X-ray image
* Processing: Feature extraction + classification
* Output: Probability score and estimated severity

---

### Note

This system performs pattern recognition based on training data and does not provide medical diagnosis.


