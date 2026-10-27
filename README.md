# 🌱 CropCare AI — Crop Disease Detection & Treatment Guidance

CropCare AI is a deep-learning-powered web application that helps users identify tomato and potato leaf diseases from uploaded images. It combines trained CNN-based image classification models with a Flask web application, disease severity assessment, and treatment guidance.

The application provides a simple interface where users can select a crop, upload a leaf image, view the predicted disease, and access relevant treatment recommendations.

## ✨ Features

- **Deep Learning-Based Disease Classification:** Uses trained TensorFlow/Keras models for tomato and potato leaf-disease classification.
- **CNN and MobileNetV2:** Includes convolutional neural network models and MobileNetV2-based classifiers.
- **Leaf Validation:** Includes a trained leaf-validation model to help validate uploaded images.
- **Disease Classification:** Supports healthy leaves and disease categories for tomato and potato.
- **Severity Assessment:** Estimates disease severity based on the application's analysis pipeline.
- **Treatment Recommendations:** Retrieves crop- and disease-specific guidance from a treatment knowledge base.
- **Interactive Web Interface:** Built with Flask, HTML, CSS, and JavaScript.
- **Image Upload and Preview:** Allows users to upload leaf images for analysis.
- **Crop Selection Validation:** Helps prevent predictions from being used for a different selected crop.
- **Modular Architecture:** Separates model inference, severity assessment, treatment lookup, and application logic.

## 🧠 Model Architecture

CropCare AI uses trained deep-learning models built with TensorFlow/Keras.

### Tomato Disease Classifier

The tomato classifier uses a trained MobileNetV2-based model for classifying tomato leaf images into four categories:

- Tomato Bacterial Spot
- Tomato Early Blight
- Tomato Late Blight
- Tomato Healthy

### Potato Disease Classifier

The potato classifier uses a trained MobileNetV2-based model for classifying potato leaf images into three categories:

- Potato Early Blight
- Potato Late Blight
- Potato Healthy

### Additional Models

The repository also includes a crop-classification model and a leaf-validation model. These models serve separate purposes in the image-processing workflow.

**Note:** Individual model files are stored in the `models/` directory. The exact inference path depends on the model-loading and prediction logic implemented in the application.

## 📊 Model Evaluation Results

The tomato and potato classifiers were evaluated using separate test datasets. The following results are taken from the evaluation outputs.

### Overall Performance

| Metric | Tomato Classifier | Potato Classifier |
|---|---:|---:|
| Test Accuracy | **96.50%** | **98.67%** |
| Macro Precision | 96% | 99% |
| Macro Recall | 97% | 99% |
| Macro F1-score | 96% | 99% |
| Number of Test Images | 400 | 300 |
| Number of Classes | 4 | 3 |

### Tomato Classifier — Class-wise Results

| Class | Precision | Recall | F1-score | Test Images |
|---|---:|---:|---:|---:|
| Bacterial Spot | 0.98 | 0.98 | 0.98 | 100 |
| Early Blight | 0.95 | 0.93 | 0.94 | 100 |
| Late Blight | 0.96 | 0.95 | 0.95 | 100 |
| Healthy | 0.97 | 1.00 | 0.99 | 100 |

The tomato classifier correctly classified **386 out of 400 test images**, achieving 96.50% test accuracy.

### Potato Classifier — Class-wise Results

| Class | Precision | Recall | F1-score | Test Images |
|---|---:|---:|---:|---:|
| Early Blight | 0.99 | 1.00 | 1.00 | 100 |
| Late Blight | 0.98 | 0.98 | 0.98 | 100 |
| Healthy | 0.99 | 0.98 | 0.98 | 100 |

The potato classifier correctly classified **296 out of 300 test images**, achieving 98.67% test accuracy.

### Evaluation Notes

- Accuracy measures the proportion of correctly classified test images.
- Precision measures how often predictions for a class are correct.
- Recall measures how many actual examples of a class are identified.
- The F1-score balances precision and recall.
- The test datasets contain 100 images per class for each evaluated crop.
- Results from the tomato and potato models should not be interpreted as a direct comparison on the same dataset.
- Individual prediction confidence is different from overall test accuracy.
- Real-world performance may vary with image quality, lighting, backgrounds, leaf orientation, and images that differ from the training data.

These results describe the evaluated classifiers and their respective test datasets. They do not guarantee identical performance for every uploaded image or every component of the complete application.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow / Keras |
| Model Architecture | CNN, MobileNetV2 |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Image Processing | Pillow |
| Numerical Processing | NumPy |
| Testing | pytest |
| Version Control | Git and GitHub |

## 📁 Project Structure

```text
CropCare-AI/
├── app.py
├── requirements.txt
├── README.md
├── core/
│   ├── __init__.py
│   ├── decision_engine.py
│   ├── image_analyzer.py
│   ├── knowledge_base.py
│   ├── potato_cnn.py
│   ├── severity.py
│   └── tomato_cnn.py
├── data/
│   ├── treatments.json
│   └── Smart_Farming_Crop_Yield_2024.csv
├── models/
│   ├── crop_classifier.pkl
│   ├── crop_classifier_backup.pkl
│   ├── crop_classifier_cnn.keras
│   ├── leaf_validator.keras
│   ├── potato_mobilenetv2.keras
│   ├── tomato_best.keras
│   └── tomato_full_mobilenetv2.keras
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── images/
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── result.html
│   ├── treatment.html
│   └── upload.html
├── tests/
│   └── test_analyzer.py
├── uploads/
│   └── .gitkeep
├── train_crop_classifier.py
├── train_crop_cnn.py
├── train_leaf_validator.py
├── train_potato_cnn.py
├── train_tomato_full_cnn.py
├── evaluate_tomato.py
├── test_crop_classifier.py
├── test_crop_mismatch.py
├── test_leaf_validator.py
├── test_potato_cnn.py
└── test_tomato_full.py
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Ishita3120/CropCare-AI.git
cd CropCare-AI
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv dl_env
.\dl_env\Scripts\Activate.ps1
```

If PowerShell does not allow activation, you can run Python using the virtual environment's executable directly.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow installation compatibility depends on your Python version and operating system. Use a supported Python and TensorFlow combination if installation fails.

### 4. Run the application

```bash
python app.py
```

Open the following URL in your browser:

[http://localhost:5000](http://localhost:5000)

## 🧪 Running Tests

Run the project tests using:

```bash
pytest tests/ -v
```

Additional model-specific test scripts are available in the repository, including:

```bash
python test_crop_classifier.py
python test_leaf_validator.py
python test_potato_cnn.py
python test_tomato_full.py
```

Some model-specific scripts may perform inference or evaluation using the corresponding model and dataset files. Check each script before running it.

## 🔄 Training and Evaluation

The repository includes scripts for training and evaluating the models.

Examples:

```bash
python train_crop_classifier.py
python train_crop_cnn.py
python train_leaf_validator.py
python train_potato_cnn.py
python train_tomato_full_cnn.py
python evaluate_tomato.py
```

**Important:** Training scripts may require datasets and preprocessing files that are not included in the GitHub repository. Ensure the expected data files and paths are available before running them. Running a training script may overwrite an existing model file.

## 🔒 Data and Model Considerations

- The `dl_env/` virtual environment is excluded from version control.
- Training datasets and uploaded test images are excluded from the repository.
- Pre-trained model files used by the application are stored in `models/`.
- The `uploads/` directory contains a `.gitkeep` file so that the folder exists after cloning. Uploaded images are not committed.
- Model evaluation results depend on the test data, preprocessing pipeline, and saved model version.

## 🚀 Future Improvements

- Evaluate all models on clearly separated and documented test datasets.
- Add per-class evaluation reports for the crop classifier and leaf-validation model.
- Improve generalization to real-world field photographs and different lighting conditions.
- Add a visual confusion matrix and model-performance charts.
- Expand the supported crops and disease categories.
- Improve application deployment using a production WSGI server.
- Add automated cleanup for uploaded images.
- Validate treatment recommendations against reliable agricultural guidance for the relevant region.

## ⚠️ Disclaimer

CropCare AI is an educational and experimental project. Model predictions may be incorrect, especially for poor-quality images or conditions not represented in the training data.

Treatment guidance should be verified against reliable agricultural extension resources and product labels before use. The application should not be treated as a substitute for professional agronomic advice.

## 👩‍💻 Author

**Ishita Sahay**

- GitHub: [Ishita3120](https://github.com/Ishita3120)
- Project Repository: [CropCare-AI](https://github.com/Ishita3120/CropCare-AI)
