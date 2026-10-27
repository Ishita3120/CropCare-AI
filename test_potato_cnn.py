from core.potato_cnn import predict_potato_disease

image_path = "C:\\Users\\ishit\\Downloads\\crop-disease-webapp\\crop-disease-webapp\\datasets\\Potato\\Test\\Potato___healthy\\Potato_healthy-26-_0_4635.jpg"

disease, confidence = predict_potato_disease(image_path)

print("\nFinal Prediction:")
print("Disease:", disease)
print("Confidence:", f"{confidence * 100:.2f}%")