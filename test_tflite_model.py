# test_tflite_model.py
import numpy as np
import tensorflow as tf
from transformers import MobileBertTokenizer, TFMobileBertModel
import joblib

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="student_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load scaler and MobileBERT
scaler = joblib.load("scaler.save")
tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
mobilebert_model = TFMobileBertModel.from_pretrained('google/mobilebert-uncased')

# Sample input
new_student = {
    'attendance_rate': 87,
    'assignment_score': 80,
    'quiz_score': 75,
    'exam_score': 82,
    'study_hours_per_week': 12,
    'feedback': "Excellent work"
}

# Preprocess numerical data
input_values = np.array([
    new_student['attendance_rate'],
    new_student['assignment_score'],
    new_student['quiz_score'],
    new_student['exam_score'],
    new_student['study_hours_per_week']
]).reshape(1, -1)
scaled_input = scaler.transform(input_values).reshape(1, 1, input_values.shape[1]).astype(np.float32)

# Preprocess textual data
inputs = tokenizer([new_student['feedback']], return_tensors='tf', padding=True, truncation=True, max_length=128)
outputs = mobilebert_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
text_input = outputs.last_hidden_state[:, 0, :].numpy().astype(np.float32)

# Set input tensors
interpreter.set_tensor(input_details[0]['index'], scaled_input)
interpreter.set_tensor(input_details[1]['index'], text_input)

# Run inference
interpreter.invoke()
tflite_prediction = interpreter.get_tensor(output_details[0]['index'])
predicted_label = np.argmax(tflite_prediction)

# Map label
label_map = {0: "At Risk ❌", 1: "Average ⚠️", 2: "Excellent ✅"}
print("\n🎯 TFLite Prediction Result:")
print(f"→ Student is predicted to be: **{label_map[predicted_label]}**")