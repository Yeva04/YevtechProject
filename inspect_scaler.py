import joblib
import numpy as np

# Load the scaler
scaler = joblib.load('scaler.save')

# Print the scaler's attributes
print("Scaler type:", type(scaler))
print("Number of features scaler was fitted on:", scaler.n_features_in_)
print("Feature names (if available):", scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else "Not available")
print("Mean of each feature:", scaler.mean_)
print("Standard deviation of each feature:", scaler.scale_)

# Test the scaler with a sample input
sample_data = np.array([[85, 80, 90]])  # Example: [assignment_score, quiz_score, exam_score]
scaled_data = scaler.transform(sample_data)
print("Sample data:", sample_data)
print("Scaled data:", scaled_data)