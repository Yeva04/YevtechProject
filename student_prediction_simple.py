import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Dense #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
from tensorflow.keras.callbacks import EarlyStopping #type: ignore
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic student data
n_samples = 1000
data = {
    'student_id': [f'STU{i:03d}' for i in range(1, n_samples + 1)],
    'attendance_rate': np.random.uniform(0, 100, n_samples),
    'assignment_score': np.random.uniform(0, 100, n_samples),
    'quiz_score': np.random.uniform(0, 100, n_samples),
    'exam_score': np.random.uniform(0, 100, n_samples),
    'study_hours_per_week': np.random.uniform(0, 70, n_samples),
    'feedback': [f"Student {i} feedback." for i in range(1, n_samples + 1)],  # Placeholder, not used
}

df = pd.DataFrame(data)

# Define performance label based on average score
df['avg_score'] = (df['exam_score'] + df['quiz_score'] + df['assignment_score']) / 3
df['performance_label'] = pd.cut(
    df['avg_score'],
    bins=[0, 30, 65, 100],
    labels=['Poor', 'Average', 'Excellent'],
    include_lowest=True
)

# Map labels to numerical values
y = df['performance_label'].map({'Poor': 0, 'Average': 1, 'Excellent': 2}).values.astype(np.int32)
logger.info(f"Target variable dtype: {y.dtype}")

# Log class distribution
class_counts = pd.Series(y).value_counts()
logger.info(f"Class distribution: {class_counts.to_dict()}")

# Use only the selected features
selected_features = ['assignment_score', 'quiz_score', 'exam_score']
X_numerical = df[selected_features]
X_train_num, X_test_num, y_train, y_test = train_test_split(X_numerical, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)
joblib.dump(scaler, 'scaler.save')

# Build a simple dense neural network
model = Sequential([
    Dense(16, activation='relu', input_shape=(len(selected_features),)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train_num_scaled, y_train,
    validation_data=(X_test_num_scaled, y_test),
    epochs=50, batch_size=32, verbose=1,
    callbacks=[early_stopping]
)

# Log final training metrics
logger.info(f"Final training loss: {history.history['loss'][-1]}")
logger.info(f"Final training accuracy: {history.history['accuracy'][-1]}")
logger.info(f"Final validation loss: {history.history['val_loss'][-1]}")
logger.info(f"Final validation accuracy: {history.history['val_accuracy'][-1]}")

# Save the model and selected features
model.save('simple_student_model.h5')
joblib.dump(selected_features, 'selected_features.joblib')

print("Simple model training complete. Saved as 'simple_student_model.h5'.")