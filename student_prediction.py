import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import joblib
from tf_keras.models import Model
from tf_keras.layers import Input, LSTM, Dense, Concatenate, Dropout
from tf_keras.optimizers import Adam
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import MobileBertTokenizer, TFMobileBertModel
from deap import base, creator, tools, algorithms
import random
import logging
from typing import List, Dict, Any
import tensorflow as tf

# Configuration
CONFIG = {
    'n_samples': 50,  # Further reduced from 200
    'random_seed': 42,
    'numerical_features': ['attendance_rate', 'assignment_score', 'quiz_score', 'exam_score', 'study_hours_per_week'],
    'text_feature': 'feedback',
    'target': 'performance_label',
    'max_length': 32,  # Reduced from 64
    'batch_size': 4,   # Reduced from 8
    'epochs': 20,      # Reduced from 30
    'lstm_units': 16,  # Reduced from 32
    'dropout_rate': 0.3,  # Reduced from 0.5
    'learning_rate': 0.0001,
    'patience': 3,    # Reduced from 5
    'population_size': 5,  # Reduced from 10
    'generations': 2,  # Reduced from 3
}


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment() -> None:
    """Set random seeds and configure TensorFlow for reproducibility."""
    np.random.seed(CONFIG['random_seed'])
    random.seed(CONFIG['random_seed'])
    tf.random.set_seed(CONFIG['random_seed'])

def generate_synthetic_data(n_samples: int) -> pd.DataFrame:
    """Generate synthetic student data with realistic distributions."""
    try:
        # Use normal distributions for scores, clipped to [0, 100]
        attendance = np.clip(np.random.normal(80, 10, n_samples), 0, 100)
        assignment = np.clip(np.random.normal(75, 15, n_samples), 0, 100)
        quiz = np.clip(np.random.normal(70, 15, n_samples), 0, 100)
        exam = np.clip(np.random.normal(70, 20, n_samples), 0, 100)
        study_hours = np.clip(np.random.normal(20, 10, n_samples), 0, 70)

        data = {
            'student_id': [f'STU{i:03d}' for i in range(1, n_samples + 1)],
            'attendance_rate': attendance,
            'assignment_score': assignment,
            'quiz_score': quiz,
            'exam_score': exam,
            'study_hours_per_week': study_hours,
        }

        # Generate diverse feedback
        feedback_options = [
            "Struggles with time management but improving in assignments.",
            "Excels in exams and actively participates in discussions.",
            "Needs support in core concepts but attends regularly.",
            "Shows great potential and consistently scores well in quizzes.",
            "Demonstrates strong effort but requires extra practice.",
            "Highly motivated and excels in group projects.",
        ]
        data['feedback'] = [f"{random.choice(feedback_options)} Student ID: STU{i:03d}." for i in range(1, n_samples + 1)]

        df = pd.DataFrame(data)

        # Validate data
        if df.isnull().any().any():
            raise ValueError("Generated data contains NaN values.")
        if (df[CONFIG['numerical_features']] < 0).any().any():
            raise ValueError("Generated data contains negative values.")

        # Derive performance label
        df['avg_score'] = (df['exam_score'] + df['quiz_score'] + df['assignment_score']) / 3
        df[CONFIG['target']] = pd.cut(
            df['avg_score'],
            bins=[0, 30, 65, 100],
            labels=['Poor', 'Average', 'Excellent'],
            include_lowest=True
        )

        logger.info(f"Generated {n_samples} synthetic samples.")
        return df
    except Exception as e:
        logger.error(f"Error generating synthetic data: {e}")
        raise

def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute class weights for imbalanced classes."""
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = dict(enumerate(weights))
    logger.info(f"Class weights: {class_weight_dict}")
    return class_weight_dict

def select_features(X: pd.DataFrame, y: np.ndarray, numerical_features: List[str]) -> List[str]:
    """Perform feature selection using a genetic algorithm."""
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(numerical_features))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate_individual(individual: List[int], X: pd.DataFrame, y: np.ndarray) -> tuple:
        selected = [numerical_features[i] for i, bit in enumerate(individual) if bit]
        if not selected:
            return 0.0,
        X_selected = X[selected]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=CONFIG['random_seed'])
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        model = tf.keras.Sequential([
            LSTM(50, activation='relu', input_shape=(1, len(selected))),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train_lstm, y_train, epochs=5, batch_size=32, verbose=0)
        _, accuracy = model.evaluate(X_test_lstm, y_test, verbose=0)
        return accuracy,

    toolbox.register("evaluate", evaluate_individual, X=X, y=y)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=CONFIG['population_size'])
    for gen in range(CONFIG['generations']):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

    best_individual = tools.selBest(population, k=1)[0]
    selected_features = [numerical_features[i] for i, bit in enumerate(best_individual) if bit]
    logger.info(f"Selected features: {selected_features}")
    return selected_features

def encode_text(texts: List[str], tokenizer: Any, mobilebert_model: Any, max_length: int, batch_size: int) -> np.ndarray:
    """Encode text using MobileBERT, processing in batches."""
    try:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="tf", padding=True, truncation=True, max_length=max_length)
            outputs = mobilebert_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(batch_embeddings)
        embeddings = np.vstack(embeddings)
        logger.info(f"Text embeddings shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Error encoding text: {e}")
        raise

def build_hybrid_model(selected_features: List[str], text_embedding_dim: int) -> Model:
    """Build and compile the hybrid LSTM-MobileBERT model."""
    numerical_input = Input(shape=(1, len(selected_features)), name='numerical_input')
    text_input = Input(shape=(text_embedding_dim,), name='text_input')

    lstm_out = LSTM(CONFIG['lstm_units'], activation='relu')(numerical_input)
    numerical_dense = Dense(32, activation='relu')(lstm_out)
    text_dense = Dense(32, activation='relu')(text_input)
    combined = Concatenate()([numerical_dense, text_dense])
    dense = Dense(64, activation='relu')(combined)
    dropout = Dropout(CONFIG['dropout_rate'])(dense)
    output = Dense(3, activation='softmax')(dropout)

    model = Model(inputs=[numerical_input, text_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=CONFIG['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

def main():
    """Main function to generate data, train model, and save outputs."""
    setup_environment()

    # Generate data
    df = generate_synthetic_data(CONFIG['n_samples'])

    # Map labels
    y = df[CONFIG['target']].map({'Poor': 0, 'Average': 1, 'Excellent': 2}).values.astype(np.int32)
    logger.info(f"Target variable dtype: {y.dtype}")
    logger.info(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")

    # Compute class weights
    class_weight_dict = compute_class_weights(y)

    # Feature selection
    selected_features = select_features(df[CONFIG['numerical_features']], y, CONFIG['numerical_features'])
    joblib.dump(selected_features, 'selected_features.joblib')

    # Prepare numerical data
    X_numerical = df[selected_features]
    X_train_num, X_test_num, y_train, y_test = train_test_split(
        X_numerical, y, test_size=0.2, random_state=CONFIG['random_seed']
    )
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)
    joblib.dump(scaler, 'scaler.save')
    X_train_lstm = X_train_num_scaled.reshape((X_train_num_scaled.shape[0], 1, len(selected_features)))
    X_test_lstm = X_test_num_scaled.reshape((X_test_num_scaled.shape[0], 1, len(selected_features)))

    # Load MobileBERT
    try:
        tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
        mobilebert_model = TFMobileBertModel.from_pretrained('google/mobilebert-uncased')
    except Exception as e:
        logger.error(f"Error loading MobileBERT: {e}")
        raise

    # Process text
    X_train_text = encode_text(
        df.loc[X_train_num.index, CONFIG['text_feature']].tolist(),
        tokenizer, mobilebert_model, CONFIG['max_length'], CONFIG['batch_size']
    )
    X_test_text = encode_text(
        df.loc[X_test_num.index, CONFIG['text_feature']].tolist(),
        tokenizer, mobilebert_model, CONFIG['max_length'], CONFIG['batch_size']
    )
    text_scaler = StandardScaler()
    X_train_text_scaled = text_scaler.fit_transform(X_train_text)
    X_test_text_scaled = text_scaler.transform(X_test_text)
    joblib.dump(text_scaler, 'text_scaler.save')

    # Build and train model
    model = build_hybrid_model(selected_features, X_train_text.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=CONFIG['patience'], restore_best_weights=True)
    checkpoint = ModelCheckpoint('hybrid_student_model.h5', monitor='val_loss', save_best_only=True)
    history = model.fit(
        [X_train_lstm, X_train_text_scaled], y_train,
        validation_data=([X_test_lstm, X_test_text_scaled], y_test),
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        verbose=1,
        callbacks=[early_stopping, checkpoint],
        class_weight=class_weight_dict
    )

    # Log metrics
    logger.info(f"Final training loss: {history.history['loss'][-1]:.4f}")
    logger.info(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    logger.info(f"Final training precision: {history.history['precision'][-1]:.4f}")
    logger.info(f"Final training recall: {history.history['recall'][-1]:.4f}")
    logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    logger.info(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    logger.info(f"Final validation precision: {history.history['val_precision'][-1]:.4f}")
    logger.info(f"Final validation recall: {history.history['val_recall'][-1]:.4f}")

    print("Hybrid model training complete. Saved as 'hybrid_student_model.h5'.")

if __name__ == "__main__":
    main()