from transformers import MobileBertTokenizer, TFMobileBertModel
import tensorflow as tf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mobilebert():
    try:
        # Load MobileBERT tokenizer and model
        logger.info("Loading MobileBERT tokenizer and model...")
        tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
        model = TFMobileBertModel.from_pretrained('google/mobilebert-uncased')
        logger.info("MobileBERT components loaded successfully.")

        # Test text encoding
        test_text = "This is a sample student feedback for testing MobileBERT."
        logger.info(f"Encoding test text: {test_text}")
        inputs = tokenizer(test_text, return_tensors="tf", padding=True, truncation=True, max_length=32)
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        logger.info(f"Text embeddings shape: {embeddings.shape}")

        print("MobileBERT test passed successfully!")
    except Exception as e:
        logger.error(f"MobileBERT test failed: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_mobilebert()