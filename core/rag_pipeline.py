# rag_pipeline.py
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import os
import re
import logging # Use logging instead of print for server apps

logging.basicConfig(level=logging.INFO)

# --- Define Paths (Adjust as needed for your server environment) ---
# Assume the script runs from the root of fashion-rag-api/
# IMPORTANT: The image dir path must be correct where the server runs!
KAGGLE_CSV_PATH = os.path.join('data', 'styles.csv') # If you include CSV in repo
# OR KAGGLE_CSV_PATH = '/path/on/server/to/styles.csv' # If data is mounted elsewhere
KAGGLE_IMAGES_DIR = 'fashion-product-images-small/images' # Relative path example
# OR KAGGLE_IMAGES_DIR = '/path/on/server/to/fashion-product-images-small/images'

# --- Mapping Rules (Keep these) ---
SKIN_TONE_COLOR_MAP = { ... }
# ... other helper functions ...

# --- Load Data and Models (Load ONCE at startup) ---
try:
    logging.info("Loading Kaggle product data...")
    product_df = pd.read_csv(KAGGLE_CSV_PATH, on_bad_lines='skip')
    product_df['image_path'] = product_df['id'].apply(lambda x: os.path.join(KAGGLE_IMAGES_DIR, str(x) + '.jpg'))
    product_df = product_df.dropna(subset=['gender', 'baseColour', 'season', 'articleType', 'productDisplayName'])
    logging.info(f"Loaded Kaggle product data: {product_df.shape[0]} rows")
except FileNotFoundError:
    logging.error(f"FATAL ERROR: Kaggle CSV not found at {KAGGLE_CSV_PATH}. Exiting.")
    product_df = None # Or raise SystemExit
except Exception as e:
    logging.error(f"An error occurred loading Kaggle data: {e}")
    product_df = None # Or raise SystemExit

try:
    logging.info("Loading Hugging Face dataset for few-shot examples...")
    hf_dataset = load_dataset("dvilasuero/clothes-assistant")
    few_shot_prompt_examples = format_hf_examples_for_prompt(hf_dataset, num_examples=2) # Call your formatting function
    logging.info("Few-shot examples prepared.")
except Exception as e:
    logging.warning(f"Could not load or process HF dataset: {e}")
    hf_dataset = None
    few_shot_prompt_examples = "" # Ensure it's an empty string if failed

try:
    logging.info("Loading LLM model and tokenizer...")
    model_id = "google/gemma-2b-it" # Or your chosen model
    # Optional quantization config
    # bnb_config = BitsAndBytesConfig(...)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        # quantization_config=bnb_config # Uncomment if needed
    )
    logging.info(f"LLM ({model_id}) loaded successfully.")
except Exception as e:
    logging.error(f"FATAL ERROR: Could not load LLM model/tokenizer: {e}. Exiting.")
    model = None
    tokenizer = None
    # raise SystemExit # Stop the app if model can't load

# --- Core Functions (Keep these as they are, using global vars now) ---
def retrieve_clothes(gender, skin_tone, season_input, max_results=10):
    # Uses the global product_df
    if product_df is None:
         logging.error("Product data not loaded, cannot retrieve.")
         return pd.DataFrame()
    logging.info(f"Retrieving for: Gender={gender}, Skin={skin_tone}, Season={season_input}")
    # ... (rest of the retrieve_clothes function) ...
    # Ensure it returns the dataframe

def generate_recommendation(user_input, retrieved_df): # Removed LLM/tokenizer/few_shot args
    # Uses the global model, tokenizer, few_shot_prompt_examples
    if model is None or tokenizer is None or product_df is None:
        logging.error("Model, tokenizer, or product data not loaded. Cannot generate.")
        return "Sorry, the recommendation service is currently unavailable."

    context = format_retrieved_context(retrieved_df)
    size_suggestion = get_size_suggestion(user_input['height_cm'], user_input['weight_kg'])
    prompt = f"""{few_shot_prompt_examples}
--- Current Task ---
# ... (rest of the prompt construction) ...
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(model.device)
    # ... (model.generate call) ...
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # ... (extract recommendation_text) ...
    return recommendation_text

def extract_image_references(recommendation_text):
    """Extracts image filenames referenced in the recommendation text."""
    image_filenames = re.findall(r'[Ii]mage\s*[:\-]?\s*(\d+\.jpg)', recommendation_text)
    return image_filenames
