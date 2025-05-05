# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS # To handle requests from frontend domain
import rag_pipeline # Import your module
import logging

app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing for all routes

logging.basicConfig(level=logging.INFO)

@app.route('/recommend', methods=['POST'])
def recommend():
    """API endpoint to get clothing recommendations."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    logging.info(f"Received request data: {data}")

    # --- Input Validation ---
    required_fields = ['gender', 'skin_tone', 'height_cm', 'weight_kg', 'season']
    if not all(field in data for field in required_fields):
        return jsonify({"error": f"Missing required fields: {required_fields}"}), 400

    # --- Basic Type/Value Validation (Example) ---
    try:
        user_input = {
            "gender": str(data['gender']),
            "skin_tone": str(data['skin_tone']),
            "height_cm": int(data['height_cm']),
            "weight_kg": int(data['weight_kg']),
            "season": str(data['season'])
        }
        # Add more specific checks if needed (e.g., skin_tone in valid list)
    except (ValueError, TypeError) as e:
         return jsonify({"error": f"Invalid input data type: {e}"}), 400

    # --- Check if pipeline components loaded ---
    if rag_pipeline.product_df is None or rag_pipeline.model is None or rag_pipeline.tokenizer is None:
         logging.error("Pipeline components not ready.")
         return jsonify({"error": "Recommendation service is initializing or unavailable. Please try again later."}), 503

    try:
        # --- Run RAG Pipeline ---
        logging.info("Starting retrieval...")
        retrieved_items = rag_pipeline.retrieve_clothes(
            gender=user_input['gender'],
            skin_tone=user_input['skin_tone'],
            season_input=user_input['season']
        )
        logging.info(f"Retrieval finished. Found {len(retrieved_items)} items.")

        logging.info("Starting generation...")
        recommendation_text = rag_pipeline.generate_recommendation(
            user_input,
            retrieved_items
        )
        logging.info("Generation finished.")

        # --- Extract Image References ---
        image_filenames = rag_pipeline.extract_image_references(recommendation_text)
        logging.info(f"Extracted image references: {image_filenames}")

        # --- Format Response ---
        response = {
            "recommendation_text": recommendation_text,
            "image_references": image_filenames # Send only filenames
        }
        return jsonify(response), 200

    except Exception as e:
        logging.exception("An error occurred during the recommendation process.") # Log full traceback
        return jsonify({"error": "An internal server error occurred."}), 500


if __name__ == '__main__':
    # Host='0.0.0.0' makes it accessible on your network, not just localhost
    # Use a proper WSGI server (like Gunicorn or Waitress) for production
    app.run(debug=False, host='0.0.0.0', port=5000) # Turn debug=False for production/sharing
