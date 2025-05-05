# AI Personalized Fashion Matching Recommendation Engine (RAG Approach)



![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)



![Flask Version](https://img.shields.io/badge/flask-2.x-brightgreen.svg)



![Transformers Version](https://img.shields.io/badge/transformers-4.x-yellow.svg)

## Project Overview

This project is an AI fashion matching recommendation system based on the Retrieval-Augmented Generation (RAG) architecture. It aims to provide personalized fashion item recommendations based on users' input, including **gender, current season, skin tone type (based on the four-season color theory), and body measurements (used to infer general size suggestions from height and weight)**. Additionally, it leverages large language models (LLMs) to generate natural-sounding recommendation rationales.

### Core Features

Accepts multi-dimensional user input.

Filters the Kaggle fashion dataset based on rules and color theory.

Optimizes LLM interaction using few-shot prompting with the Hugging Face dialogue dataset.

Calls an LLM (e.g., Google Gemma) to generate recommendation texts and rationales.

Provides services via a Flask API.

### Project Goals

Explore how to combine structured data retrieval with the generative capabilities of LLMs to offer users more accurate and interpretable personalized fashion recommendations, addressing common dilemmas like "what to wear" and "how to match outfits".

## Working Principle

*(Optional: Describe the process in words or with a simple diagram)*

**User Input**: The front-end collects the user's gender, skin tone type, height, weight, and current season.

**Size and Rule Mapping**: The back-end infers general size suggestions (S/M/L/XL) based on height and weight, maps the skin tone type to an allowed color list, and determines filtering conditions according to the season.

**Data Retrieval**: In the preprocessed Kaggle fashion product dataset (`styles.csv`), candidate fashion items are filtered based on gender, mapped colors, season, and other conditions.

**Context Augmentation**: The retrieved candidate fashion information (names, colors, image references, etc.) and few-shot examples selected from the Hugging Face dialogue dataset are formatted to construct a complete prompt context.

**Text Generation**: The prompt, which includes the user's request, few-shot examples, and retrieved context, is sent to the large language model (LLM). The LLM generates natural language text containing recommendation rationales, specific product references (including image file names), and general size suggestions.

**API Response**: The back-end API returns the generated recommendation text and the list of extracted image file names to the front-end.

## Technology Stack

**Back-end Framework**: Flask

**Core Logic**: Python 3.9+

**LLM Interaction**: Hugging Face Transformers, PyTorch

**Data Processing**: Pandas

**Datasets**:

Kaggle - Fashion Product Images (Small): Product metadata and image links ([Link to Kaggle dataset page](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small))

Hugging Face - clothes-assistant: Dialogue data for few-shot prompting ([Link to HF dataset page](https://huggingface.co/datasets/dvilasuero/clothes-assistant))

**Optional Front-end Prototype**: Android Studio (XML/Java)

**Optional LLM Model**: Google Gemma / \[Other models you use]

## Installation and Setup

**Clone the Repository**:



```
git clone https://github.com/YOUR\_USERNAME/your-repo-name.git

cd your-repo-name
```

**Create and Activate a Virtual Environment (Recommended)**:



```
\# For Windows

python -m venv venv

venv\Scripts\activate

\# For macOS/Linux

python -m venv venv

source venv/bin/activate
```

**Install Dependencies**:



```
pip install -r requirements.txt
```

**Data Preparation (Important!)**:

**Kaggle Dataset**:

Go to the Kaggle Fashion Product Images (Small) dataset page.

Download the dataset (Kaggle account required).

Place the unzipped `images` folder intact inside the `fashion-product-images-small` folder at the project root directory. The final path should be: `your-repo-name/fashion-product-images-small/images/`.

Place the `styles.csv` file from the dataset inside the `data` folder at the project root directory. The final path should be: `your-repo-name/data/styles.csv`. (Modify these instructions if the paths in the code are different.)

Note: Please comply with the license agreements of the Kaggle platform and this dataset.

**Hugging Face Dataset**:

The `dvilasuero/clothes-assistant` dataset will be automatically downloaded by the `datasets` library upon the first run. Ensure that your environment can access the Hugging Face Hub.

**Optional LLM Configuration**:

If the model you use requires authentication (e.g., Llama 3), you may need to log in using `huggingface-cli login` or set corresponding environment variables/API keys. The Gemma model used in this project usually does not require additional login.

### Running the API Service

In the project root directory, run:



```
python app.py
```

The API will start at `http://0.0.0.0:5000/` by default.

### API Usage Instructions

**Endpoint**: POST `/recommend`

**Request Body (JSON)**:



```
{

&#x20;   "gender": "Women",         // "Men", "Women", "Other" (or other values supported by the code)

&#x20;   "skin\_tone": "Summer",     // "Spring", "Summer", "Autumn", "Winter"

&#x20;   "height\_cm": 165,          // Integer, in centimeters

&#x20;   "weight\_kg": 60,           // Integer or float, in kilograms

&#x20;   "season": "Summer"         // "Spring", "Summer", "Autumn", "Winter"

}
```

**Response Body (JSON)**:



```
{

&#x20;   "recommendation\_text": "Hello! I'm happy to recommend clothing suitable for summer and summer-type skin tones... For example, this 'Blue Floral Dress (Image: 12345.jpg)' would be a great choice... Based on your height and weight, it is recommended to try size M, but please refer to the product details for specific sizing...",

&#x20;   "image\_references": \[

&#x20;       "12345.jpg",

&#x20;       "67890.jpg"

&#x20;       // ... List of image file names referenced in the LLM-generated text

&#x20;   ]

}
```

**Testing with **`curl`:



```
curl -X POST -H "Content-Type: application/json" \\

-d '{"gender": "Women", "skin\_tone": "Summer", "height\_cm": 165, "weight\_kg": 60, "season": "Summer"}' \\

http://127.0.0.1:5000/recommend
```

## Project Documentation

During the development of this project, a series of product and analysis documents were produced, reflecting a comprehensive AI product management approach. Relevant documents (such as PRD, competitor analysis, user research, evaluation and optimization plans, etc.) can be found in the `docs/` directory (if uploaded).

## Limitations

**Static Data**: Offline datasets are used, which cannot reflect real-time inventory and the latest fashion trends.

**Size Suggestions**: Height and weight are only used to infer general size suggestions (S/M/L/XL) and do not guarantee a precise fit. Users should refer to specific product size charts.

**Matching Limitations**: Currently, it mainly recommends eligible individual items, and complex full-set matching logic has not been implemented.

**Subjectivity of Color Theory**: The four-season color theory is a reference, and actual results may vary from person to person. Color mapping rules need continuous optimization.

**Performance**: There may be some latency in LLM inference, and performance optimization needs to be considered for large-scale deployment.

**Bias**: Bias in the dataset and the LLM itself may affect the recommendation results.

## Future Work

**Front-end Optimization**: Develop a more complete and user-friendly interface.

**Outfit Recommendation**: Introduce matching rules or models to achieve combined recommendations of tops, bottoms, shoes, bags, etc.

**Real-time Data**: Connect to e-commerce platform APIs to obtain real-time product information and inventory.

**User Feedback Loop**: Incorporate a user feedback mechanism to optimize the recommendation algorithm using feedback.

**A/B Testing Framework**: Build an A/B testing system to scientifically evaluate the effectiveness of different recommendation strategies.

**Model Fine-tuning**: Fine-tune the LLM for the fashion recommendation scenario to enhance professionalism and effectiveness.

**More Granular Attributes**: Introduce more dimensions of user preferences and product attributes, such as materials, styles, and occasions.

## License

The code of this project is licensed under the MIT License (if a `LICENSE` file is added).

Please note that the Kaggle and Hugging Face datasets used have their own independent license agreements, and you must comply with them when using the datasets.
