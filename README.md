Scribe's Journey: Manuscript Style Search (v1.0)

This project is a deep learning pipeline that can analyze the handwriting style (scribal style) of ancient manuscripts. It uses a Siamese network (trained with Triplet Loss) to generate a unique "fingerprint" for any given piece of handwriting.

This allows you to build a searchable database of scribal styles and identify the most likely author of a new, unknown manuscript page by comparing its "fingerprint" to the ones in your archive.

Features (v1.0 PoC)

Data Pipeline: A robust, multi-stage data preparation pipeline that binarizes, filters, and augments manuscript images to prevent model "cheating."

Encoder Model: A Siamese network (Encoder) trained with Triplet Loss to generate 512-dimension embedding vectors ("fingerprints") for handwriting patches.

Search Index: A high-speed FAISS index for performing efficient, large-scale similarity searches on thousands of fingerprints.

Web Application: A simple Streamlit app where you can upload a new manuscript page. The app analyzes the page, finds the most similar patches from the index, and predicts the most likely scribe based on a "voting" system.

Workflow & How to Run

This repository contains source code only. You must generate your own data and models by following this workflow.

Step 0: Setup

Clone the repository:

git clone [https://github.com/hellcsvishal/scribe-identification.git]
cd scribe-identification


Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate


Install all required libraries:

pip install -r requirements.txt


Step 1: Add Your Data

This is the only manual step. You must provide the raw images.

Create your scribe folders inside data/raw/. For example:

mkdir -p data/raw/scribe_A
mkdir -p data/raw/scribe_B


Place your manuscript images for the first scribe (e.g., from Sundarkand) into data/raw/scribe_A.

Place your images for the second scribe (e.g., from Uttarkand) into data/raw/scribe_B.

Step 2: Run the Data Preparation Pipeline

This script will process your raw images into a clean, binarized, and filtered dataset.

python scripts/prepare_data.py


This will populate the data/processed/ folder.

Step 3: Train the Encoder Model

This script trains the Siamese network on your new dataset. This will take some time and is best run on a machine with a GPU.

python scripts/train_siamese.py


This will create the models/best_encoder.pth file.

Step 4: Build the Search Index

This script uses your trained encoder to "fingerprint" every patch in your dataset and builds the FAISS search database.

python scripts/create_index.py


This will create scribe_index.faiss, image_paths.pkl, and image_labels.pkl in the models/ folder.

Step 5: Run the Application

You are now ready to run the final Streamlit app!

streamlit run app/app.py


Your web browser will automatically open with the application.
