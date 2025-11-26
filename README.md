Scribe's Journey: Manuscript Style Search (v2.0)

This project is a complete, end-to-end deep learning pipeline that identifies the scribes of ancient manuscripts based on their unique handwriting styles.

Unlike traditional OCR (which reads what is written), this tool analyzes how it is written. It uses a ResNet-18 Siamese Network (trained with Triplet Loss) to generate a unique "fingerprint" (embedding) for any given piece of handwriting.

This allows you to build a searchable database of scribal styles and identify the most likely author of a new, unknown manuscript page by comparing its "fingerprint" to the ones in your archive.

Features (v2.0)

Advanced Architecture: Upgraded to a pre-trained ResNet-18 backbone, fine-tuned with Triplet Loss to create robust 512-dimension stylometric embeddings.

Robust Data Pipeline: Automated pre-processing pipeline including Otsu's Binarization, text-density filtering, and data augmentation (erosion/dilation/blur) to prevent model overfitting and "cheating."

Smart Search: Implements a FAISS (Facebook AI Similarity Search) index for millisecond-level retrieval.

Intelligent Application: A Streamlit web app that includes:

Unknown Scribe Detection: Uses a confidence threshold to reject matches that are too stylistically distant.

Voting Mechanism: Aggregates results from multiple patches to predict the most likely scribe.

Version History

v2.0 (Current): Major upgrade to ResNet-18 architecture. Added "Unknown Scribe" detection logic and voting mechanism in the web app.

v1.0 (Proof of Concept): Initial release using a custom 2-layer CNN. Established the core pipeline of binarization, patch generation, and basic FAISS indexing. Validated the "handwriting fingerprint" hypothesis.

Workflow & How to Run

This repository contains source code only. You must generate your own data and models by following this workflow.

Step 0: Setup

Clone the repository:

git clone [https://github.com/hellcsvishal/scribe-identification.git](https://github.com/hellcsvishal/scribe-identification.git)
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

This script trains the ResNet-18 Siamese Network on your new dataset. This will take some time and is best run on a machine with a GPU.

python scripts/train_siamese.py


This will create the models/best_resnet_encoder.pth file.

Step 4: Build the Search Index

This script uses your trained encoder to "fingerprint" every patch in your dataset and builds the FAISS search database.

python scripts/create_index.py


This will create scribe_index.faiss, image_paths.pkl, and image_labels.pkl in the models/ folder.

Step 5: Run the Application

You are now ready to run the final Streamlit app!

streamlit run app/app.py


Your web browser will automatically open with the application.
