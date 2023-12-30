# 🖼️ Image to 🎧 Audio Story Generator

## 🌐 Webapp Link: https://huggingface.co/spaces/adi-123/Image-to-Audio_Story_Generator

This project showcases an end-to-end pipeline that transforms an image into an audio story using various AI models and tools.

## 🌟 Overview

The goal of this project is to leverage AI capabilities to convert an uploaded image into an audio story. 
It uses a combination of image captioning, text generation, and text-to-speech models.

## 🚀 Features

### 📷 Image Captioning
- Utilizes Salesforce's `blip-image-captioning-base` model to generate textual descriptions of uploaded images.

### ✍️ Text Generation (Story Creation)
- Employs Meta's `llama-2-70b-chat` model to create a short story influenced by the provided image caption within a positive conclusion of 100 words or less.

### 🔊 Text-to-Speech Conversion
- Utilizes Hugging Face's `espnet/kan-bayashi_ljspeech_vits` model to convert the generated story into an audio file.

### 🌐 Streamlit Web App
- Built using Streamlit, allowing users to upload images and visualize the generated image caption, story, and audio.

## 📝 Usage

To use this application:

1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Set up the necessary environment variables:
   - `TOGETHER_API_KEY`: TOGETHER AI API key.
   - `HUGGINGFACEHUB_API_TOKEN`: Hugging Face API token.
4. Run the Streamlit app with `streamlit run app.py`.
5. Upload an image file (supported formats: jpg, jpeg, png).
6. Wait for the AI processing to generate the story and audio.
7. Access the image caption, story, and audio outputs.

## 📁 Code Structure

- `app.py`: Contains the Streamlit web application code, integrating all functionalities.
- `README.md`: Documentation explaining the project, usage instructions, and dependencies.
- `requirements.txt`: Lists all necessary libraries.

## 🙌 Credits

This project was created with love by @Aditya-Neural-Net-Ninja. 
It makes use of cutting-edge AI models for image analysis, natural language processing, and text-to-speech conversion. 
Special thanks to Streamlit and Hugging Face for their incredible platforms.


**Note:** Please ensure you have the required API keys and tokens for TOGETHER AI and Hugging Face to run this application successfully.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
