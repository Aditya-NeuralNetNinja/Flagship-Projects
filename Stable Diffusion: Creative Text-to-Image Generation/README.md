# Stable Diffusion: Creative Text-to-Image Generation 🎨🔠

This project delves into the realm of Stable Diffusion, uncovering its potential in the domain of text-to-image generation. 
Through the utilization of KerasCV, a powerful toolset is employed to create visual representations from textual prompts. 
By seamlessly merging natural language processing and computer vision, this project showcases the transformative 
possibilities of Stable Diffusion in enabling novel creative synthesis. 

## Table of Contents 📑

- [Pre-requisites](#pre-requisites)
- [Installation](#installation)
- [Usage](#usage)
- [Example Images](#example-images)

## Pre-requisites 🧩

- Python 3.6 or higher
- TensorFlow 2.0 or higher
- Matplotlib
- keras_cv

## Usage 🎛️

Follow the steps below to run the Stable Diffusion model and generate creative images from text:

1. **Import necessary libraries**
```python
import warnings
warnings.filterwarnings('ignore')

from tensorflow import keras
import matplotlib.pyplot as plt
```
2. **Install and import the keras_cv library**
```python
!pip install keras_cv --upgrade --quiet
import keras_cv.models
```
3. **Initialize the StableDiffusion model**
```python
model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
```
4. **Generate & display images**
```python
img = model.text_to_image('''Zen garden in bloom with
a mountain range in the background''', batch_size=2)

def plot_images(img):
    plt.figure(figsize=(10,10))
    for i in range(len(img)):
        ax = plt.subplot(1, len(img), i + 1)
        plt.imshow(img[i])
        plt.axis("off")

plot_images(img)

img1 = model.text_to_image('''
A bustling cityscape at sunset with skyscrapers dominating the skyline,
the sun casting long shadows and the city lights just starting to twinkle.
In the foreground, a park with a serene lake reflecting the glowing sky''',
                           batch_size=1)

plt.figure(figsize=(10,10))
plt.imshow(img1[0])
plt.axis("off")

img2 = model.text_to_image("a red car flying in air with birds", batch_size=1)

plt.figure(figsize=(10,10))
plt.imshow(img2[0])
plt.axis("off")
```

## Example Images 🌄

Here are some example images generated from the text using the StableDiffusion model:

1. ![zen](https://github.com/Aditya-NeuralNetNinja/Flagship-Projects/assets/108260519/09e86163-35eb-4ced-ac57-b6f0271ea3d4)
   
   _Zen garden in bloom with a mountain range in the background_

2. ![skyscrapers](https://github.com/Aditya-NeuralNetNinja/Flagship-Projects/assets/108260519/0033bb48-3230-45fe-8e8f-8dcdf6b5266c)

   _A bustling cityscape at sunset with skyscrapers dominating the skyline, the sun casting long shadows and the city lights just starting to twinkle. In the foreground, a park with a serene lake reflecting the glowing sky_

3. ![flying car](https://github.com/Aditya-NeuralNetNinja/Flagship-Projects/assets/108260519/fff58441-7962-48b9-a01d-28993e39de39)

  _A red car flying in air with birds_

Enjoy creating images with text! 🖼️🎉
