📚 Importing Necessary Libraries

    import warnings
    warnings.filterwarnings('ignore')

    from tensorflow import keras
    import matplotlib.pyplot as plt

🔧 Installing and Importing the keras_cv Library

    !pip install keras_cv --upgrade --quiet
    import keras_cv.models

🏁 Initializing the StableDiffusion Model

    model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)

🌅 Generating & Displaying Images

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
