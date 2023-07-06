## Importing necessary libraries

    import time
    from tensorflow import keras
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore')

## Installing and importing the keras_cv library

    !pip install keras_cv
    import keras_cv.models

## Initializing the StableDiffusion model

    model = keras_cv.models.StableDiffusion(img_width=128, img_height=128)

## Generating and displaying an image batch of 'sunrise over beach'

    # Record the start time before generating image
    start_time = time.time()

    img1 = model.text_to_image("sunrise over beach", batch_size=100)

    # Print the time taken to generate the image
    print("Time taken to generate 'sunrise over beach': ", time.time() - start_time)

    plt.imshow(img1[0])
    plt.axis("off")

## Generating and displaying an image batch of 'pencil sketch of dog'

    # Record the start time before generating image
    start_time = time.time()

    img2 = model.text_to_image("pencil sketch of dog", batch_size=100)

    # Print the time taken to generate the image
    print("Time taken to generate 'pencil sketch of dog': ", time.time() - start_time)

    plt.imshow(img2[0])
    plt.axis("off")

## Generating and displaying an image batch of 'white car running on road'

    # Record the start time before generating image
    start_time = time.time()

    img3 = model.text_to_image("white car running on road", batch_size=100)

    # Print the time taken to generate the image
    print("Time taken to generate 'white car running on road': ", time.time() - start_time)

    plt.imshow(img3[0])
    plt.axis("off")

## Generating & displaying image batch of 'cow on moon'

    # Record the start time before generating image
    start_time = time.time()

    img4 = model.text_to_image("cow on moon", batch_size=100)

    # Print the time taken to generate the image
    print("Time taken to generate 'cow on moon': ", time.time() - start_time)

    plt.imshow(img4[0])
    plt.axis("off")

'''
Conclusion: 
While working with the StableDiffusion model from the keras_cv library,
I realized that creating images from text descriptions is a remarkable
feature but it comes with its own set of challenges:

-   Execution Environment: The output of my project was heavily
    influenced by the environment in which the code was executed. I
    found that the quality of the generated images and the time taken to
    produce them were directly linked to the computational power of my
    machine. It dawned on me that using a more powerful GPU could
    potentially yield better results.

-   Encountered Errors: During my journey with this project, I came
    across a ResourceExhaustedError. This was a wake-up call to the fact
    that my GPU was running out of memory, mainly because I was
    attempting to generate a massive batch of 10000 images. Reducing the
    batch size was a necessary step I had to take to resolve this issue.

-   Improving Image Quality: I discovered a few pathways to potentially
    improve the quality of the generated images:

    -   Hardware: I realized that executing the script on a powerful GPU
        can considerably enhance the image generation process.
    -   Model Parameters: Tweaking the model parameters was another
        lesson learned. Adjusting factors like image dimensions and
        batch size could lead to an improvement in the quality of the
        generated images.
    -   Dependencies: Lastly, it was crucial for me to ensure that all
        dependencies were compatible before executing the script to
        avoid any conflicts that could arise.

This project was a great reminder that the journey of learning machine
learning is about constant exploration, encountering errors, and
improving. Every step, every mistake, is a chance to learn and grow! 
'''
