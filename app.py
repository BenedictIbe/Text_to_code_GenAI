from dotenv import load_dotenv
load_dotenv()  # Load all the environment variables

import streamlit as st
import os
import google.generativeai as genai

# Configure genai key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load Google Gemini model and provide queries as response
def get_gemini_response(question, prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([prompt[0], question])
    return response.text

# Define your prompt
prompt = [
    """
    You are an expert in generating TensorFlow code! Follow the structure and examples provided.

    Example Task: Create a convolutional neural network for image classification.

    Example Code for Data Loading:
    ```python
    import tensorflow as tf
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    ```

    Example Code for Data Augmentation:
    ```python
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    datagen.fit(train_images)
    ```

    Example Code for Creating a Simple Neural Network:
    ```python
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    ```

    Also, you can provide more specific tasks to generate corresponding TensorFlow code.
    """
]

# Streamlit app
st.set_page_config(page_title="TensorFlow Code Generator")
st.header("Gemini App to Generate TensorFlow Code")

question = st.text_input("Describe your task: ", key="input")

submit = st.button("Generate Code")

# If Submit is clicked
if submit:
    response = get_gemini_response(question, prompt)
    st.subheader("Generated TensorFlow Code")
    st.code(response, language='python')
