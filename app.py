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
    You are an expert in generating TensorFlow code! Follow the structure and examples provided. Ensure that the code follows these steps:

    1. Data Loading: Load the dataset using TensorFlow's built-in methods.
    2. Data Preprocessing: Apply data augmentation and normalization.
    3. Model Building: Create the neural network architecture.
    4. Model Compilation: Compile the model with an optimizer, loss function, and evaluation metrics.
    5. Model Training: Train the model with a specified number of epochs and batch size.
    6. Model Evaluation: Evaluate the model on the test dataset.
    7. Model Saving: Save the trained model to a file.
    8. Code Smell Detection: Identify any code smells in the TensorFlow code and suggest improvements.
    9. Security Flaw Detection: Detect any security flaws in the TensorFlow code and provide recommendations to fix them.

    Example Task: Create a convolutional neural network for image classification.

    Example Code for Data Loading:
    ```python
    import tensorflow as tf
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    ```

    Example Code for Data Preprocessing:
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

    Example Code for Model Building:
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

    Example Code for Model Compilation:
    ```python
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    ```

    Example Code for Model Training:
    ```python
    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))
    ```

    Example Code for Model Evaluation:
    ```python
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    ```

    Example Code for Model Saving:
    ```python
    model.save('my_model.h5')
    ```

    Example Code Smell Detection:
    ```python
    # Code Smell: Using hardcoded learning rate
    # Suggested Improvement: Define learning rate as a variable and pass it to the optimizer
    learning_rate = 0.001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    ```

    Example Security Flaw Detection:
    ```python
    # Security Flaw: Saving model without encryption
    # Suggested Improvement: Save model with encryption for sensitive models
    
    from cryptography.fernet import Fernet
    
    # Save the model to a temporary file
    model.save('temp_imageModel.h5')

    # Generate a key for encryption
    key = Fernet.generate_key()
    fernet = Fernet(key)

    # Read the model file and encrypt it
    with open('temp_imageModel.h5', 'rb') as file:
        original = file.read()

    encrypted = fernet.encrypt(original)

    # Write the encrypted model to a new file
    with open('encrypted_imageModel.h5', 'wb') as encrypted_file:
        encrypted_file.write(encrypted)

    print("Model saved and encrypted. Encryption key:", key.decode())
    ```

    Example Step to Decrypt the Model File:
    ```python
    # Read the encrypted model file
    with open('encrypted_imageModel.h5', 'rb') as encrypted_file:
        encrypted = encrypted_file.read()

    # Decrypt the model file
    decrypted = fernet.decrypt(encrypted)

    # Write the decrypted model to a temporary file
    with open('temp_model_decrypted.h5', 'wb') as decrypted_file:
        decrypted_file.write(decrypted)

    # Load the model
    model = tf.keras.models.load_model('temp_model_decrypted.h5')

    print("Model loaded successfully")
    ```

    Alternative Step: Save Model to Google Cloud Storage (GCS):
    ```python
    from google.cloud import storage

    # Save the model to a temporary file
    model.save('temp_imageModel.h5')

    # Upload the model to Google Cloud Storage
    client = storage.Client()
    bucket = client.bucket('your_bucket_name')
    blob = bucket.blob('models/temp_imageModel.h5')
    blob.upload_from_filename('temp_imageModel.h5')

    print("Model saved to Google Cloud Storage.")
    ```

    Example Step to Load Model from Google Cloud Storage (GCS):
    ```python
    from google.cloud import storage
    import tensorflow as tf

    # Download the model from Google Cloud Storage
    client = storage.Client()
    bucket = client.bucket('your_bucket_name')
    blob = bucket.blob('models/temp_imageModel.h5')
    blob.download_to_filename('temp_imageModel_downloaded.h5')

    # Load the model
    model = tf.keras.models.load_model('temp_imageModel_downloaded.h5')

    print("Model loaded from Google Cloud Storage successfully.")
    ```
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
