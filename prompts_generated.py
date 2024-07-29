def create_prompts():
    tensorflow_prompt = """
    You are an expert in generating TensorFlow code. Follow the structure and examples provided.

    Example Task: Create a convolutional neural network for image classification.

    Example Code for Data Loading:
    ```python
    import tensorflow as tf
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    ```

    Example Code for Unzipping the dataset file:
    ```python
    import zipfile
    with zipfile.ZipFile('/content/your_data.zip', 'r') as zip_ref:
        zip_ref.extractall('/content/Dataset')
    ```

    Example Code for Creating a Dataframe:
    ```python
    import os
    import pandas as pd

    def create_image_dataframe(directory):
        data = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif')):  # Include image file extensions
                    file_path = os.path.join(root, file)
                    image_class = os.path.basename(root)
                    data.append({{"image_link": file_path, "class": image_class}})

        df = pd.DataFrame(data)
        return df
    ```

    Example Code for Resizing Images:
    ```python
    train_images = tf.image.resize(train_images, (32, 32))
    test_images = tf.image.resize(test_images, (32, 32))
    ```

    Example Code for Grouping the Images:
    ```python
    grouped_class = segmented_df.groupby(['Class']).count()
    ```

    Example Code for Plotting the Graph:
    ```python
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    bar_plot = plt.bar(x=grouped_class.index, height=grouped_class['IMAGE'])
    ```

    Example Code to Apply the Pre-processing Function to the Dataset Images:
    ```python
    train_tf_data = train_tf_data.map(load_and_preprocess_image)
    val_tf_data = val_tf_data.map(load_and_preprocess_image)
    test_tf_data = test_tf_data.map(load_and_preprocess_image)
    ```

    Example Code for Normalization:
    ```python
    train_images, test_images = train_images / 255.0, test_images / 255.0
    ```

    Example Code for Splitting the Dataset:
    ```python
    from sklearn.model_selection import train_test_split

    # Splitting the dataset into training and temp (temporary split is used for further splitting into validation and test)
    train_data, val_test_data = train_test_split(segmented_df, test_size=0.2, stratify=segmented_df['Class'], random_state=42)

    # Further splitting the temp dataset into validation and test dataset
    val_data, test_data = train_test_split(val_test_data, test_size=0.5, stratify=val_test_data['Class'], random_state=42)
    ```

    Example Code for Creating Image Tensors Data:
    ```python
    train_tf_data = tf.data.Dataset.from_tensor_slices((train_data['IMAGE'].values, train_data['Class'].values))
    val_tf_data = tf.data.Dataset.from_tensor_slices((val_data['IMAGE'].values, val_data['Class'].values))
    test_tf_data = tf.data.Dataset.from_tensor_slices((test_data['IMAGE'].values, test_data['Class'].values))
    ```

    Example Code for Loading and Preprocessing Image Data:
    ```python
    def load_and_preprocess_image(image_path, label):
        # Load and decode the image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)

        # Set the shape of the image
        image.set_shape((None, None, 3))

        # Set the size of the image
        image = tf.image.resize(image, (220, 220))

        return image, label
    ```

    Example Code for Building a TensorFlow Model:
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

    Example Task: Create a convolutional neural network with data augmentation layers.
    """

    pytorch_prompt = """
    You are an expert in generating PyTorch code. Follow the structure and examples provided.

    Example Task: Create a convolutional neural network for image classification.

    Example Code for Data Loading:
    ```python
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    ```

    Example Code for Resizing Images:
    ```python
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    ```

    Example Code for Normalization:
    ```python
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    ```

    Example Code for Building a PyTorch Model:
    ```python
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.conv3 = nn.Conv2d(64, 64, 3, 1)
            self.fc1 = nn.Linear(64*4*4, 64)
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2)
            x = x.view(-1, 64*4*4)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = Net()
    ```

    Example Code for Data Augmentation:
    ```python
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    ```

    Example Task: Create a convolutional neural network with data augmentation layers.
    """

    return tensorflow_prompt, pytorch_prompt

if __name__ == "__main__":
    tensorflow_prompt, pytorch_prompt = create_prompts()
    print("TensorFlow Prompt:", tensorflow_prompt)
    print("PyTorch Prompt:", pytorch_prompt)
