# Tumor detection
## Tumor classification:
* normal brain without tumor
* Astrocitoma
* Carcinoma
* Ependimoma
* Gandlioglioma
* Germinoma 
* Glioblastoma 
* Granuloma 
* Meduloblastoma
* Meningioma 
* Neurocitoma 
* Oligodendroglioma 
* Papiloma 
* Schwannoma
* Tuberculoma

## Model summary:

<table>
    <tr>
        <th>Layer Number</th>
        <th>Layer Name</th>
        <th>Output Shape</th>
        <th>Number of Parameters</th>
    </tr>
    <tr>
        <td>1</td>
        <td>Conv2d (relu)</td>
        <td>(None, 296, 296, 32)</td>
        <td>2,432</td>
    </tr>
    <tr>
        <td>2</td>
        <td>MaxPooling2D</td>
        <td>(None, 148, 148, 32)</td>
        <td>0</td>
    </tr>
    <tr>
        <td>3</td>
        <td>Conv2D (relu)</td>
        <td>(None, 146, 146, 64)</td>
        <td>18,496</td>
    </tr>
    <tr>
        <td>4</td>
        <td>MaxPooling2D</td>
        <td>(None, 73, 73, 64)</td>
        <td>0</td>
    </tr>
    <tr>
        <td>5</td>
        <td>Flatten</td>
        <td>(None, 341056)</td>
        <td>0</td>
    </tr>
    <tr>
        <td>6</td>
        <td>Dense (leaky_relu)</td>
        <td>(None, 64)</td>
        <td>21,827,648</td>
    </tr>
    <tr>
        <td>7</td>
        <td>Dense (softmax)</td>
        <td>(None, 15)</td>
        <td>975</td>
    </tr>
</table>

### Params summary:
* Total params: 65,548,655 (250.05 MB)<br>
* Trainable params: 21,849,551 (83.35 MB)<br>
* Optimizer params: 43,699,104 (166.70 MB)<br>

### Compiler Arguments:
* optimizer: adam
* loss: categorical crossentropy

# Building a Brain Tumor Classification Model with TensorFlow

This article walks through the process of building a machine learning model to classify different types of brain tumors using TensorFlow and Keras. The provided Python code serves as a comprehensive guide to preparing data, training a model, and evaluating its performance.

## Understanding the Code

### Importing Necessary Libraries

The first step is to import the necessary libraries:

```python
import tensorflow
from tensorflow import keras
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
```

### Defining Classes and Custom Exceptions

We define the different classes of brain tumors and a custom exception for handling invalid training folders:

```python
# Class codes
classes = ["nor", "ast", "car", "epe", "gan", "ger", "gli", "gra", "med", "men", "neu", "oli", "pap", "sch", "tub"]

class FolderNotFoundException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.additional_info = "Exception: " + message
```

### Helper Functions

Several helper functions are defined to handle various tasks:

1. **Extracting Class from Filename**: This function extracts the class code from a given file name.
   
    ```python
    def giveMeAnswer(file_name):
        first_alpha_sequence = ""
        for char in file_name:
            if char.isalpha():
                first_alpha_sequence += char
            else:
                break
        return first_alpha_sequence
    ```

2. **Preparing Training Folders**: Renames files in the training folder to a standardized format.

    ```python
    def prepareTrainFolder(folder):
        if not os.path.isdir(folder):
            raise FolderNotFoundException("The path specified does not exist or is not a folder.")
        files = os.listdir(folder)
        for idx, file in enumerate(files):
            new_name = f"{giveMeAnswer(folder[19:].lower())}_{idx + 1}"
            old_path = os.path.join(folder, file)
            new_path = os.path.join(folder, new_name)
            os.rename(old_path, new_path)
    ```

3. **Preparing All Training Data**: Applies `prepareTrainFolder` to all subfolders in the main training directory.

    ```python
    def prepareTrainingData(training_folder="training datasets/"):
        folders = os.listdir(training_folder)
        for folder in folders:
            prepareTrainFolder(f'{training_folder}/{folder}')
    ```

4. **Class Code Conversion**: Converts class codes to numerical labels.

    ```python
    def getClass(code):
        for i, t in enumerate(classes):
            if t == code:
                return i
        raise IndexError("Wrong class code or does not exist: " + code)
    ```

5. **Loading Images**: Loads and preprocesses images from a given folder.

    ```python
    def loadImagesFromFolder(folder_path, target_size=None):
        images = []
        ans = []
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            if os.path.isfile(img_path):
                im = image.load_img(img_path, target_size=target_size)
                img_array = image.img_to_array(im)
                ans.append(getClass(filename[:3].lower()))
                images.append(img_array)
        return images, ans
    ```

### Preparing Training Data

The `getTrainingData` function loads all training data and returns it in a format suitable for training a neural network:

```python
def getTrainingData():
    folder_name = 'training datasets/'
    folders = os.listdir(folder_name)
    all_photos = []
    all_labels = []
    for folder in folders:
        curr_folder_name = f'{folder_name}{folder}'
        temp, ans = loadImagesFromFolder(curr_folder_name, (300, 300))
        all_photos.extend(temp)
        all_labels.extend([ans[0]] * len(temp))
    all_labels = to_categorical(all_labels, num_classes=len(classes))
    return np.array(all_photos), np.array(all_labels)
```

### Training the Model

The `train_new_model` function creates, trains, and saves a new model:

```python
def train_new_model(e):
    X, y = getTrainingData()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(300, 300, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='leaky_relu'))
    model.add(Dense(15, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=e, batch_size=32, validation_data=(X_test, y_test))
    
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Loss: {loss}, acc: {accuracy}")
    
    validation_accuracy = history.history['val_accuracy']
    model.save('tumor_recognition.keras')

    return model, validation_accuracy[-1]
```

### Main Function

The main function checks if a pre-trained model exists. If not, it trains a new model:

```python
def main():
    if os.path.exists('tumor_recognition.keras') and False:
        model = keras.models.load_model('tumor_recognition.keras')
        model.summary()
    else:
        model, acc = train_new_model(5)
        print(f"Validation Accuracy: {acc}")
        model.summary()

if __name__ == "__main__":
    main()
```

## Conclusion

This code provides a detailed approach to building a neural network for classifying different types of brain tumors. It covers data preparation, model creation, training, and evaluation. By following the steps outlined, you can train a model to accurately classify brain tumors based on medical imaging data.


