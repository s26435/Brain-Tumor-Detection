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

# class codes
classes = ["nor",  # normal brain without tumor
           "ast",  # Astrocitoma
           "car",  # Carcinoma
           "epe",  # Ependimoma
           "gan",  # Gandlioglioma
           "ger",  # Germinoma
           "gli",  # Glioblastoma
           "gra",  # Granuloma
           "med",  # Meduloblastoma
           "men",  # Meningioma
           "neu",  # Neurocitoma
           "oli",  # Oligodendroglioma
           "pap",  # Papiloma
           "sch",  # Schwannoma
           "tub"]  # Tuberculoma


class FolderNotFoundException(Exception):
    """
    Exception thrown when there is invalid training folder(s)
    """
    def __init__(self, message):
        super().__init__(message)
        self.additional_info = "Exeption: " + message


def giveMeAnswer(file_name):
    """
    :param file_name: name of target file
    :return: name of class from file name
    """
    first_alpha_sequence = ""
    for char in file_name:
        if char.isalpha():
            first_alpha_sequence += char
        else:
            break
    return first_alpha_sequence


def prepareTrainFolder(folder):
    """
    Function prepars training folder
    :param folder: path to target folder
    """
    if not os.path.isdir(folder):
        raise FolderNotFoundException("The path specified does not exist or is not a folder.")
    files = os.listdir(folder)
    print(folder)
    for idx, file in enumerate(files):
        print(file)
        new_name = f"{giveMeAnswer(folder[19:].lower())}_{idx + 1}"
        old_path = os.path.join(folder, file)
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)


def prepareTrainingData(training_folder="training datasets/"):
    """
    Function prepars all training folders
    :param training_folder: path to target folder that contains all training data
    """
    folders = os.listdir(training_folder)
    for folder in folders:
        prepareTrainFolder(f'{training_folder}/{folder}')


def getClass(code):
    """
    :param code: class code
    :return: numeral class code
    """
    for i, t in enumerate(classes):
        if t == code:
            return i
    raise IndexError("Wrong class code or does not exist: " + code)


def loadImagesFromFolder(folder_path, target_size=None):
    """
    Loads all images from target folder
    :param folder_path: path
    :param target_size: size of resized images
    :return: images (numpy array), table of answers to images
    """
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


def getTrainingData():
    """
    Loads all training data from all the folders
    :return: images(numpy array), y(numpy array)
    """
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


def train_new_model(e):
    """
    Creates, train and saves new model using training folder.
    :param e: number of epochs
    :return: trained model, validation accuracy
    """
    X, y = getTrainingData()
    print(X.shape)
    print("Data downloaded!")
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
    X_train, X_test, y_train, y_test = None, None, None, None
    return model, validation_accuracy[-1]


def main():
    if os.path.exists('tumor_recognition.keras') and False:
        model = keras.models.load_model('tumor_recognition.keras')
        model.summary()
    else:
        model, acc = train_new_model(5)
        print(f": Validation Accuracy: {acc}")
        model.summary()


if __name__ == "__main__":
    main()
