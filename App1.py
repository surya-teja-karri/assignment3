import streamlit as st
import cv2
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import EfficientNetB0
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
import matplotlib.offsetbox as offsetbox
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sklearn.model_selection import train_test_split



# Define label_set
label_set = ['Dress', 'Hat', 'Longsleeve', 'Shoes', 'T-Shirt']

# Function to decode labels
def decode_labels(y, mlb):
    labels = np.array(mlb.inverse_transform(np.atleast_2d(y)))[:, 0]
    return labels

# Function to decode label probabilities
def decode_label_prob(y, classes):
    labels = []
    for i, c in enumerate(classes):
        labels.append(f'{c}: {y[i]:.2%}')
    return labels

# Load Images and Labels
def load_images_and_labels():
    base_path = '/Users/tanayparikh/Desktop/Assignment3 Part-1/clothing-dataset-master'
    df = pd.read_csv(base_path + '/images.csv')
    df = df[df['label'].isin(['Shoes', 'Dress', 'Longsleeve', 'T-Shirt', 'Hat'])]

    data, labels = [], []
    for index, row in df.iterrows():
        image_path = f'{base_path}/images/{row["image"]}.jpg'
        label = row['label']
        try:
            image = tf.keras.preprocessing.image.load_img(image_path, color_mode='rgb', target_size=(224, 224))
            image = np.array(image)
            data.append(image)
            labels.append(label)
        except Exception:
            print(f'Failed to load {image_path}')

    x = np.array(data)
    y = np.array(labels)

    x = x.astype('float32')

    mlb = MultiLabelBinarizer()
    mlb.fit([label_set])
    y = mlb.transform(np.atleast_2d(y).T)

    train_test_ratio = 0.75
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_test_ratio)

    return x_train, x_test, y_train, y_test, mlb

# Fit a Classification Model
def fit_classification_model(x_train, x_test, y_train, y_test, mlb):
    model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False

    mapper = model.output
    mapper = Flatten()(mapper)
    mapper = Dense(32, activation='relu', name='embeddings')(mapper)
    mapper = Dense(y_train.shape[1], activation='softmax')(mapper)
    transfer_model = Model(inputs=model.input, outputs=mapper)

    learning_rate = 0.001
    transfer_model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=["accuracy"])
    history = transfer_model.fit(x_train, y_train, batch_size=4, epochs=4, validation_data=(x_test, y_test))

    return transfer_model

# Visualize image class probabilities
def visualize_image_probabilities(transfer_model, x_test, mlb):
    n_col, n_row = 6, 4
    f, ax = plt.subplots(n_row, n_col, figsize=(12, 14))
    for i in range(n_row):
        for j in range(n_col):
            idx = n_col * i + j
            class_probs = transfer_model.predict(x_test[idx:idx+1])[0]
            ax[i, j].imshow(x_test[idx] / 255)
            ax[i, j].set_axis_off()
            ax[i, j].set_title('\n'.join(decode_label_prob(class_probs, mlb.classes_)))

    return f

# Compute embeddings for visualization
def compute_embeddings(x_test, y_test, mlb):
    n_samples = 1000
    embedding_outputs = transfer_model.get_layer('embeddings').output
    embedding_model = Model([transfer_model.input], embedding_outputs)

    images = x_test[0:n_samples]
    image_embeddings = embedding_model.predict(images)
    image_labels = decode_labels(y_test[0:n_samples], mlb)

    return image_embeddings, image_labels

# Visualize the 2D-projection of the embedding space with example images
def visualize_embedding_space(image_embeddings, image_labels, images):
    tsne = manifold.TSNE(n_components=2, init='pca', perplexity=20, random_state=0)
    X_tsne = tsne.fit_transform(image_embeddings)

    return X_tsne

# Streamlit App
st.title("Visual Search and Embedding Space Visualization")

# Load data and labels
x_train, x_test, y_train, y_test, mlb = load_images_and_labels()

# Fit Classification Model
transfer_model = fit_classification_model(x_train, x_test, y_train, y_test, mlb)

# Visualize Image Class Probabilities
st.subheader("Image Class Probabilities")
fig = visualize_image_probabilities(transfer_model, x_test, mlb)
st.pyplot(fig)

# Compute Embeddings for Visualization
image_embeddings, image_labels = compute_embeddings(x_test, y_test, mlb)

# Visualize Embedding Space
X_tsne = visualize_embedding_space(image_embeddings, image_labels, x_test)

st.pyplot(plt.figure(figsize=(12, 12)))


