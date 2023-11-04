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
from sklearn.model_selection import train_test_spli

# Initialize the model (outside the Streamlit functions to avoid reloading)
optimizer = tf.keras.optimizers.Adam(1e-3)
model = VAE(
    enc=encoder,
    dec=decoder,
    optimizer=optimizer,
)

# Hyperparameters
input_shape = (28, 28, 1)  # Fashion MNIST images size
batch_size = 64
latent_dim = 2  # Dimensionality of the latent space
epochs = 50
learning_rate = 0.001

# Encoder network
inputs = Input(shape=input_shape, name='encoder_input')
x = Flatten()(inputs)
x = Dense(128, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# Instantiate encoder
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# Decoder network
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(128, activation='relu')(latent_inputs)
x = Dense(28 * 28, activation='relu')(x)
x = Reshape((28, 28, 1))(x)
outputs = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

# Instantiate decoder
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

# VAE loss = mse_loss or xent_loss + kl_loss
reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
reconstruction_loss *= input_shape[0] * input_shape[1]
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer=Adam(learning_rate=learning_rate))
vae.summary()

# Train the VAE
(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0],) + input_shape)
x_test = x_test.reshape((x_test.shape[0],) + input_shape)

vae.fit(x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None))

# Save the model
vae.save('vae_fashion_mnist.h5')

# Comment out the training process if the model is already trained and simply load the model.

# for epoch in range(n_epochs):
#     for batch, train_x in zip(range(N_TRAIN_BATCHES), train_dataset):
#         model.train(train_x)

# Streamlit app starts here
st.title("Fashion MNIST VAE Explorer")

# Allow the user to choose an index and see the original image and its reconstruction
image_id = st.number_input('Enter an image ID (0-59999):', min_value=0, max_value=59999, value=15, step=1)
st.write("Original Image")
original_image = Image.fromarray((255 * train_images[image_id, :, :, 0]).astype(np.uint8))
st.image(original_image, use_column_width=True)

st.write("Reconstructed Image")
reconstructed_image = model.reconstruct(train_images[image_id:image_id+1, :, :, :]).numpy()
reconstructed_image = Image.fromarray((255 * reconstructed_image.squeeze()).astype(np.uint8))
st.image(reconstructed_image, use_column_width=True)

# Visualize nearest neighbors
k = st.slider('Select number of nearest neighbors:', min_value=1, max_value=10, value=6, step=1)
if st.button('Find Nearest Neighbors'):
    idx = query(image_id, k=k)
    
    st.write("Nearest Neighbors")
    cols = st.beta_columns(k)
    for i, col in enumerate(cols):
        neighbor_image = Image.fromarray((255 * train_images[idx[i], :, :, 0]).astype(np.uint8))
        col.image(neighbor_image, use_column_width=True)

# Optional: Allow users to input coordinates and see images from the latent space
coordinate_input = st.text_input('Enter coordinates in latent space (comma-separated):', '0.0, 0.0')
if st.button('Generate from Latent Space'):
    coordinates = np.fromstring(coordinate_input, dtype=float, sep=',')
    image_from_latent = model.decode(np.array([coordinates])).numpy()
    image_from_latent = Image.fromarray((255 * image_from_latent.squeeze()).astype(np.uint8))
    st.image(image_from_latent, caption='Decoded Image from Latent Space', use_column_width=True)

# Optional: Visualize the 2D latent space
if st.checkbox('Show 2D latent space visualization'):
    st.write("2D Latent Space")
    embeddings = embeddigns.numpy()
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=y_train, cmap='tab10')
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.add_artist(legend1)
    st.pyplot(fig)

# Optional: Save the Streamlit app as a PDF
if st.button('Save current visualization as PDF'):
    buf = io.BytesIO()
    plt.savefig(buf, format='pdf')
    st.download_button(label='Download current visualization as PDF',
                       data=buf,
                       file_name='vae_visualization.pdf')


