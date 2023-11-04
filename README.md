Assignment3
1. Visual Search with Artistic Style Transfer (visual-search-artistic-style.ipynb)
This notebook illustrates a method for searching images by artistic style, which is useful for applications such as an eCommerce website where users can search for posters that match an uploaded image's style.

Key Features:

Based on the Neural Style Transfer method, following a TensorFlow tutorial and the paper by Gatys et al.
Uses a dataset with 32 images from the tensor-house-data repository.
Implements image loading, preprocessing, and visualization of style embeddings.
Defines functions for computing style embeddings and searching within the embedding space based on style similarity.
Streamlit App Objective:
The Streamlit app should allow users to upload an image and search for other images that match its artistic style. The app will display the results that are similar in terms of artistic style, potentially allowing users to further filter by image subject or category.

2. Visual Search Based on Similarity (visual-search-similarity.ipynb)
This notebook outlines the development of visual search models that retrieve similar objects based on visual similarity or similarity in attribute space.

Key Features:

Utilizes a "Clothing Dataset" to perform visual similarity searches.
Uses a pre-trained EfficientNetB0 model, fine-tuned on the specific dataset for label prediction.
Demonstrates how to compute embeddings for visualization and perform t-SNE to project them onto a 2D plane.
Includes a function to visualize the 2D projection of the embedding space with example images.
Streamlit App Objective:
The goal is to create an app that lets users upload an image and then uses the fine-tuned EfficientNet model to find and display visually similar items from the dataset, potentially using the t-SNE visualization to show how the items relate in the embedding space.

3. Visual Search Using Variational Autoencoders (visual-search-vae.ipynb)
This notebook demonstrates the process of training a Variational Autoencoder (VAE) on the Fashion MNIST dataset. The VAE learns a manifold of embeddings that represent the data in a lower-dimensional space. This embedding can be used to perform a visual search by finding the nearest neighbors in the latent space.

Key Features:

Training a VAE model on the Fashion MNIST dataset.
Visualization of the VAE manifold and embeddings.
Query function to find nearest neighbors of a given image in the embedding space.
Streamlit App Objective:
Develop an app that allows users to select or upload a fashion item image, and uses the trained VAE model to find and display similar items from the dataset.

Codelab: https://codelabs-preview.appspot.com/?file_id=1-WgMT_Zj9-strj1uRjJY3iL8J4u6sPzUr6XDgeM3K14#0

Contributions:

Tanay: 34%
Surya: 33%
Jeevika: 33%




