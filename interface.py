import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Style Advisor Recommender 👗👠👒🕶️👚👖')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# steps
# file upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="My Style", use_column_width=True)  # Added caption here
        # feature extract
        features = feature_extraction(os.path.join("upload", uploaded_file.name), model)
        # st.text(features)
        # recommendention
        indices = recommend(features, feature_list)
        # show
        col1, col2, col3, col4, col5 = st.columns(5)

        st.title("Recommended")  # Added title here
        with col1:
            st.image(filenames[indices[0][0]], caption="Recommended 1", use_column_width=True)
        with col2:
            st.image(filenames[indices[0][1]], caption="Recommended 2", use_column_width=True)
        with col3:
            st.image(filenames[indices[0][2]], caption="Recommended 3", use_column_width=True)
        with col4:
            st.image(filenames[indices[0][3]], caption="Recommended 4", use_column_width=True)
        with col5:
            st.image(filenames[indices[0][4]], caption="Recommended 5", use_column_width=True)
    else:
        st.error("Some error occurred during file upload. Please try again.")
