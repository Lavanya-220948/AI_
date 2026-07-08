#importing necessary libraries
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

#loading the trained model
model = tf.keras.models.load_model("fashion_mnist_cnn.keras")

#model classes
class_names = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

#application title, it displays on the top of the web page
st.title("Fashion MNIST Classifier")
#adding description to the web page
st.write("Upload a grayscale clothing image (28×28) to predict its category.")

#to upload an image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

#to display the uploaded image and make predictions
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=200
    )

    #preporcessing teh image to make predictions,making it into grey scale
    image = image.convert("L")

  
    image = image.crop(image.getbbox())
    image = image.resize((28,28))

    image_array = np.array(image)

    image_array = image_array / 255.0

    image_array = image_array.reshape(1,28,28,1)

    image_array = np.array(image)

    image_array = 255 - image_array   # invert colors

    st.image(
    image_array.reshape(28,28),
    clamp=True
    )

    # Add channel dimension
    image_array = np.expand_dims(image_array, axis=-1)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    st.write(image_array.shape)

    #making predictions    
    prediction = model.predict(image_array)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction)
    #displaying the predicted class and confidence score
    st.write(f"Predicted Class: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}")
    
 