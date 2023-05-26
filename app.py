import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

def main():
    # Set up the Streamlit app
    st.markdown(
        """
        <style>
        body {
            background-image: url('BG2.jpg');
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.write("Members:")
    st.write("Kenneth Cordero")
    st.write("Grant Guriel Geslani")
    st.write("Aaron Jan Inalyo")
    st.write("Course & Section: CPE019 - CPE32S3")
    st.write("Instructor: Engr. Roman Richard")
    st.title("Gastropod Mollusk Classifier (Slug/Snail)")
    st.write("This app classifies whether an uploaded image contains a Slug or Snail using a pre-trained convolutional neural network model.")

    @st.cache(allow_output_mutation=True)
    def load_model():
        model = tf.keras.models.load_model('weights-improvement-46-0.91.hdf5')
        return model

    def import_and_predict(image_data, model):
        size = (128, 128)
        image = ImageOps.fit(image_data, size, Image.LANCZOS)
        image = np.asarray(image)
        image = image / 255.0
        img_reshape = np.reshape(image, (1, 128, 128, 3))
        prediction = model.predict(img_reshape)
        return prediction

    model = load_model()
    class_names = ["SLUG", "SNAIL"]

    file = st.file_uploader("Choose a Gastropod picture from your computer", type=["jpg", "png", "jpeg"])

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        string = "Prediction: " + class_name
        st.success(string)

if __name__ == "__main__":
    main()
