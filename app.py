import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


st.set_page_config(
    page_title="Exercise Image Classifier",
    page_icon="",
    layout="centered"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/exercise_classifier.keras"
    )


model = load_model()


class_names = [
    "Lunge",
    "Plank",
    "Push-up",
    "Squat"
]


st.title("Exercise Image Classifier")

st.write(
    "Upload an image and the model will classify it as "
    "a lunge, plank, push-up, or squat."
)

uploaded_file = st.file_uploader(
    "Upload an exercise image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )

    image_resized = image.resize((224, 224))

    image_array = np.array(image_resized)

    image_array = image_array / 255.0

    image_batch = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_batch, verbose=0)

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index]

    st.subheader("Prediction")

    st.success(
        f"{predicted_class} — {confidence * 100:.1f}% confidence"
    )

    st.subheader("Class probabilities")

    for i, class_name in enumerate(class_names):
        st.write(
            f"{class_name}: {predictions[0][i] * 100:.1f}%"
        )