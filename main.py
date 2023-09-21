import streamlit as st
import transformers
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image


# Load the transformer model
model_name = 'google/vit-base-patch16-224'
model = ViTForImageClassification.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# Define a function to describe an image
def describe_image(image):
  # Preprocess the image
  inputs = feature_extractor(images=image, return_tensors="pt")
  # Pass the image to the transformer model
  outputs = model(**inputs)
  logits = outputs.logits
  predicted_class_idx = logits.argmax(-1).item()
  output = model.config.id2label[predicted_class_idx]
  
  return output

# Add an upload button to the app
uploaded_file = st.file_uploader("Upload an image", key=6)

# If an image is uploaded, add a button to describe the image
if uploaded_file is not None:
  # Display the uploaded image
  image = st.image(uploaded_file)

  # Add a button to describe the image
  describe_button = st.button("Describe image")

  # If the describe button is clicked, describe the image
  if describe_button:
    # Preprocess the image
    image_ = Image.open(uploaded_file)

    # Describe the image
    description = describe_image(image_)

    # Display the description to the user
    st.write("Image description:", description)
