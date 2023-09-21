import streamlit as st
import transformers
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image


# Load the transformer model
processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

# Define a function to describe an image
def describe_image(image):
  # Preprocess the image
  pixel_values = processor(images=image, return_tensors="pt").pixel_values
  # Pass the image to the transformer model
  generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
  generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
  
  return generated_caption

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
