
# # import streamlit as st
# # import easyocr
# # import cv2
# # import numpy as np
# # import pandas as pd
# # from PIL import Image

# # # Initialize EasyOCR Reader
# # reader = easyocr.Reader(['en'])

# # # Streamlit UI
# # st.title("Casting Metal Text Detection & Verification")

# # # Upload Image
# # uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# # if uploaded_file is not None:
# #     # Convert uploaded image to OpenCV format
# #     image = Image.open(uploaded_file)
# #     image = np.array(image)

# #     # Perform OCR only once
# #     if "results" not in st.session_state:
# #         st.session_state.results = reader.readtext(image)

# #     # Extract detected text
# #     detected_text = [text for (_, text, _) in st.session_state.results]

# #     # Draw bounding boxes on the image
# #     annotated_image = image.copy()
# #     for (bbox, text, prob) in st.session_state.results:
# #         top_left, bottom_right = tuple(map(int, bbox[0])), tuple(map(int, bbox[2]))
# #         cv2.rectangle(annotated_image, top_left, bottom_right, (0, 255, 0), 3)
# #         cv2.putText(annotated_image, text, (top_left[0], top_left[1] - 9), 
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# #     # Convert OpenCV image to PIL format for Streamlit
# #     annotated_image = Image.fromarray(annotated_image)

# #     # Display images side by side
# #     col1, col2 = st.columns([1, 1])  # Equal column width

# #     with col1:
# #         st.subheader("Original Image")
# #         st.image(uploaded_file, width=400)

# #     with col2:
# #         st.subheader("Detected Text Image")
# #         st.image(annotated_image, width=400)

# #     # Display detected text
# #     st.subheader("Detected Text")
# #     df = pd.DataFrame(detected_text, columns=["Detected Text"])
# #     st.write(df)

# #     # Save detected text as CSV
# #     csv_path = "detected_text.csv"
# #     df.to_csv(csv_path, index=False)

# #     # Provide download link
# #     with open(csv_path, "rb") as file:
# #         st.download_button("📥 Download Detected Text", file, file_name="detected_text.csv", mime="text/csv")

# #     # Input field for original numbers AFTER displaying detected text
# #     original_text = st.text_input("Enter the Original Number(s) (comma-separated)", "")

# #     if original_text:
# #         # Convert user input into a set for easy comparison
# #         original_numbers = set(original_text.split(","))
# #         detected_numbers = set(detected_text)

# #         # Check if detected text matches original numbers
# #         is_match = original_numbers == detected_numbers

# #         # Display comparison result
# #         st.subheader("Verification Result")
# #         if is_match:
# #             st.success("✅ Match Found! The detected text matches the original numbers.")
# #         else:
# #             st.error("❌ Mismatch! The detected text does not match the original numbers.")



# # # import torch

# # # if torch.cuda.is_available():
# # #     print("PyTorch is running on GPU:", torch.cuda.get_device_name(0))
# # # else:
# # #     print("PyTorch is running on CPU")



# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model

# # Load the trained model
# model = load_model("shape_classifier.h5")

# # Load image
# image = cv2.imread("image2.png")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply GaussianBlur to reduce noise
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# # Apply edge detection
# edges = cv2.Canny(blurred, 50, 150)

# # Find contours
# contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# for cnt in contours:
#     x, y, w, h = cv2.boundingRect(cnt)

#     if w > 30 and h > 30:  # Ignore small noise
#         approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
#         num_sides = len(approx)

#         # Shape classification using contour properties
#         if num_sides == 3:
#             shape_name = "Triangle"
#         elif num_sides == 4:
#             aspect_ratio = float(w) / h
#             shape_name = "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
#         elif num_sides == 5:
#             shape_name = "Pentagon"
#         elif num_sides == 6:
#             shape_name = "Hexagon"
#         elif num_sides > 6:
#             shape_name = "Circle"
#         else:
#             # Use CNN if shape is unknown
#             shape_crop = gray[y:y+h, x:x+w]
#             shape_resized = cv2.resize(shape_crop, (64, 64)).reshape(1, 64, 64, 1) / 255.0
#             prediction = model.predict(shape_resized)
#             shape_label = np.argmax(prediction)
#             shape_names = {0: "Circle", 1: "Triangle", 2: "Rectangle"}
#             shape_name = shape_names.get(shape_label, "Unknown")

#         # Draw bounding box
#         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#         # Adjust text size dynamically based on image size
#         font_scale = max(w, h) / 150  
#         thickness = max(2, int(font_scale * 2))  

#         # Ensure text is placed properly
#         text_x, text_y = x, max(y - 10, 30)  

#         # Draw text with background for better visibility
#         text_size = cv2.getTextSize(shape_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
#         cv2.rectangle(image, (text_x, text_y - text_size[1] - 10), 
#                       (text_x + text_size[0] + 10, text_y), (0, 0, 0), -1)
#         cv2.putText(image, shape_name, (text_x + 5, text_y - 5), 
#                     cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

# # Create a resizable window
# cv2.namedWindow("Shape Detection", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Shape Detection", 800, 600)

# # Display result
# cv2.imshow("Shape Detection", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import streamlit as st
import easyocr
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

# Load the trained shape classifier model
model = load_model("shape_classifier.h5")

# Streamlit UI
st.title("Casting Metal Detection System")

# Sidebar for selecting mode
option = st.sidebar.radio("Choose a feature:", ("Text Detection", "Shape Detection"))

# Function to perform text detection
def detect_text(image):
    results = reader.readtext(image)
    detected_text = [text for (_, text, _) in results]

    # Draw bounding boxes
    annotated_image = image.copy()
    for (bbox, text, prob) in results:
        top_left, bottom_right = tuple(map(int, bbox[0])), tuple(map(int, bbox[2]))
        cv2.rectangle(annotated_image, top_left, bottom_right, (0, 255, 0), 3)
        cv2.putText(annotated_image, text, (top_left[0], top_left[1] - 9), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return detected_text, annotated_image

# Function to perform shape detection
def detect_shapes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 30 and h > 30:  # Ignore small noise
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            num_sides = len(approx)

            # Shape classification
            if num_sides == 3:
                shape_name = "Triangle"
            elif num_sides == 4:
                aspect_ratio = float(w) / h
                shape_name = "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
            elif num_sides == 5:
                shape_name = "Pentagon"
            elif num_sides == 6:
                shape_name = "Hexagon"
            elif num_sides > 6:
                shape_name = "Circle"
            else:
                # CNN-based classification
                shape_crop = gray[y:y+h, x:x+w]
                shape_resized = cv2.resize(shape_crop, (64, 64)).reshape(1, 64, 64, 1) / 255.0
                prediction = model.predict(shape_resized)
                shape_label = np.argmax(prediction)
                shape_names = {0: "Circle", 1: "Triangle", 2: "Rectangle"}
                shape_name = shape_names.get(shape_label, "Unknown")

            # Draw bounding box and text
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            font_scale = max(w, h) / 150  
            thickness = max(2, int(font_scale * 2))  
            text_x, text_y = x, max(y - 10, 30)
            text_size = cv2.getTextSize(shape_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            cv2.rectangle(image, (text_x, text_y - text_size[1] - 10), 
                          (text_x + text_size[0] + 10, text_y), (0, 0, 0), -1)
            cv2.putText(image, shape_name, (text_x + 5, text_y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    return image

# Upload Image
# Upload Image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Ensure the image is always RGB (3 channels)
    image = Image.open(uploaded_file).convert("RGB")  
    image = np.array(image)  # Convert to NumPy array

    if option == "Text Detection":
        st.subheader("Text Detection")
        detected_text, annotated_image = detect_text(image)

        # Display images
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded_file, caption="Original Image", use_container_width=True)
        with col2:
            st.image(annotated_image, caption="Detected Text Image", use_container_width=True)

        # Display detected text
        st.subheader("Detected Text")
        df = pd.DataFrame(detected_text, columns=["Detected Text"])
        st.write(df)

        # Provide CSV download
        csv_path = "detected_text.csv"
        df.to_csv(csv_path, index=False)
        with open(csv_path, "rb") as file:
            st.download_button("📥 Download Detected Text", file, file_name="detected_text.csv", mime="text/csv")

    elif option == "Shape Detection":
        st.subheader("Shape Detection")
        processed_image = detect_shapes(image)
        st.image(processed_image, caption="Detected Shapes", use_container_width=True)
