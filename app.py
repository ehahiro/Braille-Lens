import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
import zipfile
import os
import io
from PyPDF2 import PdfWriter, PdfReader
from ultralyticsplus import YOLO

# Dot-pattern to Normal letter mapping
braille_dot_to_letter = {
    "100000": "a", "110000": "b", "100100": "c", "100110": "d", "100010": "e",
    "110100": "f", "110110": "g", "110010": "h", "010100": "i", "010110": "j",
    "101000": "k", "111000": "l", "101100": "m", "101110": "n", "101010": "o",
    "111100": "p", "111110": "q", "111010": "r", "011100": "s", "011110": "t",
    "101001": "u", "111001": "v", "010111": "w", "101101": "x", "101111": "y", "111101": "y", "101011": "z",
    "000101": "capital", "001111": "number", "001001": "-", "010011": "."
}

# Load model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Draw letters using OpenCV with optional bounding boxes
def draw_letters_opencv(image, boxes, class_ids, class_map, show_bounding_boxes, font_size, bold):
    img_np = np.array(image.convert("RGB"))
    img_pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img_pil)

    # Use a larger font size
    try:
        font = ImageFont.truetype("times.ttf", font_size)
        if bold:
            font = ImageFont.truetype("timesbd.ttf", font_size)  # Using bold font
    except IOError:
        font = ImageFont.load_default()

    capital_next = False
    number_next = False

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(class_ids[i])
        dot_pattern = class_map[class_id].strip()
        letter = braille_dot_to_letter.get(dot_pattern, "?")

        if letter == "capital":
            capital_next = True
            continue
        if letter == "number":
            number_next = True
            continue

        if capital_next:
            letter = letter.upper()
            capital_next = False
        if number_next:
            number_map = {"a": "1", "b": "2", "c": "3", "d": "4", "e": "5",
                          "f": "6", "g": "7", "h": "8", "i": "9", "j": "0"}
            letter = number_map.get(letter, letter)
            number_next = False

        # Draw bounding box if option is selected
        if show_bounding_boxes:
            draw.rectangle([x1, y1, x2, y2], outline="white", width=2)

        # Place the letter just below the bounding box
        text_x = x1
        text_y = y2 + 5  # Position the text just below the bounding box

        draw.text((text_x, text_y), letter, font=font, fill="black")

    # Add watermark
    watermark_font = ImageFont.truetype("times.ttf", 20)
    watermark_text = "Translated by Dr. Claude"
    watermark_position = (img_pil.width - len(watermark_text) * 15, img_pil.height - 30)  # Adjust position as needed
    draw.text(watermark_position, watermark_text, font=watermark_font, fill=(255, 255, 255, 128))

    return np.array(img_pil)

# Rotate image
def rotate_image(image, angle):
    return image.rotate(angle, expand=True)

# Export to PDF
def create_pdf_from_images(images, output_path):
    pdf_writer = PdfWriter()
    for img in images:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        pdf_writer.addpage(PdfReader(io.BytesIO(img_byte_arr)).pages[0])
    with open(output_path, "wb") as f:
        pdf_writer.write(f)

# Export to TXT
def create_txt_from_images(images, output_path):
    with open(output_path, "w") as f:
        for img in images:
            f.write(f"Image: {img}\n")

# Main Streamlit app
def main():
    st.title("Braille Dot Pattern Detector → Normal Letters")

    if 'images' not in st.session_state:
        st.session_state.images = []

    tab1, tab2 = st.tabs(["Detection", "File Handling"])

    with tab1:
        st.header("Detection Settings")
        conf = st.slider("Class Confidence", 10, 75, 15) / 100
        iou = st.slider("IoU Threshold", 10, 75, 15) / 100
        show_bounding_boxes = st.checkbox("Show Bounding Boxes", value=True)
        font_size = st.slider("Font Size", 20, 100, 60)
        bold = st.checkbox("Bold Font", value=False)

        model_path = "yolov8n.pt"

        try:
            model = load_model(model_path)
            model.overrides["conf"] = conf
            model.overrides["iou"] = iou
            model.overrides["agnostic_nms"] = False
            model.overrides["max_det"] = 2000
        except Exception as ex:
            st.error(f"Model failed to load from {model_path}: {ex}")

        source_imgs = st.file_uploader("Upload images...", type=["jpg", "jpeg", "png", "bmp", "webp", "zip"], accept_multiple_files=True)

        if source_imgs:
            for img in source_imgs:
                if img.type == "application/zip":
                    with zipfile.ZipFile(img, 'r') as zip_ref:
                        zip_ref.extractall("extracted_images")
                    st.session_state.images.extend([Image.open(os.path.join("extracted_images", f)) for f in os.listdir("extracted_images") if f.endswith(('jpg', 'jpeg', 'png', 'bmp', 'webp'))])
                else:
                    st.session_state.images.append(Image.open(img))

        if st.button("Add More Pages"):
            st.experimental_rerun()

        if st.button("Clear All"):
            st.session_state.images = []
            st.experimental_rerun()

        for idx, image in enumerate(st.session_state.images):
            st.image(image, caption=f"Input Image {idx+1}", use_column_width=True)

            # Buttons to rotate the image before processing
            if st.button(f"Rotate Image {idx+1} 90°"):
                st.session_state.images[idx] = rotate_image(image, 90)
                st.experimental_rerun()

            if st.button(f"Rotate Image {idx+1} -90°"):
                st.session_state.images[idx] = rotate_image(image, -90)
                st.experimental_rerun()

            if st.button(f"Recognize Image {idx+1}"):
                with st.spinner("Detecting Braille..."):
                    try:
                        results = model.predict(st.session_state.images[idx], save=False, conf=conf)[0]
                        boxes = results.boxes.xyxy.cpu().numpy()
                        class_ids = results.boxes.cls.cpu().numpy()
                        processed_img_np = draw_letters_opencv(st.session_state.images[idx], boxes, class_ids, model.names, show_bounding_boxes, font_size, bold)

                        st.image(processed_img_np, caption=f"Detected Normal Letters {idx+1}", use_column_width=True)

                        # Save the processed image
                        output_path = f"output_{idx}.jpg"
                        cv2.imwrite(output_path, cv2.cvtColor(processed_img_np, cv2.COLOR_RGB2BGR))

                        # Download button for single image
                        with open(output_path, "rb") as file:
                            st.download_button(f"Download Image {idx+1}", file, file_name=output_path, mime="image/jpg")

                    except Exception as e:
                        st.error(f"Detection failed: {e}")

        if st.button("Recognize All Images"):
            with st.spinner("Detecting Braille in all images..."):
                for idx, image in enumerate(st.session_state.images):
                    try:
                        results = model.predict(image, save=False, conf=conf)[0]
                        boxes = results.boxes.xyxy.cpu().numpy()
                        class_ids = results.boxes.cls.cpu().numpy()
                        processed_img_np = draw_letters_opencv(image, boxes, class_ids, model.names, show_bounding_boxes, font_size, bold)

                        st.image(processed_img_np, caption=f"Detected Normal Letters {idx+1}", use_column_width=True)

                        # Save the processed image
                        output_path = f"output_{idx}.jpg"
                        cv2.imwrite(output_path, cv2.cvtColor(processed_img_np, cv2.COLOR_RGB2BGR))

                        # Download button for single image
                        with open(output_path, "rb") as file:
                            st.download_button(f"Download Image {idx+1}", file, file_name=output_path, mime="image/jpg")

                    except Exception as e:
                        st.error(f"Detection failed for image {idx+1}: {e}")

    with tab2:
        st.header("File Handling")

        if st.button("Share Results"):
            st.write("Sharing functionality would be implemented here.")

        if st.button("Export to TXT"):
            create_txt_from_images(st.session_state.images, "output.txt")
            with open("output.txt", "rb") as f:
                st.download_button(label="Download TXT", data=f, file_name="output.txt", mime="text/plain")

        if st.button("Export to PDF"):
            create_pdf_from_images([Image.fromarray(img) for img in st.session_state.images], "output.pdf")
            with open("output.pdf", "rb") as f:
                st.download_button(label="Download PDF", data=f, file_name="output.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
