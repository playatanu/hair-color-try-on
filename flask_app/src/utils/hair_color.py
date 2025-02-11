import onnxruntime
from PIL import Image, ImageColor
import numpy as np
import os

import cv2


def image_to_mask(image):
    onnx_path = os.path.abspath(os.path.join("src/models", "hair_seg_model.onnx"))

    image_resized = image.resize((256, 256))

    onnx_input_image = np.array(image_resized, dtype=np.float32) / 255.0  # Normalize

    # Transpose to (C, H, W)
    onnx_input_image = np.transpose(onnx_input_image, (2, 0, 1))

    # Add batch dimension
    input_tensor = np.expand_dims(onnx_input_image, axis=0)

    ort_session = onnxruntime.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )

    ort_inputs = {"input": input_tensor}

    outputs = ort_session.run(None, ort_inputs)

    output_image = outputs[0][0]

    mask_image = Image.fromarray(
        np.uint8((output_image * 255.0).clip(0, 255)[0]), mode="L"
    )

    return mask_image


def PIL_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image, dtype=np.float32), cv2.COLOR_RGB2BGR)


def cv2_to_PIL(cv2_image):
    cv2_image_rgb = cv2.cvtColor(cv2_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_image_rgb)


def convert(image_file, color_code):

    image = Image.open(image_file)
    image = image.resize((256, 256))
    mask = image_to_mask(image)

    mask = PIL_to_cv2(mask)
    image = PIL_to_cv2(image)

    _, mask = cv2.threshold(mask, thresh=180, maxval=255, type=cv2.THRESH_BINARY)

    colored_mask = np.copy(image)

    R, G, B = ImageColor.getcolor(color_code, "RGB")
    new_color = (B, G, R)
    colored_mask[(mask == 255).all(-1)] = new_color

    weight = 0.4

    colored_mask_w = cv2.addWeighted(
        colored_mask, weight, image, 1 - weight, 0, colored_mask
    )

    return cv2_to_PIL(colored_mask_w)
