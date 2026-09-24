import numpy as np
from PIL import Image
import torch

def min_max_normalize(signal):

    min_val = np.min(signal)
    max_val = np.max(signal)

    if max_val == min_val:
        return np.zeros_like(signal)

    normalized = (2*(signal - min_val)/(max_val - min_val)) - 1

    return normalized

def polar_signal(signal):

    return np.arccos(signal)

def create_gasf(phi):

    gasf = np.cos(phi[:, None] + phi[None, :])

    return gasf

def create_gadf(phi):

    gadf = np.sin(phi[:, None] - phi[None, :])

    return gadf

def matrix_to_image(matrix):
    image = ((matrix + 1)*127.5).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image, mode="L")
    image = image.resize((224, 224), Image.Resampling.LANCZOS)

    return image

def create_cnn_input(gasf_image, gadf_image):
    gasf_tensor = torch.from_numpy(np.array(gasf_image)).float()/ 255.0
    gadf_tensor = torch.from_numpy(np.array(gadf_image)).float()/ 255.0

    gasf_tensor = gasf_tensor.unsqueeze(0)
    gadf_tensor = gadf_tensor.unsqueeze(0)

    image = torch.cat([gasf_tensor, gadf_tensor], dim=0)
    image = image.unsqueeze(0)

    return image

def process_window(window):
    processed = min_max_normalize(window)
    phi = polar_signal(processed)

    gasf = create_gasf(phi)
    gadf = create_gadf(phi)

    gasf_image = matrix_to_image(gasf)
    gadf_image = matrix_to_image(gadf)

    return gasf_image, gadf_image
