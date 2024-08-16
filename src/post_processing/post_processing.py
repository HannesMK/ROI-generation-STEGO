import math
import os
from typing import List

import cv2
import numpy as np
import pandas as pd
import skimage.measure
from matplotlib import pyplot as plt


def clean_up_mask(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Closes any gaps in the mask using the provided kernel.
    Afterwards reduces the mask to the largest connected component.

    Parameters
    ----------
    mask : np.ndarray
        Binary (i.e., black and white) mask
    kernel : np.ndarray
        Kernel used to close the mask (passed to cv2.morphologyEx())

    Returns
    -------
    np.ndarray
        Cleaned-up version of the binary mask
    """
    mask_clean = cv2.morphologyEx(src=mask, op=cv2.MORPH_CLOSE, kernel=kernel)

    mask_clean = extract_largest_connected_compenent(mask=mask_clean)
    
    return mask_clean


def create_binary_mask(mask: np.ndarray, target_colors: List[tuple]) -> np.ndarray:
    """
    Creates a binary (i.e., black and white) mask from a multi-colored mask

    Parameters
    ----------
    mask : np.ndarray
        Multi-colored mask
    target_colors : List[tuple]
        Colors of the areas to include in the binary mask

    Returns
    -------
    np.ndarray
        Binary mask
    """
    mask_binary = mask.copy()
    
    # Set all values of the valid colors to white
    for color in target_colors:
        indices = np.where(np.all(mask_binary == color, axis=-1))
        mask_binary[indices[0], indices[1], :] = 255

    # Set all values that are not white to black
    indices = np.where(np.all(mask_binary != (255, 255, 255), axis=-1))
    mask_binary[indices[0], indices[1], :] = 0
    
    return mask_binary


def degrees2pixels(x: float, viewing_distance: float, monitor_width: float, monitor_resolution_x: int) -> int:
    """
    Converts degrees of visual angle to pixels

    Parameters
    ----------
    x : float
        Degrees of visual angle
    viewing_distance : float
        Distance between observer and screen (unit must match the unit of monitor_width)
    monitor_width : float
        Width of the screen (unit must match the unit of viewing_distance)
    monitor_resolution_x : int
        Screen resolution in pixels

    Returns
    -------
    int
        Amount of pixel corresponding to the provided degrees of visual angle
    """
    pixels_per_unit = monitor_resolution_x / monitor_width

    x = degrees2radians(x)

    element_size = abs(2 * viewing_distance * np.tan(x / 2))

    pixels = element_size * pixels_per_unit
    pixels = int(np.round(pixels))

    return pixels


def degrees2radians(x: float) -> float:
    """
    Helper function to convert degrees to radians

    Parameters
    ----------
    x : float
        Degrees

    Returns
    -------
    float
        Radians
    """
    radians = x * math.pi / 180
    return radians


def dilate_mask(
    mask: np.ndarray,
    margin_degrees: float,
    viewing_distance: float,
    monitor_width: float,
    monitor_resolution_x: int,
) -> np.ndarray:
    """
    Adds a margin around a given mask

    Parameters
    ----------
    mask : np.ndarray
        Mask to be dilated
    margin_degrees : float
        Degrees of visual angle by which to dilate the mask
    viewing_distance : float
        Distance between observer and screen (unit must match the unit of monitor_width)
    monitor_width : float
        Width of the screen (unit must match the unit of viewing_distance)
    monitor_resolution_x : int
        Screen resolution in pixels

    Returns
    -------
    np.ndarray
        Dilated mask
    """
    
    margin_pixels = degrees2pixels(
        margin_degrees,
        viewing_distance=viewing_distance,
        monitor_width=monitor_width,
        monitor_resolution_x=monitor_resolution_x,
    )
    
    kernel = cv2.getStructuringElement(
        shape=cv2.MORPH_ELLIPSE,
        ksize=((margin_pixels + 1) * 2,) * 2,
    )
    
    mask_dilated = cv2.dilate(src=mask, kernel=kernel)
    
    return mask_dilated


def draw_mask_on_image(
        image: np.ndarray,
        mask: np.ndarray,
        color: tuple,
        alpha: float = .4,
    ) -> np.ndarray:
    """
    Draws a transparend overlay on an image based on a mask

    Parameters
    ----------
    image : np.ndarray
        Image on which to draw the overlay
    mask : np.ndarray
        Mask indicating the area of the overlay
    color : tuple
        Color of the overlay
    alpha : float, optional
        Transparency level of the overlay, by default .4

    Returns
    -------
    np.ndarray
        Image with added overlay
    """
    overlay = image.copy()
    overlay[np.where(mask[:, :, 0] > 0)] = color

    image_with_overlay = cv2.addWeighted(
        src1=overlay,
        alpha=alpha,
        src2=image,
        beta=1 - alpha,
        gamma=0,
    )

    return image_with_overlay


def extract_largest_connected_compenent(mask: np.ndarray) -> np.ndarray:
    """
    Reduces a binary mask to its largest connected compentent (excluding the background)

    Parameters
    ----------
    mask : np.ndarray
        Binary (i.e., black and white) mask

    Returns
    -------
    np.ndarray
        Binary mask for only the largest connected component of the input mask
    """
    
    mask_boolean = np.all(mask == (255, 255, 255), axis=-1)

    labels = skimage.measure.label(label_image=mask_boolean)

    # If there is only one label (0), return the mask as is
    if labels.max() == 0:
        return mask

    occurances_per_label = np.bincount(labels.flat)

    # Ignore the first element as this is the background (see https://stackoverflow.com/a/55110923)
    label_with_most_occurances = np.argmax(occurances_per_label[1:]) + 1

    largest_connected_component = labels == label_with_most_occurances

    mask_largest_connected_component = np.array(largest_connected_component * 255, dtype="uint8")

    return mask_largest_connected_component


def load_image_rgb(filepath: str) -> np.ndarray:
    """
    Wrapper function to load an RGB image using OpenCV (instead of BGR)

    Parameters
    ----------
    filepath : str
        Path from which to load the image

    Returns
    -------
    np.ndarray
        RGB image
    """
    
    image = cv2.imread(filename=filepath)
    
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
    
    return image
    

def plot_binary_masks(path_to_mask: str) -> tuple:
    """
    For a multi-colored mask, plots corresponding binary masks for each color

    Parameters
    ----------
    path_to_mask : str
        File path to the mask image

    Returns
    -------
    tuple
        (figure, axes)
    """
    mask = load_image_rgb(filepath=path_to_mask)

    mask_collapsed_x_y = mask.reshape(-1, mask.shape[2])

    unique_colors = np.unique(mask_collapsed_x_y, axis=0)

    figure, axes = plt.subplots(nrows=2, ncols=len(unique_colors), figsize=(len(unique_colors) * 3, 2 * 3))

    for index, color in enumerate(unique_colors):
        mask_current_color = create_binary_mask(mask=mask, target_colors=[color])

        axes[0, index].imshow(mask)

        axes[1, index].imshow(mask_current_color)

        axes[1, index].set_xlabel(f"{color}")

        axes[0, index].set_xticks([])
        axes[0, index].set_yticks([])
        axes[1, index].set_xticks([])
        axes[1, index].set_yticks([])
        
    return figure, axes


def process_masks(
    row_settings: pd.Series,
    path: str,
    margin_degrees: float,
    viewing_distance: float,
    monitor_width: float,
    monitor_resolution_x: int,
) -> None:
    """
    For a given set of multi-colored masks, creates masks resized to the original image resolution as well as undilated and dilated binary masks

    Parameters
    ----------
    row_settings : pd.Series
        Settings for "object_type", "target_colors", and "close_kernel"
    path_images : str
        Path to the parent folder of the masks (with the actual masks located at path/[OBJECT_TYPE]/cluster)
    margin_degrees : float
        Degrees of visual angle by which to dilate the mask
    viewing_distance : float
        Distance between observer and screen (unit must match the unit of monitor_width)
    monitor_width : float
        Width of the screen (unit must match the unit of viewing_distance)
    monitor_resolution_x : int
        Screen resolution in pixels
    """
    path_object_type = os.path.join(path, row_settings["object_type"])
    
    path_masks = os.path.join(path_object_type, "cluster")
    
    path_masks_binary = os.path.join(path_object_type, "masks_binary")
    path_masks_binary_dilated = os.path.join(path_object_type, "masks_binary_dilated")
    
    os.makedirs(path_masks_binary, exist_ok=True)
    os.makedirs(path_masks_binary_dilated, exist_ok=True)

    filenames = [x for x in os.listdir(path_masks) if x.endswith(".png")]

    for filename in filenames:
        mask = load_image_rgb(filepath=os.path.join(path_masks, filename))

        mask = resize_image(image=mask, target_resolution=row_settings["original_resolution"])
        
        mask = create_binary_mask(mask=mask, target_colors=row_settings["target_colors"])

        mask = clean_up_mask(mask=mask, kernel=row_settings["close_kernel"])
        
        save_iamge_rgb(image=mask, filepath=os.path.join(path_masks_binary, filename))
        
        mask_dilated = dilate_mask(
            mask=mask,
            margin_degrees = margin_degrees,
            viewing_distance = viewing_distance,
            monitor_width = monitor_width,
            monitor_resolution_x = monitor_resolution_x,
        )
        
        save_iamge_rgb(image=mask_dilated, filepath=os.path.join(path_masks_binary_dilated, filename))


def resize_image(image: np.ndarray, target_resolution: int) -> np.ndarray:
    """
    Resizes a square image to the specified resolution

    Parameters
    ----------
    image : np.ndarray
        Image to be resized
    target_resolution : int
        Resolution of the output image (will be used as both x and y resolution)

    Returns
    -------
    np.ndarray
        Resized image
    """
    image_resized = cv2.resize(
        src=image,
        dsize=(target_resolution,) * 2,
        interpolation=cv2.INTER_NEAREST_EXACT,
    )
    
    return image_resized

    
def save_iamge_rgb(image: np.ndarray, filepath: str) -> None:
    """
    Wrapper function to save an RGB image using OpenCV (instead of BGR)

    Parameters
    ----------
    image : np.ndarray
        RGB image to save
    filepath : str
        Path to save the image to
    """
    image_rgb = cv2.cvtColor(src=image, code=cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(filename=filepath, img=image_rgb)
