"""
Utility module for visualizing images and masks in the context of object detection or segmentation tasks.
The module contains several functions for generating visually distinct colors, applying masks to images, and
displaying instances of detected objects on an image.


Copied and lightly modified from: https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
"""

import os
import sys
import random
import itertools
import colorsys

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Polygon


def random_colors(N, bright=True):
    """
    Generates N visually distinct colors.

    Parameters
    ----------
    N : int
        The number of colors to generate.
    bright : bool, optional
        If True, generate brighter, more saturated colors.  If False, generate
        more muted colors.

    Returns
    -------
    colors : list of (r, g, b) tuples
        A list of N colors, each represented as a tuple of (r, g, b) values.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    # To get visually distinct colors, generate them in HSV space then convert to RGB.
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # Generate random colors.
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """
    Apply the given mask to the image.

    Parameters
    ----------
    image : array of shape (H, W, 3)
        The image to apply the mask to.
    mask : array of shape (H, W)
        The mask to apply, where 1 is the foreground and 0 is the background.
    color : (r, g, b)
        The color to apply to the foreground.
    alpha : float, optional
        The alpha value to apply to the foreground. Defaults to 0.5.

    Returns
    -------
    image : array of shape (H, W, 3)
        The image with the mask applied.
    """
    for c in range(3):
        # Apply the given mask to the image.
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    Displays instances of detected objects on an image with their
    corresponding bounding boxes, masks, class labels, and optional
    scores and captions.

    Parameters
    ----------
    image : array
        The image on which to display the instances.
    boxes : array
        Array of shape [num_instance, (y1, x1, y2, x2, class_id)] containing
        the bounding box coordinates and class IDs for each instance.
    masks : array
        Array of shape [height, width, num_instances] containing the masks
        for each instance.
    class_ids : array
        Array of class IDs for each instance.
    class_names : list
        List of class names corresponding to the class IDs.
    scores : array, optional
        Confidence scores for each instance, default is None.
    title : str, optional
        Title for the plot, default is an empty string.
    figsize : tuple, optional
        Size of the figure, default is (16, 16).
    ax : matplotlib.axes.Axes, optional
        Axes object to draw on, default is None.
    show_mask : bool, optional
        Whether to display instance masks, default is True.
    show_bbox : bool, optional
        Whether to display bounding boxes, default is True.
    colors : list, optional
        List of colors for each instance, default is None.
    captions : list, optional
        List of captions for each instance, default is None.

    Returns
    -------
    None
    """

    # Number of instances
    N = boxes.shape[0]
    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        x1, y1, width, height = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), width, height, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()
