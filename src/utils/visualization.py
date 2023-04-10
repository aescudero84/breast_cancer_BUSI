import matplotlib.pyplot as plt
import numpy as np
from numpy import logical_and as l_and, logical_not as l_not
from skimage import morphology, measure
import pandas as pd
pd.options.display.float_format = '{:.4f}'.format


def plot_overlapping(image, mask, segmentation):

    # Define the colors for true positive, false positive, and false negative pixels
    tp_color = [0, 1, 0]  # green
    fp_color = [1, 0, 0]  # red
    fn_color = [0, 0, 1]  # blue

    # Compute the intersection, false positive, and false negative between the mask and output
    intersection = np.logical_and(mask, segmentation)
    fp = np.logical_and(segmentation, np.logical_not(mask))
    fn = np.logical_and(mask, np.logical_not(segmentation))

    # Create the final image with colors for true positive, false positive, and false negative pixels
    result = np.zeros((mask.shape[0], mask.shape[1], 3))
    result[..., 0] = fp_color[0] * fp + fn_color[0] * fn + tp_color[0] * intersection
    result[..., 1] = fp_color[1] * fp + fn_color[1] * fn + tp_color[1] * intersection
    result[..., 2] = fp_color[2] * fp + fn_color[2] * fn + tp_color[2] * intersection

    # Define the legend
    legend_elements = [
        plt.Line2D([0], [0], color='w', marker='o', markerfacecolor=tp_color, markersize=16, label='True Positive'),
        plt.Line2D([0], [0], color='w', marker='o', markerfacecolor=fp_color, markersize=16, label='False Positive'),
        plt.Line2D([0], [0], color='w', marker='o', markerfacecolor=fn_color, markersize=16, label='False Negative')
    ]

    # Plot the image, mask, output, overlap and legend using subplots
    fig, axs = plt.subplots(1, 4, figsize=(25, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Image')
    axs[0].axis('off')
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title('Mask')
    axs[1].axis('off')
    axs[2].imshow(segmentation, cmap='gray')
    axs[2].set_title('Segmentation')
    axs[2].axis('off')
    axs[3].imshow(result)
    axs[3].set_title('Overlap')
    axs[3].axis('off')
    plt.legend(handles=legend_elements, bbox_to_anchor=(0, 0), ncol=3, fontsize=24)
    plt.show()
