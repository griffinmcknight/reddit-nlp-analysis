# color_palette.py

import matplotlib.pyplot as plt
import numpy as np

# Load the Plasma colormap from matplotlib
plasma_cmap = plt.cm.plasma(np.linspace(0, 1, 256))  # Define 256 colors along the Plasma colormap

# Utility function to convert RGBA to HEX for Streamlit/Plotly compatibility
def rgba_to_hex(rgba):
    return '#{:02x}{:02x}{:02x}'.format(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))

# Generate 10 evenly spaced colors from the Plasma colormap
categorical_colors = [rgba_to_hex(plasma_cmap[i]) for i in np.linspace(0, 255, 10, dtype=int)]

# Define main color scheme in HEX format
COLOR_PALETTE = {
    "background": rgba_to_hex((0,0,0,0.1)),
    "primary": rgba_to_hex(plasma_cmap[120]),
    "secondary": rgba_to_hex(plasma_cmap[180]),
    "outlier": rgba_to_hex(plasma_cmap[255]),
    "text": "#333333",       # Dark color for main text
    "axes": rgba_to_hex((1,1,1,0.4)),       # Soft gray for axes
    "continuous": [rgba_to_hex(color) for color in plasma_cmap],
    "categorical": categorical_colors  # List of 10 evenly spaced categorical colors
}
