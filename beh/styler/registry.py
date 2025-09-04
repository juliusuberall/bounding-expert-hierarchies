import matplotlib as mplt

# Project wide default cmap for plots
cmap = 'gray'
expert_cmap = 'nipy_spectral'
white_gray = "whitesmoke" # Replacement for #FFFFFF because otherwise no difference on white background

# Custom gray to black gradient
wb_gradient = mplt.colors.LinearSegmentedColormap.from_list("mono_custom", [white_gray, "black"])

# Model colors
cols = {
    'moe' : '#00A517',
    'mlp' : '#ff00c2',
    'light_gray' : '#f5f5f5',
    'dark_gray' : '#B3B3B3'
}