import matplotlib as mplt

# Project wide default cmap for plots
cmap = 'gray'
expert_cmap = 'nipy_spectral'
back_col = "#f2f2e9" # Replacement for #FFFFFF because otherwise no difference on white background

# Custom gray to black gradient
wb_gradient = mplt.colors.LinearSegmentedColormap.from_list("mono_custom", [back_col, "black"])

# Model colors used for create graph plot of tabular data
cols = {
    'moe' : '#FF8200',
    'mlp' : '#00A517',
    'light_gray' : '#f5f5f5',
    'dark_gray' : '#B3B3B3'
}