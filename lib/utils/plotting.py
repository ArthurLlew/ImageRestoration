###########
# IMPORTS #
###########


# Handy arrays
import numpy as np
# FFTs
import scipy.fft as fft
import scipy.fftpack as fftp
# Plotting
import matplotlib.pyplot as plt
import matplotlib.image as img
# Custom modules
import lib.utils.utils as utils


##################
# Plotting Utils #
##################


def add_colorbar(fig: plt.Figure, axes: img.AxesImage, image: np.ndarray) -> None:
    """Adds colorbar to provided figure subplot.
    """
    
    # Plot colorbar for grayscale image only
    if len(image.shape) == 2:
        # Check if values are valid for colorbar
        if np.any(np.isnan(image)):
            raise Exception('Image has NaN values')
        if np.any(np.isinf(image)):
            raise Exception('Image has Inf values')
        
        # Choose colorbar format depending on values count
        if len(np.unique(image)) != 2:
            # Use colorbar with 3 values
            cbar = fig.colorbar(axes, shrink=0.5, ticks=np.linspace(image.min(), image.max(), 3, endpoint=True))
        else:
            # Use colorbar with 2 values
            cbar = fig.colorbar(axes, shrink=0.5, ticks=np.linspace(image.min(), image.max(), 2, endpoint=True))
            # Colorbar labels formatting
            cbar.set_ticklabels(['{:.1f}'.format(image.min()), '{:.1f}'.format(image.max())])
        
        # Color bar tick labels size
        cbar.ax.tick_params(axis='both', which='major', labelsize=20)


def add_subplot(fig: plt.Figure, axes_names: tuple, subplot_pos=(1,1,1), fancy=False, y_axis_position=0, y_labels_pos='bottom',
                y_label_padding=-60) -> plt.Axes:
    """Creates customized subplot.
    """
    
    # Subplot init
    subplot = fig.add_subplot(subplot_pos[0], subplot_pos[1], subplot_pos[2])
    
    if fancy:
        # Move bottom line to 0 (creates Ox)
        subplot.spines['bottom'].set_position(('data', 0))
        # Move left line to provided position (creates Oy)
        subplot.spines['left'].set_position(('data', y_axis_position))
        
        # Hide top and right lines
        subplot.spines[['top', 'right']].set_visible(False)
    
    # Set axis linewidth
    subplot.spines[['top', 'bottom', 'left', 'right']].set_linewidth(1.5)
    
    if fancy:
        # Draw arrows (as black triangles: ">k"/"^k") at the end of the axes.  In each
        # case, one of the coordinates (0) is a data coordinate (i.e., y = 0 or x = 0,
        # respectively) and the other one (1) is an axes coordinate (i.e., at the very
        # right/top of the axes).  Also, disable clipping (clip_on=False) as the marker
        # actually spills out of the axes.
        # Ox arrow:
        subplot.plot(1, 0, '>k', transform=subplot.get_yaxis_transform(), linewidth=2.8, clip_on=False)
        # Oy arrow:
        subplot.plot(y_axis_position, 1, '^k', transform=subplot.get_xaxis_transform(), linewidth=2.8, clip_on=False)
    
    # Ox and Oy tick labels size
    subplot.tick_params(axis='both', which='major', width=1.7, length=5.5, labelsize=20)
    
    # Align Ox labels to the right of the ticks
    for label in subplot.get_xticklabels():
        label.set_horizontalalignment('left')
    fig.align_xlabels()
    # Align Oy tick labels above the ticks
    for label in subplot.get_yticklabels():
        label.set_verticalalignment(y_labels_pos)
    fig.align_ylabels()
    
    # Axes labels name, location and rotation
    if fancy:
        subplot.set_xlabel(axes_names[0], loc='right', fontsize=22)
        subplot.set_ylabel(axes_names[1], loc='top', rotation=0, labelpad=y_label_padding, fontsize=22)
    else:
        subplot.set_xlabel(axes_names[0], fontsize=22)
        subplot.set_ylabel(axes_names[1], fontsize=22)
    
    if not fancy:
        # Grid
        subplot.grid()
    
    return subplot


def plot_image(image: np.ndarray, title='', axes_labels=[r'$-\pi$', '0', r'$\pi$'], plot_image=True,
               plot_cut=False, cut_index=0, y_axis_position=0, fancy_cut=False,
               plot_colorbar=True,
               plot_file='') -> None:
    """Plots image.
    """
    
    utils.check_array_is_not_complex(image)
    
    # Reject incorrect shape
    if len(image.shape) != 2 and len(image.shape) != 3:
        raise Exception('Invalid image shape! Expected 2D or 3D but found:', image.shape)
    
    # Init figure
    fig = plt.figure(figsize=(7*plot_image + plot_cut*8, 7 - plot_cut))
    
    # Figure title
    fig.suptitle(title, fontsize=16, x=0.435, y=0.85)
    
    if plot_image:
        # Image subplot
        subplot = fig.add_subplot(1, 1+plot_cut, 1+plot_cut)
        
        # Ox and Oy axis points
        subplot.set_xticks([-image.shape[0] // 2, 0, image.shape[0] // 2], axes_labels)
        subplot.set_yticks([-image.shape[1] // 2, 0, image.shape[1] // 2], axes_labels)
        # Ox and Oy labels size
        subplot.tick_params(axis='both', which='major', labelsize=20)
        
        # Coordinate center should be in image's center
        coordinates=(-image.shape[0] // 2, image.shape[0] // 2, -image.shape[1] // 2, image.shape[1] // 2)
        # Plot 2D image (depending on image type: grayscale/rgb)
        if len(image.shape) == 2:
            axes = subplot.imshow(image, cmap='gray', extent=coordinates)
        elif len(image.shape) == 3:
            axes = subplot.imshow(image, extent=coordinates)
    
    # Plot grayscale image cut
    if plot_cut and len(image.shape) == 2:
        # 2D image cut (1D function)
        image_cut = image[cut_index]
        
        # Init Ox points (with coordinate center being in the image cut center)
        x_axis_points = np.arange(-(image.shape[0]//2), image.shape[0]//2)
        
        # Subplot init
        subplot = add_subplot(fig, ('', ''), subplot_pos=(1,1+plot_image,1), fancy=fancy_cut, y_axis_position=y_axis_position)
        
        # Ox labels
        subplot.set_xticks([-(image.shape[0]//2), 0, (image.shape[0]//2)-1], [axes_labels[0], 0, axes_labels[2]])
        # Oy ticks
        if fancy_cut:
            subplot.set_yticks([], [])

        # Bigger tick labels
        subplot.tick_params(axis='both', which='major', labelsize=30)
        
        # Plot image
        subplot.plot(x_axis_points, image_cut, linewidth=2)
    
    if plot_image and plot_colorbar:
        # Add colorbar
        add_colorbar(fig, axes, image)
    
    # Save plot as vector image
    if plot_file != '':
        fig.savefig(plot_file, format='svg')


def plot_colormesh(x_ticks: np.ndarray, y_ticks: np.ndarray, image: np.ndarray, plot_file='') -> None:
    """Plots image with custom ticks.
    """
    
    # Init figure and subplot
    fig = plt.figure(figsize=(7,6))
    subplot = fig.add_subplot()

    # Add colormesh and colorbar
    mesh = subplot.pcolormesh(x_ticks, y_ticks, image, cmap='gray')
    add_colorbar(fig, mesh, image)
    
    # Save plot as vector image
    if plot_file != '':
        fig.savefig(plot_file, format='svg')


def plot_squared_complex_module(image: np.ndarray, title='', plot_file='') -> None:
    """Plots image FFT squared complex module.
    """
    
    utils.check_shape_is_2d(image.shape)
    
    # Check image type
    if image.dtype == np.dtype('complex'):
        image_fft = image
    else:
        # Image FFT
        image_fft = fft.fft2(image)
    
    # Plot (shifted) magnitude
    plot_image(fftp.fftshift(utils.get_squared_complex_module(image_fft)), title=title, plot_file=plot_file)


def plot_spectrum(image: np.ndarray, title='', plot_file='') -> None:
    """Plots image FFT magnitude.
    """
    
    utils.check_shape_is_2d(image.shape)
    
    # Check image type
    if image.dtype == np.dtype('complex'):
        image_fft = image
    else:
        # Image FFT
        image_fft = fft.fft2(image)
    
    # Plot (shifted) spectrum
    plot_image(fftp.fftshift(utils.get_spectrum(image_fft)), title=title, plot_file=plot_file)


class FuncDesc:
    """Stores info about 1d function.
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray, legend_name: str, style=None):
        # X points
        self.x = x
        # Y points
        self.y = y
        # Name in the legend
        self.legend_name = legend_name
        # Line style (string)
        self.style = style


def plot_functions(functions: list[FuncDesc], title='', x_axis_name='', y_axis_name='УЕ', plot_file='', figsize=(14, 7)) -> None:
    """Plots given 1d functions in one plot.
    """
    
    # Init figure
    fig = plt.figure(figsize=figsize)
    
    # Figure title
    fig.suptitle(title, fontsize=30, x=0.5, y=0.91)
    
    # Collect all minimum X positions
    mins = np.empty((len(functions)))
    for i in range(len(functions)):
        mins[i] = functions[i].x[0]
    # Y axis position is at 0 or minimum value of combined grid
    y_axis_position=max(0, np.min(mins))
    
    # Subplot init
    subplot = add_subplot(fig, (x_axis_name, y_axis_name), y_axis_position=y_axis_position)
    
    # Plot data
    for function in functions:
        if function.style == None:
            subplot.plot(function.x, function.y, label=function.legend_name, linewidth=2.7)
        else:
            subplot.plot(function.x, function.y, function.style, label=function.legend_name, linewidth=2.7, markersize=12)
    
    # Show legend  
    subplot.legend(fontsize=20, loc='upper right')
    
    # Save plot as vector image
    if plot_file != '':
        fig.savefig(plot_file, format='svg')