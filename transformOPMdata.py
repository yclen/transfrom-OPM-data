import numpy as np
import tifffile
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.ndimage import affine_transform
import time
start = time.perf_counter()

#location of volume we wish to transform
grid = "C:\\Users\\y2cle\\important images\\10um-grid_crop.tif"
beads = "C:\\Users\\y2cle\\important images\\561nmEx_500nm-beads_0.173-um_y-step.tif"
bigGrid = "C:\\Users\\y2cle\\important images\\new-10-um-grid_0.5-um-y-step_camera-2.tif"

#location to save the transformed volume
save_location = "C:\\Users\\y2cle\\important images\\transformedVol.tif"

#define the transform function
def transform(vol, y_step):

    xypixelsize = 0.08667
    angle = 30
    dz_stage = y_step

    dz = np.sin(angle*np.pi/180.0)*dz_stage
    dx = xypixelsize
    deskewfactor = np.cos(angle*np.pi/180.0)*dz_stage/dx
    dzdx_aspect = dz/dx

    print("Parameter summary:")
    print("==================")
    print("dx, dy:", dx)
    print("dz:", dz)
    print("deskewfactor:", deskewfactor)
    print("voxel aspect ratio z-voxel/xy-voxel:", dzdx_aspect)
    print("original shape:", vol.shape)

    #define functions

    def get_projection_montage(vol):
        image1 = np.max(vol, axis=0)
        image2 = np.max(vol, axis=1)
        image3 = np.max(vol, axis=2)
        # Create a 2x2 grid of subplots
        fig, axes = plt.subplots(nrows=2, ncols=2)

        # Plot the first image in the upper left subplot
        axes[0, 0].imshow(image1)
        axes[0, 0].set_xlabel("X")
        axes[0, 0].set_ylabel("Y")

        # Plot the second image in the upper right subplot
        axes[0, 1].imshow(image3)
        axes[0, 1].set_xlabel("Y")
        axes[0, 1].set_ylabel("Z")

        # Plot the third image in the lower left subplot
        axes[1, 0].imshow(image2)
        axes[1, 0].set_xlabel("X")
        axes[1, 0].set_ylabel("Z")

        # Hide the lower right subplot
        axes[1, 1].axis("off")

        # Adjust the spacing between subplots
        fig.tight_layout()

        # Show the plot
        plt.show()

    def display_image_stack(img_stack):
    # Create a figure and axis for the image display
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)  # Make space for the slider

        # Display the first image in the stack
        img_display = ax.imshow(img_stack[10], cmap='gray')
        ax.axis('off')  # Hide the axes

        # Create a slider for browsing the z-axis
        slider_ax = plt.axes([0.1, 0.05, 0.8, 0.03])
        slider = Slider(slider_ax, 'Z-axis', 0, len(img_stack) - 1, valinit=0, valstep=1)

        def update(val):
            z = int(slider.val)
            img_display.set_data(img_stack[z])
            fig.canvas.draw_idle()

        # Connect the slider to the update function
        slider.on_changed(update)

        plt.show()

    def get_transformed_shape(shape, matrix):
        # Create an array of all 8 corners of the 3D shape
        corners = np.array([
            [0, 0, 0, 1],
            [shape[0], 0, 0, 1],
            [0, shape[1], 0, 1],
            [0, 0, shape[2], 1],
            [shape[0], shape[1], 0, 1],
            [shape[0], 0, shape[2], 1],
            [0, shape[1], shape[2], 1],
            [shape[0], shape[1], shape[2], 1]
        ])

        # Apply the transformation matrix to all corners
        transformed_corners = np.round(matrix @ corners.T).astype(int).T

        # Find the new bounding box
        min_corner = np.min(transformed_corners, axis=0)
        max_corner = np.max(transformed_corners, axis=0)

        # Calculate the dimensions of the transformed shape
        new_shape = max_corner - min_corner

        return tuple(new_shape[:-1])

    def rotate_around_x(angle):

        matrix = np.eye(4)
        matrix[0,0] = np.cos(angle*np.pi/180.0)
        matrix[1,1] = np.cos(angle*np.pi/180.0)
        matrix[1,0] = np.sin(angle*np.pi/180.0)
        matrix[0,1] = -np.sin(angle*np.pi/180.0)

        return matrix

    def shift_center(shape):
        matrix = np.eye(4)

        matrix[0,3] = -shape[0]/2
        matrix[1,3] = -shape[1]/2
        matrix[2,3] = -shape[2]/2

        return matrix

    def unshift_center(shape):
        matrix = np.eye(4)

        matrix[0,3] = shape[0]/2
        matrix[1,3] = shape[1]/2
        matrix[2,3] = shape[2]/2

        return matrix

    #define matrices
    skew = np.eye(4)
    skew[1,0] = deskewfactor

    scale = np.eye(4)
    scale[0,0] = dzdx_aspect

    rotate = rotate_around_x(angle)

    shift = shift_center(vol.shape)

    #calculate final shape and unshift
    combined = rotate @ scale @ skew @ shift
    output_shape = get_transformed_shape(vol.shape, combined)
    unshift = unshift_center(output_shape)
    print("final shape:", output_shape, "\n\nPreparing to transform...\n")

    #transform
    matrix = unshift @ rotate @ scale @ skew @ shift
    new_vol = affine_transform(vol, np.linalg.inv(matrix), output_shape=output_shape, order=1)


    #write the file and project
    tifffile.imwrite(save_location, new_vol)
    print("total time:", int(time.perf_counter() - start), "seconds")
    display_image_stack(new_vol)


#use the transform function (be sure to enter the correct y_step_size!)
volume = tifffile.imread(bigGrid)
y_step_size = 0.5



transform(volume, y_step_size)
