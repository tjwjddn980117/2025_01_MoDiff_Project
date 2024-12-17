import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def load_scan(path):
    scan = nib.load(path)
    scan_array = scan.get_fdata()
    print(f'The scan data array has the shape: {scan_array.shape}')
    return scan, scan_array


def plot_scan_views(scan_array):
    fig, axes = plt.subplots(1, 3)
    fig.suptitle("Scan array from 3 views")
    mid_slices = [scan_array.shape[i] // 2 for i in range(3)]
    axes[0].imshow(scan_array[mid_slices[0], :, :], cmap='gray')
    axes[1].imshow(scan_array[:, mid_slices[1], :], cmap='gray')
    axes[2].imshow(scan_array[:, :, mid_slices[2]], cmap='gray')
    plt.tight_layout()
    plt.show()


def calculate_aspect_ratios(scan):
    pix_dim = scan.header['pixdim'][1:4]
    aspect_ratios = [pix_dim[1] / pix_dim[2], pix_dim[0] / pix_dim[2], pix_dim[0] / pix_dim[1]]
    print(f'The required aspect ratios are: {aspect_ratios}')
    return aspect_ratios


def save_slices(volume_array, mask_array, aspect_ratios, output_path):
    new_dims = np.multiply(volume_array.shape, aspect_ratios)
    new_dims = tuple(map(round, new_dims))

    volume_output_path = os.path.join(output_path, 'images')
    mask_output_path = os.path.join(output_path, 'masks')

    # Ensure output directories exist
    os.makedirs(volume_output_path, exist_ok=True)
    os.makedirs(mask_output_path, exist_ok=True)
    # Save slices along each dimension where mask has annotations
    for dim in range(3):
        for i in range(volume_array.shape[dim]):
            if dim == 0:
                volume_slice = volume_array[i, :, :]
                mask_slice = mask_array[i, :, :]
            elif dim == 1:
                volume_slice = volume_array[:, i, :]
                mask_slice = mask_array[:, i, :]
            else:
                volume_slice = volume_array[:, :, i]
                mask_slice = mask_array[:, :, i]

            # Check if the mask slice has annotations
            if np.any(mask_slice):
                # Process and save both volume and mask slices

                # Resize and normalize volume slice
                resampled_volume_slice = resize_and_normalize(volume_slice, new_dims, (dim + 1) % 3, (dim + 2) % 3)
                cv2.imwrite(os.path.join(volume_output_path, f'Volume_Dim{dim + 1}_Slice{i}.png'), resampled_volume_slice)

                # Resize and normalize mask slice
                resampled_mask_slice = resize_and_normalize(mask_slice, new_dims, (dim + 1) % 3, (dim + 2) % 3)
                cv2.imwrite(os.path.join(mask_output_path, f'Mask_Dim{dim + 1}_Slice{i}.png'), resampled_mask_slice)


def resize_and_normalize(slice, new_dims, dim1, dim2):
    new_size = (new_dims[dim1], new_dims[dim2])
    resampled_slice = cv2.resize(slice, new_size, interpolation=cv2.INTER_LINEAR)

    # Normalize and scale to 8-bit range
    resampled_slice = cv2.normalize(resampled_slice, None, 0, 255, cv2.NORM_MINMAX)
    resampled_slice = np.uint8(resampled_slice)

    return resampled_slice


def process_modality_files(training_folder, modality, output_path):
    preprocessed_path = os.path.join(training_folder, 'preprocessed')
    masks_path = os.path.join(training_folder, 'masks')

    # Loop through preprocessed images of a specific modality
    for file in os.listdir(preprocessed_path):
        if modality in file:  # Select the specific modality
            image_path = os.path.join(preprocessed_path, file)
            # Construct mask file name and path
            mask_file = file.replace(modality, 'mask2')
            mask_path = os.path.join(masks_path, mask_file)
            # If the mask file exists, process it
            if os.path.exists(mask_path):
                print(f"Processing {file} and corresponding masks")
                # Load the image and mask
                image, image_array = load_scan(image_path)
                mask, mask_array = load_scan(mask_path)

                aspect_ratios = calculate_aspect_ratios(image)
                modality_output_path = os.path.join(output_path, file.split('.')[0])

                save_slices(image_array, mask_array, aspect_ratios, modality_output_path)


def process_dataset(root_path, output_path):
    # Iterate over each training folder
    for training_folder_name in os.listdir(root_path):
        training_folder_path = os.path.join(root_path, training_folder_name)
        if os.path.isdir(training_folder_path):
            for modality in ['flair', 't2', 'pd', 'mprage']:
                process_modality_files(training_folder_path, modality, output_path)


# Main execution
dataset_path = 'training'  # Adjust this path
output_path = 'output_data'  # Adjust this path
os.makedirs(output_path, exist_ok=True)
process_dataset(dataset_path, output_path)