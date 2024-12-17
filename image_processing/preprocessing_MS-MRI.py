import nibabel as nib
import numpy as np
import cv2
import os

def load_scan(path):
    """Load a NIfTI file and return its data array."""
    scan = nib.load(path)
    scan_array = scan.get_fdata()
    print(f'The scan data array has the shape: {scan_array.shape}')
    return scan, scan_array

def calculate_aspect_ratios(scan):
    """Calculate the aspect ratios using the pixel dimensions from the scan header."""
    pix_dim = scan.header['pixdim'][1:4]
    aspect_ratios = [pix_dim[1] / pix_dim[2], pix_dim[0] / pix_dim[2], pix_dim[0] / pix_dim[1]]
    print(f'The required aspect ratios are: {aspect_ratios}')
    return aspect_ratios

def resize_and_normalize(slice, new_dims, dim1, dim2):
    """Resize the given slice and normalize the intensity to an 8-bit range."""
    new_size = (128, 128)
    resampled_slice = cv2.resize(slice, new_size, interpolation=cv2.INTER_LINEAR)

    # Normalize and scale to 8-bit range
    resampled_slice = cv2.normalize(resampled_slice, None, 0, 255, cv2.NORM_MINMAX)
    resampled_slice = np.uint8(resampled_slice)

    return resampled_slice

def save_slices_custom(volume_arrays, aspect_ratios, output_path, slice_id):
    """Save slices for each modality and label in the given folder."""
    new_dims = np.multiply(volume_arrays['flair'].shape, aspect_ratios)
    new_dims = tuple(map(round, new_dims))

    selected_dim = [2] # include axial(0), coronal(1), sagital(2), get all => selected_dim = [0, 1, 2]
    # Process and save each slice
    for dim in selected_dim:  # Iterate over the three dimensions (axial, coronal, sagittal)
        for i in range(volume_arrays['flair'].shape[dim]):
            # Create a variable to track if the slice contains any non-zero data
            is_empty = False

            # Check if any of the modality or mask slices have non-zero values
            for modality, modality_array in volume_arrays.items():
                if dim == 0:
                    volume_slice = modality_array[i, :, :]
                elif dim == 1:
                    volume_slice = modality_array[:, i, :]
                else:
                    volume_slice = modality_array[:, :, i]

                if not(np.any(volume_slice)) and modality not in ['mask1', 'mask2']:
                    is_empty = True
                    break
                    
                slice_folder = os.path.join(output_path, f'{slice_id}')
                os.makedirs(slice_folder, exist_ok=True)
                
                resampled_volume_slice = resize_and_normalize(volume_slice, new_dims, (dim + 1) % 3, (dim + 2) % 3)
                
                name = f'image-{modality}_{slice_id}.jpg' if modality not in ['mask1', 'mask2'] else 'label0_.jpg' if modality == 'mask1' else 'label1_.jpg'
                cv2.imwrite(os.path.join(slice_folder, name), resampled_volume_slice)
                
            # Skip saving if the slice contains no data
            if is_empty:
                continue

            slice_id += 1

    return slice_id

def process_modality_files_custom(training_folder, output_path, slice_id):
    """Process each modality and corresponding masks in a given training folder."""
    preprocessed_path = os.path.join(training_folder, 'preprocessed')
    masks_path = os.path.join(training_folder, 'masks')

    all_files = sorted(os.listdir(preprocessed_path) + os.listdir(masks_path))

    files = {}
    
    for f in all_files:
        temp = f.split('_')
        if temp[1] not in files.keys():
            files[temp[1]] = [f]
        else:
            files[temp[1]].append(f)

    volume_arrays = {'flair': [],
                    'mprage': [],
                    'pd' : [],
                    't2' : [],
                    'mask1' : [],
                    'mask2' : []}
    
    for fs in files.values():
        for f in fs:
            k = f.split('_')[2].split('.')[0]
            if k in ['mask1', 'mask2']:
                image_path = os.path.join(masks_path, f)
            else:
                image_path = os.path.join(preprocessed_path, f)
            
            image, image_array = load_scan(image_path)
            volume_arrays[k] = image_array
    
        aspect_ratios = calculate_aspect_ratios(image)
        
        slice_id = save_slices_custom(volume_arrays, aspect_ratios, output_path, slice_id)

    return slice_id

def process_dataset(root_path, output_path):
    """Process all training folders in the dataset and store the results."""
    slice_id = 0  # Initialize the slice ID for naming each image uniquely
    # Iterate over each training folder
    for training_folder_name in sorted(os.listdir(root_path)):
        training_folder_path = os.path.join(root_path, training_folder_name)
        if os.path.isdir(training_folder_path):
            slice_id = process_modality_files_custom(training_folder_path, output_path, slice_id)

# Main execution
dataset_path = 'training'  # Adjust this path to your dataset directory
output_path = 'training_output1'  # Adjust this path to your output directory
os.makedirs(output_path, exist_ok=True)
process_dataset(dataset_path, output_path)
