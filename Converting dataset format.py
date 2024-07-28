#Converting nrrd files into nii files
import os
import nibabel as nib
import numpy as np
from skimage import io

# Set the directory containing your NRRD dataset
dataset_dir = 'content/sample_data/datanrrd'

# Create a directory to save the NIfTI files
output_dir = 'content/sample_data/output'
os.makedirs(output_dir, exist_ok=True)

# Get the list of NRRD files in the dataset directory
nrrd_files = [f for f in os.listdir(dataset_dir) if f.endswith('.nrrd')]

# Loop over each NRRD file in the dataset
for nrrd_file in nrrd_files:
    # Construct the full path to the NRRD file
    nrrd_path = os.path.join(dataset_dir, nrrd_file)

    # Load the NRRD data using scikit-image
    nrrd_data = io.imread(nrrd_path)

    # Extract the data array and header information
    nrrd_array = np.array(nrrd_data, dtype=np.float32)
    nrrd_header = None  # You can provide the header information if available

    # Create a NIfTI image from the data and header
    nifti_img = nib.Nifti1Image(nrrd_array, None, header=nrrd_header)

    # Save the NIfTI image as a file
    nifti_file = nrrd_file.replace('.nrrd', '.nii')
    nifti_path = os.path.join(output_dir, nifti_file)
    nib.save(nifti_img, nifti_path)
