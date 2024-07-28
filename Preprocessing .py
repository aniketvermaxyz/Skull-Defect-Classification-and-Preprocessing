import os
import matplotlib.pyplot as plt
import nibabel as nib
from skimage import filters, exposure
from skimage.transform import resize

def preprocess_image(image_path, target_size=(256, 256, 256)):
    # Step 1: Load the image
    image = nib.load(image_path).get_fdata()

    # Step 2: Resize the image to the target size
    resized_image = resize(image, target_size, mode='constant', anti_aliasing=True)

    # Step 3: Apply noise reduction using Gaussian blur
    denoised_image = filters.gaussian(resized_image, sigma=1)

    # Step 4: Normalize the pixel intensities to the range [0, 1]
    normalized_image = exposure.rescale_intensity(denoised_image)

    return normalized_image

# Set the directory containing your dataset
dataset_dir = '/content/sample_data/output'

# Get the list of image files in the dataset directory
image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]

# Create a directory to save the preprocessed images
output_dir = '/content/sample_data/preprocessed'
os.makedirs(output_dir, exist_ok=True)

# Loop over each image file in the dataset
for image_file in image_files:
    # Construct the full path to the image file
    image_path = os.path.join(dataset_dir, image_file)

    # Load the original image to get its size
    original_image = nib.load(image_path).get_fdata()
    original_size = original_image.shape

    # Preprocess the image
    preprocessed_image = preprocess_image(image_path, target_size=original_size)

    # Save the preprocessed image
    output_path = os.path.join(output_dir, image_file)
    nib.save(nib.Nifti1Image(preprocessed_image, affine=nib.load(image_path).affine), output_path)

    # Display the original and preprocessed images (optional)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(nib.load(image_path).get_fdata()[:, :, original_size[2]//2], cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(preprocessed_image[:, :, original_size[2]//2], cmap='gray')
    axes[1].set_title('Preprocessed Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
