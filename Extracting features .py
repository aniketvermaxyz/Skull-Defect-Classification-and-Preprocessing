import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, feature, filters, morphology, measure, exposure
from skimage.feature import greycomatrix, greycoprops
from skimage.transform import resize

def extract_features(image):
    # Pixel Intensity Statistics
    intensity_mean = np.mean(image)
    intensity_std = np.std(image)

    # Geometric Features
    contours = measure.find_contours(image, 0.8)
    num_contours = len(contours)

    # Texture Analysis using Gray Level Co-occurrence Matrix (GLCM)
    glcm = greycomatrix((image * 255).astype(np.uint8), [1], [0], 256, symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]

    # Edge Detection using Canny Edge Detector
    edges = feature.canny(image)

    return intensity_mean, intensity_std, num_contours, contrast, energy, correlation, homogeneity, edges

# Set the directory containing your dataset
dataset_dir = '/content/sample_data/preprocessed'

# Get the list of image files in the dataset directory
image_files = os.listdir(dataset_dir)

# Create a directory to save the extracted features
output_dir = '/path/to/extracted_features'
os.makedirs(output_dir, exist_ok=True)

# Loop over each image file in the dataset
for image_file in image_files:
    # Construct the full path to the image file
    image_path = os.path.join(dataset_dir, image_file)

    # Load the image
    image = io.imread(image_path)

    # Convert the RGB image to grayscale
    grayscale_image = color.rgb2gray(image)

    # Resize the grayscale image if necessary
    resized_image = resize(grayscale_image, (256, 256))

    # Extract features from the image
    features = extract_features(resized_image)

    # Save the extracted features as a numpy array
    output_path = os.path.join(output_dir, f"{image_file}.npy")
    np.save(output_path, features)

    # Print the extracted features
    print("Image:", image_file)
    print("Pixel Intensity Statistics:")
    print("Mean Intensity:", features[0])
    print("Standard Deviation:", features[1])
    print("")

    print("Geometric Features:")
    print("Number of Contours:", features[2])
    print("")

    print("Texture Analysis:")
    print("Contrast:", features[3])
    print("Energy:", features[4])
    print("Correlation:", features[5])
    print("Homogeneity:", features[6])
    print("")

    # Display the original image and edges (optional)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(resized_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].axis('off')

    axes[1].imshow(features[-1], cmap='gray')
    axes[1].set_title('Edges')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


