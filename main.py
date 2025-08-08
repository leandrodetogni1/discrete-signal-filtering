import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from scipy import ndimage


def loadAndPreprocessImage(imagePath):
    if not os.path.exists(imagePath):
        basePath = os.path.splitext(imagePath)[0]
        for ext in [".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
            altPath = basePath + ext
            if os.path.exists(altPath):
                imagePath = altPath
                break
        else:
            raise FileNotFoundError(f"Image not found: {imagePath}")

    image = Image.open(imagePath)
    grayImage = image.convert("L")
    imageArray = np.array(grayImage, dtype=np.float32)

    return grayImage, imageArray


# Manual 2D convolution implementation // NOT USED - ndimage is more efficient
def convolve2d(image, kernel):
    kHeight, kWidth = kernel.shape
    imgHeight, imgWidth = image.shape

    outputH = imgHeight - kHeight + 1
    outputW = imgWidth - kWidth + 1
    finalOutput = np.zeros((outputH, outputW), dtype=np.float32)

    for y in range(outputH):
        for x in range(outputW):
            roi = image[y:y + kHeight, x:x + kWidth] 
            finalOutput[y, x] = np.sum(roi * kernel)

    return finalOutput

def applyMostSimilarEmbossingFilter(imageArray):
    embossingKernel = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
    processedArray = ndimage.convolve(imageArray, embossingKernel)
    processedArray = processedArray + 128
    processedArray = np.clip(processedArray, 0, 255)

    return Image.fromarray(processedArray.astype(np.uint8))


def applyEmbossingFilter(imageArray):
    embossingKernel = (1 / 9) * np.array([[-2, 1, 0], [-1, 1, 1], [0, 1, 2]])
    processedArray = ndimage.convolve(imageArray, embossingKernel)
    processedArray = np.clip(processedArray, 0, 255)

    return Image.fromarray(processedArray.astype(np.uint8))


def applyEdgeDetection(imageArray):
    edgeKernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    edgesArray = ndimage.convolve(imageArray, edgeKernel)
    edgesArray = np.clip(edgesArray, 0, 255)
    binaryEdges = np.where(edgesArray < 128, 0, 255)

    return Image.fromarray(binaryEdges.astype(np.uint8))


def main():
    n = np.arange(0, 200)

    originalSignal = np.cos(0.08 * n) + 0.3 * np.sin(0.20 * n)

    noise = np.random.uniform(low=-0.3, high=0.3, size=len(n))
    noisySignal = originalSignal + noise

    unitStepFunction = np.where(n >= 0, 1, 0)
    filterParameters = [-0.25, 0.25, 0.5, 0.99]

    totalPlots = 2 + len(filterParameters)
    
    plt.figure(figsize=(15, 12))
    
    plt.subplot(totalPlots, 1, 1)
    plt.stem(n, originalSignal, linefmt='green', markerfmt='go')
    plt.title('Original Signal')
    plt.xlabel('n')
    plt.grid(True)
    
    plt.subplot(totalPlots, 1, 2)
    plt.stem(n, noisySignal, linefmt='red', markerfmt='ro')
    plt.title('Noisy Signal')
    plt.xlabel('n')
    plt.grid(True)
    
    for i, a in enumerate(filterParameters):
        hN = (1 - a) * (a**n) * unitStepFunction
        filteredOutput = np.convolve(noisySignal, hN, mode="full")
        filteredSignal = filteredOutput[:200]
        
        plt.subplot(totalPlots, 1, i + 3)
        plt.stem(n, filteredSignal, linefmt="blue")
        plt.title(f'Filtered Signal (a = {a})')
        plt.xlabel('n')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # Part 2 - Image Processing 
    imagePaths = ["images/image1.jpg", "images/image2.jpg"]

    originalImages = []
    embossedImages = []
    similarEmbossedImages = []
    edgeImages = []
    imageNames = []

    for i, imagePath in enumerate(imagePaths, 1):
        originalImage, imageArray = loadAndPreprocessImage(imagePath)

        embossedImage = applyEmbossingFilter(imageArray)
        edgeImage = applyEdgeDetection(imageArray)
        similarEmbossedImage = applyMostSimilarEmbossingFilter(imageArray)

        originalImages.append(originalImage)
        embossedImages.append(embossedImage)
        similarEmbossedImages.append(similarEmbossedImage)

        edgeImages.append(edgeImage)
        imageNames.append(f"Image {i}")


    if originalImages:
        plt.figure(figsize=(20, 10))

        for i in range(len(originalImages)):
            base_idx = i * 4
            
            plt.subplot(2, 4, base_idx + 1)
            plt.imshow(originalImages[i], cmap="gray")
            plt.title("Original")
            plt.axis("off")

            plt.subplot(2, 4, base_idx + 2)
            plt.imshow(embossedImages[i], cmap="gray")
            plt.title("Embossed Filter")
            plt.axis("off")

            plt.subplot(2, 4, base_idx + 3)
            plt.imshow(similarEmbossedImages[i], cmap="gray")
            plt.title("Most Similar Embossed")
            plt.axis("off")

            plt.subplot(2, 4, base_idx + 4)
            plt.imshow(edgeImages[i], cmap="gray")
            plt.title("Edge Detection")
            plt.axis("off")

          
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
