# Discrete Signal Filtering

This project implements discrete signal filtering using convolution in both 1D and 2D domains. It demonstrates noise reduction in discrete signals and image processing techniques including edge detection and embossing effects.

## Features

- **Signal Filtering**: Filters noisy discrete signals using convolution with different filter parameters
- **Image Processing**: Applies convolution kernels to images for:
  - Edge detection
  - Embossing effects
  - Noise reduction

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd discrete-signal-filtering
```

2. Install the required dependencies:

```bash
pip3 install -r requirements.txt
```

3. Add your images to the `images/` folder:

   - `image1.jpg` - First image for processing
   - `image2.jpg` - Second image for processing

   **Note**: The script will run without images, but 2D processing examples will be skipped.

## Usage

Run the main script:

```bash
python3 main.py
```

## Signal Processing Details

### Signal Filtering

- Original signal: `s(n) = cos(0.08*n) + 0.3*sin(0.20*n)`
- Noise: Uniform random noise in range [-0.3, 0.3]
- Filter: `h(n) = (1-a) * a^n * u(n)` where u(n) is the unit step function

### Image Processing

**Note**: The 1D signal processing will work immediately. For 2D image processing, add your JPG images to the `images/` folder.

- **Embossing kernel**:
  ```
  [-2, 1, 0]
  [-1, 1, 1] * (1/9)
  [0, 1, 2]
  ```
- **Edge detection kernel**:
  ```
  [-1, -2, -1]
  [0,   0,  0]
  [1,   2,  1]
  ```
