# JPEG encoder-decoder
This script is intended to be used solely for academic purposes:

1. Analysis of the effects of approximating an image with a partial set of DCT coefficients. It applies a 8x8 pixels block DCT and keeps only the first K selected coefficients. The IDCT of these coefficients is shown and image quality assessment metrics are computed.

2. Analysis of the effects of quantizing the DCT coefficients of an image using the JPEG quantization matrix and the provided scaling factor. The IDCT of the quantized coefficients is shown and image quality assessment metrics are computed.
No subsamping us applied (4:4:4)

More info: https://en.wikipedia.org/wiki/JPEG#JPEG_codec_example

---

## Requirements
- Python3
- Python3 pip
- Python3 venv

## Usage

Create an environment to install the required pip3 packages

```bash
python3 -m venv env
```

Activate the environment

```bash
source env/bin/activate
```

Install the required packages

```bash
pip3 install -r requirements.txt
```

Run the script and pass an image as an argument

```bash
python3 main.py --input <path_to_image> --coeffs <number_of_coefficients>
```

### Example 

```bash
python3 main.py --input ~/Pictures/flowers.jpg --coeffs 16
```