# Bilateral Texture Filtering

## Overview
This repository hosts the MATLAB and C++ implementations of the Bilateral Texture Filter as presented in the paper by Cho, H., Lee, H., Kang, H., and Lee, S. (2014) titled "Bilateral texture filtering", published in ACM Transactions on Graphics, Vol. 33, No. 4, Article 128.

## Contents
The repository includes:
- MATLAB implementation of the bilateral texture filter.
- C++ implementation of the same algorithm.
- Over 30 sample images in the `images/` directory, demonstrating various textures. These images are named in the format `name-k-iter.ext`, where `k` is the optimal patch size and `iter` is the number of iterations for that image.

## Getting Started

### MATLAB Usage
1. Open the `main.m` file.
2. Set the input image `I`, patch size `K`, and number of iterations `ITER`.
3. Run the script by typing `main` in the MATLAB console.

### C++ Installation and Usage
```bash
cd Cpp
mkdir build
cd build
cmake ..
make -j
```

Note: While the code is optimized to utilize all CPU threads, it is not real-time. The image `compression4-5-15.png` is the most time-consuming, averaging 12.4206 seconds over 10 runs on an i7-13700K CPU with 24 cores.

![Alt text](Cpp/compression4-5-15-comparison.jpg)