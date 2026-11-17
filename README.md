# Denoising Autoencoder on MNIST

A small PyTorch denoising autoencoder trained on a 10,000-image subset of MNIST.
The model receives a noisy digit and learns to reconstruct the clean original.
Built as a first project for learning about autoencoders.

![Denoising results](results.png)
*Top: clean test images. Middle: with Gaussian noise (σ = 0.3). Bottom: model output.*

## How it works

- **Noise:** Gaussian noise with standard deviation `sigma` is added to each image
  and clipped to [0, 1]. Fresh noise is sampled every batch.
- **Model:** a fully connected autoencoder. The encoder compresses 784 pixels into a
  `latent`-dimensional vector, and the decoder reconstructs the image.
- **Training:** MSE loss between the reconstruction of the *noisy* input and the
  *clean* image, optimized with Adam (lr = 1e-3, 10 epochs).

## Run it

```bash
pip install -r requirements.txt
jupyter notebook autoencoder.ipynb
```
MNIST downloads automatically on first run.

## Experiments

Each run changes a single variable from the baseline ('latent=32', 'sigma=0.3)

| latent | sigma | final train loss | observation |
|--------|-------|------------------|-------------|
| 32     | 0.3   |   0.0245         | baseline    |
| 2      | 0.3   |   0.0516         | clearer image but less precise|
| 32     | 0.6   | 0.0340           | better quality, low precision |
| 128    | 0.3   |   0.0222         | better quality, better precision |

### Effect of latent size
![latent=2](results/latent2_sigma0.3.png)
*Smooth and clean outputs, but often gets the number wrong*

![latent=128](results/latent128_sigma0.3.png)
*image is very clean and numbers are correctly identified*
### Effect of noise level

![sigma=0.6](results/latent32_sigma0.6.png)
*Cleaner outputs but sometimes gets the number wrong*

## What I learned

latent reflect model's capacity. it's important to find the optimal latent value such that it's not too small (to be able to store information) and not too big (model just copies data and doesn't learn anything). 

sigma reflects the noise. higher values lead to unclear input and the model has to guess more.

## Next steps

- Try a convolutional autoencoder