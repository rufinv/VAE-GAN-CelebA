# VAE-GAN-CelebA
Code and notebooks related to the paper: "Reconstructing Faces from fMRI Patterns using Deep Generative Neural Networks" by VanRullen &amp; Reddy, 2019

This folder contains:
* a pre-trained VAE-GAN model checkpoint (~15 epochs on CelebA dataset=50,000 batches of 64 images)
* a set of .py functions for the VAE-GAN face decomposition/reconstruction model, in particular:
  * VAEGAN_image2latent.py => goes from any image file to the corresponding 1024D latent encoding (saved as a Matlab .mat file)
  * VAEGAN_latent2image.py => goes from a 1024D latent encoding (Matlab .mat file) to the corresponding image(s)
* together with a .ipynb notebook showing examples of usage
* (optional) a link to download the fMRI datasets (4 subjects, each saw > 8,000 faces in the scanner) and some Matlab analysis code

Requirements:
* Python >= 3.4
* Tensorflow >= 1.2
* git-lfs to download the pre-trained model checkpoint (>500MB)
