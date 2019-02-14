# VAE-GAN-CelebA
Python code related to the paper: ["Reconstructing Faces from fMRI Patterns using Deep Generative Neural Networks"](https://arxiv.org/abs/1810.03856) by VanRullen &amp; Reddy (2019)

### This folder contains:
* a pre-trained VAE-GAN model checkpoint 'vaegan_celeba.ckpt' (~15 epochs on CelebA dataset=50,000 batches of 64 images)
* a set of .py functions for the VAE-GAN face decomposition/reconstruction model, in particular:
  * VAEGAN_image2latent.py => goes from any image file to the corresponding 1024D latent encoding (saved as a Matlab .mat file)
  * VAEGAN_latent2image.py => goes from a 1024D latent encoding (Matlab .mat file) to the corresponding image(s)
* (optional) [a link to download the fMRI datasets](https://openneuro.org/datasets/ds001759) (4 subjects, each saw > 8,000 faces in the scanner) and some Matlab analysis code

### Example usage:
    VAEGAN_image2latent.py -i example.jpg     #this will create example_z.mat with the 1024 latent vars
    VAEGAN_latent2image.py -i example_z.mat   #this will generate example_z_g.jpg (and also example_z_g.mat)

### Requirements:
* Python >= 3.4
* Tensorflow >= 1.8
* git-lfs to download the pre-trained model checkpoint (>500MB)
* matplotlib, numpy, scipy, skimage
