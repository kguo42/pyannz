# pyannz
The code use MLP to estimate redshfits with photometry. A Pytorch Version of ANNz with modifications.

This is for CE397 class project. The goal is this project is to implement a Pytorch Version of ANNz (with modifications).

In the project, we follow the steps outlined in <em>ANNz: Estimating Photometric Redshifts Using Artificial Neural Networks</em> [Collister, Adrian A. & Lahav, Ofer, (2004)](https://ui.adsabs.harvard.edu/abs/2004PASP..116..345C/abstract), construct a neural network model to learn to map from photometry to redshift. 

Test, Training, and validation data are obtained from: 
https://www.homepages.ucl.ac.uk/~ucapola/annz.html

An extra notebook trained on CANDELS data has been added.

For package dependencies, check requirement.txt or do the following interminal:

pip install -r requirements.txt