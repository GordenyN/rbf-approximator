##  What the project does:
* Generates data for the function `x^2 * sin(x)`
* Builds a Radial Basis Function Neural Network (RBFNet)
* Trains the model on the generated data
* Visualizes:

  * The comparison between the predicted output and the true function
  * The activation curves of all RBF neurons

##  Model Architecture
* Custom implementation of RBF (Gaussian) functions
* Trainable RBF centers
* A linear output layer

## Additional functionality
tune_sigma.py — a separate script to automatically search for the optimal sigma parameter in the RBF network by training multiple models with different sigma values, and plotting the loss vs sigma graph.


##  Requirements
* TensorFlow
* NumPy
* Matplotlib

