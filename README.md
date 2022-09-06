#Plain C Neural Network - MNIST Classification - 97% Accuracy

###To download and use the MNIST dataset :<br />
https://github.com/takafumihoriuchi/MNIST_for_C

##How it works

###Training
1. Make sure you have downloaded the MNIST dataset.
2. Go to **training** folder, type `make` and press *Enter*.<br />
3. Type `./training` and press *Enter*.<br />
The training process should take about 10 minutes.

###Validation
1. Go to **test** folder, type `make` and press *Enter*.<br />
2. Type `./test` and press *Enter*.<br />
You should get an accuracy of approximately 97% depending on the training session.

##Neural Network Details

###Neural Network Architecture :
Input layer: 784 (28x28 pixels)<br />
Hidden layer 1: 100 (ReLU)<br />
Hidden layer 2: 100 (ReLU)<br />
Output layer: 10 (Softmax)

###Neural Network Training :
Training sets: 60000<br />
Batch size: 10<br />
Learning rate: 0.01<br />
Epochs: 10

###Neural Network Validation:
Validation sets: 10000<br />
Accuracy: 97%
