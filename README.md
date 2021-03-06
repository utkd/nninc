nninc
=====

##Neural Networks in C

Neural Network implementation in C. Support for adaptable learning rate, multiple activation functions and softmax output (planned).

####Command line arguments
* `-train | -test <training/testing file>` (REQUIRED)

  Specify whether to perform training or testing using the given file. Testing requires a model file.
    
* `-config <filename>` 

  Provide a custom config file. If not provided, the program looks for a file `config.conf`.
  
* `-model <filename>`

  Provide a custom model file. If not specified, the program looks for a file `model.mdl`. This file is automatically generated after training, overwriting previous model files.

* `-validation <filename>`

  Use the specified validation file during training.
  
* `-output <filename>`

  File to print testset predictions to. If not provided, predictions are printed to `stdout`.

* `-saveacts`

  Save the activations of the hidden layer in a separate file. Use this option in the testing phase to save hidden layer activations to a file `activations.out`. This is particularly useful when creating a network of stacked autoencoders. The saved activations can be used as input to the next autoencoder.

####Instructions:
1. Build the code using the provided Makefile.
2. Setup a config file to describe the network and learning parameters.
3. Generate training, validation(optional) and test files.
4. Train and test the network

####Building the code
Run `make` in the project root directory to build the project. The executable will be named `nninc` .

####Config File
A config file should be present in the same directory as the executable. 
The program searches for a file named `config.conf` by default. You can provide any other file in the proper format through a command line argument `-config <filename>`.

###### A sample config file is included. Modify the values of the parameters as required.

####Data Formats
The training, testing and validation files are required to be in the same format. Only numerical values are accepted. Each line represents one data instance. All attributes (input and output) should be separated by spaces, with output values following inputs. The number of input and output values per line must match the numbers specified in the config file.

######Example:
For a dataset with 6 input attributes and 2 output variables, a line in any of the input files should look like this:
`0.3 0.404 -0.11 0.005 1.0 0.95 0.01`

