nninc
=====

##Neural Networks in C

Neural Network implementation in C. Support for adaptable learning rate, multiple activation functions and softmax output (planned).

####Command line arguments
* -train | -test <training/testing file> (REQUIRED)

  Specify whether to perform training or testing using the required file. Testing requires a model file.
    
* -config <filename> 

  Provide a custom config file. If not provided, the program looks for a file `config.conf`.

####Instructions:
1. Build the code using the provided Makefile.
2. Setup a config file to describe the network and learning parameters.
3. Generate training, validation(optional) and test files.
4. Train and test the network

####Building the code
Run `make` in the project root directory to build the project. The executable will be named `nninc` .

####Config File
A config file should be present in the same directory as the executable. 
The program searches for a file name `config.conf` by default. You can provide any other file in the proper format through a command line argument `-config <filename>`.

