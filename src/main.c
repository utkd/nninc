/*
	nninc version 0.1
	Neural Networks in C
	Author: Utkarsh Desai

	main.c
	Starting point of the program
*/

#include <stdio.h>
#include <string.h>
#include "config.h"
#include "data.h"

/*
	main function
	Arguments Required:
	1. -train | -test

	Optional Arguments:
	1. custom config file: -config <filename>
	2. custom model file: -model <filename>
	3. validation file: -validation <filename>
*/
int main(int argc, char* argv[]) {
	struct network_config configuration;
	const char* USAGE_STRING = "Usage: nninc -[train|test] <train/test file> <options>\nCheck README.txt for more details on options\n";
	const char* config_option = "-config";
	const char* model_option = "-model";
	const char* validation_option = "-validation";
	const char* output_option = "-output";
	char *config_file = "config.conf";
	char *model_file = "model.mdl";
	char *validation_file = NULL;
	char *data_file;
	char *output_file = NULL;
	int option;
	int do_training = 1;
	int do_validation = 0;
	int i;
	
	/* Check if required arguments are provided */
	if(argc < 3){
		printf("Insufficient arguments.\n%s", USAGE_STRING);
		return 1;
	}
	/* Check if training or testing is specified, if yes set a flag */
	if(strcmp("-train", argv[1]) == 0){
		do_training = 1;
	}
	else if(strcmp("-test", argv[1]) == 0) {
		do_training = 0;
	}
	else{
		printf("Incorrect usage.\n%s", USAGE_STRING);
		return 1;	
	}

	data_file = argv[2];

	/* Load optional arguments if specified */
	if(argc > 2) {
		for(option = 2; option < argc; option++){
			if(strcmp(argv[option], config_option) == 0)
				config_file = argv[++option];
			if(strcmp(argv[option], model_option) == 0)
				model_file = argv[++option];
			if(strcmp(argv[option], validation_option) == 0)
				validation_file = argv[++option];
			if(strcmp(argv[option], output_option) == 0)
				output_file = argv[++option];
		}
	}

	if(validation_file)
		do_validation = 1;

	/* Read configuration settings */
	int result = read_config(config_file, &configuration);
	if(result == 0) {
		printf("Error parsing config file.\n");
		return 1;
	}
	else {
		printf("Configuration loaded successfully.\n");
	}

	/* Read the data file*/
	struct data_instance* dataset = NULL;
	int num_instances = read_data(data_file, &dataset, configuration.num_input_nodes, configuration.num_output_nodes);
	if(num_instances == -1){
		printf("Something went wrong in reading data set. Exiting.\n");
		return 1;
	}
	else {
		printf("Read %d training instances successfully.\n", num_instances);
	}

	/* Read the validation file*/
	struct data_instance* validationset = NULL;
	if(do_validation) {
		int num_validation_instances = read_data(validation_file, &validationset, configuration.num_input_nodes, configuration.num_output_nodes);
		if(num_validation_instances == -1){
			printf("Something went wrong in reading validation set. Exiting.\n");
			return 1;
		}
		else {
			printf("Read %d validation instances successfully.\n", num_validation_instances);
		}
	}

	/* Perform training or testing, as specified */
	if(do_training) {
		result = train(dataset, validationset, num_instances, &configuration);
		if(result == 0){
			printf("Training failed.\n");
		}
	}
	else {
		result = test(dataset, model_file, output_file, num_instances);
		if(result == 0){
			printf("Testing failed.\n");
		}
	}
	return 0;
}
