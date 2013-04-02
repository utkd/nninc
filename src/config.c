/*
	nninc version 0.1
	Neural Networks in C
	Author: Utkarsh Desai

	config.c
	Code to read configuration file
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config.h"

char* keys[] = {"input", "hidden", "output", "iter", "learn", "seed", "momentum"};

/*
	Reads configuration settings from the config file
*/
int read_config(char* filename, struct network_config* config) {

	FILE *config_fh = fopen(filename, "r");
	if(!config_fh) {
		printf("Error opening config file %s.\n", filename);
		return 0;
	}

	char k[7][10];
	fscanf(config_fh, "%s %d", k[0], &config->num_input_nodes);
	fscanf(config_fh, "%s %d", k[1], &config->num_hidden_nodes);
	fscanf(config_fh, "%s %d", k[2], &config->num_output_nodes);
	fscanf(config_fh, "%s %d", k[3], &config->num_iterations);
	fscanf(config_fh, "%s %lf", k[4], &config->learning_rate);
	fscanf(config_fh, "%s %d", k[5], &config->seed_value);
	fscanf(config_fh, "%s %lf", k[6], &config->momentum);

	fclose(config_fh);

	return (validate_key(keys[0], k[0]) && validate_key(keys[1], k[1]) 
			&& validate_key(keys[2], k[2]) && validate_key(keys[3], k[3])
			&& validate_key(keys[4], k[4]) && validate_key(keys[5], k[5])
			&& validate_key(keys[6], k[6]));
}

int validate_key(char* required_key, char* input_key) {
	return (strcmp(required_key, input_key) == 0);
}
