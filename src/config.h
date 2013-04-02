/*
	nninc version 0.1
	Neural Networks in C
	Author: Utkarsh Desai

	config.h
	Header file for config.c
*/

#ifndef __CONFIG_H

#define __CONFIG_H

char* keys[];

struct network_config {
	int num_input_nodes;
	int num_hidden_nodes;
	int num_output_nodes;
	int num_iterations;
	double learning_rate;
	int seed_value;
	double momentum;
	int batch_size;
};

int read_config(char* filename, struct network_config* config);
int validate_key(char* required_key, char* input_key);

#endif
