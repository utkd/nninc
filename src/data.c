/*
	nninc version 0.1
	Neural Networks in C
	Author: Utkarsh Desai

	data.c
	Code to deal with data input and conversions
*/

#include <stdio.h>
#include <stdlib.h>
#include "data.h"

int read_data(char* filename, struct data_instance** dataset_head, int num_input, int num_output) {
	int i, j;
	double value;
	int num_data_instances = 0;
	FILE* datafp = fopen(filename, "r");
	if(!datafp) {
		printf("Error reading data file %s\n", filename);
		return -1;
	}

	struct data_instance* current = NULL;
	printf("Reading input file ...\n");
	while(1) {
		struct data_instance* datanode = (struct data_instance*)malloc(sizeof(struct data_instance));
		double* inputs = (double*)malloc(sizeof(double) * (num_input + 1));	// +1 for bias
		double* outputs = (double*)malloc(sizeof(double) * num_output);
		if(!inputs || !outputs){
			return -1;
		}

		inputs[0] = 1.0;
		/* Read in values from 1 line*/
		for(i = 1; i < num_input+1; i++){
			j = fscanf(datafp, "%lf", &value);
			if(j == EOF && i == 1) {
				fclose(datafp);
				return num_data_instances;
			}
				
			inputs[i] = value;
		}
		if(j == EOF)
			return -1;
		for(i = 0; i < num_output; i++){
			j = fscanf(datafp, "%lf", &value);
			outputs[i] = value;
		}
		if(j == EOF)
			return -1;
		datanode->input = inputs;
		datanode->output = outputs;
		datanode->next = NULL;

		num_data_instances++;

		if(*dataset_head == NULL) 
			*dataset_head = datanode;
		if(current != NULL)
			current->next = datanode;
		current = datanode;
	}
}