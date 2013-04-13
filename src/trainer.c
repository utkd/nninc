/*
	nninc version 0.1
	Neural Networks in C
	Author: Utkarsh Desai

	trainer.c
	Code for training a neural network with given data
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "trainer.h"

int train(struct data_instance* dataset, struct data_instance* validationset, int dataset_size, struct network_config* configuration) {
	int iter;
	struct data_instance* datanode = NULL;
	double* inp_vals;
	double* out_vals;
	double sum;
	int i, j;
	double learning_rate = configuration->learning_rate;

	/* Allocate variables to store weights */
	int num_input = configuration->num_input_nodes;
	int num_hidden = configuration->num_hidden_nodes ;
	int num_output = configuration->num_output_nodes;
	double ih_wts[num_hidden][num_input + 1];
	double ho_wts[num_output][num_hidden + 1];
	double hid_acts[num_hidden + 1];
	double out_acts[num_output];
	double hid_deltas[num_hidden];
	double out_deltas[num_output];

	double range = MAX_INITWT - MIN_INITWT;

	/* Randomly initialize weights */
	
	int seed = configuration->seed_value;
	if(seed < 0) {
		seed = time(0);
	}
	srand(seed);

	for(i = 0; i <num_hidden; i++){
		for(j = 0; j < num_input+1; j++) {
			ih_wts[i][j] = get_random(range);
		}
	}
	for(i = 0; i <num_output; i++){
		for(j = 0; j < num_hidden+1; j++) {
			ho_wts[i][j] = get_random(range);
		}
	}

	/* Set bias value in the hidden layer*/
	hid_acts[0] = 1.0;

	printf("Beginning Training ... \n");
	/* Iterate over the training set */
	for(iter = 0 ; iter < configuration->num_iterations; iter++) {
		double cost = 0;
		datanode = dataset;
		while(datanode != NULL) {
			inp_vals = datanode->input;
			out_vals = datanode->output;

			/***  Perform forward propagation  ***/
			/* Compute hidden layer activations */
			for(i = 0; i < num_hidden; i++){
				sum = 0;
				for(j = 0; j < num_input+1; j++){
					sum += inp_vals[j] * ih_wts[i][j];
				}
				hid_acts[i + 1] = apply_actfn(sum);			
			}

			/* Compute output layer activations */
			for(i = 0; i < num_output; i++){
				sum = 0;
				for(j = 0; j < num_hidden+1; j++) {				
					sum += hid_acts[j] * ho_wts[i][j];
				}
				out_acts[i] = apply_actfn(sum);
			}

			/* Compute cost */
			for(i = 0; i < num_output; i++)
				cost = cost + (out_acts[i] - out_vals[i]) * (out_acts[i] - out_vals[i]);

			/*** Perform backpropagation ***/
			/* Compute output layer deltas */
			for(i = 0; i < num_output; i++)
				out_deltas[i] = out_acts[i] * (1 - out_acts[i]) * (out_vals[i] - out_acts[i]);
			/* Compute hidden later deltas */
			for(i = 0; i < num_hidden; i++) {
				sum = 0;
				for(j = 0; j < num_output; j++)
					sum += out_deltas[j] * ho_wts[j][i+1];				
				hid_deltas[i] = hid_acts[i] * (1 - hid_acts[i]) * sum;
			}

			/* Adjust weights between hidden and output layers*/
			for(i = 0; i < num_output; i++){
				for(j = 0; j < num_hidden+1; j++)
					ho_wts[i][j] += learning_rate * hid_acts[j] * out_deltas[i];
			}
			/* Adjust weights between input and hidden layers*/
			for(i = 0; i < num_hidden; i++){
				for(j = 0; j < num_input+1; j++)
					ih_wts[i][j] += learning_rate * inp_vals[j] * hid_deltas[i];
			}

			datanode = datanode->next;
		}
		cost = cost / dataset_size;
		printf("Cost after iteration %3d: %lf\n", iter+1, cost);
	}
	
	/* Compute error on validation set if one is defined */
	if(validationset) {
		struct data_instance* validation_node = validationset;
		double cost = 0.0;
		while(validation_node != NULL) {
			inp_vals = validation_node->input;
			out_vals = validation_node->output;

			/***  Perform forward propagation  ***/
			/* Compute hidden layer activations */
			for(i = 0; i < num_hidden; i++){
				sum = 0;
				for(j = 0; j < num_input+1; j++){
					sum += inp_vals[j] * ih_wts[i][j];
				}
				hid_acts[i + 1] = apply_actfn(sum);			
			}

			/* Compute output layer activations */
			for(i = 0; i < num_output; i++){
				sum = 0;
				for(j = 0; j < num_hidden+1; j++) {				
					sum += hid_acts[j] * ho_wts[i][j];
				}
				out_acts[i] = apply_actfn(sum);
			}

			/* Compute cost */
			for(i = 0; i < num_output; i++)
				cost = cost + (out_acts[i] - out_vals[i]) * (out_acts[i] - out_vals[i]);
			validation_node = validation_node->next;
		}
		printf("Total error on validation set: %lf\n", cost);
	}
	
	/* Save the model file */
	FILE* fp = fopen("model.mdl", "w");
	if(fp) {
		printf("Saving model ... ");
		fprintf(fp, "%d %d %d\n", num_input, num_hidden, num_output);
		for(i = 0; i <num_hidden; i++){
			for(j = 0; j < num_input+1; j++) {
				fprintf(fp, " %lf", ih_wts[i][j]);
			}
		}
		fprintf(fp, "\n");
		for(i = 0; i <num_output; i++){
			for(j = 0; j < num_hidden+1; j++) {
				fprintf(fp, " %lf", ho_wts[i][j]);
			}
		}
		fclose(fp);
		printf("Saved\n");	
	}
	else
		printf("Unable to create model file.\n");
	
	return 1;
}

double get_random(double range) {
	double rvalue = (double)rand() / RAND_MAX;
	rvalue = MIN_INITWT + (rvalue * range);
	return rvalue;
}

double apply_actfn(double z) {
	return 1.0 / (1.0 + exp(-z));
}