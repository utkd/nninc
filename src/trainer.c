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
	double momentum = configuration->momentum;
	int batchsz = configuration->batch_size;

	/* Allocate variables to store weights, activations and deltas */
	int num_input = configuration->num_input_nodes;
	int num_hidden = configuration->num_hidden_nodes ;
	int num_output = configuration->num_output_nodes;
	double ih_wts[num_hidden][num_input + 1];
	double ho_wts[num_output][num_hidden + 1];
	double hid_acts[num_hidden + 1];
	double out_acts[num_output];

	double prev_ih_wtupdt[num_hidden][num_input + 1];
	double prev_ho_wtupdt[num_output][num_hidden + 1];

	double range = MAX_INITWT - MIN_INITWT;

	/* Allocate memory for deltas */
	hid_deltas = (double*)malloc(sizeof(double) * num_hidden);
	out_deltas = (double*)malloc(sizeof(double) * num_output);

	/* Randomly initialize weights */
	int seed = configuration->seed_value;
	if(seed < 0) {
		seed = time(0);
	}
	srand(seed);

	for(i = 0; i <num_hidden; i++)
		for(j = 0; j < num_input+1; j++)
			ih_wts[i][j] = get_random(range);
	for(i = 0; i <num_output; i++)
		for(j = 0; j < num_hidden+1; j++)
			ho_wts[i][j] = get_random(range);

	/* Initialize previous weight change values for momentum  */
	for(i = 0; i < num_hidden; i++)
		for(j = 0; j < num_input+1; j++)
			prev_ih_wtupdt[i][j] = 0;
	for(i = 0; i < num_output; i++)
		for(j = 0; j < num_hidden+1; j++)
			prev_ho_wtupdt[i][j] = 0;

	/* If a proper batch size is not specified, do online gradient descent */
	if(batchsz < 1)
		batchsz = 1;

	/* Initialize deltas to 0, for accumulation in minibatch gradient descent */
	for(i = 0; i < num_hidden; i++)
		hid_deltas[i] = 0;
	for(i = 0; i < num_output; i++)
		out_deltas[i] = 0;

	/* Set bias value in the hidden layer*/
	hid_acts[0] = 1.0;

	printf("Beginning Training ... \n");
	/* Iterate over the training set */
	for(iter = 0 ; iter < configuration->num_iterations; iter++) {
		double cost = 0;
		double wt_update = 0;
		int instance_num = 0;
		datanode = dataset;
		while(datanode != NULL) {
			inp_vals = datanode->input;
			out_vals = datanode->output;
			
			instance_num++;

			/***  Perform forward propagation  ***/
			forward_propogate(num_input, num_hidden, num_output, &ih_wts[0][0], &ho_wts[0][0], inp_vals, out_vals, hid_acts, out_acts);
			
			/* Compute cost */
			for(i = 0; i < num_output; i++)
				cost = cost + (out_acts[i] - out_vals[i]) * (out_acts[i] - out_vals[i]);

			/*** Perform backpropagation ***/
			/* Compute output layer deltas */
			for(i = 0; i < num_output; i++)
				out_deltas[i] += out_acts[i] * (1 - out_acts[i]) * (out_vals[i] - out_acts[i]);
			/* Compute hidden later deltas */
			for(i = 0; i < num_hidden; i++) {
				sum = 0;
				for(j = 0; j < num_output; j++)
					sum += out_deltas[j] * ho_wts[j][i+1];				
				hid_deltas[i] += hid_acts[i] * (1 - hid_acts[i]) * sum;
			}

			/* If we are not finished accumulating for this mini batch and not reached the end of the dataset  */
			if(instance_num % batchsz != 0 && instance_num < dataset_size)
				continue;

			/* Adjust weights between hidden and output layers*/
			for(i = 0; i < num_output; i++){
				for(j = 0; j < num_hidden+1; j++) {
					wt_update = learning_rate * hid_acts[j] * out_deltas[i] + momentum * prev_ho_wtupdt[i][j];
					ho_wts[i][j] += wt_update;
					prev_ho_wtupdt[i][j] = wt_update;
				}
			}
			/* Adjust weights between input and hidden layers*/
			for(i = 0; i < num_hidden; i++){
				for(j = 0; j < num_input+1; j++) {
					wt_update = learning_rate * inp_vals[j] * hid_deltas[i] + momentum * prev_ih_wtupdt[i][j];
					ih_wts[i][j] += wt_update;
					prev_ih_wtupdt[i][j] = wt_update;
				}
			}
	
			/* Reset accumulated deltas for next mini batch */
			for(i = 0; i < num_hidden; i++)
				hid_deltas[i] = 0;
			for(i = 0; i < num_output; i++)
				out_deltas[i] = 0;

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
				
			/* Forward propagation */
			forward_propogate(num_input, num_hidden, num_output, &ih_wts[0][0], &ho_wts[0][0], inp_vals, out_vals, hid_acts, out_acts);
			
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

void forward_propogate(int ninp, int nhid, int nout, double* ihwts, double* howts, double* inpvals, double* outvals, double* hidacts, double* outacts) {
	double sum;
	int i, j;
	/***  Perform forward propagation  ***/
	/* Compute hidden layer activations */
	for(i = 0; i < nhid; i++){
		sum = 0;
		for(j = 0; j < ninp+1; j++){
			sum += inpvals[j] * ihwts[i*(ninp+1)+j];
		}
		hidacts[i + 1] = apply_actfn(sum);			
	}
	/* Compute output layer activations */
	for(i = 0; i < nout; i++){
		sum = 0;
		for(j = 0; j < nhid+1; j++) {				
			sum += hidacts[j] * howts[i*(nhid+1)+j];
		}
		outacts[i] = apply_actfn(sum);
	}
}
