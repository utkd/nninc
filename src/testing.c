/*
	nninc version 0.1
	Neural Networks in C
	Author: Utkarsh Desai

	testing.c
	Code for testing a trained neural network
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "trainer.h"
#include "testing.h"

int test(struct data_instance* dataset, char* model_filename, char* output_filename, int dataset_size) {
	struct data_instance* datanode = NULL;
	int num_input, num_hidden, num_output;
	int i, j, t;
	FILE *mdl_fp;
	
	double* ih_wts = NULL;
	double* ho_wts = NULL;
	double* hid_acts = NULL;
	double* out_acts = NULL;
	
	/*** Load Model ***/
	mdl_fp = fopen(model_filename, "r");
	if(!mdl_fp){
		printf("Unable to open model file %s.\n", model_filename);
		return 0;
	}
	
	fscanf(mdl_fp, "%d", &num_input);
	fscanf(mdl_fp, "%d", &num_hidden);
	t = fscanf(mdl_fp, "%d", &num_output);
	if(t == EOF){
		printf("Model file not in correct format.\n");
		return 0;
	}

	/* Allocate memory for variables to weights and activations */
	ih_wts = (double*)malloc(sizeof(double) * ((num_input + 1) * num_hidden));
	ho_wts = (double*)malloc(sizeof(double) * ((num_hidden + 1) * num_output));

	hid_acts = (double*)malloc(sizeof(double) * (num_hidden + 1));
	out_acts = (double*)malloc(sizeof(double) * num_output);

	if(!ih_wts || !ho_wts || !hid_acts || !out_acts){
		printf("Unable to allocate memory for loading model.\n");
		return 0;
	}
	
	for(i = 0; i <num_hidden; i++)
		for(j = 0; j < num_input+1; j++)
			t = fscanf(mdl_fp, "%lf", &ih_wts[i*(num_input+1) + j]);
	for(i = 0; i <num_output; i++)
		for(j = 0; j < num_hidden+1; j++)
			t = fscanf(mdl_fp, "%lf", &ho_wts[i*(num_hidden+1) + j]);
	if(t == EOF){
		printf("Model file not in correct format.\n");
		return 0;	
	}
	fclose(mdl_fp);	
	
	printf("Model loaded successfully.\n");

	/* Set bias in hidden layer */
	hid_acts[0] = 1;

	FILE* out_fp = NULL;
	if(output_filename)
		out_fp = fopen(output_filename, "w");
	if(!out_fp)
		out_fp = stdout;
	
	double cost = 0;
	double* inp_vals;
	double* out_vals;
	/*** Iterate over the test set ***/
	datanode = dataset;
	while(datanode != NULL) {
		inp_vals = datanode->input;
		out_vals = datanode->output;
		
		/*  Perform forward propagation  */
		forward_propogate(num_input, num_hidden, num_output, &ih_wts[0], &ho_wts[0], inp_vals, out_vals, hid_acts, out_acts);	

		/* Compute cost */
		for(i = 0; i < num_output; i++)
			cost = cost + (out_acts[i] - out_vals[i]) * (out_acts[i] - out_vals[i]);

		/* Print output */
		for(i = 0; i < num_output; i++)
			fprintf(out_fp, "%lf ", out_acts[i]);
		fprintf(out_fp, "\n");

		datanode = datanode->next;
	}
	
	printf("Average squared error on test set: %lf\n", (cost/dataset_size));
	fclose(out_fp);
	return 1; 
}	
