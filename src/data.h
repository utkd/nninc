/*
	nninc version 0.1
	Neural Networks in C
	Author: Utkarsh Desai

	data.h
	Header file for data.c
*/

#ifndef __DATA_H

#define __DATA_H

struct data_instance {
	double* input;
	double* output;
	struct data_instance* next;
};

int read_data(char* filename, struct data_instance** dataset_head, int num_input, int num_output);

#endif

