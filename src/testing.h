/*
	nninc version 0.1
	Neural Networks in C
	Author: Utkarsh Desai

	testing.h
	Header file for testing.c
*/

#ifndef __TESTING_H

#define __TESTING_H

#include "data.h"

int test(struct data_instance* dataset, char* model_filename, char* output_filename, int dataset_size);

#endif
