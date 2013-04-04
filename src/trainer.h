/*
	nninc version 0.1
	Neural Networks in C
	Author: Utkarsh Desai

	trainer.h
	Header file for trainer.c
*/

#ifndef __TRAINER_H

#define __TRAINER_H

#include "config.h"
#include "data.h"

const double MIN_INITWT = -0.02;
const double MAX_INITWT = 0.02;
	
double* hid_deltas;
double* out_deltas;

int train(struct data_instance* dataset, struct data_instance* validationset, int dataset_size, struct network_config* configuration);

double get_random(double range);

double apply_actfn(double z);

void forward_propogate(int ninp, int nhid, int nout, double* ihwts, double* howts, double* inpvals, double* outvals, double* hidacts, double* outacts);

#endif
