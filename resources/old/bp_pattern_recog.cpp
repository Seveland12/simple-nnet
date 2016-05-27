// bp pattern recog.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

// Back-Prop for Pattern Recognition

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_SPACE 15
#define DIMENSION 4
#define OUTPUT_SPACE 4

#define NUM_TRIALS 500

#define RHO 2.0
#define MAX_EPOCHS 5000

static int minarg1,minarg2;
#define MIN(a,b) (minarg1=(a),minarg2=(b),(minarg1) < (minarg2) ? (minarg1) : (minarg2))
#define FSIGNUM(a) ((a) >= (0.0) ? (1) : (-1))

#define LEARN_DIR_LIMIT 25 // number of consecutive iterations with the same gradient sign before learning rate adjustment

#define sigmoid_activation_function(x) ((1.7159 * tanh(0.66666*(x))))
#define sigmoid_activation_function_deriv(x) ((1.1439333) / pow((cosh(0.66666*(x))),2.0))

/**********************************************************************/

typedef struct{
    int count[INPUT_SPACE + 1][DIMENSION];
    int prev[INPUT_SPACE + 1][DIMENSION];
    int direction[INPUT_SPACE + 1][DIMENSION];
} wi_adjuster;

typedef struct {
    int count[DIMENSION + 1][OUTPUT_SPACE];
    int prev[DIMENSION + 1][OUTPUT_SPACE];
    int direction[DIMENSION + 1][OUTPUT_SPACE];
} wo_adjuster;

typedef struct {
    double wi[INPUT_SPACE + 1][DIMENSION];
    double wi_prev[INPUT_SPACE + 1][DIMENSION];

    double learning_rate_wi[INPUT_SPACE + 1][DIMENSION];
    double momentum_constant_wi[INPUT_SPACE + 1][DIMENSION];
    wi_adjuster lr_wi_adjust;

    double wo[DIMENSION + 1][OUTPUT_SPACE];
    double wo_prev[DIMENSION + 1][OUTPUT_SPACE];
    wo_adjuster lr_wo_adjust;

    double learning_rate_wo[DIMENSION + 1][OUTPUT_SPACE];
    double momentum_constant_wo[DIMENSION + 1][OUTPUT_SPACE];

    double input[INPUT_SPACE + 1];
    double output[OUTPUT_SPACE];

    double hidden[DIMENSION + 1];

    double desired_response[OUTPUT_SPACE];
    double error_signal[OUTPUT_SPACE];

    double d_output[OUTPUT_SPACE];
    double d_hidden[DIMENSION];

    double v_output[OUTPUT_SPACE];
    double v_hidden[DIMENSION];
} neural_network;

double digits[10][INPUT_SPACE]  = {{1.0,1.0,1.0,1.0,0.0,1.0,1.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0},
                                   {0.0,1.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,1.0,1.0,1.0},
                                   {1.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0},
                                   {1.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,1.0},
                                   {1.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0},
                                   {1.0,1.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,1.0},
                                   {1.0,1.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0},
                                   {1.0,1.0,1.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0},
                                   {1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0},
                                   {1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0}};


double input_mean[INPUT_SPACE] = {0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5};


double zero_mean_digits[10][INPUT_SPACE] = {{0.5,0.5,0.5,0.5,-0.5,0.5,0.5,-0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5},
                                   {-0.5,0.5,-0.5,0.5,0.5,-0.5,-0.5,0.5,-0.5,-0.5,0.5,-0.5,0.5,0.5,0.5},
                                   {0.5,0.5,0.5,-0.5,-0.5,0.5,0.5,0.5,0.5,0.5,-0.5,-0.5,0.5,0.5,0.5},
                                   {0.5,0.5,0.5,-0.5,-0.5,0.5,0.5,0.5,0.5,-0.5,-0.5,0.5,0.5,0.5,0.5},
                                   {0.5,-0.5,0.5,0.5,-0.5,0.5,0.5,0.5,0.5,-0.5,-0.5,0.5,-0.5,-0.5,0.5},
                                   {0.5,0.5,0.5,0.5,-0.5,-0.5,0.5,0.5,0.5,-0.5,-0.5,0.5,0.5,0.5,0.5},
                                   {0.5,0.5,0.5,0.5,-0.5,-0.5,0.5,0.5,0.5,0.5,-0.5,0.5,0.5,0.5,0.5},
                                   {0.5,0.5,0.5,-0.5,-0.5,0.5,-0.5,0.5,-0.5,-0.5,0.5,-0.5,-0.5,0.5,-0.5},
                                   {0.5,0.5,0.5,0.5,-0.5,0.5,0.5,0.5,0.5,0.5,-0.5,0.5,0.5,0.5,0.5},
                                   {0.5,0.5,0.5,0.5,-0.5,0.5,0.5,0.5,0.5,-0.5,-0.5,0.5,-0.5,-0.5,0.5}};


double one_pixel_off_zero_mean_digits[10][INPUT_SPACE][INPUT_SPACE]; //this is the matrix of one-pixel-off digits for testing generalization performance

/*********************************************************************/

int binary_outputs[10][OUTPUT_SPACE]  = {{0,0,0,0},
                                         {1,0,0,0},
                                         {0,1,0,0},
                                         {1,1,0,0},
                                         {0,0,1,0},
                                         {1,0,1,0},
                                         {0,1,1,0},
                                         {1,1,1,0},
                                         {0,0,0,1},
                                         {1,0,0,1}};

double zero_mean_binary_outputs[10][OUTPUT_SPACE] = {{-0.5,-0.5,-0.5,-0.5},
													{0.5,-0.5,-0.5,-0.5},
													{-0.5,0.5,-0.5,-0.5},
													{0.5,0.5,-0.5,-0.5},
													{-0.5,-0.5,0.5,-0.5},
													{0.5,-0.5,0.5,-0.5},
													{-0.5,0.5,0.5,-0.5},
													{0.5,0.5,0.5,-0.5},
													{-0.5,-0.5,-0.5,0.5},
													{0.5,-0.5,-0.5,0.5}};

double binary_output_mean[OUTPUT_SPACE] = {0.5, 0.5, 0.5, 0.5};


/***************************************************************/



/**********************************************************************/
void calculate_local_gradients(neural_network *nnet)
{
    int j,k;
    double sum;

    for(j = 0; j < OUTPUT_SPACE; j++)
    {
        //the vector d_output[j] is the local gradient vector of the synaptic weights between the output and hidden layers
        nnet->d_output[j] = (RHO / 2.0) * pow((nnet->desired_response[j] - nnet->output[j]), RHO - 1.0) * sigmoid_activation_function_deriv(nnet->v_output[j]);
    }

    for(j = 0; j < DIMENSION; j++)
    {
        sum = 0.0;
        for(k = 0; k < OUTPUT_SPACE; k++)
        {
            sum += (nnet->d_output[k] * nnet->wo[j][k]);
        }
        // the vector d_hidden[j] is the local gradient vector of the synaptic weights between the input and hidden layers
        nnet->d_hidden[j] = sigmoid_activation_function_deriv(nnet->v_hidden[j]) * sum;
    }
}

/*********************************************************************/
void initialize_learning_rates_and_momentums(neural_network *nnet)
{
    int x,y,z;

    for (x = 0; x < INPUT_SPACE + 1; x++)
    {
        for(y = 0; y < DIMENSION; y++)
        {
            nnet->learning_rate_wi[x][y] = 0.01;
            nnet->momentum_constant_wi[x][y] = 0.000;

            nnet->lr_wi_adjust.count[x][y] = 0;
            nnet->lr_wi_adjust.prev[x][y] = 0;
            nnet->lr_wi_adjust.direction[x][y] = 1;
        }
    }

    for(y = 0; y < DIMENSION + 1; y++)
    {
        for(z = 0; z < OUTPUT_SPACE; z++)
        {
            nnet->learning_rate_wo[y][z] = 0.01;
            nnet->momentum_constant_wo[y][z] = 0.000;

            nnet->lr_wo_adjust.count[y][z] = 0;
            nnet->lr_wo_adjust.prev[y][z] = 0;
            nnet->lr_wo_adjust.direction[y][z] = 1;
        }
    }
}


/*********************************************************************/
void initialize_weights(neural_network *nnet)
{
	int x;
	int y;
    int z;

	double r,m;

    for(x = 0;x < INPUT_SPACE + 1; x++)
    {
        for(y = 0; y < DIMENSION; y++)
        {
            r = 0.0 + (double) (10.0*rand()/(RAND_MAX+1.0));
            m = (double) INPUT_SPACE;
            nnet->wi[x][y] = pow(2.0/m,0.5)*((2.0*r/9.0)-1.0);
        }
    }

    for(y = 0; y < DIMENSION + 1; y++)
    {
        for(z = 0; z < OUTPUT_SPACE; z++)
        {
            r = 0.0 + (double) (10.0*rand()/(RAND_MAX+1.0));
            m = (double) DIMENSION;
            nnet->wo[y][z] = pow(2.0/m,0.5)*((2.0*r/9.0)-1.0);
        }
    }
}

/*********************************************************************/
void set_previous_weight_vectors(neural_network *nnet)
{
    int x,y;

	for (x = 0; x < INPUT_SPACE + 1; x++)
	{
		for (y = 0; y < DIMENSION; y++)
        {
            nnet->wi_prev[x][y] = nnet->wi[x][y];
        }
	}
	for (x = 0; x < DIMENSION + 1; x++)
	{
        for (y = 0; y < OUTPUT_SPACE; y++)
		{
		    nnet->wo_prev[x][y] = nnet->wo[x][y];
		}
	}
}

/*********************************************************************/
double output_value(neural_network nnet)
{
	int x;
	double val;

	val = 0.0;
	for (x = 0; x < OUTPUT_SPACE; x++)
		{
		//val = val + threshold(output[x] + binary_output_mean[x]) * (double)(pow(2.0,x));
		val = val + ((nnet.output[x] + binary_output_mean[x]) * (pow(2.0,x)));

		//val = val + (output[x] * (pow(2.0,x)));
		}
	return val;
}

/**************************************************************/
void set_input(int n, neural_network *nnet)
{
	int x;

	for (x = 0; x < INPUT_SPACE; x++)
		nnet->input[x] = zero_mean_digits[n][x];
}

/**************************************************************/
void compute_hidden_layer(neural_network *nnet)
{
	int i;
	int j;

	for (j = 0; j < DIMENSION; j++)
	{
		nnet->v_hidden[j] = 0.0;
		for (i = 0; i < INPUT_SPACE + 1; i++)
		{
		    nnet->v_hidden[j] += (nnet->wi[i][j] * nnet->input[i]);
		}
		nnet->hidden[j] = sigmoid_activation_function(nnet->v_hidden[j]);
		//printf("hidden[%d] = %f\n",j,nnet->wo[i][j]);
	}
}

/***************************************************************/
void compute_output_layer(neural_network *nnet)
{
	int i, j;

	for (j = 0; j < OUTPUT_SPACE; j++)
	{
		nnet->v_output[j] = 0.0;
		for (i = 0; i < DIMENSION + 1; i++)
		{
        	nnet->v_output[j] += (nnet->wo[i][j] * nnet->hidden[i]);
		}
		//output[j] = threshold(v_output[j]);
		nnet->output[j] = sigmoid_activation_function(nnet->v_output[j]);
	}
}

/***************************************************************/
double calculated_error(neural_network nnet)
{
    int x;
    double sub_total = 0.0;

    for(x = 0; x < OUTPUT_SPACE; x ++)
    {
        sub_total += pow((nnet.desired_response[x] - nnet.output[x]),RHO);
    }
    return (0.5*sub_total);
}

/***************************************************************/
void adjust_output_layer_weights(neural_network *nnet)
{
    int i,j;
    double holder;
    double partial_dE_dWo;
    int curr_sign;

    for(j = 0; j < OUTPUT_SPACE; j++)
    {
        for (i = 0; i < DIMENSION + 1; i++)
        {
            holder = nnet->wo_prev[i][j];
            nnet->wo_prev[i][j] = nnet->wo[i][j];
            partial_dE_dWo = (nnet->learning_rate_wo[i][j]) * nnet->d_output[j] * nnet->hidden[i];
            nnet->wo[i][j] = nnet->wo_prev[i][j] + ((nnet->momentum_constant_wo[i][j]) * holder) + partial_dE_dWo;
            //nnet->wo[i][j] = nnet->wo_prev[i][j] + ((nnet->momentum_constant_wo[i][j]) * holder) + ((nnet->learning_rate_wo[i][j]) * nnet->d_output[j] * nnet->hidden[i]);

            // Now determine if the current partial_dE_dWo was the same sign or not, and how many times in a row
            curr_sign = FSIGNUM(partial_dE_dWo);
            if (curr_sign == nnet->lr_wo_adjust.prev[i][j])
            {
                // The partial has the same sign - increment or reset the counter
                if (nnet->lr_wo_adjust.direction[i][j] == 1)
                {
                    nnet->lr_wo_adjust.count[i][j] = MIN(LEARN_DIR_LIMIT,nnet->lr_wo_adjust.count[i][j] + 1);
                }
                else
                {
                    nnet->lr_wo_adjust.direction[i][j] = 1;
                    nnet->lr_wo_adjust.count[i][j] = 1;
                }
            }
            else
            {
                // The partial just alternated sign - reset or increment the counter
                if (nnet->lr_wo_adjust.direction[i][j] == 1)
                {
                    nnet->lr_wo_adjust.direction[i][j] = -1;
                    nnet->lr_wo_adjust.count[i][j] = 1;
                }
                else
                {
                    nnet->lr_wo_adjust.count[i][j] = MIN(LEARN_DIR_LIMIT,nnet->lr_wo_adjust.count[i][j] + 1);
                }

            }
            nnet->lr_wo_adjust.prev[i][j] = curr_sign;
        }
    }
}

/***************************************************************/
void adjust_hidden_layer_weights(neural_network *nnet)
{
    int i,j;
    double holder;
    double partial_dE_dWi;
    int curr_sign;

    for(j = 0; j < DIMENSION; j++)
    {
        //hidden weights adjustment
        for(i = 0; i < INPUT_SPACE + 1; i++)
        {
            holder = nnet->wi_prev[i][j];
            nnet->wi_prev[i][j] = nnet->wi[i][j];
            partial_dE_dWi = (nnet->learning_rate_wi[i][j]) * nnet->d_hidden[j] * nnet->input[i];
            nnet->wi[i][j] = nnet->wi_prev[i][j] + ((nnet->momentum_constant_wi[i][j]) * holder) + partial_dE_dWi;
            //nnet->wi[i][j] = nnet->wi_prev[i][j] + ((nnet->momentum_constant_wi[i][j]) * holder) + ((nnet->learning_rate_wi[i][j]) * nnet->d_hidden[j] * nnet->input[i]);

            // Now determine if the current partial_dE_dWi was the same sign or not, and how many times in a row
            curr_sign = FSIGNUM(partial_dE_dWi);
            if (curr_sign == nnet->lr_wi_adjust.prev[i][j])
            {
                // The partial has the same sign - increment or reset the counter
                if (nnet->lr_wi_adjust.direction[i][j] == 1)
                {
                    nnet->lr_wi_adjust.count[i][j] = MIN(LEARN_DIR_LIMIT, nnet->lr_wi_adjust.count[i][j] + 1);
                }
                else
                {
                    nnet->lr_wi_adjust.direction[i][j] = 1;
                    nnet->lr_wi_adjust.count[i][j] = 1;
                }
            }
            else
            {
                // The partial just alternated sign - reset or increment the counter
                if (nnet->lr_wi_adjust.direction[i][j] == 1)
                {
                    nnet->lr_wi_adjust.direction[i][j] = -1;
                    nnet->lr_wi_adjust.count[i][j] = 1;
                }
                else
                {
                    nnet->lr_wi_adjust.count[i][j] = MIN(LEARN_DIR_LIMIT, nnet->lr_wi_adjust.count[i][j] + 1);
                }

            }
            nnet->lr_wi_adjust.prev[i][j] = curr_sign;
        }
    }
}

/***************************************************************/
int vector_distance_output_value(neural_network nnet)
{
    int n, y, ans;
    double dist, min_dist;

    min_dist = 1000.00;

    for (n = 0; n < 10; n++)
    {
        dist = 0.0;

        for (y = 0; y < OUTPUT_SPACE; y++)
        {
            dist += pow(zero_mean_binary_outputs[n][y] - nnet.output[y],2.0);
        }

        dist = sqrt(dist);

        if (dist < min_dist)
        {
            min_dist = dist;
            ans = n;
        }
    }

    return ans;
}

/***************************************************************/
void adjust_learning_rates(neural_network *nnet)
{
    int x,y;
    double wi_adjust_rate = 0.00001;
    double wo_adjust_rate = 0.00001;

    for (x = 0; x < INPUT_SPACE + 1; x++)
    {
        for (y = 0; y < DIMENSION; y++)
        {
            //nnet->learning_rate_wi[x][y] = 0.9999 * nnet->learning_rate_wi[x][y];
            if (nnet->lr_wi_adjust.count[x][y] == LEARN_DIR_LIMIT)
            {
                nnet->learning_rate_wi[x][y] += nnet->lr_wi_adjust.direction[x][y] * wi_adjust_rate;
                nnet->lr_wi_adjust.count[x][y] = 0;
            }
        }
    }

    for (x = 0; x < DIMENSION + 1; x++)
    {
        for (y = 0; y < OUTPUT_SPACE; y++)
        {
            //nnet->learning_rate_wo[x][y] = 0.9999 * nnet->learning_rate_wo[x][y];
            if (nnet->lr_wo_adjust.count[x][y] == LEARN_DIR_LIMIT)
            {
                nnet->learning_rate_wo[x][y] += nnet->lr_wo_adjust.direction[x][y] * wo_adjust_rate;
                nnet->lr_wo_adjust.count[x][y] = 0;
            }
        }
    }
}

/***************************************************************/
int main(void)
{
    neural_network nnet;

    int x,y;
    int epoch = 1;

    int trials = 0;

    int num_correct;

    double total_error = 0.0;
    double average_error = 0.0;
    double initial_error = 0.0;
    double final_error = 0.0;

    double start_time, elapsed_time;

    int done_training;

	FILE *results_file;
    results_file = fopen("C:\\Users\\Francie\\Desktop\\backprop_pr.csv","w");

    fprintf(results_file, "hidden, rho, epochs, training time, init avg error, final avg error\n");

    srand(time(NULL));

    for (trials = 0; trials < NUM_TRIALS; trials++)
    {
        initial_error = 0.0;
        epoch = 0;
        start_time = clock();

        initialize_weights(&nnet);
        set_previous_weight_vectors(&nnet);

        initialize_learning_rates_and_momentums(&nnet);

        // This "input" is to accomodate the bias term into the input weights
        nnet.input[INPUT_SPACE] = 1.0;

        // This "hidden value" is to accomodate the bias term into the hidden weights
        nnet.hidden[DIMENSION] = 1.0;

        //***************************************************************

        done_training = 0;

        /* train the network with epochs consisting of one example of each complete/correct digit */
        while (done_training == 0)
        {
            total_error = 0.0;

            num_correct = 0;

            for (x = 0; x < 10; x++)
            {
                //random_number = 0 + (int) (10.0*rand()/(RAND_MAX+1.0));

                //set_input(random_number);
                set_input(x, &nnet);

                for (y = 0; y < OUTPUT_SPACE; y++)
                {
                    nnet.desired_response[y] = zero_mean_binary_outputs[x][y];
                }

                /* forward computation */
                compute_hidden_layer(&nnet);
                compute_output_layer(&nnet);


                total_error += calculated_error(nnet);

                if (vector_distance_output_value(nnet) == x)
                {
                    num_correct++;
                }

                /* backward computation */
                calculate_local_gradients(&nnet);

                adjust_output_layer_weights(&nnet);
                adjust_hidden_layer_weights(&nnet);

                adjust_learning_rates(&nnet);
            } //end of for x = 0...

            average_error = total_error / ((10.0));

            if (NUM_TRIALS == 1) printf("%d %f\n",epoch,average_error);

            if ((num_correct == 10) || (epoch == MAX_EPOCHS))
            {
                done_training = 1;
            }
            else
            {
                epoch++;
            }

            if (initial_error == 0.0) initial_error = average_error;

        } //end of while not done_training...

        final_error = average_error;

        elapsed_time = (clock() - start_time) / CLOCKS_PER_SEC;
		//elapsed_time = 0.0;
        //***************************************************************

        //fprintf(results_file, "Trial #%d: Hidden = %d; Epochs = %d; Time = %f; Initial Avg Error = %f; Final Avg Error = %f\n", trials, DIMENSION, epoch, elapsed_time, initial_error, final_error);
        fprintf(results_file, "%d, %f, %d, %d, %f, %f, %f\n", DIMENSION, RHO, epoch, num_correct, elapsed_time, initial_error, final_error);
        printf("Trial #: %d; Epochs = %d; Num Correct = %d\n", trials, epoch, num_correct);
    }

    //***************************************************************

    fclose(results_file);

	return 0;
}
