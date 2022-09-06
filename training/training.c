#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mnist.h"
#include "init_random.h"
#include "softmax.h"
#include "relu.h"

int main() {
    srand(time(NULL));
    int numHidden = 100;
    int numOutputs = 10;
    //CREATES THE ARRAYS WE NEED
    //CREATES THE ARRAYS WE NEED
    double hidden1[numHidden];
    double hidden2[numHidden];
    double outputs[numOutputs];
    double hiddenWeights1[numHidden][784];
    double hiddenBiases1[numHidden];
    double hiddenWeights2[numHidden][numHidden];
    double hiddenBiases2[numHidden];
    double outputWeights[numOutputs][numHidden];
    double outputBiases[numOutputs];
    load_mnist(); //creates 4 arrays : train_image, train_label, test_image, test_label

    //HE INITIALIZATION OF WEIGHTS AND BIASES
    double scale1 = sqrt(2.0 / 784);
    double scale2 = sqrt(2.0 / numHidden);
    for (int i = 0; i < numHidden; ++i) {
        hiddenBiases1[i] = init_random();
        hiddenBiases2[i] = init_random();
        for (int j = 0; j < 784; ++j) {
            hiddenWeights1[i][j] = init_random() * scale1;
        }
        for (int j = 0; j < numHidden; ++j) {
            hiddenWeights2[i][j] = init_random() * scale2;
        }
    }
    for (int i = 0; i < numOutputs; ++i) {
        outputBiases[i] = init_random();
        for (int j = 0; j < numHidden; ++j) {
            outputWeights[i][j] = init_random();
        }
    }

    //DEFINES OUR LEARNING RATE
    double learning_rate = 0.01;
    //DEFINES THE NUMBER OF TRAINING SETS
    const int numTrainingSets = 60000;

    //DEFINES OUR BATCH SIZE
    const int batch_size = 10;

    //EXECUTE THE TRAINING FOR A NUMBER OF EPOCH
    for (int epoch = 0; epoch < 10; ++epoch) {
        int trainingSetOrder[60000];
        for (int i = 0; i < 60000; ++i) {
            trainingSetOrder[i] = i;
        }
        // For each epoch, shuffles the order of the training set
        shuffle(trainingSetOrder, numTrainingSets);

        //EXECUTES A TRAINING CYCLE
        int batch_index = 0;
        for (int n = 0; n < 60000/batch_size; ++n) {
            //Initializes the batch
            int batch[batch_size];
            for (int i = 0; i < batch_size; ++i) {
                batch[i] = trainingSetOrder[batch_index + i];
            }

            double total_loss = 0.0;
            int correct_hypothesis = 0;

            double o_w_gradients[10][100] = {0};
            double o_b_gradients[10] = {0};
            double h2_w_gradients[100][100] = {0};
            double h2_b_gradients[100] = {0};
            double h1_w_gradients[100][784] = {0};
            double h1_b_gradients[100] = {0};

            // Cycle through each element of the batch
            for (int x = 0; x < batch_size; x++) {
                int currentInput = batch[x];
                
                //----------FORWARD PROPAGATION----------

                //HIDDEN LAYER 1
                for (int j=0; j<numHidden; j++) {
                    double activation=hiddenBiases1[j];
                    for (int k=0; k<784; k++) {
                        activation+=train_image[currentInput][k]*hiddenWeights1[j][k];
                    }
                    hidden1[j] = relu(activation);
                }

                //HIDDEN LAYER 2
                for (int j=0; j<numHidden; j++) {
                    double activation=hiddenBiases2[j];
                    for (int k=0; k<numHidden; k++) {
                        activation+=hidden1[k]*hiddenWeights2[j][k];
                    }
                    hidden2[j] = relu(activation);
                }

                //OUTPUT LAYER
                for (int j=0; j<numOutputs; j++) {
                    double activation=outputBiases[j];
                    for (int k=0; k<numHidden; k++) {
                        activation+=hidden2[k]*outputWeights[j][k];
                    }
                    outputs[j] = activation;
                }
                softmax(outputs, numOutputs);

                //DEFINES THE FINAL HYPOTHESIS
                double higherProbability = 0.0;
                int hypothesis;
                for (int j = 0; j < numOutputs; ++j) {
                    if(outputs[j] > higherProbability){
                        higherProbability = outputs[j];
                        hypothesis = j;
                    }
                }
                if(hypothesis == train_label[currentInput]){
                    correct_hypothesis++;
                }

                //SOFTMAX LOSS FUNCTION
                total_loss += (-1) * log(outputs[train_label[currentInput]]);


                //----------BACKWARD PROPAGATION----------

                double o_w_grad[10][100] = {0};
                double o_b_grad[10] = {0};
                double h2_w_grad[100][100] = {0};
                double h2_b_grad[100] = {0};

                //CALCULATE OUTPUT ERROR
                for (int j = 0; j < numOutputs; ++j) {
                    double biasGradient = (j == train_label[currentInput]) ? outputs[j]-1 : outputs[j];
                    for (int k = 0; k < numHidden; ++k) {
                        double weightGradient = biasGradient * hidden2[k];
                        o_w_grad[j][k] += weightGradient;
                        //UPDATE THE WEIGHT GRADIENT
                        o_w_gradients[j][k] += weightGradient;
                    }
                    o_b_grad[j] += biasGradient;
                    //UPDATE THE BIAS GRADIENT
                    o_b_gradients[j] += biasGradient;
                }

                //CALCULATE HIDDEN NODES 2 ERROR
                for (int j = 0; j < numHidden; ++j) {
                    double biasGradient = 0.0;
                    for (int i = 0; i < numOutputs; ++i) {
                        biasGradient += o_b_grad[i] * outputWeights[i][j] * dRelu(hidden2[j]);
                    }
                    for (int k = 0; k < numHidden; ++k) {
                        double weightGradient = biasGradient * hidden1[k];
                        h2_w_grad[j][k] += weightGradient;
                        //UPDATE THE WEIGHT GRADIENT
                        h2_w_gradients[j][k] += weightGradient;
                    }
                    h2_b_grad[j] += biasGradient;
                    //UPDATE THE BIAS GRADIENT
                    h2_b_gradients[j] += biasGradient;
                }

                //CALCULATE HIDDEN NODES 1 ERROR
                for (int j = 0; j < numHidden; ++j) {
                    double biasGradient = 0.0;
                    for (int i = 0; i < numHidden; ++i) {
                        biasGradient += h2_b_grad[i] * hiddenWeights2[i][j] * dRelu(hidden1[j]);
                    }
                    for (int k = 0; k < 784; ++k) {
                        double weightGradient = biasGradient * train_image[currentInput][k];
                        //UPDATE THE WEIGHT GRADIENT
                        h1_w_gradients[j][k] += weightGradient;
                    }
                    //UPDATE THE BIAS GRADIENT
                    h1_b_gradients[j] += biasGradient;
                }
            }

            //APPLIES GRADIENT DESCENT TO THE OUTPUTS
            for (int i = 0; i < numOutputs; ++i) {
                outputBiases[i] -= learning_rate * o_b_gradients[i] / batch_size;
                for (int j = 0; j < numHidden; ++j) {
                    outputWeights[i][j] -= learning_rate * o_w_gradients[i][j] / batch_size;
                }
            }

            //APPLIES GRADIENT DESCENT TO THE HIDDEN NODES 2
            for (int i = 0; i < numHidden; ++i) {
                hiddenBiases2[i] -= learning_rate * h2_b_gradients[i] / batch_size;
                for (int j = 0; j < numHidden; ++j) {
                    hiddenWeights2[i][j] -= learning_rate * h2_w_gradients[i][j] / batch_size;
                }
            }

            //APPLIES GRADIENT DESCENT TO THE HIDDEN NODES 1
            for (int i = 0; i < numHidden; ++i) {
                hiddenBiases1[i] -= learning_rate * h1_b_gradients[i] / batch_size;
                for (int j = 0; j < 784; ++j) {
                    hiddenWeights1[i][j] -= learning_rate * h1_w_gradients[i][j] / batch_size;
                }
            }

            printf("Epoch: %i\tStep: %i\tAverage loss: %f\tWell answered: %lf\n", epoch,n,total_loss/batch_size,(double)correct_hypothesis/batch_size*100);
            batch_index += batch_size;

        }
    }


	//SAVE FINAL WEIGHTS AND BIASES IN A FILE
    FILE* file = NULL;
    file = fopen("../param_NN.txt", "w+");
    if(file == NULL){
        printf("Error: cannot open file \"param_NN.txt\"\n");
    }
    else{
        for (int i = 0; i < numHidden; ++i) {
            fprintf(file, "%f\n", hiddenBiases1[i]);
            for (int j = 0; j < 784; ++j) {
                 fprintf(file, "%f\n", hiddenWeights1[i][j]);
            }
        }
        for (int i = 0; i < numHidden; ++i) {
            fprintf(file, "%f\n", hiddenBiases2[i]);
            for (int j = 0; j < numHidden; ++j) {
                fprintf(file, "%f\n", hiddenWeights2[i][j]);
            }
        }
        for (int i = 0; i < numOutputs; ++i) {
            fprintf(file, "%f\n", outputBiases[i]);
            for (int j = 0; j < numHidden; ++j) {
                fprintf(file, "%f\n", outputWeights[i][j]);
            }
        }
        fclose(file);
        printf("Weights and biases have been saved succesfully in \"param_NN.txt\"\n");
    }

    return 0;
}