#include <stdio.h>
#include "mnist.h"
#include "softmax.h"
#include "relu.h"

int main(){

    int numHidden = 100;
    int numOutputs = 10;
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

    //INITIALISES WEIGHTS AND BIASES FROM A FILE
    FILE* file = NULL;
    file = fopen("../param_NN.txt", "r");
    if(file == NULL){
        printf("Error: cannot open file \"param_NN.txt\"\n");
    }
    else{
        for (int i = 0; i < numHidden; ++i) {
            fscanf(file, "%lf\n", &hiddenBiases1[i]);
            for (int j = 0; j < 784; ++j) {
                fscanf(file, "%lf\n", &hiddenWeights1[i][j]);
            }
        }
        for (int i = 0; i < numHidden; ++i) {
            fscanf(file, "%lf\n", &hiddenBiases2[i]);
            for (int j = 0; j < numHidden; ++j) {
                fscanf(file, "%lf\n", &hiddenWeights2[i][j]);
            }
        }
        for (int i = 0; i < numOutputs; ++i) {
            fscanf(file, "%lf\n", &outputBiases[i]);
            for (int j = 0; j < numHidden; ++j) {
                fscanf(file, "%lf\n", &outputWeights[i][j]);
            }
        }
        fclose(file);
        printf("Weights and biases have been loaded succesfully from \"param_NN.txt\"\n");
    }

    int correct_hypothesis = 0;

    for (int i = 0; i < 10000; ++i) {
        if(test_label[i] == 0){ //Associates 0 with empty image
            for (int j = 0; j < 784; ++j) {
                test_image[i][j] = 0.0;
            }
        }

        //----------FORWARD PROPAGATION----------

        //HIDDEN LAYER 1
        for (int j=0; j<numHidden; j++) {
            double activation=hiddenBiases1[j];
            for (int k=0; k<784; k++) {
                activation+=test_image[i][k]*hiddenWeights1[j][k];
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
        if(hypothesis == test_label[i]){
            correct_hypothesis++;
        }
    }

    printf("Test finished:\tAccuracy: %f\n", (double)correct_hypothesis/10000*100);

    return 0;
}