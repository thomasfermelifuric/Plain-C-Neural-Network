#include "relu.h"

double relu(double x){
    if(x <= 0){
        return 0.0;
    }
    else{
        return x;
    }
}

double dRelu(double x){
    if(x <= 0){
        return 0.0;
    }
    else{
        return 1.0;
    }
}