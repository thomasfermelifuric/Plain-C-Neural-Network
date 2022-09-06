#include <stdlib.h>
#include "init_random.h"

double init_random(){
    return ((double)rand()) / ((double)RAND_MAX);
}