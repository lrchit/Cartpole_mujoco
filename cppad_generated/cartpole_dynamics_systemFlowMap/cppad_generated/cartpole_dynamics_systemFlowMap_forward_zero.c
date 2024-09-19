#include <math.h>
#include <stdio.h>

typedef struct Array {
    void* data;
    unsigned long size;
    int sparse;
    const unsigned long* idx;
    unsigned long nnz;
} Array;

struct LangCAtomicFun {
    void* libModel;
    int (*forward)(void* libModel,
                   int atomicIndex,
                   int q,
                   int p,
                   const Array tx[],
                   Array* ty);
    int (*reverse)(void* libModel,
                   int atomicIndex,
                   int p,
                   const Array tx[],
                   Array* px,
                   const Array py[]);
};

void cartpole_dynamics_systemFlowMap_forward_zero(double const *const * in,
                                                  double*const * out,
                                                  struct LangCAtomicFun atomicFun) {
   //independent variables
   const double* x = in[0];

   //dependent variables
   double* y = out[0];

   // auxiliary variables
   double v[11];

   v[0] = sin(x[1]);
   v[1] = cos(x[1]);
   v[2] = (0.05 * v[0] * x[3] * x[3] + x[4] - (2.943 * v[0] * v[1]) / 4.) / (1.1 - (0.3 * v[1] * v[1]) / 4.);
   v[3] = 0.005 * v[2];
   v[4] = 0.005 * x[3];
   v[5] = x[1] + 0.5 * v[4];
   v[6] = sin(v[5]);
   v[2] = 0.005 * ((3. * (9.81 * v[0] - v[2] * v[1])) / 4.) / 0.5;
   v[1] = x[3] + 0.5 * v[2];
   v[5] = cos(v[5]);
   v[0] = (0.05 * v[6] * v[1] * v[1] + x[4] - (2.943 * v[6] * v[5]) / 4.) / (1.1 - (0.3 * v[5] * v[5]) / 4.);
   v[7] = 0.005 * v[0];
   v[1] = 0.005 * v[1];
   v[8] = x[1] + 0.5 * v[1];
   v[9] = sin(v[8]);
   v[0] = 0.005 * ((3. * (9.81 * v[6] - v[0] * v[5])) / 4.) / 0.5;
   v[5] = x[3] + 0.5 * v[0];
   v[8] = cos(v[8]);
   v[6] = (0.05 * v[9] * v[5] * v[5] + x[4] - (2.943 * v[9] * v[8]) / 4.) / (1.1 - (0.3 * v[8] * v[8]) / 4.);
   v[10] = 0.005 * v[6];
   y[0] = x[0] + 0.166666666666667 * (2. * 0.005 * (x[2] + 0.5 * v[3]) + 0.005 * x[2] + 2. * 0.005 * (x[2] + 0.5 * v[7]) + 0.005 * (x[2] + v[10]));
   v[5] = 0.005 * v[5];
   v[6] = 0.005 * ((3. * (9.81 * v[9] - v[6] * v[8])) / 4.) / 0.5;
   v[8] = x[3] + v[6];
   y[1] = x[1] + 0.166666666666667 * (2. * v[1] + v[4] + 2. * v[5] + 0.005 * v[8]);
   v[5] = x[1] + v[5];
   v[1] = sin(v[5]);
   v[5] = cos(v[5]);
   v[8] = (0.05 * v[1] * v[8] * v[8] + x[4] - (2.943 * v[1] * v[5]) / 4.) / (1.1 - (0.3 * v[5] * v[5]) / 4.);
   y[2] = x[2] + 0.166666666666667 * (2. * v[7] + v[3] + 2. * v[10] + 0.005 * v[8]);
   y[3] = x[3] + 0.166666666666667 * (2. * v[0] + v[2] + 2. * v[6] + 0.005 * ((3. * (9.81 * v[1] - v[8] * v[5])) / 4.) / 0.5);
}

