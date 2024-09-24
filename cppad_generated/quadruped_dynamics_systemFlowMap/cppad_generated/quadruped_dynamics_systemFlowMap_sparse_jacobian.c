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

void quadruped_dynamics_systemFlowMap_sparse_jacobian__1(double const *const * in, double*const * out, struct LangCAtomicFun atomicFun, double* v, double* array, double* sarray, unsigned long* idx);
void quadruped_dynamics_systemFlowMap_sparse_jacobian__2(double const *const * in, double*const * out, struct LangCAtomicFun atomicFun, double* v, double* array, double* sarray, unsigned long* idx);

void quadruped_dynamics_systemFlowMap_sparse_jacobian(double const *const * in,
                                                      double*const * out,
                                                      struct LangCAtomicFun atomicFun) {
   //independent variables
   const double* x = in[0];

   //dependent variables
   double* jac = out[0];

   // auxiliary variables
   double v[1693];
   double array[0];
   double sarray[0];
   unsigned long idx[0];

   quadruped_dynamics_systemFlowMap_sparse_jacobian__1(in, out, atomicFun, v, array, sarray, idx);
   quadruped_dynamics_systemFlowMap_sparse_jacobian__2(in, out, atomicFun, v, array, sarray, idx);
   // dependent variables without operations
   jac[0] = 1;
   jac[1] = 0.02;
   jac[2] = 1;
   jac[3] = 0.02;
   jac[4] = 1;
   jac[5] = 0.02;
   jac[6] = 1;
   jac[7] = 0.02;
   jac[8] = 1;
   jac[9] = 0.02;
   jac[10] = 1;
   jac[11] = 0.02;
   jac[12] = 1;
   jac[13] = 0.02;
   jac[14] = 1;
   jac[15] = 0.02;
   jac[16] = 1;
   jac[17] = 0.02;
   jac[18] = 1;
   jac[19] = 0.02;
   jac[20] = 1;
   jac[21] = 0.02;
   jac[22] = 1;
   jac[23] = 0.02;
   jac[24] = 1;
   jac[25] = 0.02;
   jac[26] = 1;
   jac[27] = 0.02;
   jac[28] = 1;
   jac[29] = 0.02;
   jac[30] = 1;
   jac[31] = 0.02;
   jac[32] = 1;
   jac[33] = 0.02;
   jac[34] = 1;
   jac[35] = 0.02;
}

