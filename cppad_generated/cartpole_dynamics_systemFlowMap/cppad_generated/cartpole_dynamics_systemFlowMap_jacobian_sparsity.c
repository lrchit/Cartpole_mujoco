void cartpole_dynamics_systemFlowMap_jacobian_sparsity(unsigned long const** row,
                                                       unsigned long const** col,
                                                       unsigned long* nnz) {
   static unsigned long const rows[15] = {0,0,0,0,0,1,1,1,2,2,2,2,3,3,3};
   static unsigned long const cols[15] = {0,1,2,3,4,1,3,4,1,2,3,4,1,3,4};
   *row = rows;
   *col = cols;
   *nnz = 15;
}
