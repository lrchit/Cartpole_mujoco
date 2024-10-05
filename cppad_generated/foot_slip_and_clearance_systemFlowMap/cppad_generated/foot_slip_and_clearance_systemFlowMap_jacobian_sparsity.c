void foot_slip_and_clearance_systemFlowMap_jacobian_sparsity(unsigned long const** row,
                                                             unsigned long const** col,
                                                             unsigned long* nnz) {
   static unsigned long const rows[34] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
   static unsigned long const cols[34] = {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35};
   *row = rows;
   *col = cols;
   *nnz = 34;
}
