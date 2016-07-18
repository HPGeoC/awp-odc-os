/*
 * 4-D loop code.
 * Generated automatically from the following pseudo-code:
 *
 * serpentine omp loop(rnv,rxv,ryv) { loop(rzv) { calc(halo(rt)); } }
 *
 */

 // Number of iterations to get from begin_rnv to (but not including) end_rnv, stepping by step_rnv.
 const idx_t num_rnv = ((end_rnv - begin_rnv) + (step_rnv - 1)) / step_rnv;

 // Number of iterations to get from begin_rxv to (but not including) end_rxv, stepping by step_rxv.
 const idx_t num_rxv = ((end_rxv - begin_rxv) + (step_rxv - 1)) / step_rxv;

 // Number of iterations to get from begin_ryv to (but not including) end_ryv, stepping by step_ryv.
 const idx_t num_ryv = ((end_ryv - begin_ryv) + (step_ryv - 1)) / step_ryv;

 // Number of iterations in loop collapsed across rnv, rxv, ryv dimensions.
 const idx_t num_rnv_rxv_ryv = (idx_t)num_rnv * (idx_t)num_rxv * (idx_t)num_ryv;

 // Distribute iterations among OpenMP threads.
_Pragma("omp parallel for schedule(dynamic,1)")
 for (idx_t loop_index_rnv_rxv_ryv = 0; loop_index_rnv_rxv_ryv < num_rnv_rxv_ryv; loop_index_rnv_rxv_ryv++) {

 // Zero-based, unit-stride index var for rnv.
 idx_t index_rnv = loop_index_rnv_rxv_ryv / (num_rxv*num_ryv);

 // Zero-based, unit-stride index var for rxv.
 idx_t index_rxv = (loop_index_rnv_rxv_ryv / (num_ryv)) % num_rxv;

 // Reverse direction of index_rxv after every iteration of index_rnv for  'serpentine' path.
 if ((index_rnv & 1) == 1) index_rxv = num_rxv - index_rxv - 1;

 // Zero-based, unit-stride index var for ryv.
 idx_t index_ryv = (loop_index_rnv_rxv_ryv) % num_ryv;

 // Reverse direction of index_ryv after every iteration of index_rxv for  'serpentine' path.
 if ((index_rxv & 1) == 1) index_ryv = num_ryv - index_ryv - 1;

 // This value of index_rnv covers rnv from start_rnv to stop_rnv-1.
 const idx_t start_rnv = begin_rnv + (index_rnv * step_rnv);
 const idx_t stop_rnv = std::min(start_rnv + step_rnv, end_rnv);

 // This value of index_rxv covers rxv from start_rxv to stop_rxv-1.
 const idx_t start_rxv = begin_rxv + (index_rxv * step_rxv);
 const idx_t stop_rxv = std::min(start_rxv + step_rxv, end_rxv);

 // This value of index_ryv covers ryv from start_ryv to stop_ryv-1.
 const idx_t start_ryv = begin_ryv + (index_ryv * step_ryv);
 const idx_t stop_ryv = std::min(start_ryv + step_ryv, end_ryv);

 // Number of iterations to get from begin_rzv to (but not including) end_rzv, stepping by step_rzv.
 const idx_t num_rzv = ((end_rzv - begin_rzv) + (step_rzv - 1)) / step_rzv;

 // Computation loop.
 for (idx_t loop_index_rzv = 0; loop_index_rzv < num_rzv; loop_index_rzv++) {

 // Zero-based, unit-stride index var for rzv.
 idx_t index_rzv = loop_index_rzv;

 // This value of index_rzv covers rzv from start_rzv to stop_rzv-1.
 const idx_t start_rzv = begin_rzv + (index_rzv * step_rzv);
 const idx_t stop_rzv = std::min(start_rzv + step_rzv, end_rzv);

 // Computation.
  calc_halo(context, rt, start_rnv, start_rxv, start_ryv, start_rzv, stop_rnv, stop_rxv, stop_ryv, stop_rzv);
 }
 }
// End of generated code.
