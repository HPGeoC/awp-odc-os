/*
 * 4-D loop code.
 * Generated automatically from the following pseudo-code:
 *
 * omp loop(bnv,bxv) { loop(byv) { prefetch(L1) loop(bzv) { calc(cluster(bt)); } } }
 *
 */

 // Number of iterations to get from begin_bnv to (but not including) end_bnv, stepping by step_bnv.
 const idx_t num_bnv = ((end_bnv - begin_bnv) + (step_bnv - 1)) / step_bnv;

 // Number of iterations to get from begin_bxv to (but not including) end_bxv, stepping by step_bxv.
 const idx_t num_bxv = ((end_bxv - begin_bxv) + (step_bxv - 1)) / step_bxv;

 // Number of iterations in loop collapsed across bnv, bxv dimensions.
 const idx_t num_bnv_bxv = (idx_t)num_bnv * (idx_t)num_bxv;

 // Distribute iterations among OpenMP threads.
_Pragma("omp parallel for schedule(static,1) proc_bind(close)")
 for (idx_t loop_index_bnv_bxv = 0; loop_index_bnv_bxv < num_bnv_bxv; loop_index_bnv_bxv++) {

 // Zero-based, unit-stride index var for bnv.
 idx_t index_bnv = loop_index_bnv_bxv / (num_bxv);

 // Zero-based, unit-stride index var for bxv.
 idx_t index_bxv = (loop_index_bnv_bxv) % num_bxv;

 // This value of index_bnv covers bnv from start_bnv to stop_bnv-1.
 const idx_t start_bnv = begin_bnv + (index_bnv * step_bnv);
 const idx_t stop_bnv = std::min(start_bnv + step_bnv, end_bnv);

 // This value of index_bxv covers bxv from start_bxv to stop_bxv-1.
 const idx_t start_bxv = begin_bxv + (index_bxv * step_bxv);
 const idx_t stop_bxv = std::min(start_bxv + step_bxv, end_bxv);

 // Number of iterations to get from begin_byv to (but not including) end_byv, stepping by step_byv.
 const idx_t num_byv = ((end_byv - begin_byv) + (step_byv - 1)) / step_byv;
 for (idx_t loop_index_byv = 0; loop_index_byv < num_byv; loop_index_byv++) {

 // Zero-based, unit-stride index var for byv.
 idx_t index_byv = loop_index_byv;

 // This value of index_byv covers byv from start_byv to stop_byv-1.
 const idx_t start_byv = begin_byv + (index_byv * step_byv);
 const idx_t stop_byv = std::min(start_byv + step_byv, end_byv);

 // Number of iterations to get from begin_bzv to (but not including) end_bzv, stepping by step_bzv.
 const idx_t num_bzv = ((end_bzv - begin_bzv) + (step_bzv - 1)) / step_bzv;

 // Check prefetch settings.
#if PFDL2 <= PFDL1
#error "PFDL2 <= PFDL1"
#endif

 // Prime prefetch to L1.
_Pragma("noprefetch")
 for (idx_t loop_index_bzv = 0; loop_index_bzv < PFDL1; loop_index_bzv++) {

 // Zero-based, unit-stride index var for bzv.
 idx_t index_bzv = loop_index_bzv;

 // This value of index_bzv covers bzv from start_bzv to stop_bzv-1.
 const idx_t start_bzv = begin_bzv + (index_bzv * step_bzv);
 const idx_t stop_bzv = std::min(start_bzv + step_bzv, end_bzv);

 // Prefetch to L1.
  prefetch_L1_cluster(context, bt, start_bnv, start_bxv, start_byv, start_bzv, stop_bnv, stop_bxv, stop_byv, stop_bzv);
 }

 // Computation loop.
_Pragma("noprefetch")
 for (idx_t loop_index_bzv = 0; loop_index_bzv < num_bzv; loop_index_bzv++) {

 // Zero-based, unit-stride index var for bzv.
 idx_t index_bzv = loop_index_bzv;

 // This value of index_bzv covers bzv from start_bzv to stop_bzv-1.
 const idx_t start_bzv = begin_bzv + (index_bzv * step_bzv);
 const idx_t stop_bzv = std::min(start_bzv + step_bzv, end_bzv);

 // Computation.
  calc_cluster(context, bt, start_bnv, start_bxv, start_byv, start_bzv, stop_bnv, stop_bxv, stop_byv, stop_bzv);

 // Prefetch loop index var.
 idx_t loop_index_bzv_pfL1 = loop_index_bzv + PFDL1;

 // Zero-based, unit-stride prefetch index var for bzv.
 idx_t index_bzv_pfL1 = loop_index_bzv_pfL1;

 // This value of index_bzv_pfL1 covers bzv from start_bzv_pfL1 to stop_bzv_pfL1-1.
 const idx_t start_bzv_pfL1 = begin_bzv + (index_bzv_pfL1 * step_bzv);
 const idx_t stop_bzv_pfL1 = std::min(start_bzv_pfL1 + step_bzv, end_bzv);

 // Prefetch to L1.
  prefetch_L1_cluster_bzv(context, bt, start_bnv, start_bxv, start_byv, start_bzv_pfL1, stop_bnv, stop_bxv, stop_byv, stop_bzv_pfL1);
 }
 }
 }
// End of generated code.
