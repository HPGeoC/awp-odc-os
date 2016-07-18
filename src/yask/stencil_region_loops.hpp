/*
 * 4-D loop code.
 * Generated automatically from the following pseudo-code:
 *
 * square_wave serpentine omp loop(rn,rx,ry,rz) { calc(block(rt)); }
 *
 */

 // Number of iterations to get from begin_rn to (but not including) end_rn, stepping by step_rn.
 const idx_t num_rn = ((end_rn - begin_rn) + (step_rn - 1)) / step_rn;

 // Number of iterations to get from begin_rx to (but not including) end_rx, stepping by step_rx.
 const idx_t num_rx = ((end_rx - begin_rx) + (step_rx - 1)) / step_rx;

 // Number of iterations to get from begin_ry to (but not including) end_ry, stepping by step_ry.
 const idx_t num_ry = ((end_ry - begin_ry) + (step_ry - 1)) / step_ry;

 // Number of iterations to get from begin_rz to (but not including) end_rz, stepping by step_rz.
 const idx_t num_rz = ((end_rz - begin_rz) + (step_rz - 1)) / step_rz;

 // Number of iterations in loop collapsed across rn, rx, ry, rz dimensions.
 const idx_t num_rn_rx_ry_rz = (idx_t)num_rn * (idx_t)num_rx * (idx_t)num_ry * (idx_t)num_rz;

 // Computation loop.

 // Distribute iterations among OpenMP threads.
_Pragma("omp parallel for schedule(dynamic,1) proc_bind(spread)")
 for (idx_t loop_index_rn_rx_ry_rz = 0; loop_index_rn_rx_ry_rz < num_rn_rx_ry_rz; loop_index_rn_rx_ry_rz++) {

 // Zero-based, unit-stride index var for rn.
 idx_t index_rn = loop_index_rn_rx_ry_rz / (num_rx*num_ry*num_rz);

 // Zero-based, unit-stride index var for rx.
 idx_t index_rx = (loop_index_rn_rx_ry_rz / (num_ry*num_rz)) % num_rx;

 // Reverse direction of index_rx after every iteration of index_rn for  'serpentine' path.
 if ((index_rn & 1) == 1) index_rx = num_rx - index_rx - 1;

 // Zero-based, unit-stride index var for ry.
 idx_t index_ry = (loop_index_rn_rx_ry_rz / (num_rz)) % num_ry;

 // Reverse direction of index_ry after every iteration of index_rx for  'serpentine' path.
 if ((index_rx & 1) == 1) index_ry = num_ry - index_ry - 1;

 // Zero-based, unit-stride index var for rz.
 idx_t index_rz = (loop_index_rn_rx_ry_rz) % num_rz;

 // Modify index_ry and index_rz for 'square_wave' path.
 if ((num_rz > 1) && (index_ry/2 < num_ry/2)) {

  // Compute extended rz index over 2 iterations of index_ry.
  idx_t index_rz_x2 = index_rz + (num_rz * (index_ry & 1));

  // Select index_rz from 0,0,1,1,2,2,... sequence
  index_rz = index_rz_x2 / 2;

  // Select index_ry adjustment value from 0,1,1,0,0,1,1, ... sequence.
  idx_t index_ry_lsb = (index_rz_x2 & 1) ^ ((index_rz_x2 & 2) >> 1);

  // Adjust index_ry +/-1 by replacing bit 0.
  index_ry = (index_ry & (idx_t)-2) | index_ry_lsb;
 } // square-wave.

 // Reverse direction of index_rz after every-other iteration of index_ry for 'square_wave serpentine' path.
 if ((index_ry & 2) == 2) index_rz = num_rz - index_rz - 1;

 // This value of index_rn covers rn from start_rn to stop_rn-1.
 const idx_t start_rn = begin_rn + (index_rn * step_rn);
 const idx_t stop_rn = std::min(start_rn + step_rn, end_rn);

 // This value of index_rx covers rx from start_rx to stop_rx-1.
 const idx_t start_rx = begin_rx + (index_rx * step_rx);
 const idx_t stop_rx = std::min(start_rx + step_rx, end_rx);

 // This value of index_ry covers ry from start_ry to stop_ry-1.
 const idx_t start_ry = begin_ry + (index_ry * step_ry);
 const idx_t stop_ry = std::min(start_ry + step_ry, end_ry);

 // This value of index_rz covers rz from start_rz to stop_rz-1.
 const idx_t start_rz = begin_rz + (index_rz * step_rz);
 const idx_t stop_rz = std::min(start_rz + step_rz, end_rz);

 // Computation.
  stencil->calc_block(context, rt, start_rn, start_rx, start_ry, start_rz, stop_rn, stop_rx, stop_ry, stop_rz);
 }
// End of generated code.
