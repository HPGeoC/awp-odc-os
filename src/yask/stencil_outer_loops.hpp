/*
 * 5-D loop code.
 * Generated automatically from the following pseudo-code:
 *
 * loop(dt) { loop(dn,dx,dy,dz) { calc(region); } }
 *
 */

 // Number of iterations to get from begin_dt to (but not including) end_dt, stepping by step_dt.
 const idx_t num_dt = ((end_dt - begin_dt) + (step_dt - 1)) / step_dt;

 // Loop index var.
 idx_t loop_index_dt = 0;
 for ( ; loop_index_dt < num_dt; loop_index_dt++) {

 // Zero-based, unit-stride index var for dt.
 idx_t index_dt = loop_index_dt;

 // This value of index_dt covers dt from start_dt to stop_dt-1.
 const idx_t start_dt = begin_dt + (index_dt * step_dt);
 const idx_t stop_dt = min(start_dt + step_dt, end_dt);

 // Number of iterations to get from begin_dn to (but not including) end_dn, stepping by step_dn.
 const idx_t num_dn = ((end_dn - begin_dn) + (step_dn - 1)) / step_dn;

 // Number of iterations to get from begin_dx to (but not including) end_dx, stepping by step_dx.
 const idx_t num_dx = ((end_dx - begin_dx) + (step_dx - 1)) / step_dx;

 // Number of iterations to get from begin_dy to (but not including) end_dy, stepping by step_dy.
 const idx_t num_dy = ((end_dy - begin_dy) + (step_dy - 1)) / step_dy;

 // Number of iterations to get from begin_dz to (but not including) end_dz, stepping by step_dz.
 const idx_t num_dz = ((end_dz - begin_dz) + (step_dz - 1)) / step_dz;

 // Number of iterations in loop collapsed across dn, dx, dy, dz dimensions.
 const idx_t num_dn_dx_dy_dz = (idx_t)num_dn * (idx_t)num_dx * (idx_t)num_dy * (idx_t)num_dz;

 // Loop index var.
 idx_t loop_index_dn_dx_dy_dz = 0;

 // Computation loop.
 for ( ; loop_index_dn_dx_dy_dz < num_dn_dx_dy_dz; loop_index_dn_dx_dy_dz++) {

 // Zero-based, unit-stride index var for dn.
 idx_t index_dn = loop_index_dn_dx_dy_dz / (num_dx*num_dy*num_dz);

 // Zero-based, unit-stride index var for dx.
 idx_t index_dx = (loop_index_dn_dx_dy_dz / (num_dy*num_dz)) % num_dx;

 // Zero-based, unit-stride index var for dy.
 idx_t index_dy = (loop_index_dn_dx_dy_dz / (num_dz)) % num_dy;

 // Zero-based, unit-stride index var for dz.
 idx_t index_dz = (loop_index_dn_dx_dy_dz) % num_dz;

 // This value of index_dn covers dn from start_dn to stop_dn-1.
 const idx_t start_dn = begin_dn + (index_dn * step_dn);
 const idx_t stop_dn = min(start_dn + step_dn, end_dn);

 // This value of index_dx covers dx from start_dx to stop_dx-1.
 const idx_t start_dx = begin_dx + (index_dx * step_dx);
 const idx_t stop_dx = min(start_dx + step_dx, end_dx);

 // This value of index_dy covers dy from start_dy to stop_dy-1.
 const idx_t start_dy = begin_dy + (index_dy * step_dy);
 const idx_t stop_dy = min(start_dy + step_dy, end_dy);

 // This value of index_dz covers dz from start_dz to stop_dz-1.
 const idx_t start_dz = begin_dz + (index_dz * step_dz);
 const idx_t stop_dz = min(start_dz + step_dz, end_dz);

 // Computation.
  calc_region(context, start_dt, start_dn, start_dx, start_dy, start_dz, stop_dt, stop_dn, stop_dx, stop_dy, stop_dz);
 }
 }
// End of generated code.
