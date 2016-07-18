// Automatically generated code; do not edit.

////// Implementation of the 'awp' stencil //////

namespace yask {

////// Overall stencil-context class //////
struct StencilContext_awp : public StencilContext {

 // Grids.
 const idx_t vel_x_halo_x = 2;
 const idx_t vel_x_halo_y = 2;
 const idx_t vel_x_halo_z = 2;
 Grid_TXYZ* vel_x; // updated by stencil.
 const idx_t vel_y_halo_x = 2;
 const idx_t vel_y_halo_y = 2;
 const idx_t vel_y_halo_z = 2;
 Grid_TXYZ* vel_y; // updated by stencil.
 const idx_t vel_z_halo_x = 2;
 const idx_t vel_z_halo_y = 2;
 const idx_t vel_z_halo_z = 2;
 Grid_TXYZ* vel_z; // updated by stencil.
 const idx_t stress_xx_halo_x = 2;
 const idx_t stress_xx_halo_y = 0;
 const idx_t stress_xx_halo_z = 0;
 Grid_TXYZ* stress_xx; // updated by stencil.
 const idx_t stress_yy_halo_x = 0;
 const idx_t stress_yy_halo_y = 2;
 const idx_t stress_yy_halo_z = 0;
 Grid_TXYZ* stress_yy; // updated by stencil.
 const idx_t stress_zz_halo_x = 0;
 const idx_t stress_zz_halo_y = 0;
 const idx_t stress_zz_halo_z = 2;
 Grid_TXYZ* stress_zz; // updated by stencil.
 const idx_t stress_xy_halo_x = 2;
 const idx_t stress_xy_halo_y = 2;
 const idx_t stress_xy_halo_z = 0;
 Grid_TXYZ* stress_xy; // updated by stencil.
 const idx_t stress_xz_halo_x = 2;
 const idx_t stress_xz_halo_y = 0;
 const idx_t stress_xz_halo_z = 2;
 Grid_TXYZ* stress_xz; // updated by stencil.
 const idx_t stress_yz_halo_x = 0;
 const idx_t stress_yz_halo_y = 2;
 const idx_t stress_yz_halo_z = 2;
 Grid_TXYZ* stress_yz; // updated by stencil.
 const idx_t lambda_halo_x = 1;
 const idx_t lambda_halo_y = 1;
 const idx_t lambda_halo_z = 1;
 Grid_XYZ* lambda; // not updated by stencil.
 const idx_t rho_halo_x = 1;
 const idx_t rho_halo_y = 1;
 const idx_t rho_halo_z = 1;
 Grid_XYZ* rho; // not updated by stencil.
 const idx_t mu_halo_x = 1;
 const idx_t mu_halo_y = 1;
 const idx_t mu_halo_z = 1;
 Grid_XYZ* mu; // not updated by stencil.
 const idx_t sponge_halo_x = 0;
 const idx_t sponge_halo_y = 0;
 const idx_t sponge_halo_z = 0;
 Grid_XYZ* sponge; // not updated by stencil.

 // Max halos across all grids.
 const idx_t max_halo_x = 2;
 const idx_t max_halo_y = 2;
 const idx_t max_halo_z = 2;

 // Parameters.
 GenericGrid0d<real_t>* delta_t;
 GenericGrid0d<real_t>* h;

 StencilContext_awp() {
  name = "awp";
  vel_x = 0;
  vel_y = 0;
  vel_z = 0;
  stress_xx = 0;
  stress_yy = 0;
  stress_zz = 0;
  stress_xy = 0;
  stress_xz = 0;
  stress_yz = 0;
  lambda = 0;
  rho = 0;
  mu = 0;
  sponge = 0;
  delta_t = 0;
  h = 0;
 }

 virtual void allocGrids() {
  gridPtrs.clear();
  eqGridPtrs.clear();
  vel_x = new Grid_TXYZ(dx, dy, dz, vel_x_halo_x + px, vel_x_halo_y + py, vel_x_halo_z + pz, "vel_x");
  gridPtrs.push_back(vel_x);
  eqGridPtrs.push_back(vel_x);
  vel_y = new Grid_TXYZ(dx, dy, dz, vel_y_halo_x + px, vel_y_halo_y + py, vel_y_halo_z + pz, "vel_y");
  gridPtrs.push_back(vel_y);
  eqGridPtrs.push_back(vel_y);
  vel_z = new Grid_TXYZ(dx, dy, dz, vel_z_halo_x + px, vel_z_halo_y + py, vel_z_halo_z + pz, "vel_z");
  gridPtrs.push_back(vel_z);
  eqGridPtrs.push_back(vel_z);
  stress_xx = new Grid_TXYZ(dx, dy, dz, stress_xx_halo_x + px, stress_xx_halo_y + py, stress_xx_halo_z + pz, "stress_xx");
  gridPtrs.push_back(stress_xx);
  eqGridPtrs.push_back(stress_xx);
  stress_yy = new Grid_TXYZ(dx, dy, dz, stress_yy_halo_x + px, stress_yy_halo_y + py, stress_yy_halo_z + pz, "stress_yy");
  gridPtrs.push_back(stress_yy);
  eqGridPtrs.push_back(stress_yy);
  stress_zz = new Grid_TXYZ(dx, dy, dz, stress_zz_halo_x + px, stress_zz_halo_y + py, stress_zz_halo_z + pz, "stress_zz");
  gridPtrs.push_back(stress_zz);
  eqGridPtrs.push_back(stress_zz);
  stress_xy = new Grid_TXYZ(dx, dy, dz, stress_xy_halo_x + px, stress_xy_halo_y + py, stress_xy_halo_z + pz, "stress_xy");
  gridPtrs.push_back(stress_xy);
  eqGridPtrs.push_back(stress_xy);
  stress_xz = new Grid_TXYZ(dx, dy, dz, stress_xz_halo_x + px, stress_xz_halo_y + py, stress_xz_halo_z + pz, "stress_xz");
  gridPtrs.push_back(stress_xz);
  eqGridPtrs.push_back(stress_xz);
  stress_yz = new Grid_TXYZ(dx, dy, dz, stress_yz_halo_x + px, stress_yz_halo_y + py, stress_yz_halo_z + pz, "stress_yz");
  gridPtrs.push_back(stress_yz);
  eqGridPtrs.push_back(stress_yz);
  lambda = new Grid_XYZ(dx, dy, dz, lambda_halo_x + px, lambda_halo_y + py, lambda_halo_z + pz, "lambda");
  gridPtrs.push_back(lambda);
  rho = new Grid_XYZ(dx, dy, dz, rho_halo_x + px, rho_halo_y + py, rho_halo_z + pz, "rho");
  gridPtrs.push_back(rho);
  mu = new Grid_XYZ(dx, dy, dz, mu_halo_x + px, mu_halo_y + py, mu_halo_z + pz, "mu");
  gridPtrs.push_back(mu);
  sponge = new Grid_XYZ(dx, dy, dz, sponge_halo_x + px, sponge_halo_y + py, sponge_halo_z + pz, "sponge");
  gridPtrs.push_back(sponge);
 }

 virtual void allocParams() {
  paramPtrs.clear();
  delta_t = new GenericGrid0d<real_t>();
  paramPtrs.push_back(delta_t);
  h = new GenericGrid0d<real_t>();
  paramPtrs.push_back(h);
 }
};

////// Stencil equation 'velocity' //////

struct Stencil_velocity {
 std::string name = "velocity";

 // 78 FP operation(s) per point:
 // vel_x(t+1, x, y, z) = ((vel_x(t, x, y, z) + ((delta_t / (h * (rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25)) * ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))))) * sponge(x, y, z)).
 // vel_y(t+1, x, y, z) = ((vel_y(t, x, y, z) + ((delta_t / (h * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25)) * ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))))) * sponge(x, y, z)).
 // vel_z(t+1, x, y, z) = ((vel_z(t, x, y, z) + ((delta_t / (h * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25)) * ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))))) * sponge(x, y, z)).
 const int scalar_fp_ops = 78;

 // All grids updated by this equation.
 std::vector<RealVecGridBase*> eqGridPtrs;
 void init(StencilContext_awp& context) {
  eqGridPtrs.clear();
  eqGridPtrs.push_back(context.vel_x);
  eqGridPtrs.push_back(context.vel_y);
  eqGridPtrs.push_back(context.vel_z);
 }

 // Calculate one scalar result relative to indices t, x, y, z.
 void calc_scalar(StencilContext_awp& context, idx_t t, idx_t x, idx_t y, idx_t z) {

 // temp1 = delta_t().
 real_t temp1 = (*context.delta_t)();

 // temp2 = h().
 real_t temp2 = (*context.h)();

 // temp3 = rho(x, y, z).
 real_t temp3 = context.rho->readElem(x, y, z, __LINE__);

 // temp4 = rho(x, y-1, z).
 real_t temp4 = context.rho->readElem(x, y-1, z, __LINE__);

 // temp5 = rho(x, y, z) + rho(x, y-1, z).
 real_t temp5 = temp3 + temp4;

 // temp6 = rho(x, y, z-1).
 real_t temp6 = context.rho->readElem(x, y, z-1, __LINE__);

 // temp7 = rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1).
 real_t temp7 = temp5 + temp6;

 // temp8 = rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1).
 real_t temp8 = temp7 + context.rho->readElem(x, y-1, z-1, __LINE__);

 // temp9 = h() * (rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)).
 real_t temp9 = temp2 * temp8;

 // temp10 = 0.25.
 real_t temp10 = 2.50000000000000000e-01;

 // temp11 = h() * (rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25.
 real_t temp11 = temp9 * temp10;

 // temp12 = (delta_t / (h * (rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25)).
 real_t temp12 = temp1 / temp11;

 // temp13 = 1.125.
 real_t temp13 = 1.12500000000000000e+00;

 // temp14 = 1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z)).
 real_t temp14 = temp13 * (context.stress_xx->readElem(t, x, y, z, __LINE__) - context.stress_xx->readElem(t, x-1, y, z, __LINE__));

 // temp15 = -0.0416667.
 real_t temp15 = -4.16666666666666644e-02;

 // temp16 = -0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z)).
 real_t temp16 = temp15 * (context.stress_xx->readElem(t, x+1, y, z, __LINE__) - context.stress_xx->readElem(t, x-2, y, z, __LINE__));

 // temp17 = (1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))).
 real_t temp17 = temp14 + temp16;

 // temp18 = stress_xy(t, x, y, z).
 real_t temp18 = context.stress_xy->readElem(t, x, y, z, __LINE__);

 // temp19 = (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z)).
 real_t temp19 = temp18 - context.stress_xy->readElem(t, x, y-1, z, __LINE__);

 // temp20 = 1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z)).
 real_t temp20 = temp13 * temp19;

 // temp21 = (1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))).
 real_t temp21 = temp17 + temp20;

 // temp22 = -0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z)).
 real_t temp22 = temp15 * (context.stress_xy->readElem(t, x, y+1, z, __LINE__) - context.stress_xy->readElem(t, x, y-2, z, __LINE__));

 // temp23 = (1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))).
 real_t temp23 = temp21 + temp22;

 // temp24 = stress_xz(t, x, y, z).
 real_t temp24 = context.stress_xz->readElem(t, x, y, z, __LINE__);

 // temp25 = (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1)).
 real_t temp25 = temp24 - context.stress_xz->readElem(t, x, y, z-1, __LINE__);

 // temp26 = 1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1)).
 real_t temp26 = temp13 * temp25;

 // temp27 = (1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))).
 real_t temp27 = temp23 + temp26;

 // temp28 = -0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)).
 real_t temp28 = temp15 * (context.stress_xz->readElem(t, x, y, z+1, __LINE__) - context.stress_xz->readElem(t, x, y, z-2, __LINE__));

 // temp29 = (1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2))).
 real_t temp29 = temp27 + temp28;

 // temp30 = (delta_t / (h * (rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25)) * ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))).
 real_t temp30 = temp12 * temp29;

 // temp31 = vel_x(t, x, y, z) + ((delta_t / (h * (rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25)) * ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2))))).
 real_t temp31 = context.vel_x->readElem(t, x, y, z, __LINE__) + temp30;

 // temp32 = sponge(x, y, z).
 real_t temp32 = context.sponge->readElem(x, y, z, __LINE__);

 // temp33 = (vel_x(t, x, y, z) + ((delta_t / (h * (rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25)) * ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))))) * sponge(x, y, z).
 real_t temp33 = temp31 * temp32;

 // temp34 = ((vel_x(t, x, y, z) + ((delta_t / (h * (rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25)) * ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))))) * sponge(x, y, z)).
 real_t temp34 = temp33;

 // Save result to vel_x(t+1, x, y, z):
 context.vel_x->writeElem(temp34, t+1, x, y, z, __LINE__);

 // temp35 = rho(x+1, y, z).
 real_t temp35 = context.rho->readElem(x+1, y, z, __LINE__);

 // temp36 = rho(x, y, z) + rho(x+1, y, z).
 real_t temp36 = temp3 + temp35;

 // temp37 = rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1).
 real_t temp37 = temp36 + temp6;

 // temp38 = rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1).
 real_t temp38 = temp37 + context.rho->readElem(x+1, y, z-1, __LINE__);

 // temp39 = h() * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)).
 real_t temp39 = temp2 * temp38;

 // temp40 = h() * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25.
 real_t temp40 = temp39 * temp10;

 // temp41 = (delta_t / (h * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25)).
 real_t temp41 = temp1 / temp40;

 // temp42 = (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z)).
 real_t temp42 = context.stress_xy->readElem(t, x+1, y, z, __LINE__) - temp18;

 // temp43 = 1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z)).
 real_t temp43 = temp13 * temp42;

 // temp44 = -0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z)).
 real_t temp44 = temp15 * (context.stress_xy->readElem(t, x+2, y, z, __LINE__) - context.stress_xy->readElem(t, x-1, y, z, __LINE__));

 // temp45 = (1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))).
 real_t temp45 = temp43 + temp44;

 // temp46 = 1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z)).
 real_t temp46 = temp13 * (context.stress_yy->readElem(t, x, y+1, z, __LINE__) - context.stress_yy->readElem(t, x, y, z, __LINE__));

 // temp47 = (1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))).
 real_t temp47 = temp45 + temp46;

 // temp48 = -0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z)).
 real_t temp48 = temp15 * (context.stress_yy->readElem(t, x, y+2, z, __LINE__) - context.stress_yy->readElem(t, x, y-1, z, __LINE__));

 // temp49 = (1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))).
 real_t temp49 = temp47 + temp48;

 // temp50 = stress_yz(t, x, y, z).
 real_t temp50 = context.stress_yz->readElem(t, x, y, z, __LINE__);

 // temp51 = (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1)).
 real_t temp51 = temp50 - context.stress_yz->readElem(t, x, y, z-1, __LINE__);

 // temp52 = 1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1)).
 real_t temp52 = temp13 * temp51;

 // temp53 = (1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))).
 real_t temp53 = temp49 + temp52;

 // temp54 = -0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)).
 real_t temp54 = temp15 * (context.stress_yz->readElem(t, x, y, z+1, __LINE__) - context.stress_yz->readElem(t, x, y, z-2, __LINE__));

 // temp55 = (1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2))).
 real_t temp55 = temp53 + temp54;

 // temp56 = (delta_t / (h * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25)) * ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))).
 real_t temp56 = temp41 * temp55;

 // temp57 = vel_y(t, x, y, z) + ((delta_t / (h * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25)) * ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2))))).
 real_t temp57 = context.vel_y->readElem(t, x, y, z, __LINE__) + temp56;

 // temp58 = (vel_y(t, x, y, z) + ((delta_t / (h * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25)) * ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))))) * sponge(x, y, z).
 real_t temp58 = temp57 * temp32;

 // temp59 = ((vel_y(t, x, y, z) + ((delta_t / (h * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25)) * ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))))) * sponge(x, y, z)).
 real_t temp59 = temp58;

 // Save result to vel_y(t+1, x, y, z):
 context.vel_y->writeElem(temp59, t+1, x, y, z, __LINE__);

 // temp60 = rho(x, y, z) + rho(x+1, y, z).
 real_t temp60 = temp3 + temp35;

 // temp61 = rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z).
 real_t temp61 = temp60 + temp4;

 // temp62 = rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z).
 real_t temp62 = temp61 + context.rho->readElem(x+1, y-1, z, __LINE__);

 // temp63 = h() * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)).
 real_t temp63 = temp2 * temp62;

 // temp64 = h() * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25.
 real_t temp64 = temp63 * temp10;

 // temp65 = (delta_t / (h * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25)).
 real_t temp65 = temp1 / temp64;

 // temp66 = (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z)).
 real_t temp66 = context.stress_xz->readElem(t, x+1, y, z, __LINE__) - temp24;

 // temp67 = 1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z)).
 real_t temp67 = temp13 * temp66;

 // temp68 = -0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z)).
 real_t temp68 = temp15 * (context.stress_xz->readElem(t, x+2, y, z, __LINE__) - context.stress_xz->readElem(t, x-1, y, z, __LINE__));

 // temp69 = (1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))).
 real_t temp69 = temp67 + temp68;

 // temp70 = (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z)).
 real_t temp70 = temp50 - context.stress_yz->readElem(t, x, y-1, z, __LINE__);

 // temp71 = 1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z)).
 real_t temp71 = temp13 * temp70;

 // temp72 = (1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))).
 real_t temp72 = temp69 + temp71;

 // temp73 = -0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z)).
 real_t temp73 = temp15 * (context.stress_yz->readElem(t, x, y+1, z, __LINE__) - context.stress_yz->readElem(t, x, y-2, z, __LINE__));

 // temp74 = (1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))).
 real_t temp74 = temp72 + temp73;

 // temp75 = 1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z)).
 real_t temp75 = temp13 * (context.stress_zz->readElem(t, x, y, z+1, __LINE__) - context.stress_zz->readElem(t, x, y, z, __LINE__));

 // temp76 = (1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))).
 real_t temp76 = temp74 + temp75;

 // temp77 = -0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)).
 real_t temp77 = temp15 * (context.stress_zz->readElem(t, x, y, z+2, __LINE__) - context.stress_zz->readElem(t, x, y, z-1, __LINE__));

 // temp78 = (1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1))).
 real_t temp78 = temp76 + temp77;

 // temp79 = (delta_t / (h * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25)) * ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))).
 real_t temp79 = temp65 * temp78;

 // temp80 = vel_z(t, x, y, z) + ((delta_t / (h * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25)) * ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1))))).
 real_t temp80 = context.vel_z->readElem(t, x, y, z, __LINE__) + temp79;

 // temp81 = (vel_z(t, x, y, z) + ((delta_t / (h * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25)) * ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))))) * sponge(x, y, z).
 real_t temp81 = temp80 * temp32;

 // temp82 = ((vel_z(t, x, y, z) + ((delta_t / (h * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25)) * ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))))) * sponge(x, y, z)).
 real_t temp82 = temp81;

 // Save result to vel_z(t+1, x, y, z):
 context.vel_z->writeElem(temp82, t+1, x, y, z, __LINE__);
} // scalar calculation.

 // Calculate 16 result(s) relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=4 * y=4 * z=1' vector(s).
 // Indices must be normalized, i.e., already divided by VLEN_*.
 // SIMD calculations use 44 vector block(s) created from 38 aligned vector-block(s).
 // There are 1248 FP operation(s) per cluster.
 void calc_vector(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {

 // Un-normalized indices.
 idx_t t = tv;
 idx_t x = xv * 4;
 idx_t y = yv * 4;
 idx_t z = zv * 1;

 // Read aligned vector block from vel_x at t, x, y, z.
 real_vec_t temp_vec1 = context.vel_x->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from rho at x, y, z.
 real_vec_t temp_vec2 = context.rho->readVecNorm(xv, yv, zv, __LINE__);

 // Read aligned vector block from rho at x, y-4, z.
 real_vec_t temp_vec3 = context.rho->readVecNorm(xv, yv-(4/4), zv, __LINE__);

 // Construct unaligned vector block from rho at x, y-1, z.
 real_vec_t temp_vec4;
 // temp_vec4[0] = temp_vec3[12];  // for x, y-1, z;
 // temp_vec4[1] = temp_vec3[13];  // for x+1, y-1, z;
 // temp_vec4[2] = temp_vec3[14];  // for x+2, y-1, z;
 // temp_vec4[3] = temp_vec3[15];  // for x+3, y-1, z;
 // temp_vec4[4] = temp_vec2[0];  // for x, y, z;
 // temp_vec4[5] = temp_vec2[1];  // for x+1, y, z;
 // temp_vec4[6] = temp_vec2[2];  // for x+2, y, z;
 // temp_vec4[7] = temp_vec2[3];  // for x+3, y, z;
 // temp_vec4[8] = temp_vec2[4];  // for x, y+1, z;
 // temp_vec4[9] = temp_vec2[5];  // for x+1, y+1, z;
 // temp_vec4[10] = temp_vec2[6];  // for x+2, y+1, z;
 // temp_vec4[11] = temp_vec2[7];  // for x+3, y+1, z;
 // temp_vec4[12] = temp_vec2[8];  // for x, y+2, z;
 // temp_vec4[13] = temp_vec2[9];  // for x+1, y+2, z;
 // temp_vec4[14] = temp_vec2[10];  // for x+2, y+2, z;
 // temp_vec4[15] = temp_vec2[11];  // for x+3, y+2, z;
 // Get 12 element(s) from temp_vec2 and 4 from temp_vec3.
 real_vec_align<12>(temp_vec4, temp_vec2, temp_vec3);

 // Read aligned vector block from rho at x, y, z-1.
 real_vec_t temp_vec5 = context.rho->readVecNorm(xv, yv, zv-(1/1), __LINE__);

 // Read aligned vector block from rho at x, y-4, z-1.
 real_vec_t temp_vec6 = context.rho->readVecNorm(xv, yv-(4/4), zv-(1/1), __LINE__);

 // Construct unaligned vector block from rho at x, y-1, z-1.
 real_vec_t temp_vec7;
 // temp_vec7[0] = temp_vec6[12];  // for x, y-1, z-1;
 // temp_vec7[1] = temp_vec6[13];  // for x+1, y-1, z-1;
 // temp_vec7[2] = temp_vec6[14];  // for x+2, y-1, z-1;
 // temp_vec7[3] = temp_vec6[15];  // for x+3, y-1, z-1;
 // temp_vec7[4] = temp_vec5[0];  // for x, y, z-1;
 // temp_vec7[5] = temp_vec5[1];  // for x+1, y, z-1;
 // temp_vec7[6] = temp_vec5[2];  // for x+2, y, z-1;
 // temp_vec7[7] = temp_vec5[3];  // for x+3, y, z-1;
 // temp_vec7[8] = temp_vec5[4];  // for x, y+1, z-1;
 // temp_vec7[9] = temp_vec5[5];  // for x+1, y+1, z-1;
 // temp_vec7[10] = temp_vec5[6];  // for x+2, y+1, z-1;
 // temp_vec7[11] = temp_vec5[7];  // for x+3, y+1, z-1;
 // temp_vec7[12] = temp_vec5[8];  // for x, y+2, z-1;
 // temp_vec7[13] = temp_vec5[9];  // for x+1, y+2, z-1;
 // temp_vec7[14] = temp_vec5[10];  // for x+2, y+2, z-1;
 // temp_vec7[15] = temp_vec5[11];  // for x+3, y+2, z-1;
 // Get 12 element(s) from temp_vec5 and 4 from temp_vec6.
 real_vec_align<12>(temp_vec7, temp_vec5, temp_vec6);

 // Read aligned vector block from stress_xx at t, x, y, z.
 real_vec_t temp_vec8 = context.stress_xx->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from stress_xx at t, x-4, y, z.
 real_vec_t temp_vec9 = context.stress_xx->readVecNorm(tv, xv-(4/4), yv, zv, __LINE__);

 // Construct unaligned vector block from stress_xx at t, x-1, y, z.
 real_vec_t temp_vec10;
 // temp_vec10[0] = temp_vec9[3];  // for t, x-1, y, z;
 // temp_vec10[1] = temp_vec8[0];  // for t, x, y, z;
 // temp_vec10[2] = temp_vec8[1];  // for t, x+1, y, z;
 // temp_vec10[3] = temp_vec8[2];  // for t, x+2, y, z;
 // temp_vec10[4] = temp_vec9[7];  // for t, x-1, y+1, z;
 // temp_vec10[5] = temp_vec8[4];  // for t, x, y+1, z;
 // temp_vec10[6] = temp_vec8[5];  // for t, x+1, y+1, z;
 // temp_vec10[7] = temp_vec8[6];  // for t, x+2, y+1, z;
 // temp_vec10[8] = temp_vec9[11];  // for t, x-1, y+2, z;
 // temp_vec10[9] = temp_vec8[8];  // for t, x, y+2, z;
 // temp_vec10[10] = temp_vec8[9];  // for t, x+1, y+2, z;
 // temp_vec10[11] = temp_vec8[10];  // for t, x+2, y+2, z;
 // temp_vec10[12] = temp_vec9[15];  // for t, x-1, y+3, z;
 // temp_vec10[13] = temp_vec8[12];  // for t, x, y+3, z;
 // temp_vec10[14] = temp_vec8[13];  // for t, x+1, y+3, z;
 // temp_vec10[15] = temp_vec8[14];  // for t, x+2, y+3, z;
 const real_vec_t_data ctrl_data_A3_B0_B1_B2_A7_B4_B5_B6_A11_B8_B9_B10_A15_B12_B13_B14 = { .ci = { 3, ctrl_sel_bit |0, ctrl_sel_bit |1, ctrl_sel_bit |2, 7, ctrl_sel_bit |4, ctrl_sel_bit |5, ctrl_sel_bit |6, 11, ctrl_sel_bit |8, ctrl_sel_bit |9, ctrl_sel_bit |10, 15, ctrl_sel_bit |12, ctrl_sel_bit |13, ctrl_sel_bit |14 } };
 const real_vec_t ctrl_A3_B0_B1_B2_A7_B4_B5_B6_A11_B8_B9_B10_A15_B12_B13_B14(ctrl_data_A3_B0_B1_B2_A7_B4_B5_B6_A11_B8_B9_B10_A15_B12_B13_B14);
 real_vec_permute2(temp_vec10, ctrl_A3_B0_B1_B2_A7_B4_B5_B6_A11_B8_B9_B10_A15_B12_B13_B14, temp_vec9, temp_vec8);

 // Read aligned vector block from stress_xx at t, x+4, y, z.
 real_vec_t temp_vec11 = context.stress_xx->readVecNorm(tv, xv+(4/4), yv, zv, __LINE__);

 // Construct unaligned vector block from stress_xx at t, x+1, y, z.
 real_vec_t temp_vec12;
 // temp_vec12[0] = temp_vec8[1];  // for t, x+1, y, z;
 // temp_vec12[1] = temp_vec8[2];  // for t, x+2, y, z;
 // temp_vec12[2] = temp_vec8[3];  // for t, x+3, y, z;
 // temp_vec12[3] = temp_vec11[0];  // for t, x+4, y, z;
 // temp_vec12[4] = temp_vec8[5];  // for t, x+1, y+1, z;
 // temp_vec12[5] = temp_vec8[6];  // for t, x+2, y+1, z;
 // temp_vec12[6] = temp_vec8[7];  // for t, x+3, y+1, z;
 // temp_vec12[7] = temp_vec11[4];  // for t, x+4, y+1, z;
 // temp_vec12[8] = temp_vec8[9];  // for t, x+1, y+2, z;
 // temp_vec12[9] = temp_vec8[10];  // for t, x+2, y+2, z;
 // temp_vec12[10] = temp_vec8[11];  // for t, x+3, y+2, z;
 // temp_vec12[11] = temp_vec11[8];  // for t, x+4, y+2, z;
 // temp_vec12[12] = temp_vec8[13];  // for t, x+1, y+3, z;
 // temp_vec12[13] = temp_vec8[14];  // for t, x+2, y+3, z;
 // temp_vec12[14] = temp_vec8[15];  // for t, x+3, y+3, z;
 // temp_vec12[15] = temp_vec11[12];  // for t, x+4, y+3, z;
 const real_vec_t_data ctrl_data_A1_A2_A3_B0_A5_A6_A7_B4_A9_A10_A11_B8_A13_A14_A15_B12 = { .ci = { 1, 2, 3, ctrl_sel_bit |0, 5, 6, 7, ctrl_sel_bit |4, 9, 10, 11, ctrl_sel_bit |8, 13, 14, 15, ctrl_sel_bit |12 } };
 const real_vec_t ctrl_A1_A2_A3_B0_A5_A6_A7_B4_A9_A10_A11_B8_A13_A14_A15_B12(ctrl_data_A1_A2_A3_B0_A5_A6_A7_B4_A9_A10_A11_B8_A13_A14_A15_B12);
 real_vec_permute2(temp_vec12, ctrl_A1_A2_A3_B0_A5_A6_A7_B4_A9_A10_A11_B8_A13_A14_A15_B12, temp_vec8, temp_vec11);

 // Construct unaligned vector block from stress_xx at t, x-2, y, z.
 real_vec_t temp_vec13;
 // temp_vec13[0] = temp_vec9[2];  // for t, x-2, y, z;
 // temp_vec13[1] = temp_vec9[3];  // for t, x-1, y, z;
 // temp_vec13[2] = temp_vec8[0];  // for t, x, y, z;
 // temp_vec13[3] = temp_vec8[1];  // for t, x+1, y, z;
 // temp_vec13[4] = temp_vec9[6];  // for t, x-2, y+1, z;
 // temp_vec13[5] = temp_vec9[7];  // for t, x-1, y+1, z;
 // temp_vec13[6] = temp_vec8[4];  // for t, x, y+1, z;
 // temp_vec13[7] = temp_vec8[5];  // for t, x+1, y+1, z;
 // temp_vec13[8] = temp_vec9[10];  // for t, x-2, y+2, z;
 // temp_vec13[9] = temp_vec9[11];  // for t, x-1, y+2, z;
 // temp_vec13[10] = temp_vec8[8];  // for t, x, y+2, z;
 // temp_vec13[11] = temp_vec8[9];  // for t, x+1, y+2, z;
 // temp_vec13[12] = temp_vec9[14];  // for t, x-2, y+3, z;
 // temp_vec13[13] = temp_vec9[15];  // for t, x-1, y+3, z;
 // temp_vec13[14] = temp_vec8[12];  // for t, x, y+3, z;
 // temp_vec13[15] = temp_vec8[13];  // for t, x+1, y+3, z;
 const real_vec_t_data ctrl_data_A2_A3_B0_B1_A6_A7_B4_B5_A10_A11_B8_B9_A14_A15_B12_B13 = { .ci = { 2, 3, ctrl_sel_bit |0, ctrl_sel_bit |1, 6, 7, ctrl_sel_bit |4, ctrl_sel_bit |5, 10, 11, ctrl_sel_bit |8, ctrl_sel_bit |9, 14, 15, ctrl_sel_bit |12, ctrl_sel_bit |13 } };
 const real_vec_t ctrl_A2_A3_B0_B1_A6_A7_B4_B5_A10_A11_B8_B9_A14_A15_B12_B13(ctrl_data_A2_A3_B0_B1_A6_A7_B4_B5_A10_A11_B8_B9_A14_A15_B12_B13);
 real_vec_permute2(temp_vec13, ctrl_A2_A3_B0_B1_A6_A7_B4_B5_A10_A11_B8_B9_A14_A15_B12_B13, temp_vec9, temp_vec8);

 // Read aligned vector block from stress_xy at t, x, y, z.
 real_vec_t temp_vec14 = context.stress_xy->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from stress_xy at t, x, y-4, z.
 real_vec_t temp_vec15 = context.stress_xy->readVecNorm(tv, xv, yv-(4/4), zv, __LINE__);

 // Construct unaligned vector block from stress_xy at t, x, y-1, z.
 real_vec_t temp_vec16;
 // temp_vec16[0] = temp_vec15[12];  // for t, x, y-1, z;
 // temp_vec16[1] = temp_vec15[13];  // for t, x+1, y-1, z;
 // temp_vec16[2] = temp_vec15[14];  // for t, x+2, y-1, z;
 // temp_vec16[3] = temp_vec15[15];  // for t, x+3, y-1, z;
 // temp_vec16[4] = temp_vec14[0];  // for t, x, y, z;
 // temp_vec16[5] = temp_vec14[1];  // for t, x+1, y, z;
 // temp_vec16[6] = temp_vec14[2];  // for t, x+2, y, z;
 // temp_vec16[7] = temp_vec14[3];  // for t, x+3, y, z;
 // temp_vec16[8] = temp_vec14[4];  // for t, x, y+1, z;
 // temp_vec16[9] = temp_vec14[5];  // for t, x+1, y+1, z;
 // temp_vec16[10] = temp_vec14[6];  // for t, x+2, y+1, z;
 // temp_vec16[11] = temp_vec14[7];  // for t, x+3, y+1, z;
 // temp_vec16[12] = temp_vec14[8];  // for t, x, y+2, z;
 // temp_vec16[13] = temp_vec14[9];  // for t, x+1, y+2, z;
 // temp_vec16[14] = temp_vec14[10];  // for t, x+2, y+2, z;
 // temp_vec16[15] = temp_vec14[11];  // for t, x+3, y+2, z;
 // Get 12 element(s) from temp_vec14 and 4 from temp_vec15.
 real_vec_align<12>(temp_vec16, temp_vec14, temp_vec15);

 // Read aligned vector block from stress_xy at t, x, y+4, z.
 real_vec_t temp_vec17 = context.stress_xy->readVecNorm(tv, xv, yv+(4/4), zv, __LINE__);

 // Construct unaligned vector block from stress_xy at t, x, y+1, z.
 real_vec_t temp_vec18;
 // temp_vec18[0] = temp_vec14[4];  // for t, x, y+1, z;
 // temp_vec18[1] = temp_vec14[5];  // for t, x+1, y+1, z;
 // temp_vec18[2] = temp_vec14[6];  // for t, x+2, y+1, z;
 // temp_vec18[3] = temp_vec14[7];  // for t, x+3, y+1, z;
 // temp_vec18[4] = temp_vec14[8];  // for t, x, y+2, z;
 // temp_vec18[5] = temp_vec14[9];  // for t, x+1, y+2, z;
 // temp_vec18[6] = temp_vec14[10];  // for t, x+2, y+2, z;
 // temp_vec18[7] = temp_vec14[11];  // for t, x+3, y+2, z;
 // temp_vec18[8] = temp_vec14[12];  // for t, x, y+3, z;
 // temp_vec18[9] = temp_vec14[13];  // for t, x+1, y+3, z;
 // temp_vec18[10] = temp_vec14[14];  // for t, x+2, y+3, z;
 // temp_vec18[11] = temp_vec14[15];  // for t, x+3, y+3, z;
 // temp_vec18[12] = temp_vec17[0];  // for t, x, y+4, z;
 // temp_vec18[13] = temp_vec17[1];  // for t, x+1, y+4, z;
 // temp_vec18[14] = temp_vec17[2];  // for t, x+2, y+4, z;
 // temp_vec18[15] = temp_vec17[3];  // for t, x+3, y+4, z;
 // Get 4 element(s) from temp_vec17 and 12 from temp_vec14.
 real_vec_align<4>(temp_vec18, temp_vec17, temp_vec14);

 // Construct unaligned vector block from stress_xy at t, x, y-2, z.
 real_vec_t temp_vec19;
 // temp_vec19[0] = temp_vec15[8];  // for t, x, y-2, z;
 // temp_vec19[1] = temp_vec15[9];  // for t, x+1, y-2, z;
 // temp_vec19[2] = temp_vec15[10];  // for t, x+2, y-2, z;
 // temp_vec19[3] = temp_vec15[11];  // for t, x+3, y-2, z;
 // temp_vec19[4] = temp_vec15[12];  // for t, x, y-1, z;
 // temp_vec19[5] = temp_vec15[13];  // for t, x+1, y-1, z;
 // temp_vec19[6] = temp_vec15[14];  // for t, x+2, y-1, z;
 // temp_vec19[7] = temp_vec15[15];  // for t, x+3, y-1, z;
 // temp_vec19[8] = temp_vec14[0];  // for t, x, y, z;
 // temp_vec19[9] = temp_vec14[1];  // for t, x+1, y, z;
 // temp_vec19[10] = temp_vec14[2];  // for t, x+2, y, z;
 // temp_vec19[11] = temp_vec14[3];  // for t, x+3, y, z;
 // temp_vec19[12] = temp_vec14[4];  // for t, x, y+1, z;
 // temp_vec19[13] = temp_vec14[5];  // for t, x+1, y+1, z;
 // temp_vec19[14] = temp_vec14[6];  // for t, x+2, y+1, z;
 // temp_vec19[15] = temp_vec14[7];  // for t, x+3, y+1, z;
 // Get 8 element(s) from temp_vec14 and 8 from temp_vec15.
 real_vec_align<8>(temp_vec19, temp_vec14, temp_vec15);

 // Read aligned vector block from stress_xz at t, x, y, z.
 real_vec_t temp_vec20 = context.stress_xz->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from stress_xz at t, x, y, z-1.
 real_vec_t temp_vec21 = context.stress_xz->readVecNorm(tv, xv, yv, zv-(1/1), __LINE__);

 // Read aligned vector block from stress_xz at t, x, y, z+1.
 real_vec_t temp_vec22 = context.stress_xz->readVecNorm(tv, xv, yv, zv+(1/1), __LINE__);

 // Read aligned vector block from stress_xz at t, x, y, z-2.
 real_vec_t temp_vec23 = context.stress_xz->readVecNorm(tv, xv, yv, zv-(2/1), __LINE__);

 // Read aligned vector block from sponge at x, y, z.
 real_vec_t temp_vec24 = context.sponge->readVecNorm(xv, yv, zv, __LINE__);

 // temp_vec25 = delta_t().
 real_vec_t temp_vec25 = (*context.delta_t)();

 // temp_vec26 = h().
 real_vec_t temp_vec26 = (*context.h)();

 // temp_vec27 = rho(x, y, z).
 real_vec_t temp_vec27 = temp_vec2;

 // temp_vec28 = rho(x, y-1, z).
 real_vec_t temp_vec28 = temp_vec4;

 // temp_vec29 = rho(x, y, z) + rho(x, y-1, z).
 real_vec_t temp_vec29 = temp_vec27 + temp_vec28;

 // temp_vec30 = rho(x, y, z-1).
 real_vec_t temp_vec30 = temp_vec5;

 // temp_vec31 = rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1).
 real_vec_t temp_vec31 = temp_vec29 + temp_vec30;

 // temp_vec32 = rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1).
 real_vec_t temp_vec32 = temp_vec31 + temp_vec7;

 // temp_vec33 = h() * (rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)).
 real_vec_t temp_vec33 = temp_vec26 * temp_vec32;

 // temp_vec34 = 0.25.
 real_vec_t temp_vec34 = 2.50000000000000000e-01;

 // temp_vec35 = h() * (rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25.
 real_vec_t temp_vec35 = temp_vec33 * temp_vec34;

 // temp_vec36 = (delta_t / (h * (rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25)).
 real_vec_t temp_vec36 = temp_vec25 / temp_vec35;

 // temp_vec37 = 1.125.
 real_vec_t temp_vec37 = 1.12500000000000000e+00;

 // temp_vec38 = 1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z)).
 real_vec_t temp_vec38 = temp_vec37 * (temp_vec8 - temp_vec10);

 // temp_vec39 = -0.0416667.
 real_vec_t temp_vec39 = -4.16666666666666644e-02;

 // temp_vec40 = -0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z)).
 real_vec_t temp_vec40 = temp_vec39 * (temp_vec12 - temp_vec13);

 // temp_vec41 = (1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))).
 real_vec_t temp_vec41 = temp_vec38 + temp_vec40;

 // temp_vec42 = stress_xy(t, x, y, z).
 real_vec_t temp_vec42 = temp_vec14;

 // temp_vec43 = (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z)).
 real_vec_t temp_vec43 = temp_vec42 - temp_vec16;

 // temp_vec44 = 1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z)).
 real_vec_t temp_vec44 = temp_vec37 * temp_vec43;

 // temp_vec45 = (1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))).
 real_vec_t temp_vec45 = temp_vec41 + temp_vec44;

 // temp_vec46 = -0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z)).
 real_vec_t temp_vec46 = temp_vec39 * (temp_vec18 - temp_vec19);

 // temp_vec47 = (1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))).
 real_vec_t temp_vec47 = temp_vec45 + temp_vec46;

 // temp_vec48 = stress_xz(t, x, y, z).
 real_vec_t temp_vec48 = temp_vec20;

 // temp_vec49 = (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1)).
 real_vec_t temp_vec49 = temp_vec48 - temp_vec21;

 // temp_vec50 = 1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1)).
 real_vec_t temp_vec50 = temp_vec37 * temp_vec49;

 // temp_vec51 = (1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))).
 real_vec_t temp_vec51 = temp_vec47 + temp_vec50;

 // temp_vec52 = -0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)).
 real_vec_t temp_vec52 = temp_vec39 * (temp_vec22 - temp_vec23);

 // temp_vec53 = (1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2))).
 real_vec_t temp_vec53 = temp_vec51 + temp_vec52;

 // temp_vec54 = (delta_t / (h * (rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25)) * ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))).
 real_vec_t temp_vec54 = temp_vec36 * temp_vec53;

 // temp_vec55 = vel_x(t, x, y, z) + ((delta_t / (h * (rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25)) * ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2))))).
 real_vec_t temp_vec55 = temp_vec1 + temp_vec54;

 // temp_vec56 = sponge(x, y, z).
 real_vec_t temp_vec56 = temp_vec24;

 // temp_vec57 = (vel_x(t, x, y, z) + ((delta_t / (h * (rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25)) * ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))))) * sponge(x, y, z).
 real_vec_t temp_vec57 = temp_vec55 * temp_vec56;

 // temp_vec58 = ((vel_x(t, x, y, z) + ((delta_t / (h * (rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25)) * ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))))) * sponge(x, y, z)).
 real_vec_t temp_vec58 = temp_vec57;

 // Save result to vel_x(t+1, x, y, z):
 
 // Write aligned vector block to vel_x at t+1, x, y, z.
context.vel_x->writeVecNorm(temp_vec58, tv+(1/1), xv, yv, zv, __LINE__);
;

 // Read aligned vector block from vel_y at t, x, y, z.
 real_vec_t temp_vec59 = context.vel_y->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from rho at x+4, y, z.
 real_vec_t temp_vec60 = context.rho->readVecNorm(xv+(4/4), yv, zv, __LINE__);

 // Construct unaligned vector block from rho at x+1, y, z.
 real_vec_t temp_vec61;
 // temp_vec61[0] = temp_vec2[1];  // for x+1, y, z;
 // temp_vec61[1] = temp_vec2[2];  // for x+2, y, z;
 // temp_vec61[2] = temp_vec2[3];  // for x+3, y, z;
 // temp_vec61[3] = temp_vec60[0];  // for x+4, y, z;
 // temp_vec61[4] = temp_vec2[5];  // for x+1, y+1, z;
 // temp_vec61[5] = temp_vec2[6];  // for x+2, y+1, z;
 // temp_vec61[6] = temp_vec2[7];  // for x+3, y+1, z;
 // temp_vec61[7] = temp_vec60[4];  // for x+4, y+1, z;
 // temp_vec61[8] = temp_vec2[9];  // for x+1, y+2, z;
 // temp_vec61[9] = temp_vec2[10];  // for x+2, y+2, z;
 // temp_vec61[10] = temp_vec2[11];  // for x+3, y+2, z;
 // temp_vec61[11] = temp_vec60[8];  // for x+4, y+2, z;
 // temp_vec61[12] = temp_vec2[13];  // for x+1, y+3, z;
 // temp_vec61[13] = temp_vec2[14];  // for x+2, y+3, z;
 // temp_vec61[14] = temp_vec2[15];  // for x+3, y+3, z;
 // temp_vec61[15] = temp_vec60[12];  // for x+4, y+3, z;
 real_vec_permute2(temp_vec61, ctrl_A1_A2_A3_B0_A5_A6_A7_B4_A9_A10_A11_B8_A13_A14_A15_B12, temp_vec2, temp_vec60);

 // Read aligned vector block from rho at x+4, y, z-1.
 real_vec_t temp_vec62 = context.rho->readVecNorm(xv+(4/4), yv, zv-(1/1), __LINE__);

 // Construct unaligned vector block from rho at x+1, y, z-1.
 real_vec_t temp_vec63;
 // temp_vec63[0] = temp_vec5[1];  // for x+1, y, z-1;
 // temp_vec63[1] = temp_vec5[2];  // for x+2, y, z-1;
 // temp_vec63[2] = temp_vec5[3];  // for x+3, y, z-1;
 // temp_vec63[3] = temp_vec62[0];  // for x+4, y, z-1;
 // temp_vec63[4] = temp_vec5[5];  // for x+1, y+1, z-1;
 // temp_vec63[5] = temp_vec5[6];  // for x+2, y+1, z-1;
 // temp_vec63[6] = temp_vec5[7];  // for x+3, y+1, z-1;
 // temp_vec63[7] = temp_vec62[4];  // for x+4, y+1, z-1;
 // temp_vec63[8] = temp_vec5[9];  // for x+1, y+2, z-1;
 // temp_vec63[9] = temp_vec5[10];  // for x+2, y+2, z-1;
 // temp_vec63[10] = temp_vec5[11];  // for x+3, y+2, z-1;
 // temp_vec63[11] = temp_vec62[8];  // for x+4, y+2, z-1;
 // temp_vec63[12] = temp_vec5[13];  // for x+1, y+3, z-1;
 // temp_vec63[13] = temp_vec5[14];  // for x+2, y+3, z-1;
 // temp_vec63[14] = temp_vec5[15];  // for x+3, y+3, z-1;
 // temp_vec63[15] = temp_vec62[12];  // for x+4, y+3, z-1;
 real_vec_permute2(temp_vec63, ctrl_A1_A2_A3_B0_A5_A6_A7_B4_A9_A10_A11_B8_A13_A14_A15_B12, temp_vec5, temp_vec62);

 // Read aligned vector block from stress_xy at t, x+4, y, z.
 real_vec_t temp_vec64 = context.stress_xy->readVecNorm(tv, xv+(4/4), yv, zv, __LINE__);

 // Construct unaligned vector block from stress_xy at t, x+1, y, z.
 real_vec_t temp_vec65;
 // temp_vec65[0] = temp_vec14[1];  // for t, x+1, y, z;
 // temp_vec65[1] = temp_vec14[2];  // for t, x+2, y, z;
 // temp_vec65[2] = temp_vec14[3];  // for t, x+3, y, z;
 // temp_vec65[3] = temp_vec64[0];  // for t, x+4, y, z;
 // temp_vec65[4] = temp_vec14[5];  // for t, x+1, y+1, z;
 // temp_vec65[5] = temp_vec14[6];  // for t, x+2, y+1, z;
 // temp_vec65[6] = temp_vec14[7];  // for t, x+3, y+1, z;
 // temp_vec65[7] = temp_vec64[4];  // for t, x+4, y+1, z;
 // temp_vec65[8] = temp_vec14[9];  // for t, x+1, y+2, z;
 // temp_vec65[9] = temp_vec14[10];  // for t, x+2, y+2, z;
 // temp_vec65[10] = temp_vec14[11];  // for t, x+3, y+2, z;
 // temp_vec65[11] = temp_vec64[8];  // for t, x+4, y+2, z;
 // temp_vec65[12] = temp_vec14[13];  // for t, x+1, y+3, z;
 // temp_vec65[13] = temp_vec14[14];  // for t, x+2, y+3, z;
 // temp_vec65[14] = temp_vec14[15];  // for t, x+3, y+3, z;
 // temp_vec65[15] = temp_vec64[12];  // for t, x+4, y+3, z;
 real_vec_permute2(temp_vec65, ctrl_A1_A2_A3_B0_A5_A6_A7_B4_A9_A10_A11_B8_A13_A14_A15_B12, temp_vec14, temp_vec64);

 // Construct unaligned vector block from stress_xy at t, x+2, y, z.
 real_vec_t temp_vec66;
 // temp_vec66[0] = temp_vec14[2];  // for t, x+2, y, z;
 // temp_vec66[1] = temp_vec14[3];  // for t, x+3, y, z;
 // temp_vec66[2] = temp_vec64[0];  // for t, x+4, y, z;
 // temp_vec66[3] = temp_vec64[1];  // for t, x+5, y, z;
 // temp_vec66[4] = temp_vec14[6];  // for t, x+2, y+1, z;
 // temp_vec66[5] = temp_vec14[7];  // for t, x+3, y+1, z;
 // temp_vec66[6] = temp_vec64[4];  // for t, x+4, y+1, z;
 // temp_vec66[7] = temp_vec64[5];  // for t, x+5, y+1, z;
 // temp_vec66[8] = temp_vec14[10];  // for t, x+2, y+2, z;
 // temp_vec66[9] = temp_vec14[11];  // for t, x+3, y+2, z;
 // temp_vec66[10] = temp_vec64[8];  // for t, x+4, y+2, z;
 // temp_vec66[11] = temp_vec64[9];  // for t, x+5, y+2, z;
 // temp_vec66[12] = temp_vec14[14];  // for t, x+2, y+3, z;
 // temp_vec66[13] = temp_vec14[15];  // for t, x+3, y+3, z;
 // temp_vec66[14] = temp_vec64[12];  // for t, x+4, y+3, z;
 // temp_vec66[15] = temp_vec64[13];  // for t, x+5, y+3, z;
 real_vec_permute2(temp_vec66, ctrl_A2_A3_B0_B1_A6_A7_B4_B5_A10_A11_B8_B9_A14_A15_B12_B13, temp_vec14, temp_vec64);

 // Read aligned vector block from stress_xy at t, x-4, y, z.
 real_vec_t temp_vec67 = context.stress_xy->readVecNorm(tv, xv-(4/4), yv, zv, __LINE__);

 // Construct unaligned vector block from stress_xy at t, x-1, y, z.
 real_vec_t temp_vec68;
 // temp_vec68[0] = temp_vec67[3];  // for t, x-1, y, z;
 // temp_vec68[1] = temp_vec14[0];  // for t, x, y, z;
 // temp_vec68[2] = temp_vec14[1];  // for t, x+1, y, z;
 // temp_vec68[3] = temp_vec14[2];  // for t, x+2, y, z;
 // temp_vec68[4] = temp_vec67[7];  // for t, x-1, y+1, z;
 // temp_vec68[5] = temp_vec14[4];  // for t, x, y+1, z;
 // temp_vec68[6] = temp_vec14[5];  // for t, x+1, y+1, z;
 // temp_vec68[7] = temp_vec14[6];  // for t, x+2, y+1, z;
 // temp_vec68[8] = temp_vec67[11];  // for t, x-1, y+2, z;
 // temp_vec68[9] = temp_vec14[8];  // for t, x, y+2, z;
 // temp_vec68[10] = temp_vec14[9];  // for t, x+1, y+2, z;
 // temp_vec68[11] = temp_vec14[10];  // for t, x+2, y+2, z;
 // temp_vec68[12] = temp_vec67[15];  // for t, x-1, y+3, z;
 // temp_vec68[13] = temp_vec14[12];  // for t, x, y+3, z;
 // temp_vec68[14] = temp_vec14[13];  // for t, x+1, y+3, z;
 // temp_vec68[15] = temp_vec14[14];  // for t, x+2, y+3, z;
 real_vec_permute2(temp_vec68, ctrl_A3_B0_B1_B2_A7_B4_B5_B6_A11_B8_B9_B10_A15_B12_B13_B14, temp_vec67, temp_vec14);

 // Read aligned vector block from stress_yy at t, x, y, z.
 real_vec_t temp_vec69 = context.stress_yy->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from stress_yy at t, x, y+4, z.
 real_vec_t temp_vec70 = context.stress_yy->readVecNorm(tv, xv, yv+(4/4), zv, __LINE__);

 // Construct unaligned vector block from stress_yy at t, x, y+1, z.
 real_vec_t temp_vec71;
 // temp_vec71[0] = temp_vec69[4];  // for t, x, y+1, z;
 // temp_vec71[1] = temp_vec69[5];  // for t, x+1, y+1, z;
 // temp_vec71[2] = temp_vec69[6];  // for t, x+2, y+1, z;
 // temp_vec71[3] = temp_vec69[7];  // for t, x+3, y+1, z;
 // temp_vec71[4] = temp_vec69[8];  // for t, x, y+2, z;
 // temp_vec71[5] = temp_vec69[9];  // for t, x+1, y+2, z;
 // temp_vec71[6] = temp_vec69[10];  // for t, x+2, y+2, z;
 // temp_vec71[7] = temp_vec69[11];  // for t, x+3, y+2, z;
 // temp_vec71[8] = temp_vec69[12];  // for t, x, y+3, z;
 // temp_vec71[9] = temp_vec69[13];  // for t, x+1, y+3, z;
 // temp_vec71[10] = temp_vec69[14];  // for t, x+2, y+3, z;
 // temp_vec71[11] = temp_vec69[15];  // for t, x+3, y+3, z;
 // temp_vec71[12] = temp_vec70[0];  // for t, x, y+4, z;
 // temp_vec71[13] = temp_vec70[1];  // for t, x+1, y+4, z;
 // temp_vec71[14] = temp_vec70[2];  // for t, x+2, y+4, z;
 // temp_vec71[15] = temp_vec70[3];  // for t, x+3, y+4, z;
 // Get 4 element(s) from temp_vec70 and 12 from temp_vec69.
 real_vec_align<4>(temp_vec71, temp_vec70, temp_vec69);

 // Construct unaligned vector block from stress_yy at t, x, y+2, z.
 real_vec_t temp_vec72;
 // temp_vec72[0] = temp_vec69[8];  // for t, x, y+2, z;
 // temp_vec72[1] = temp_vec69[9];  // for t, x+1, y+2, z;
 // temp_vec72[2] = temp_vec69[10];  // for t, x+2, y+2, z;
 // temp_vec72[3] = temp_vec69[11];  // for t, x+3, y+2, z;
 // temp_vec72[4] = temp_vec69[12];  // for t, x, y+3, z;
 // temp_vec72[5] = temp_vec69[13];  // for t, x+1, y+3, z;
 // temp_vec72[6] = temp_vec69[14];  // for t, x+2, y+3, z;
 // temp_vec72[7] = temp_vec69[15];  // for t, x+3, y+3, z;
 // temp_vec72[8] = temp_vec70[0];  // for t, x, y+4, z;
 // temp_vec72[9] = temp_vec70[1];  // for t, x+1, y+4, z;
 // temp_vec72[10] = temp_vec70[2];  // for t, x+2, y+4, z;
 // temp_vec72[11] = temp_vec70[3];  // for t, x+3, y+4, z;
 // temp_vec72[12] = temp_vec70[4];  // for t, x, y+5, z;
 // temp_vec72[13] = temp_vec70[5];  // for t, x+1, y+5, z;
 // temp_vec72[14] = temp_vec70[6];  // for t, x+2, y+5, z;
 // temp_vec72[15] = temp_vec70[7];  // for t, x+3, y+5, z;
 // Get 8 element(s) from temp_vec70 and 8 from temp_vec69.
 real_vec_align<8>(temp_vec72, temp_vec70, temp_vec69);

 // Read aligned vector block from stress_yy at t, x, y-4, z.
 real_vec_t temp_vec73 = context.stress_yy->readVecNorm(tv, xv, yv-(4/4), zv, __LINE__);

 // Construct unaligned vector block from stress_yy at t, x, y-1, z.
 real_vec_t temp_vec74;
 // temp_vec74[0] = temp_vec73[12];  // for t, x, y-1, z;
 // temp_vec74[1] = temp_vec73[13];  // for t, x+1, y-1, z;
 // temp_vec74[2] = temp_vec73[14];  // for t, x+2, y-1, z;
 // temp_vec74[3] = temp_vec73[15];  // for t, x+3, y-1, z;
 // temp_vec74[4] = temp_vec69[0];  // for t, x, y, z;
 // temp_vec74[5] = temp_vec69[1];  // for t, x+1, y, z;
 // temp_vec74[6] = temp_vec69[2];  // for t, x+2, y, z;
 // temp_vec74[7] = temp_vec69[3];  // for t, x+3, y, z;
 // temp_vec74[8] = temp_vec69[4];  // for t, x, y+1, z;
 // temp_vec74[9] = temp_vec69[5];  // for t, x+1, y+1, z;
 // temp_vec74[10] = temp_vec69[6];  // for t, x+2, y+1, z;
 // temp_vec74[11] = temp_vec69[7];  // for t, x+3, y+1, z;
 // temp_vec74[12] = temp_vec69[8];  // for t, x, y+2, z;
 // temp_vec74[13] = temp_vec69[9];  // for t, x+1, y+2, z;
 // temp_vec74[14] = temp_vec69[10];  // for t, x+2, y+2, z;
 // temp_vec74[15] = temp_vec69[11];  // for t, x+3, y+2, z;
 // Get 12 element(s) from temp_vec69 and 4 from temp_vec73.
 real_vec_align<12>(temp_vec74, temp_vec69, temp_vec73);

 // Read aligned vector block from stress_yz at t, x, y, z.
 real_vec_t temp_vec75 = context.stress_yz->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from stress_yz at t, x, y, z-1.
 real_vec_t temp_vec76 = context.stress_yz->readVecNorm(tv, xv, yv, zv-(1/1), __LINE__);

 // Read aligned vector block from stress_yz at t, x, y, z+1.
 real_vec_t temp_vec77 = context.stress_yz->readVecNorm(tv, xv, yv, zv+(1/1), __LINE__);

 // Read aligned vector block from stress_yz at t, x, y, z-2.
 real_vec_t temp_vec78 = context.stress_yz->readVecNorm(tv, xv, yv, zv-(2/1), __LINE__);

 // temp_vec79 = rho(x+1, y, z).
 real_vec_t temp_vec79 = temp_vec61;

 // temp_vec80 = rho(x, y, z) + rho(x+1, y, z).
 real_vec_t temp_vec80 = temp_vec27 + temp_vec79;

 // temp_vec81 = rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1).
 real_vec_t temp_vec81 = temp_vec80 + temp_vec30;

 // temp_vec82 = rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1).
 real_vec_t temp_vec82 = temp_vec81 + temp_vec63;

 // temp_vec83 = h() * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)).
 real_vec_t temp_vec83 = temp_vec26 * temp_vec82;

 // temp_vec84 = h() * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25.
 real_vec_t temp_vec84 = temp_vec83 * temp_vec34;

 // temp_vec85 = (delta_t / (h * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25)).
 real_vec_t temp_vec85 = temp_vec25 / temp_vec84;

 // temp_vec86 = (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z)).
 real_vec_t temp_vec86 = temp_vec65 - temp_vec42;

 // temp_vec87 = 1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z)).
 real_vec_t temp_vec87 = temp_vec37 * temp_vec86;

 // temp_vec88 = -0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z)).
 real_vec_t temp_vec88 = temp_vec39 * (temp_vec66 - temp_vec68);

 // temp_vec89 = (1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))).
 real_vec_t temp_vec89 = temp_vec87 + temp_vec88;

 // temp_vec90 = 1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z)).
 real_vec_t temp_vec90 = temp_vec37 * (temp_vec71 - temp_vec69);

 // temp_vec91 = (1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))).
 real_vec_t temp_vec91 = temp_vec89 + temp_vec90;

 // temp_vec92 = -0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z)).
 real_vec_t temp_vec92 = temp_vec39 * (temp_vec72 - temp_vec74);

 // temp_vec93 = (1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))).
 real_vec_t temp_vec93 = temp_vec91 + temp_vec92;

 // temp_vec94 = stress_yz(t, x, y, z).
 real_vec_t temp_vec94 = temp_vec75;

 // temp_vec95 = (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1)).
 real_vec_t temp_vec95 = temp_vec94 - temp_vec76;

 // temp_vec96 = 1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1)).
 real_vec_t temp_vec96 = temp_vec37 * temp_vec95;

 // temp_vec97 = (1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))).
 real_vec_t temp_vec97 = temp_vec93 + temp_vec96;

 // temp_vec98 = -0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)).
 real_vec_t temp_vec98 = temp_vec39 * (temp_vec77 - temp_vec78);

 // temp_vec99 = (1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2))).
 real_vec_t temp_vec99 = temp_vec97 + temp_vec98;

 // temp_vec100 = (delta_t / (h * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25)) * ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))).
 real_vec_t temp_vec100 = temp_vec85 * temp_vec99;

 // temp_vec101 = vel_y(t, x, y, z) + ((delta_t / (h * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25)) * ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2))))).
 real_vec_t temp_vec101 = temp_vec59 + temp_vec100;

 // temp_vec102 = (vel_y(t, x, y, z) + ((delta_t / (h * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25)) * ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))))) * sponge(x, y, z).
 real_vec_t temp_vec102 = temp_vec101 * temp_vec56;

 // temp_vec103 = ((vel_y(t, x, y, z) + ((delta_t / (h * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25)) * ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))))) * sponge(x, y, z)).
 real_vec_t temp_vec103 = temp_vec102;

 // Save result to vel_y(t+1, x, y, z):
 
 // Write aligned vector block to vel_y at t+1, x, y, z.
context.vel_y->writeVecNorm(temp_vec103, tv+(1/1), xv, yv, zv, __LINE__);
;

 // Read aligned vector block from vel_z at t, x, y, z.
 real_vec_t temp_vec104 = context.vel_z->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from rho at x+4, y-4, z.
 real_vec_t temp_vec105 = context.rho->readVecNorm(xv+(4/4), yv-(4/4), zv, __LINE__);

 // Construct unaligned vector block from rho at x+1, y-1, z.
 real_vec_t temp_vec106;
 // temp_vec106[0] = temp_vec3[13];  // for x+1, y-1, z;
 // temp_vec106[1] = temp_vec3[14];  // for x+2, y-1, z;
 // temp_vec106[2] = temp_vec3[15];  // for x+3, y-1, z;
 // temp_vec106[3] = temp_vec105[12];  // for x+4, y-1, z;
 // temp_vec106[4] = temp_vec2[1];  // for x+1, y, z;
 // temp_vec106[5] = temp_vec2[2];  // for x+2, y, z;
 // temp_vec106[6] = temp_vec2[3];  // for x+3, y, z;
 // temp_vec106[7] = temp_vec60[0];  // for x+4, y, z;
 // temp_vec106[8] = temp_vec2[5];  // for x+1, y+1, z;
 // temp_vec106[9] = temp_vec2[6];  // for x+2, y+1, z;
 // temp_vec106[10] = temp_vec2[7];  // for x+3, y+1, z;
 // temp_vec106[11] = temp_vec60[4];  // for x+4, y+1, z;
 // temp_vec106[12] = temp_vec2[9];  // for x+1, y+2, z;
 // temp_vec106[13] = temp_vec2[10];  // for x+2, y+2, z;
 // temp_vec106[14] = temp_vec2[11];  // for x+3, y+2, z;
 // temp_vec106[15] = temp_vec60[8];  // for x+4, y+2, z;
 // Get 9 element(s) from temp_vec2 and 3 from temp_vec3.
 real_vec_align<13>(temp_vec106, temp_vec2, temp_vec3);
 // Get 3 element(s) from temp_vec60 and 1 from temp_vec105.
 real_vec_align_masked<9>(temp_vec106, temp_vec60, temp_vec105, 0x8888);

 // Read aligned vector block from stress_xz at t, x+4, y, z.
 real_vec_t temp_vec107 = context.stress_xz->readVecNorm(tv, xv+(4/4), yv, zv, __LINE__);

 // Construct unaligned vector block from stress_xz at t, x+1, y, z.
 real_vec_t temp_vec108;
 // temp_vec108[0] = temp_vec20[1];  // for t, x+1, y, z;
 // temp_vec108[1] = temp_vec20[2];  // for t, x+2, y, z;
 // temp_vec108[2] = temp_vec20[3];  // for t, x+3, y, z;
 // temp_vec108[3] = temp_vec107[0];  // for t, x+4, y, z;
 // temp_vec108[4] = temp_vec20[5];  // for t, x+1, y+1, z;
 // temp_vec108[5] = temp_vec20[6];  // for t, x+2, y+1, z;
 // temp_vec108[6] = temp_vec20[7];  // for t, x+3, y+1, z;
 // temp_vec108[7] = temp_vec107[4];  // for t, x+4, y+1, z;
 // temp_vec108[8] = temp_vec20[9];  // for t, x+1, y+2, z;
 // temp_vec108[9] = temp_vec20[10];  // for t, x+2, y+2, z;
 // temp_vec108[10] = temp_vec20[11];  // for t, x+3, y+2, z;
 // temp_vec108[11] = temp_vec107[8];  // for t, x+4, y+2, z;
 // temp_vec108[12] = temp_vec20[13];  // for t, x+1, y+3, z;
 // temp_vec108[13] = temp_vec20[14];  // for t, x+2, y+3, z;
 // temp_vec108[14] = temp_vec20[15];  // for t, x+3, y+3, z;
 // temp_vec108[15] = temp_vec107[12];  // for t, x+4, y+3, z;
 real_vec_permute2(temp_vec108, ctrl_A1_A2_A3_B0_A5_A6_A7_B4_A9_A10_A11_B8_A13_A14_A15_B12, temp_vec20, temp_vec107);

 // Construct unaligned vector block from stress_xz at t, x+2, y, z.
 real_vec_t temp_vec109;
 // temp_vec109[0] = temp_vec20[2];  // for t, x+2, y, z;
 // temp_vec109[1] = temp_vec20[3];  // for t, x+3, y, z;
 // temp_vec109[2] = temp_vec107[0];  // for t, x+4, y, z;
 // temp_vec109[3] = temp_vec107[1];  // for t, x+5, y, z;
 // temp_vec109[4] = temp_vec20[6];  // for t, x+2, y+1, z;
 // temp_vec109[5] = temp_vec20[7];  // for t, x+3, y+1, z;
 // temp_vec109[6] = temp_vec107[4];  // for t, x+4, y+1, z;
 // temp_vec109[7] = temp_vec107[5];  // for t, x+5, y+1, z;
 // temp_vec109[8] = temp_vec20[10];  // for t, x+2, y+2, z;
 // temp_vec109[9] = temp_vec20[11];  // for t, x+3, y+2, z;
 // temp_vec109[10] = temp_vec107[8];  // for t, x+4, y+2, z;
 // temp_vec109[11] = temp_vec107[9];  // for t, x+5, y+2, z;
 // temp_vec109[12] = temp_vec20[14];  // for t, x+2, y+3, z;
 // temp_vec109[13] = temp_vec20[15];  // for t, x+3, y+3, z;
 // temp_vec109[14] = temp_vec107[12];  // for t, x+4, y+3, z;
 // temp_vec109[15] = temp_vec107[13];  // for t, x+5, y+3, z;
 real_vec_permute2(temp_vec109, ctrl_A2_A3_B0_B1_A6_A7_B4_B5_A10_A11_B8_B9_A14_A15_B12_B13, temp_vec20, temp_vec107);

 // Read aligned vector block from stress_xz at t, x-4, y, z.
 real_vec_t temp_vec110 = context.stress_xz->readVecNorm(tv, xv-(4/4), yv, zv, __LINE__);

 // Construct unaligned vector block from stress_xz at t, x-1, y, z.
 real_vec_t temp_vec111;
 // temp_vec111[0] = temp_vec110[3];  // for t, x-1, y, z;
 // temp_vec111[1] = temp_vec20[0];  // for t, x, y, z;
 // temp_vec111[2] = temp_vec20[1];  // for t, x+1, y, z;
 // temp_vec111[3] = temp_vec20[2];  // for t, x+2, y, z;
 // temp_vec111[4] = temp_vec110[7];  // for t, x-1, y+1, z;
 // temp_vec111[5] = temp_vec20[4];  // for t, x, y+1, z;
 // temp_vec111[6] = temp_vec20[5];  // for t, x+1, y+1, z;
 // temp_vec111[7] = temp_vec20[6];  // for t, x+2, y+1, z;
 // temp_vec111[8] = temp_vec110[11];  // for t, x-1, y+2, z;
 // temp_vec111[9] = temp_vec20[8];  // for t, x, y+2, z;
 // temp_vec111[10] = temp_vec20[9];  // for t, x+1, y+2, z;
 // temp_vec111[11] = temp_vec20[10];  // for t, x+2, y+2, z;
 // temp_vec111[12] = temp_vec110[15];  // for t, x-1, y+3, z;
 // temp_vec111[13] = temp_vec20[12];  // for t, x, y+3, z;
 // temp_vec111[14] = temp_vec20[13];  // for t, x+1, y+3, z;
 // temp_vec111[15] = temp_vec20[14];  // for t, x+2, y+3, z;
 real_vec_permute2(temp_vec111, ctrl_A3_B0_B1_B2_A7_B4_B5_B6_A11_B8_B9_B10_A15_B12_B13_B14, temp_vec110, temp_vec20);

 // Read aligned vector block from stress_yz at t, x, y-4, z.
 real_vec_t temp_vec112 = context.stress_yz->readVecNorm(tv, xv, yv-(4/4), zv, __LINE__);

 // Construct unaligned vector block from stress_yz at t, x, y-1, z.
 real_vec_t temp_vec113;
 // temp_vec113[0] = temp_vec112[12];  // for t, x, y-1, z;
 // temp_vec113[1] = temp_vec112[13];  // for t, x+1, y-1, z;
 // temp_vec113[2] = temp_vec112[14];  // for t, x+2, y-1, z;
 // temp_vec113[3] = temp_vec112[15];  // for t, x+3, y-1, z;
 // temp_vec113[4] = temp_vec75[0];  // for t, x, y, z;
 // temp_vec113[5] = temp_vec75[1];  // for t, x+1, y, z;
 // temp_vec113[6] = temp_vec75[2];  // for t, x+2, y, z;
 // temp_vec113[7] = temp_vec75[3];  // for t, x+3, y, z;
 // temp_vec113[8] = temp_vec75[4];  // for t, x, y+1, z;
 // temp_vec113[9] = temp_vec75[5];  // for t, x+1, y+1, z;
 // temp_vec113[10] = temp_vec75[6];  // for t, x+2, y+1, z;
 // temp_vec113[11] = temp_vec75[7];  // for t, x+3, y+1, z;
 // temp_vec113[12] = temp_vec75[8];  // for t, x, y+2, z;
 // temp_vec113[13] = temp_vec75[9];  // for t, x+1, y+2, z;
 // temp_vec113[14] = temp_vec75[10];  // for t, x+2, y+2, z;
 // temp_vec113[15] = temp_vec75[11];  // for t, x+3, y+2, z;
 // Get 12 element(s) from temp_vec75 and 4 from temp_vec112.
 real_vec_align<12>(temp_vec113, temp_vec75, temp_vec112);

 // Read aligned vector block from stress_yz at t, x, y+4, z.
 real_vec_t temp_vec114 = context.stress_yz->readVecNorm(tv, xv, yv+(4/4), zv, __LINE__);

 // Construct unaligned vector block from stress_yz at t, x, y+1, z.
 real_vec_t temp_vec115;
 // temp_vec115[0] = temp_vec75[4];  // for t, x, y+1, z;
 // temp_vec115[1] = temp_vec75[5];  // for t, x+1, y+1, z;
 // temp_vec115[2] = temp_vec75[6];  // for t, x+2, y+1, z;
 // temp_vec115[3] = temp_vec75[7];  // for t, x+3, y+1, z;
 // temp_vec115[4] = temp_vec75[8];  // for t, x, y+2, z;
 // temp_vec115[5] = temp_vec75[9];  // for t, x+1, y+2, z;
 // temp_vec115[6] = temp_vec75[10];  // for t, x+2, y+2, z;
 // temp_vec115[7] = temp_vec75[11];  // for t, x+3, y+2, z;
 // temp_vec115[8] = temp_vec75[12];  // for t, x, y+3, z;
 // temp_vec115[9] = temp_vec75[13];  // for t, x+1, y+3, z;
 // temp_vec115[10] = temp_vec75[14];  // for t, x+2, y+3, z;
 // temp_vec115[11] = temp_vec75[15];  // for t, x+3, y+3, z;
 // temp_vec115[12] = temp_vec114[0];  // for t, x, y+4, z;
 // temp_vec115[13] = temp_vec114[1];  // for t, x+1, y+4, z;
 // temp_vec115[14] = temp_vec114[2];  // for t, x+2, y+4, z;
 // temp_vec115[15] = temp_vec114[3];  // for t, x+3, y+4, z;
 // Get 4 element(s) from temp_vec114 and 12 from temp_vec75.
 real_vec_align<4>(temp_vec115, temp_vec114, temp_vec75);

 // Construct unaligned vector block from stress_yz at t, x, y-2, z.
 real_vec_t temp_vec116;
 // temp_vec116[0] = temp_vec112[8];  // for t, x, y-2, z;
 // temp_vec116[1] = temp_vec112[9];  // for t, x+1, y-2, z;
 // temp_vec116[2] = temp_vec112[10];  // for t, x+2, y-2, z;
 // temp_vec116[3] = temp_vec112[11];  // for t, x+3, y-2, z;
 // temp_vec116[4] = temp_vec112[12];  // for t, x, y-1, z;
 // temp_vec116[5] = temp_vec112[13];  // for t, x+1, y-1, z;
 // temp_vec116[6] = temp_vec112[14];  // for t, x+2, y-1, z;
 // temp_vec116[7] = temp_vec112[15];  // for t, x+3, y-1, z;
 // temp_vec116[8] = temp_vec75[0];  // for t, x, y, z;
 // temp_vec116[9] = temp_vec75[1];  // for t, x+1, y, z;
 // temp_vec116[10] = temp_vec75[2];  // for t, x+2, y, z;
 // temp_vec116[11] = temp_vec75[3];  // for t, x+3, y, z;
 // temp_vec116[12] = temp_vec75[4];  // for t, x, y+1, z;
 // temp_vec116[13] = temp_vec75[5];  // for t, x+1, y+1, z;
 // temp_vec116[14] = temp_vec75[6];  // for t, x+2, y+1, z;
 // temp_vec116[15] = temp_vec75[7];  // for t, x+3, y+1, z;
 // Get 8 element(s) from temp_vec75 and 8 from temp_vec112.
 real_vec_align<8>(temp_vec116, temp_vec75, temp_vec112);

 // Read aligned vector block from stress_zz at t, x, y, z+1.
 real_vec_t temp_vec117 = context.stress_zz->readVecNorm(tv, xv, yv, zv+(1/1), __LINE__);

 // Read aligned vector block from stress_zz at t, x, y, z.
 real_vec_t temp_vec118 = context.stress_zz->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from stress_zz at t, x, y, z+2.
 real_vec_t temp_vec119 = context.stress_zz->readVecNorm(tv, xv, yv, zv+(2/1), __LINE__);

 // Read aligned vector block from stress_zz at t, x, y, z-1.
 real_vec_t temp_vec120 = context.stress_zz->readVecNorm(tv, xv, yv, zv-(1/1), __LINE__);

 // temp_vec121 = rho(x, y, z) + rho(x+1, y, z).
 real_vec_t temp_vec121 = temp_vec27 + temp_vec79;

 // temp_vec122 = rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z).
 real_vec_t temp_vec122 = temp_vec121 + temp_vec28;

 // temp_vec123 = rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z).
 real_vec_t temp_vec123 = temp_vec122 + temp_vec106;

 // temp_vec124 = h() * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)).
 real_vec_t temp_vec124 = temp_vec26 * temp_vec123;

 // temp_vec125 = h() * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25.
 real_vec_t temp_vec125 = temp_vec124 * temp_vec34;

 // temp_vec126 = (delta_t / (h * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25)).
 real_vec_t temp_vec126 = temp_vec25 / temp_vec125;

 // temp_vec127 = (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z)).
 real_vec_t temp_vec127 = temp_vec108 - temp_vec48;

 // temp_vec128 = 1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z)).
 real_vec_t temp_vec128 = temp_vec37 * temp_vec127;

 // temp_vec129 = -0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z)).
 real_vec_t temp_vec129 = temp_vec39 * (temp_vec109 - temp_vec111);

 // temp_vec130 = (1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))).
 real_vec_t temp_vec130 = temp_vec128 + temp_vec129;

 // temp_vec131 = (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z)).
 real_vec_t temp_vec131 = temp_vec94 - temp_vec113;

 // temp_vec132 = 1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z)).
 real_vec_t temp_vec132 = temp_vec37 * temp_vec131;

 // temp_vec133 = (1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))).
 real_vec_t temp_vec133 = temp_vec130 + temp_vec132;

 // temp_vec134 = -0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z)).
 real_vec_t temp_vec134 = temp_vec39 * (temp_vec115 - temp_vec116);

 // temp_vec135 = (1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))).
 real_vec_t temp_vec135 = temp_vec133 + temp_vec134;

 // temp_vec136 = 1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z)).
 real_vec_t temp_vec136 = temp_vec37 * (temp_vec117 - temp_vec118);

 // temp_vec137 = (1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))).
 real_vec_t temp_vec137 = temp_vec135 + temp_vec136;

 // temp_vec138 = -0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)).
 real_vec_t temp_vec138 = temp_vec39 * (temp_vec119 - temp_vec120);

 // temp_vec139 = (1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1))).
 real_vec_t temp_vec139 = temp_vec137 + temp_vec138;

 // temp_vec140 = (delta_t / (h * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25)) * ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))).
 real_vec_t temp_vec140 = temp_vec126 * temp_vec139;

 // temp_vec141 = vel_z(t, x, y, z) + ((delta_t / (h * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25)) * ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1))))).
 real_vec_t temp_vec141 = temp_vec104 + temp_vec140;

 // temp_vec142 = (vel_z(t, x, y, z) + ((delta_t / (h * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25)) * ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))))) * sponge(x, y, z).
 real_vec_t temp_vec142 = temp_vec141 * temp_vec56;

 // temp_vec143 = ((vel_z(t, x, y, z) + ((delta_t / (h * (rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25)) * ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))))) * sponge(x, y, z)).
 real_vec_t temp_vec143 = temp_vec142;

 // Save result to vel_z(t+1, x, y, z):
 
 // Write aligned vector block to vel_z at t+1, x, y, z.
context.vel_z->writeVecNorm(temp_vec143, tv+(1/1), xv, yv, zv, __LINE__);
;
} // vector calculation.

// Prefetches cache line(s) for entire stencil to L1 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=4 * y=4 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L1_vector(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned rho at x, y-4, z-1.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv-(4/4), zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned rho at x, y-4, z.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned rho at x, y, z-1.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned rho at x, y, z.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned rho at x+4, y-4, z.
 p = (const char*)context.rho->getVecPtrNorm(xv+(4/4), yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned rho at x+4, y, z-1.
 p = (const char*)context.rho->getVecPtrNorm(xv+(4/4), yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned rho at x+4, y, z.
 p = (const char*)context.rho->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned sponge at x, y, z.
 p = (const char*)context.sponge->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xx at t, x-4, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xx at t, x, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xx at t, x+4, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x-4, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x, y-4, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x, y+4, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x+4, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x-4, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z-2.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv-(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z-1.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z+1.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x+4, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yy at t, x, y-4, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yy at t, x, y, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yy at t, x, y+4, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y-4, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z-2.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv-(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z-1.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z+1.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y+4, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z-1.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z+1.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z+2.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv+(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t, x, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t, x, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t, x, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);
}

// Prefetches cache line(s) for entire stencil to L2 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=4 * y=4 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L2_vector(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned rho at x, y-4, z-1.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv-(4/4), zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned rho at x, y-4, z.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned rho at x, y, z-1.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned rho at x, y, z.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned rho at x+4, y-4, z.
 p = (const char*)context.rho->getVecPtrNorm(xv+(4/4), yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned rho at x+4, y, z-1.
 p = (const char*)context.rho->getVecPtrNorm(xv+(4/4), yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned rho at x+4, y, z.
 p = (const char*)context.rho->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned sponge at x, y, z.
 p = (const char*)context.sponge->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xx at t, x-4, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xx at t, x, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xx at t, x+4, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x-4, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x, y-4, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x, y+4, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x+4, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x-4, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z-2.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv-(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z-1.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z+1.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x+4, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yy at t, x, y-4, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yy at t, x, y, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yy at t, x, y+4, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y-4, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z-2.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv-(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z-1.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z+1.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y+4, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z-1.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z+1.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z+2.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv+(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t, x, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t, x, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t, x, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);
}

// Prefetches cache line(s) for leading edge of stencil in '+t' direction to L1 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=4 * y=4 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L1_vector_t(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned stress_xx at t, x-4, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xx at t, x, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xx at t, x+4, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x-4, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x, y-4, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x, y+4, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x+4, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x-4, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z-2.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv-(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z-1.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z+1.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x+4, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yy at t, x, y-4, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yy at t, x, y, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yy at t, x, y+4, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y-4, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z-2.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv-(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z-1.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z+1.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y+4, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z-1.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z+1.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z+2.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv+(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t, x, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t, x, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t, x, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);
}

// Prefetches cache line(s) for leading edge of stencil in '+t' direction to L2 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=4 * y=4 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L2_vector_t(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned stress_xx at t, x-4, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xx at t, x, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xx at t, x+4, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x-4, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x, y-4, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x, y+4, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x+4, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x-4, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z-2.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv-(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z-1.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z+1.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x+4, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yy at t, x, y-4, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yy at t, x, y, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yy at t, x, y+4, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y-4, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z-2.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv-(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z-1.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z+1.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y+4, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z-1.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z+1.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z+2.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv+(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t, x, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t, x, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t, x, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);
}

// Prefetches cache line(s) for leading edge of stencil in '+x' direction to L1 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=4 * y=4 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L1_vector_x(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned rho at x, y-4, z-1.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv-(4/4), zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned rho at x+4, y-4, z.
 p = (const char*)context.rho->getVecPtrNorm(xv+(4/4), yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned rho at x+4, y, z-1.
 p = (const char*)context.rho->getVecPtrNorm(xv+(4/4), yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned rho at x+4, y, z.
 p = (const char*)context.rho->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned sponge at x, y, z.
 p = (const char*)context.sponge->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xx at t, x+4, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x, y-4, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x, y+4, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x+4, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z-2.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv-(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z-1.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z+1.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x+4, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yy at t, x, y-4, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yy at t, x, y, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yy at t, x, y+4, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y-4, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z-2.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv-(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z-1.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z+1.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y+4, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z-1.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z+1.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z+2.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv+(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t, x, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t, x, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t, x, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);
}

// Prefetches cache line(s) for leading edge of stencil in '+x' direction to L2 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=4 * y=4 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L2_vector_x(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned rho at x, y-4, z-1.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv-(4/4), zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned rho at x+4, y-4, z.
 p = (const char*)context.rho->getVecPtrNorm(xv+(4/4), yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned rho at x+4, y, z-1.
 p = (const char*)context.rho->getVecPtrNorm(xv+(4/4), yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned rho at x+4, y, z.
 p = (const char*)context.rho->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned sponge at x, y, z.
 p = (const char*)context.sponge->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xx at t, x+4, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x, y-4, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x, y+4, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x+4, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z-2.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv-(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z-1.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z+1.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x+4, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yy at t, x, y-4, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yy at t, x, y, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yy at t, x, y+4, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y-4, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z-2.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv-(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z-1.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z+1.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y+4, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z-1.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z+1.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z+2.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv+(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t, x, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t, x, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t, x, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);
}

// Prefetches cache line(s) for leading edge of stencil in '+y' direction to L1 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=4 * y=4 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L1_vector_y(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned rho at x, y, z-1.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned rho at x, y, z.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned rho at x+4, y, z-1.
 p = (const char*)context.rho->getVecPtrNorm(xv+(4/4), yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned rho at x+4, y, z.
 p = (const char*)context.rho->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned sponge at x, y, z.
 p = (const char*)context.sponge->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xx at t, x-4, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xx at t, x, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xx at t, x+4, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x-4, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x, y+4, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x+4, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x-4, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z-2.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv-(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z-1.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z+1.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x+4, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yy at t, x, y+4, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z-2.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv-(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z-1.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z+1.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y+4, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z-1.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z+1.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z+2.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv+(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t, x, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t, x, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t, x, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);
}

// Prefetches cache line(s) for leading edge of stencil in '+y' direction to L2 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=4 * y=4 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L2_vector_y(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned rho at x, y, z-1.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned rho at x, y, z.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned rho at x+4, y, z-1.
 p = (const char*)context.rho->getVecPtrNorm(xv+(4/4), yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned rho at x+4, y, z.
 p = (const char*)context.rho->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned sponge at x, y, z.
 p = (const char*)context.sponge->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xx at t, x-4, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xx at t, x, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xx at t, x+4, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x-4, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x, y+4, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x+4, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x-4, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z-2.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv-(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z-1.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z+1.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x+4, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yy at t, x, y+4, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z-2.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv-(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z-1.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z+1.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y+4, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z-1.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z+1.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z+2.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv+(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t, x, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t, x, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t, x, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);
}

// Prefetches cache line(s) for leading edge of stencil in '+z' direction to L1 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=4 * y=4 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L1_vector_z(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned rho at x, y-4, z.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned rho at x, y, z.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned rho at x+4, y-4, z.
 p = (const char*)context.rho->getVecPtrNorm(xv+(4/4), yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned rho at x+4, y, z.
 p = (const char*)context.rho->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned sponge at x, y, z.
 p = (const char*)context.sponge->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xx at t, x-4, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xx at t, x, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xx at t, x+4, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x-4, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x, y-4, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x, y+4, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x+4, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x-4, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z+1.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x+4, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yy at t, x, y-4, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yy at t, x, y, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yy at t, x, y+4, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y-4, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z+1.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y+4, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z+2.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv+(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t, x, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t, x, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t, x, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);
}

// Prefetches cache line(s) for leading edge of stencil in '+z' direction to L2 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=4 * y=4 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L2_vector_z(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned rho at x, y-4, z.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned rho at x, y, z.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned rho at x+4, y-4, z.
 p = (const char*)context.rho->getVecPtrNorm(xv+(4/4), yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned rho at x+4, y, z.
 p = (const char*)context.rho->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned sponge at x, y, z.
 p = (const char*)context.sponge->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xx at t, x-4, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xx at t, x, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xx at t, x+4, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x-4, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x, y-4, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x, y+4, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x+4, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x-4, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z+1.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x+4, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yy at t, x, y-4, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yy at t, x, y, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yy at t, x, y+4, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y-4, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z+1.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y+4, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z+2.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv+(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t, x, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t, x, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t, x, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);
}
};

////// Stencil equation 'stress' //////

struct Stencil_stress {
 std::string name = "stress";

 // 110 FP operation(s) per point:
 // stress_xx(t+1, x, y, z) = ((stress_xx(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))))) * sponge(x, y, z)).
 // stress_yy(t+1, x, y, z) = ((stress_yy(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))))) * sponge(x, y, z)).
 // stress_zz(t+1, x, y, z) = ((stress_zz(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))))) * sponge(x, y, z)).
 // stress_xy(t+1, x, y, z) = ((stress_xy(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x, y, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z)))))) * sponge(x, y, z)).
 // stress_xz(t+1, x, y, z) = ((stress_xz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x, y-1, z))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z)))))) * sponge(x, y, z)).
 // stress_yz(t+1, x, y, z) = ((stress_yz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z))) * delta_t) / h) * ((1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+2, z) - vel_z(t+1, x, y-1, z)))))) * sponge(x, y, z)).
 const int scalar_fp_ops = 110;

 // All grids updated by this equation.
 std::vector<RealVecGridBase*> eqGridPtrs;
 void init(StencilContext_awp& context) {
  eqGridPtrs.clear();
  eqGridPtrs.push_back(context.stress_xx);
  eqGridPtrs.push_back(context.stress_yy);
  eqGridPtrs.push_back(context.stress_zz);
  eqGridPtrs.push_back(context.stress_xy);
  eqGridPtrs.push_back(context.stress_xz);
  eqGridPtrs.push_back(context.stress_yz);
 }

 // Calculate one scalar result relative to indices t, x, y, z.
 void calc_scalar(StencilContext_awp& context, idx_t t, idx_t x, idx_t y, idx_t z) {

 // temp1 = delta_t().
 real_t temp1 = (*context.delta_t)();

 // temp2 = h().
 real_t temp2 = (*context.h)();

 // temp3 = (delta_t / h).
 real_t temp3 = temp1 / temp2;

 // temp4 = 2.
 real_t temp4 = 2.00000000000000000e+00;

 // temp5 = 8.
 real_t temp5 = 8.00000000000000000e+00;

 // temp6 = mu(x, y, z).
 real_t temp6 = context.mu->readElem(x, y, z, __LINE__);

 // temp7 = mu(x+1, y, z).
 real_t temp7 = context.mu->readElem(x+1, y, z, __LINE__);

 // temp8 = mu(x, y, z) + mu(x+1, y, z).
 real_t temp8 = temp6 + temp7;

 // temp9 = mu(x, y-1, z).
 real_t temp9 = context.mu->readElem(x, y-1, z, __LINE__);

 // temp10 = mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z).
 real_t temp10 = temp8 + temp9;

 // temp11 = mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z).
 real_t temp11 = temp10 + context.mu->readElem(x+1, y-1, z, __LINE__);

 // temp12 = mu(x, y, z-1).
 real_t temp12 = context.mu->readElem(x, y, z-1, __LINE__);

 // temp13 = mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1).
 real_t temp13 = temp11 + temp12;

 // temp14 = mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1).
 real_t temp14 = temp13 + context.mu->readElem(x+1, y, z-1, __LINE__);

 // temp15 = mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1).
 real_t temp15 = temp14 + context.mu->readElem(x, y-1, z-1, __LINE__);

 // temp16 = mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1).
 real_t temp16 = temp15 + context.mu->readElem(x+1, y-1, z-1, __LINE__);

 // temp17 = (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))).
 real_t temp17 = temp5 / temp16;

 // temp18 = 2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))).
 real_t temp18 = temp4 * temp17;

 // temp19 = 1.125.
 real_t temp19 = 1.12500000000000000e+00;

 // temp20 = vel_x(t+1, x, y, z).
 real_t temp20 = context.vel_x->readElem(t+1, x, y, z, __LINE__);

 // temp21 = (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z)).
 real_t temp21 = context.vel_x->readElem(t+1, x+1, y, z, __LINE__) - temp20;

 // temp22 = 1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z)).
 real_t temp22 = temp19 * temp21;

 // temp23 = -0.0416667.
 real_t temp23 = -4.16666666666666644e-02;

 // temp24 = -0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z)).
 real_t temp24 = temp23 * (context.vel_x->readElem(t+1, x+2, y, z, __LINE__) - context.vel_x->readElem(t+1, x-1, y, z, __LINE__));

 // temp25 = (1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))).
 real_t temp25 = temp22 + temp24;

 // temp26 = 2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z)))).
 real_t temp26 = temp18 * temp25;

 // temp27 = (8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))).
 real_t temp27 = temp5 / (context.lambda->readElem(x, y, z, __LINE__) + context.lambda->readElem(x+1, y, z, __LINE__) + context.lambda->readElem(x, y-1, z, __LINE__) + context.lambda->readElem(x+1, y-1, z, __LINE__) + context.lambda->readElem(x, y, z-1, __LINE__) + context.lambda->readElem(x+1, y, z-1, __LINE__) + context.lambda->readElem(x, y-1, z-1, __LINE__) + context.lambda->readElem(x+1, y-1, z-1, __LINE__));

 // temp28 = (1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))).
 real_t temp28 = temp22 + temp24;

 // temp29 = vel_y(t+1, x, y, z).
 real_t temp29 = context.vel_y->readElem(t+1, x, y, z, __LINE__);

 // temp30 = (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z)).
 real_t temp30 = temp29 - context.vel_y->readElem(t+1, x, y-1, z, __LINE__);

 // temp31 = 1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z)).
 real_t temp31 = temp19 * temp30;

 // temp32 = (1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))).
 real_t temp32 = temp28 + temp31;

 // temp33 = -0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)).
 real_t temp33 = temp23 * (context.vel_y->readElem(t+1, x, y+1, z, __LINE__) - context.vel_y->readElem(t+1, x, y-2, z, __LINE__));

 // temp34 = (1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))).
 real_t temp34 = temp32 + temp33;

 // temp35 = vel_z(t+1, x, y, z).
 real_t temp35 = context.vel_z->readElem(t+1, x, y, z, __LINE__);

 // temp36 = (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1)).
 real_t temp36 = temp35 - context.vel_z->readElem(t+1, x, y, z-1, __LINE__);

 // temp37 = 1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1)).
 real_t temp37 = temp19 * temp36;

 // temp38 = (1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))).
 real_t temp38 = temp34 + temp37;

 // temp39 = -0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)).
 real_t temp39 = temp23 * (context.vel_z->readElem(t+1, x, y, z+1, __LINE__) - context.vel_z->readElem(t+1, x, y, z-2, __LINE__));

 // temp40 = (1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))).
 real_t temp40 = temp38 + temp39;

 // temp41 = (8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))).
 real_t temp41 = temp27 * temp40;

 // temp42 = (2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 real_t temp42 = temp26 + temp41;

 // temp43 = (delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))).
 real_t temp43 = temp3 * temp42;

 // temp44 = stress_xx(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))).
 real_t temp44 = context.stress_xx->readElem(t, x, y, z, __LINE__) + temp43;

 // temp45 = sponge(x, y, z).
 real_t temp45 = context.sponge->readElem(x, y, z, __LINE__);

 // temp46 = (stress_xx(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))))) * sponge(x, y, z).
 real_t temp46 = temp44 * temp45;

 // temp47 = ((stress_xx(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))))) * sponge(x, y, z)).
 real_t temp47 = temp46;

 // Save result to stress_xx(t+1, x, y, z):
 context.stress_xx->writeElem(temp47, t+1, x, y, z, __LINE__);

 // temp48 = 2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))).
 real_t temp48 = temp4 * temp17;

 // temp49 = (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))).
 real_t temp49 = temp31 + temp33;

 // temp50 = 2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))).
 real_t temp50 = temp48 * temp49;

 // temp51 = (2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 real_t temp51 = temp50 + temp41;

 // temp52 = (delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))).
 real_t temp52 = temp3 * temp51;

 // temp53 = stress_yy(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))).
 real_t temp53 = context.stress_yy->readElem(t, x, y, z, __LINE__) + temp52;

 // temp54 = (stress_yy(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))))) * sponge(x, y, z).
 real_t temp54 = temp53 * temp45;

 // temp55 = ((stress_yy(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))))) * sponge(x, y, z)).
 real_t temp55 = temp54;

 // Save result to stress_yy(t+1, x, y, z):
 context.stress_yy->writeElem(temp55, t+1, x, y, z, __LINE__);

 // temp56 = 2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))).
 real_t temp56 = temp4 * temp17;

 // temp57 = (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))).
 real_t temp57 = temp37 + temp39;

 // temp58 = 2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))).
 real_t temp58 = temp56 * temp57;

 // temp59 = (2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 real_t temp59 = temp58 + temp41;

 // temp60 = (delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))).
 real_t temp60 = temp3 * temp59;

 // temp61 = stress_zz(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))).
 real_t temp61 = context.stress_zz->readElem(t, x, y, z, __LINE__) + temp60;

 // temp62 = (stress_zz(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))))) * sponge(x, y, z).
 real_t temp62 = temp61 * temp45;

 // temp63 = ((stress_zz(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))))) * sponge(x, y, z)).
 real_t temp63 = temp62;

 // Save result to stress_zz(t+1, x, y, z):
 context.stress_zz->writeElem(temp63, t+1, x, y, z, __LINE__);

 // temp64 = mu(x, y, z) + mu(x, y, z-1).
 real_t temp64 = temp6 + temp12;

 // temp65 = (2 / (mu(x, y, z) + mu(x, y, z-1))).
 real_t temp65 = temp4 / temp64;

 // temp66 = (2 / (mu(x, y, z) + mu(x, y, z-1))) * delta_t().
 real_t temp66 = temp65 * temp1;

 // temp67 = (((2 / (mu(x, y, z) + mu(x, y, z-1))) * delta_t) / h).
 real_t temp67 = temp66 / temp2;

 // temp68 = (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z)).
 real_t temp68 = context.vel_x->readElem(t+1, x, y+1, z, __LINE__) - temp20;

 // temp69 = 1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z)).
 real_t temp69 = temp19 * temp68;

 // temp70 = -0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z)).
 real_t temp70 = temp23 * (context.vel_x->readElem(t+1, x, y+2, z, __LINE__) - context.vel_x->readElem(t+1, x, y-1, z, __LINE__));

 // temp71 = (1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))).
 real_t temp71 = temp69 + temp70;

 // temp72 = (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z)).
 real_t temp72 = temp29 - context.vel_y->readElem(t+1, x-1, y, z, __LINE__);

 // temp73 = 1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z)).
 real_t temp73 = temp19 * temp72;

 // temp74 = (1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))).
 real_t temp74 = temp71 + temp73;

 // temp75 = -0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z)).
 real_t temp75 = temp23 * (context.vel_y->readElem(t+1, x+1, y, z, __LINE__) - context.vel_y->readElem(t+1, x-2, y, z, __LINE__));

 // temp76 = (1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z))).
 real_t temp76 = temp74 + temp75;

 // temp77 = (((2 / (mu(x, y, z) + mu(x, y, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z)))).
 real_t temp77 = temp67 * temp76;

 // temp78 = stress_xy(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x, y, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z))))).
 real_t temp78 = context.stress_xy->readElem(t, x, y, z, __LINE__) + temp77;

 // temp79 = (stress_xy(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x, y, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z)))))) * sponge(x, y, z).
 real_t temp79 = temp78 * temp45;

 // temp80 = ((stress_xy(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x, y, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z)))))) * sponge(x, y, z)).
 real_t temp80 = temp79;

 // Save result to stress_xy(t+1, x, y, z):
 context.stress_xy->writeElem(temp80, t+1, x, y, z, __LINE__);

 // temp81 = mu(x, y, z) + mu(x, y-1, z).
 real_t temp81 = temp6 + temp9;

 // temp82 = (2 / (mu(x, y, z) + mu(x, y-1, z))).
 real_t temp82 = temp4 / temp81;

 // temp83 = (2 / (mu(x, y, z) + mu(x, y-1, z))) * delta_t().
 real_t temp83 = temp82 * temp1;

 // temp84 = (((2 / (mu(x, y, z) + mu(x, y-1, z))) * delta_t) / h).
 real_t temp84 = temp83 / temp2;

 // temp85 = (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z)).
 real_t temp85 = context.vel_x->readElem(t+1, x, y, z+1, __LINE__) - temp20;

 // temp86 = 1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z)).
 real_t temp86 = temp19 * temp85;

 // temp87 = -0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1)).
 real_t temp87 = temp23 * (context.vel_x->readElem(t+1, x, y, z+2, __LINE__) - context.vel_x->readElem(t+1, x, y, z-1, __LINE__));

 // temp88 = (1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))).
 real_t temp88 = temp86 + temp87;

 // temp89 = (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z)).
 real_t temp89 = temp35 - context.vel_z->readElem(t+1, x-1, y, z, __LINE__);

 // temp90 = 1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z)).
 real_t temp90 = temp19 * temp89;

 // temp91 = (1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))).
 real_t temp91 = temp88 + temp90;

 // temp92 = -0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z)).
 real_t temp92 = temp23 * (context.vel_z->readElem(t+1, x+1, y, z, __LINE__) - context.vel_z->readElem(t+1, x-2, y, z, __LINE__));

 // temp93 = (1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z))).
 real_t temp93 = temp91 + temp92;

 // temp94 = (((2 / (mu(x, y, z) + mu(x, y-1, z))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z)))).
 real_t temp94 = temp84 * temp93;

 // temp95 = stress_xz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x, y-1, z))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z))))).
 real_t temp95 = context.stress_xz->readElem(t, x, y, z, __LINE__) + temp94;

 // temp96 = (stress_xz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x, y-1, z))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z)))))) * sponge(x, y, z).
 real_t temp96 = temp95 * temp45;

 // temp97 = ((stress_xz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x, y-1, z))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z)))))) * sponge(x, y, z)).
 real_t temp97 = temp96;

 // Save result to stress_xz(t+1, x, y, z):
 context.stress_xz->writeElem(temp97, t+1, x, y, z, __LINE__);

 // temp98 = mu(x, y, z) + mu(x+1, y, z).
 real_t temp98 = temp6 + temp7;

 // temp99 = (2 / (mu(x, y, z) + mu(x+1, y, z))).
 real_t temp99 = temp4 / temp98;

 // temp100 = (2 / (mu(x, y, z) + mu(x+1, y, z))) * delta_t().
 real_t temp100 = temp99 * temp1;

 // temp101 = (((2 / (mu(x, y, z) + mu(x+1, y, z))) * delta_t) / h).
 real_t temp101 = temp100 / temp2;

 // temp102 = (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z)).
 real_t temp102 = context.vel_y->readElem(t+1, x, y, z+1, __LINE__) - temp29;

 // temp103 = 1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z)).
 real_t temp103 = temp19 * temp102;

 // temp104 = -0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1)).
 real_t temp104 = temp23 * (context.vel_y->readElem(t+1, x, y, z+2, __LINE__) - context.vel_y->readElem(t+1, x, y, z-1, __LINE__));

 // temp105 = (1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))).
 real_t temp105 = temp103 + temp104;

 // temp106 = (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z)).
 real_t temp106 = context.vel_z->readElem(t+1, x, y+1, z, __LINE__) - temp35;

 // temp107 = 1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z)).
 real_t temp107 = temp19 * temp106;

 // temp108 = (1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))).
 real_t temp108 = temp105 + temp107;

 // temp109 = -0.0416667 * (vel_z(t+1, x, y+2, z) - vel_z(t+1, x, y-1, z)).
 real_t temp109 = temp23 * (context.vel_z->readElem(t+1, x, y+2, z, __LINE__) - context.vel_z->readElem(t+1, x, y-1, z, __LINE__));

 // temp110 = (1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+2, z) - vel_z(t+1, x, y-1, z))).
 real_t temp110 = temp108 + temp109;

 // temp111 = (((2 / (mu(x, y, z) + mu(x+1, y, z))) * delta_t) / h) * ((1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+2, z) - vel_z(t+1, x, y-1, z)))).
 real_t temp111 = temp101 * temp110;

 // temp112 = stress_yz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z))) * delta_t) / h) * ((1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+2, z) - vel_z(t+1, x, y-1, z))))).
 real_t temp112 = context.stress_yz->readElem(t, x, y, z, __LINE__) + temp111;

 // temp113 = (stress_yz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z))) * delta_t) / h) * ((1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+2, z) - vel_z(t+1, x, y-1, z)))))) * sponge(x, y, z).
 real_t temp113 = temp112 * temp45;

 // temp114 = ((stress_yz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z))) * delta_t) / h) * ((1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+2, z) - vel_z(t+1, x, y-1, z)))))) * sponge(x, y, z)).
 real_t temp114 = temp113;

 // Save result to stress_yz(t+1, x, y, z):
 context.stress_yz->writeElem(temp114, t+1, x, y, z, __LINE__);
} // scalar calculation.

 // Calculate 16 result(s) relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=4 * y=4 * z=1' vector(s).
 // Indices must be normalized, i.e., already divided by VLEN_*.
 // SIMD calculations use 53 vector block(s) created from 47 aligned vector-block(s).
 // There are 1760 FP operation(s) per cluster.
 void calc_vector(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {

 // Un-normalized indices.
 idx_t t = tv;
 idx_t x = xv * 4;
 idx_t y = yv * 4;
 idx_t z = zv * 1;

 // Read aligned vector block from stress_xx at t, x, y, z.
 real_vec_t temp_vec1 = context.stress_xx->readVecNorm(tv, xv, yv, zv, __LINE__);

 // temp_vec2 = delta_t().
 real_vec_t temp_vec2 = (*context.delta_t)();

 // temp_vec3 = h().
 real_vec_t temp_vec3 = (*context.h)();

 // temp_vec4 = (delta_t / h).
 real_vec_t temp_vec4 = temp_vec2 / temp_vec3;

 // Read aligned vector block from mu at x, y, z.
 real_vec_t temp_vec5 = context.mu->readVecNorm(xv, yv, zv, __LINE__);

 // Read aligned vector block from mu at x+4, y, z.
 real_vec_t temp_vec6 = context.mu->readVecNorm(xv+(4/4), yv, zv, __LINE__);

 // Construct unaligned vector block from mu at x+1, y, z.
 real_vec_t temp_vec7;
 // temp_vec7[0] = temp_vec5[1];  // for x+1, y, z;
 // temp_vec7[1] = temp_vec5[2];  // for x+2, y, z;
 // temp_vec7[2] = temp_vec5[3];  // for x+3, y, z;
 // temp_vec7[3] = temp_vec6[0];  // for x+4, y, z;
 // temp_vec7[4] = temp_vec5[5];  // for x+1, y+1, z;
 // temp_vec7[5] = temp_vec5[6];  // for x+2, y+1, z;
 // temp_vec7[6] = temp_vec5[7];  // for x+3, y+1, z;
 // temp_vec7[7] = temp_vec6[4];  // for x+4, y+1, z;
 // temp_vec7[8] = temp_vec5[9];  // for x+1, y+2, z;
 // temp_vec7[9] = temp_vec5[10];  // for x+2, y+2, z;
 // temp_vec7[10] = temp_vec5[11];  // for x+3, y+2, z;
 // temp_vec7[11] = temp_vec6[8];  // for x+4, y+2, z;
 // temp_vec7[12] = temp_vec5[13];  // for x+1, y+3, z;
 // temp_vec7[13] = temp_vec5[14];  // for x+2, y+3, z;
 // temp_vec7[14] = temp_vec5[15];  // for x+3, y+3, z;
 // temp_vec7[15] = temp_vec6[12];  // for x+4, y+3, z;
 const real_vec_t_data ctrl_data_A1_A2_A3_B0_A5_A6_A7_B4_A9_A10_A11_B8_A13_A14_A15_B12 = { .ci = { 1, 2, 3, ctrl_sel_bit |0, 5, 6, 7, ctrl_sel_bit |4, 9, 10, 11, ctrl_sel_bit |8, 13, 14, 15, ctrl_sel_bit |12 } };
 const real_vec_t ctrl_A1_A2_A3_B0_A5_A6_A7_B4_A9_A10_A11_B8_A13_A14_A15_B12(ctrl_data_A1_A2_A3_B0_A5_A6_A7_B4_A9_A10_A11_B8_A13_A14_A15_B12);
 real_vec_permute2(temp_vec7, ctrl_A1_A2_A3_B0_A5_A6_A7_B4_A9_A10_A11_B8_A13_A14_A15_B12, temp_vec5, temp_vec6);

 // Read aligned vector block from mu at x, y-4, z.
 real_vec_t temp_vec8 = context.mu->readVecNorm(xv, yv-(4/4), zv, __LINE__);

 // Construct unaligned vector block from mu at x, y-1, z.
 real_vec_t temp_vec9;
 // temp_vec9[0] = temp_vec8[12];  // for x, y-1, z;
 // temp_vec9[1] = temp_vec8[13];  // for x+1, y-1, z;
 // temp_vec9[2] = temp_vec8[14];  // for x+2, y-1, z;
 // temp_vec9[3] = temp_vec8[15];  // for x+3, y-1, z;
 // temp_vec9[4] = temp_vec5[0];  // for x, y, z;
 // temp_vec9[5] = temp_vec5[1];  // for x+1, y, z;
 // temp_vec9[6] = temp_vec5[2];  // for x+2, y, z;
 // temp_vec9[7] = temp_vec5[3];  // for x+3, y, z;
 // temp_vec9[8] = temp_vec5[4];  // for x, y+1, z;
 // temp_vec9[9] = temp_vec5[5];  // for x+1, y+1, z;
 // temp_vec9[10] = temp_vec5[6];  // for x+2, y+1, z;
 // temp_vec9[11] = temp_vec5[7];  // for x+3, y+1, z;
 // temp_vec9[12] = temp_vec5[8];  // for x, y+2, z;
 // temp_vec9[13] = temp_vec5[9];  // for x+1, y+2, z;
 // temp_vec9[14] = temp_vec5[10];  // for x+2, y+2, z;
 // temp_vec9[15] = temp_vec5[11];  // for x+3, y+2, z;
 // Get 12 element(s) from temp_vec5 and 4 from temp_vec8.
 real_vec_align<12>(temp_vec9, temp_vec5, temp_vec8);

 // Read aligned vector block from mu at x+4, y-4, z.
 real_vec_t temp_vec10 = context.mu->readVecNorm(xv+(4/4), yv-(4/4), zv, __LINE__);

 // Construct unaligned vector block from mu at x+1, y-1, z.
 real_vec_t temp_vec11;
 // temp_vec11[0] = temp_vec8[13];  // for x+1, y-1, z;
 // temp_vec11[1] = temp_vec8[14];  // for x+2, y-1, z;
 // temp_vec11[2] = temp_vec8[15];  // for x+3, y-1, z;
 // temp_vec11[3] = temp_vec10[12];  // for x+4, y-1, z;
 // temp_vec11[4] = temp_vec5[1];  // for x+1, y, z;
 // temp_vec11[5] = temp_vec5[2];  // for x+2, y, z;
 // temp_vec11[6] = temp_vec5[3];  // for x+3, y, z;
 // temp_vec11[7] = temp_vec6[0];  // for x+4, y, z;
 // temp_vec11[8] = temp_vec5[5];  // for x+1, y+1, z;
 // temp_vec11[9] = temp_vec5[6];  // for x+2, y+1, z;
 // temp_vec11[10] = temp_vec5[7];  // for x+3, y+1, z;
 // temp_vec11[11] = temp_vec6[4];  // for x+4, y+1, z;
 // temp_vec11[12] = temp_vec5[9];  // for x+1, y+2, z;
 // temp_vec11[13] = temp_vec5[10];  // for x+2, y+2, z;
 // temp_vec11[14] = temp_vec5[11];  // for x+3, y+2, z;
 // temp_vec11[15] = temp_vec6[8];  // for x+4, y+2, z;
 // Get 9 element(s) from temp_vec5 and 3 from temp_vec8.
 real_vec_align<13>(temp_vec11, temp_vec5, temp_vec8);
 // Get 3 element(s) from temp_vec6 and 1 from temp_vec10.
 real_vec_align_masked<9>(temp_vec11, temp_vec6, temp_vec10, 0x8888);

 // Read aligned vector block from mu at x, y, z-1.
 real_vec_t temp_vec12 = context.mu->readVecNorm(xv, yv, zv-(1/1), __LINE__);

 // Read aligned vector block from mu at x+4, y, z-1.
 real_vec_t temp_vec13 = context.mu->readVecNorm(xv+(4/4), yv, zv-(1/1), __LINE__);

 // Construct unaligned vector block from mu at x+1, y, z-1.
 real_vec_t temp_vec14;
 // temp_vec14[0] = temp_vec12[1];  // for x+1, y, z-1;
 // temp_vec14[1] = temp_vec12[2];  // for x+2, y, z-1;
 // temp_vec14[2] = temp_vec12[3];  // for x+3, y, z-1;
 // temp_vec14[3] = temp_vec13[0];  // for x+4, y, z-1;
 // temp_vec14[4] = temp_vec12[5];  // for x+1, y+1, z-1;
 // temp_vec14[5] = temp_vec12[6];  // for x+2, y+1, z-1;
 // temp_vec14[6] = temp_vec12[7];  // for x+3, y+1, z-1;
 // temp_vec14[7] = temp_vec13[4];  // for x+4, y+1, z-1;
 // temp_vec14[8] = temp_vec12[9];  // for x+1, y+2, z-1;
 // temp_vec14[9] = temp_vec12[10];  // for x+2, y+2, z-1;
 // temp_vec14[10] = temp_vec12[11];  // for x+3, y+2, z-1;
 // temp_vec14[11] = temp_vec13[8];  // for x+4, y+2, z-1;
 // temp_vec14[12] = temp_vec12[13];  // for x+1, y+3, z-1;
 // temp_vec14[13] = temp_vec12[14];  // for x+2, y+3, z-1;
 // temp_vec14[14] = temp_vec12[15];  // for x+3, y+3, z-1;
 // temp_vec14[15] = temp_vec13[12];  // for x+4, y+3, z-1;
 real_vec_permute2(temp_vec14, ctrl_A1_A2_A3_B0_A5_A6_A7_B4_A9_A10_A11_B8_A13_A14_A15_B12, temp_vec12, temp_vec13);

 // Read aligned vector block from mu at x, y-4, z-1.
 real_vec_t temp_vec15 = context.mu->readVecNorm(xv, yv-(4/4), zv-(1/1), __LINE__);

 // Construct unaligned vector block from mu at x, y-1, z-1.
 real_vec_t temp_vec16;
 // temp_vec16[0] = temp_vec15[12];  // for x, y-1, z-1;
 // temp_vec16[1] = temp_vec15[13];  // for x+1, y-1, z-1;
 // temp_vec16[2] = temp_vec15[14];  // for x+2, y-1, z-1;
 // temp_vec16[3] = temp_vec15[15];  // for x+3, y-1, z-1;
 // temp_vec16[4] = temp_vec12[0];  // for x, y, z-1;
 // temp_vec16[5] = temp_vec12[1];  // for x+1, y, z-1;
 // temp_vec16[6] = temp_vec12[2];  // for x+2, y, z-1;
 // temp_vec16[7] = temp_vec12[3];  // for x+3, y, z-1;
 // temp_vec16[8] = temp_vec12[4];  // for x, y+1, z-1;
 // temp_vec16[9] = temp_vec12[5];  // for x+1, y+1, z-1;
 // temp_vec16[10] = temp_vec12[6];  // for x+2, y+1, z-1;
 // temp_vec16[11] = temp_vec12[7];  // for x+3, y+1, z-1;
 // temp_vec16[12] = temp_vec12[8];  // for x, y+2, z-1;
 // temp_vec16[13] = temp_vec12[9];  // for x+1, y+2, z-1;
 // temp_vec16[14] = temp_vec12[10];  // for x+2, y+2, z-1;
 // temp_vec16[15] = temp_vec12[11];  // for x+3, y+2, z-1;
 // Get 12 element(s) from temp_vec12 and 4 from temp_vec15.
 real_vec_align<12>(temp_vec16, temp_vec12, temp_vec15);

 // Read aligned vector block from mu at x+4, y-4, z-1.
 real_vec_t temp_vec17 = context.mu->readVecNorm(xv+(4/4), yv-(4/4), zv-(1/1), __LINE__);

 // Construct unaligned vector block from mu at x+1, y-1, z-1.
 real_vec_t temp_vec18;
 // temp_vec18[0] = temp_vec15[13];  // for x+1, y-1, z-1;
 // temp_vec18[1] = temp_vec15[14];  // for x+2, y-1, z-1;
 // temp_vec18[2] = temp_vec15[15];  // for x+3, y-1, z-1;
 // temp_vec18[3] = temp_vec17[12];  // for x+4, y-1, z-1;
 // temp_vec18[4] = temp_vec12[1];  // for x+1, y, z-1;
 // temp_vec18[5] = temp_vec12[2];  // for x+2, y, z-1;
 // temp_vec18[6] = temp_vec12[3];  // for x+3, y, z-1;
 // temp_vec18[7] = temp_vec13[0];  // for x+4, y, z-1;
 // temp_vec18[8] = temp_vec12[5];  // for x+1, y+1, z-1;
 // temp_vec18[9] = temp_vec12[6];  // for x+2, y+1, z-1;
 // temp_vec18[10] = temp_vec12[7];  // for x+3, y+1, z-1;
 // temp_vec18[11] = temp_vec13[4];  // for x+4, y+1, z-1;
 // temp_vec18[12] = temp_vec12[9];  // for x+1, y+2, z-1;
 // temp_vec18[13] = temp_vec12[10];  // for x+2, y+2, z-1;
 // temp_vec18[14] = temp_vec12[11];  // for x+3, y+2, z-1;
 // temp_vec18[15] = temp_vec13[8];  // for x+4, y+2, z-1;
 // Get 9 element(s) from temp_vec12 and 3 from temp_vec15.
 real_vec_align<13>(temp_vec18, temp_vec12, temp_vec15);
 // Get 3 element(s) from temp_vec13 and 1 from temp_vec17.
 real_vec_align_masked<9>(temp_vec18, temp_vec13, temp_vec17, 0x8888);

 // Read aligned vector block from vel_x at t+1, x, y, z.
 real_vec_t temp_vec19 = context.vel_x->readVecNorm(tv+(1/1), xv, yv, zv, __LINE__);

 // Read aligned vector block from vel_x at t+1, x+4, y, z.
 real_vec_t temp_vec20 = context.vel_x->readVecNorm(tv+(1/1), xv+(4/4), yv, zv, __LINE__);

 // Construct unaligned vector block from vel_x at t+1, x+1, y, z.
 real_vec_t temp_vec21;
 // temp_vec21[0] = temp_vec19[1];  // for t+1, x+1, y, z;
 // temp_vec21[1] = temp_vec19[2];  // for t+1, x+2, y, z;
 // temp_vec21[2] = temp_vec19[3];  // for t+1, x+3, y, z;
 // temp_vec21[3] = temp_vec20[0];  // for t+1, x+4, y, z;
 // temp_vec21[4] = temp_vec19[5];  // for t+1, x+1, y+1, z;
 // temp_vec21[5] = temp_vec19[6];  // for t+1, x+2, y+1, z;
 // temp_vec21[6] = temp_vec19[7];  // for t+1, x+3, y+1, z;
 // temp_vec21[7] = temp_vec20[4];  // for t+1, x+4, y+1, z;
 // temp_vec21[8] = temp_vec19[9];  // for t+1, x+1, y+2, z;
 // temp_vec21[9] = temp_vec19[10];  // for t+1, x+2, y+2, z;
 // temp_vec21[10] = temp_vec19[11];  // for t+1, x+3, y+2, z;
 // temp_vec21[11] = temp_vec20[8];  // for t+1, x+4, y+2, z;
 // temp_vec21[12] = temp_vec19[13];  // for t+1, x+1, y+3, z;
 // temp_vec21[13] = temp_vec19[14];  // for t+1, x+2, y+3, z;
 // temp_vec21[14] = temp_vec19[15];  // for t+1, x+3, y+3, z;
 // temp_vec21[15] = temp_vec20[12];  // for t+1, x+4, y+3, z;
 real_vec_permute2(temp_vec21, ctrl_A1_A2_A3_B0_A5_A6_A7_B4_A9_A10_A11_B8_A13_A14_A15_B12, temp_vec19, temp_vec20);

 // Construct unaligned vector block from vel_x at t+1, x+2, y, z.
 real_vec_t temp_vec22;
 // temp_vec22[0] = temp_vec19[2];  // for t+1, x+2, y, z;
 // temp_vec22[1] = temp_vec19[3];  // for t+1, x+3, y, z;
 // temp_vec22[2] = temp_vec20[0];  // for t+1, x+4, y, z;
 // temp_vec22[3] = temp_vec20[1];  // for t+1, x+5, y, z;
 // temp_vec22[4] = temp_vec19[6];  // for t+1, x+2, y+1, z;
 // temp_vec22[5] = temp_vec19[7];  // for t+1, x+3, y+1, z;
 // temp_vec22[6] = temp_vec20[4];  // for t+1, x+4, y+1, z;
 // temp_vec22[7] = temp_vec20[5];  // for t+1, x+5, y+1, z;
 // temp_vec22[8] = temp_vec19[10];  // for t+1, x+2, y+2, z;
 // temp_vec22[9] = temp_vec19[11];  // for t+1, x+3, y+2, z;
 // temp_vec22[10] = temp_vec20[8];  // for t+1, x+4, y+2, z;
 // temp_vec22[11] = temp_vec20[9];  // for t+1, x+5, y+2, z;
 // temp_vec22[12] = temp_vec19[14];  // for t+1, x+2, y+3, z;
 // temp_vec22[13] = temp_vec19[15];  // for t+1, x+3, y+3, z;
 // temp_vec22[14] = temp_vec20[12];  // for t+1, x+4, y+3, z;
 // temp_vec22[15] = temp_vec20[13];  // for t+1, x+5, y+3, z;
 const real_vec_t_data ctrl_data_A2_A3_B0_B1_A6_A7_B4_B5_A10_A11_B8_B9_A14_A15_B12_B13 = { .ci = { 2, 3, ctrl_sel_bit |0, ctrl_sel_bit |1, 6, 7, ctrl_sel_bit |4, ctrl_sel_bit |5, 10, 11, ctrl_sel_bit |8, ctrl_sel_bit |9, 14, 15, ctrl_sel_bit |12, ctrl_sel_bit |13 } };
 const real_vec_t ctrl_A2_A3_B0_B1_A6_A7_B4_B5_A10_A11_B8_B9_A14_A15_B12_B13(ctrl_data_A2_A3_B0_B1_A6_A7_B4_B5_A10_A11_B8_B9_A14_A15_B12_B13);
 real_vec_permute2(temp_vec22, ctrl_A2_A3_B0_B1_A6_A7_B4_B5_A10_A11_B8_B9_A14_A15_B12_B13, temp_vec19, temp_vec20);

 // Read aligned vector block from vel_x at t+1, x-4, y, z.
 real_vec_t temp_vec23 = context.vel_x->readVecNorm(tv+(1/1), xv-(4/4), yv, zv, __LINE__);

 // Construct unaligned vector block from vel_x at t+1, x-1, y, z.
 real_vec_t temp_vec24;
 // temp_vec24[0] = temp_vec23[3];  // for t+1, x-1, y, z;
 // temp_vec24[1] = temp_vec19[0];  // for t+1, x, y, z;
 // temp_vec24[2] = temp_vec19[1];  // for t+1, x+1, y, z;
 // temp_vec24[3] = temp_vec19[2];  // for t+1, x+2, y, z;
 // temp_vec24[4] = temp_vec23[7];  // for t+1, x-1, y+1, z;
 // temp_vec24[5] = temp_vec19[4];  // for t+1, x, y+1, z;
 // temp_vec24[6] = temp_vec19[5];  // for t+1, x+1, y+1, z;
 // temp_vec24[7] = temp_vec19[6];  // for t+1, x+2, y+1, z;
 // temp_vec24[8] = temp_vec23[11];  // for t+1, x-1, y+2, z;
 // temp_vec24[9] = temp_vec19[8];  // for t+1, x, y+2, z;
 // temp_vec24[10] = temp_vec19[9];  // for t+1, x+1, y+2, z;
 // temp_vec24[11] = temp_vec19[10];  // for t+1, x+2, y+2, z;
 // temp_vec24[12] = temp_vec23[15];  // for t+1, x-1, y+3, z;
 // temp_vec24[13] = temp_vec19[12];  // for t+1, x, y+3, z;
 // temp_vec24[14] = temp_vec19[13];  // for t+1, x+1, y+3, z;
 // temp_vec24[15] = temp_vec19[14];  // for t+1, x+2, y+3, z;
 const real_vec_t_data ctrl_data_A3_B0_B1_B2_A7_B4_B5_B6_A11_B8_B9_B10_A15_B12_B13_B14 = { .ci = { 3, ctrl_sel_bit |0, ctrl_sel_bit |1, ctrl_sel_bit |2, 7, ctrl_sel_bit |4, ctrl_sel_bit |5, ctrl_sel_bit |6, 11, ctrl_sel_bit |8, ctrl_sel_bit |9, ctrl_sel_bit |10, 15, ctrl_sel_bit |12, ctrl_sel_bit |13, ctrl_sel_bit |14 } };
 const real_vec_t ctrl_A3_B0_B1_B2_A7_B4_B5_B6_A11_B8_B9_B10_A15_B12_B13_B14(ctrl_data_A3_B0_B1_B2_A7_B4_B5_B6_A11_B8_B9_B10_A15_B12_B13_B14);
 real_vec_permute2(temp_vec24, ctrl_A3_B0_B1_B2_A7_B4_B5_B6_A11_B8_B9_B10_A15_B12_B13_B14, temp_vec23, temp_vec19);

 // temp_vec25 = 2.
 real_vec_t temp_vec25 = 2.00000000000000000e+00;

 // temp_vec26 = 8.
 real_vec_t temp_vec26 = 8.00000000000000000e+00;

 // temp_vec27 = mu(x, y, z).
 real_vec_t temp_vec27 = temp_vec5;

 // temp_vec28 = mu(x+1, y, z).
 real_vec_t temp_vec28 = temp_vec7;

 // temp_vec29 = mu(x, y, z) + mu(x+1, y, z).
 real_vec_t temp_vec29 = temp_vec27 + temp_vec28;

 // temp_vec30 = mu(x, y-1, z).
 real_vec_t temp_vec30 = temp_vec9;

 // temp_vec31 = mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z).
 real_vec_t temp_vec31 = temp_vec29 + temp_vec30;

 // temp_vec32 = mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z).
 real_vec_t temp_vec32 = temp_vec31 + temp_vec11;

 // temp_vec33 = mu(x, y, z-1).
 real_vec_t temp_vec33 = temp_vec12;

 // temp_vec34 = mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1).
 real_vec_t temp_vec34 = temp_vec32 + temp_vec33;

 // temp_vec35 = mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1).
 real_vec_t temp_vec35 = temp_vec34 + temp_vec14;

 // temp_vec36 = mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1).
 real_vec_t temp_vec36 = temp_vec35 + temp_vec16;

 // temp_vec37 = mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1).
 real_vec_t temp_vec37 = temp_vec36 + temp_vec18;

 // temp_vec38 = (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))).
 real_vec_t temp_vec38 = temp_vec26 / temp_vec37;

 // temp_vec39 = 2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))).
 real_vec_t temp_vec39 = temp_vec25 * temp_vec38;

 // temp_vec40 = 1.125.
 real_vec_t temp_vec40 = 1.12500000000000000e+00;

 // temp_vec41 = vel_x(t+1, x, y, z).
 real_vec_t temp_vec41 = temp_vec19;

 // temp_vec42 = (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z)).
 real_vec_t temp_vec42 = temp_vec21 - temp_vec41;

 // temp_vec43 = 1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z)).
 real_vec_t temp_vec43 = temp_vec40 * temp_vec42;

 // temp_vec44 = -0.0416667.
 real_vec_t temp_vec44 = -4.16666666666666644e-02;

 // temp_vec45 = -0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z)).
 real_vec_t temp_vec45 = temp_vec44 * (temp_vec22 - temp_vec24);

 // temp_vec46 = (1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))).
 real_vec_t temp_vec46 = temp_vec43 + temp_vec45;

 // temp_vec47 = 2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z)))).
 real_vec_t temp_vec47 = temp_vec39 * temp_vec46;

 // Read aligned vector block from lambda at x, y, z.
 real_vec_t temp_vec48 = context.lambda->readVecNorm(xv, yv, zv, __LINE__);

 // Read aligned vector block from lambda at x+4, y, z.
 real_vec_t temp_vec49 = context.lambda->readVecNorm(xv+(4/4), yv, zv, __LINE__);

 // Construct unaligned vector block from lambda at x+1, y, z.
 real_vec_t temp_vec50;
 // temp_vec50[0] = temp_vec48[1];  // for x+1, y, z;
 // temp_vec50[1] = temp_vec48[2];  // for x+2, y, z;
 // temp_vec50[2] = temp_vec48[3];  // for x+3, y, z;
 // temp_vec50[3] = temp_vec49[0];  // for x+4, y, z;
 // temp_vec50[4] = temp_vec48[5];  // for x+1, y+1, z;
 // temp_vec50[5] = temp_vec48[6];  // for x+2, y+1, z;
 // temp_vec50[6] = temp_vec48[7];  // for x+3, y+1, z;
 // temp_vec50[7] = temp_vec49[4];  // for x+4, y+1, z;
 // temp_vec50[8] = temp_vec48[9];  // for x+1, y+2, z;
 // temp_vec50[9] = temp_vec48[10];  // for x+2, y+2, z;
 // temp_vec50[10] = temp_vec48[11];  // for x+3, y+2, z;
 // temp_vec50[11] = temp_vec49[8];  // for x+4, y+2, z;
 // temp_vec50[12] = temp_vec48[13];  // for x+1, y+3, z;
 // temp_vec50[13] = temp_vec48[14];  // for x+2, y+3, z;
 // temp_vec50[14] = temp_vec48[15];  // for x+3, y+3, z;
 // temp_vec50[15] = temp_vec49[12];  // for x+4, y+3, z;
 real_vec_permute2(temp_vec50, ctrl_A1_A2_A3_B0_A5_A6_A7_B4_A9_A10_A11_B8_A13_A14_A15_B12, temp_vec48, temp_vec49);

 // Read aligned vector block from lambda at x, y-4, z.
 real_vec_t temp_vec51 = context.lambda->readVecNorm(xv, yv-(4/4), zv, __LINE__);

 // Construct unaligned vector block from lambda at x, y-1, z.
 real_vec_t temp_vec52;
 // temp_vec52[0] = temp_vec51[12];  // for x, y-1, z;
 // temp_vec52[1] = temp_vec51[13];  // for x+1, y-1, z;
 // temp_vec52[2] = temp_vec51[14];  // for x+2, y-1, z;
 // temp_vec52[3] = temp_vec51[15];  // for x+3, y-1, z;
 // temp_vec52[4] = temp_vec48[0];  // for x, y, z;
 // temp_vec52[5] = temp_vec48[1];  // for x+1, y, z;
 // temp_vec52[6] = temp_vec48[2];  // for x+2, y, z;
 // temp_vec52[7] = temp_vec48[3];  // for x+3, y, z;
 // temp_vec52[8] = temp_vec48[4];  // for x, y+1, z;
 // temp_vec52[9] = temp_vec48[5];  // for x+1, y+1, z;
 // temp_vec52[10] = temp_vec48[6];  // for x+2, y+1, z;
 // temp_vec52[11] = temp_vec48[7];  // for x+3, y+1, z;
 // temp_vec52[12] = temp_vec48[8];  // for x, y+2, z;
 // temp_vec52[13] = temp_vec48[9];  // for x+1, y+2, z;
 // temp_vec52[14] = temp_vec48[10];  // for x+2, y+2, z;
 // temp_vec52[15] = temp_vec48[11];  // for x+3, y+2, z;
 // Get 12 element(s) from temp_vec48 and 4 from temp_vec51.
 real_vec_align<12>(temp_vec52, temp_vec48, temp_vec51);

 // Read aligned vector block from lambda at x+4, y-4, z.
 real_vec_t temp_vec53 = context.lambda->readVecNorm(xv+(4/4), yv-(4/4), zv, __LINE__);

 // Construct unaligned vector block from lambda at x+1, y-1, z.
 real_vec_t temp_vec54;
 // temp_vec54[0] = temp_vec51[13];  // for x+1, y-1, z;
 // temp_vec54[1] = temp_vec51[14];  // for x+2, y-1, z;
 // temp_vec54[2] = temp_vec51[15];  // for x+3, y-1, z;
 // temp_vec54[3] = temp_vec53[12];  // for x+4, y-1, z;
 // temp_vec54[4] = temp_vec48[1];  // for x+1, y, z;
 // temp_vec54[5] = temp_vec48[2];  // for x+2, y, z;
 // temp_vec54[6] = temp_vec48[3];  // for x+3, y, z;
 // temp_vec54[7] = temp_vec49[0];  // for x+4, y, z;
 // temp_vec54[8] = temp_vec48[5];  // for x+1, y+1, z;
 // temp_vec54[9] = temp_vec48[6];  // for x+2, y+1, z;
 // temp_vec54[10] = temp_vec48[7];  // for x+3, y+1, z;
 // temp_vec54[11] = temp_vec49[4];  // for x+4, y+1, z;
 // temp_vec54[12] = temp_vec48[9];  // for x+1, y+2, z;
 // temp_vec54[13] = temp_vec48[10];  // for x+2, y+2, z;
 // temp_vec54[14] = temp_vec48[11];  // for x+3, y+2, z;
 // temp_vec54[15] = temp_vec49[8];  // for x+4, y+2, z;
 // Get 9 element(s) from temp_vec48 and 3 from temp_vec51.
 real_vec_align<13>(temp_vec54, temp_vec48, temp_vec51);
 // Get 3 element(s) from temp_vec49 and 1 from temp_vec53.
 real_vec_align_masked<9>(temp_vec54, temp_vec49, temp_vec53, 0x8888);

 // Read aligned vector block from lambda at x, y, z-1.
 real_vec_t temp_vec55 = context.lambda->readVecNorm(xv, yv, zv-(1/1), __LINE__);

 // Read aligned vector block from lambda at x+4, y, z-1.
 real_vec_t temp_vec56 = context.lambda->readVecNorm(xv+(4/4), yv, zv-(1/1), __LINE__);

 // Construct unaligned vector block from lambda at x+1, y, z-1.
 real_vec_t temp_vec57;
 // temp_vec57[0] = temp_vec55[1];  // for x+1, y, z-1;
 // temp_vec57[1] = temp_vec55[2];  // for x+2, y, z-1;
 // temp_vec57[2] = temp_vec55[3];  // for x+3, y, z-1;
 // temp_vec57[3] = temp_vec56[0];  // for x+4, y, z-1;
 // temp_vec57[4] = temp_vec55[5];  // for x+1, y+1, z-1;
 // temp_vec57[5] = temp_vec55[6];  // for x+2, y+1, z-1;
 // temp_vec57[6] = temp_vec55[7];  // for x+3, y+1, z-1;
 // temp_vec57[7] = temp_vec56[4];  // for x+4, y+1, z-1;
 // temp_vec57[8] = temp_vec55[9];  // for x+1, y+2, z-1;
 // temp_vec57[9] = temp_vec55[10];  // for x+2, y+2, z-1;
 // temp_vec57[10] = temp_vec55[11];  // for x+3, y+2, z-1;
 // temp_vec57[11] = temp_vec56[8];  // for x+4, y+2, z-1;
 // temp_vec57[12] = temp_vec55[13];  // for x+1, y+3, z-1;
 // temp_vec57[13] = temp_vec55[14];  // for x+2, y+3, z-1;
 // temp_vec57[14] = temp_vec55[15];  // for x+3, y+3, z-1;
 // temp_vec57[15] = temp_vec56[12];  // for x+4, y+3, z-1;
 real_vec_permute2(temp_vec57, ctrl_A1_A2_A3_B0_A5_A6_A7_B4_A9_A10_A11_B8_A13_A14_A15_B12, temp_vec55, temp_vec56);

 // Read aligned vector block from lambda at x, y-4, z-1.
 real_vec_t temp_vec58 = context.lambda->readVecNorm(xv, yv-(4/4), zv-(1/1), __LINE__);

 // Construct unaligned vector block from lambda at x, y-1, z-1.
 real_vec_t temp_vec59;
 // temp_vec59[0] = temp_vec58[12];  // for x, y-1, z-1;
 // temp_vec59[1] = temp_vec58[13];  // for x+1, y-1, z-1;
 // temp_vec59[2] = temp_vec58[14];  // for x+2, y-1, z-1;
 // temp_vec59[3] = temp_vec58[15];  // for x+3, y-1, z-1;
 // temp_vec59[4] = temp_vec55[0];  // for x, y, z-1;
 // temp_vec59[5] = temp_vec55[1];  // for x+1, y, z-1;
 // temp_vec59[6] = temp_vec55[2];  // for x+2, y, z-1;
 // temp_vec59[7] = temp_vec55[3];  // for x+3, y, z-1;
 // temp_vec59[8] = temp_vec55[4];  // for x, y+1, z-1;
 // temp_vec59[9] = temp_vec55[5];  // for x+1, y+1, z-1;
 // temp_vec59[10] = temp_vec55[6];  // for x+2, y+1, z-1;
 // temp_vec59[11] = temp_vec55[7];  // for x+3, y+1, z-1;
 // temp_vec59[12] = temp_vec55[8];  // for x, y+2, z-1;
 // temp_vec59[13] = temp_vec55[9];  // for x+1, y+2, z-1;
 // temp_vec59[14] = temp_vec55[10];  // for x+2, y+2, z-1;
 // temp_vec59[15] = temp_vec55[11];  // for x+3, y+2, z-1;
 // Get 12 element(s) from temp_vec55 and 4 from temp_vec58.
 real_vec_align<12>(temp_vec59, temp_vec55, temp_vec58);

 // Read aligned vector block from lambda at x+4, y-4, z-1.
 real_vec_t temp_vec60 = context.lambda->readVecNorm(xv+(4/4), yv-(4/4), zv-(1/1), __LINE__);

 // Construct unaligned vector block from lambda at x+1, y-1, z-1.
 real_vec_t temp_vec61;
 // temp_vec61[0] = temp_vec58[13];  // for x+1, y-1, z-1;
 // temp_vec61[1] = temp_vec58[14];  // for x+2, y-1, z-1;
 // temp_vec61[2] = temp_vec58[15];  // for x+3, y-1, z-1;
 // temp_vec61[3] = temp_vec60[12];  // for x+4, y-1, z-1;
 // temp_vec61[4] = temp_vec55[1];  // for x+1, y, z-1;
 // temp_vec61[5] = temp_vec55[2];  // for x+2, y, z-1;
 // temp_vec61[6] = temp_vec55[3];  // for x+3, y, z-1;
 // temp_vec61[7] = temp_vec56[0];  // for x+4, y, z-1;
 // temp_vec61[8] = temp_vec55[5];  // for x+1, y+1, z-1;
 // temp_vec61[9] = temp_vec55[6];  // for x+2, y+1, z-1;
 // temp_vec61[10] = temp_vec55[7];  // for x+3, y+1, z-1;
 // temp_vec61[11] = temp_vec56[4];  // for x+4, y+1, z-1;
 // temp_vec61[12] = temp_vec55[9];  // for x+1, y+2, z-1;
 // temp_vec61[13] = temp_vec55[10];  // for x+2, y+2, z-1;
 // temp_vec61[14] = temp_vec55[11];  // for x+3, y+2, z-1;
 // temp_vec61[15] = temp_vec56[8];  // for x+4, y+2, z-1;
 // Get 9 element(s) from temp_vec55 and 3 from temp_vec58.
 real_vec_align<13>(temp_vec61, temp_vec55, temp_vec58);
 // Get 3 element(s) from temp_vec56 and 1 from temp_vec60.
 real_vec_align_masked<9>(temp_vec61, temp_vec56, temp_vec60, 0x8888);

 // Read aligned vector block from vel_y at t+1, x, y, z.
 real_vec_t temp_vec62 = context.vel_y->readVecNorm(tv+(1/1), xv, yv, zv, __LINE__);

 // Read aligned vector block from vel_y at t+1, x, y-4, z.
 real_vec_t temp_vec63 = context.vel_y->readVecNorm(tv+(1/1), xv, yv-(4/4), zv, __LINE__);

 // Construct unaligned vector block from vel_y at t+1, x, y-1, z.
 real_vec_t temp_vec64;
 // temp_vec64[0] = temp_vec63[12];  // for t+1, x, y-1, z;
 // temp_vec64[1] = temp_vec63[13];  // for t+1, x+1, y-1, z;
 // temp_vec64[2] = temp_vec63[14];  // for t+1, x+2, y-1, z;
 // temp_vec64[3] = temp_vec63[15];  // for t+1, x+3, y-1, z;
 // temp_vec64[4] = temp_vec62[0];  // for t+1, x, y, z;
 // temp_vec64[5] = temp_vec62[1];  // for t+1, x+1, y, z;
 // temp_vec64[6] = temp_vec62[2];  // for t+1, x+2, y, z;
 // temp_vec64[7] = temp_vec62[3];  // for t+1, x+3, y, z;
 // temp_vec64[8] = temp_vec62[4];  // for t+1, x, y+1, z;
 // temp_vec64[9] = temp_vec62[5];  // for t+1, x+1, y+1, z;
 // temp_vec64[10] = temp_vec62[6];  // for t+1, x+2, y+1, z;
 // temp_vec64[11] = temp_vec62[7];  // for t+1, x+3, y+1, z;
 // temp_vec64[12] = temp_vec62[8];  // for t+1, x, y+2, z;
 // temp_vec64[13] = temp_vec62[9];  // for t+1, x+1, y+2, z;
 // temp_vec64[14] = temp_vec62[10];  // for t+1, x+2, y+2, z;
 // temp_vec64[15] = temp_vec62[11];  // for t+1, x+3, y+2, z;
 // Get 12 element(s) from temp_vec62 and 4 from temp_vec63.
 real_vec_align<12>(temp_vec64, temp_vec62, temp_vec63);

 // Read aligned vector block from vel_y at t+1, x, y+4, z.
 real_vec_t temp_vec65 = context.vel_y->readVecNorm(tv+(1/1), xv, yv+(4/4), zv, __LINE__);

 // Construct unaligned vector block from vel_y at t+1, x, y+1, z.
 real_vec_t temp_vec66;
 // temp_vec66[0] = temp_vec62[4];  // for t+1, x, y+1, z;
 // temp_vec66[1] = temp_vec62[5];  // for t+1, x+1, y+1, z;
 // temp_vec66[2] = temp_vec62[6];  // for t+1, x+2, y+1, z;
 // temp_vec66[3] = temp_vec62[7];  // for t+1, x+3, y+1, z;
 // temp_vec66[4] = temp_vec62[8];  // for t+1, x, y+2, z;
 // temp_vec66[5] = temp_vec62[9];  // for t+1, x+1, y+2, z;
 // temp_vec66[6] = temp_vec62[10];  // for t+1, x+2, y+2, z;
 // temp_vec66[7] = temp_vec62[11];  // for t+1, x+3, y+2, z;
 // temp_vec66[8] = temp_vec62[12];  // for t+1, x, y+3, z;
 // temp_vec66[9] = temp_vec62[13];  // for t+1, x+1, y+3, z;
 // temp_vec66[10] = temp_vec62[14];  // for t+1, x+2, y+3, z;
 // temp_vec66[11] = temp_vec62[15];  // for t+1, x+3, y+3, z;
 // temp_vec66[12] = temp_vec65[0];  // for t+1, x, y+4, z;
 // temp_vec66[13] = temp_vec65[1];  // for t+1, x+1, y+4, z;
 // temp_vec66[14] = temp_vec65[2];  // for t+1, x+2, y+4, z;
 // temp_vec66[15] = temp_vec65[3];  // for t+1, x+3, y+4, z;
 // Get 4 element(s) from temp_vec65 and 12 from temp_vec62.
 real_vec_align<4>(temp_vec66, temp_vec65, temp_vec62);

 // Construct unaligned vector block from vel_y at t+1, x, y-2, z.
 real_vec_t temp_vec67;
 // temp_vec67[0] = temp_vec63[8];  // for t+1, x, y-2, z;
 // temp_vec67[1] = temp_vec63[9];  // for t+1, x+1, y-2, z;
 // temp_vec67[2] = temp_vec63[10];  // for t+1, x+2, y-2, z;
 // temp_vec67[3] = temp_vec63[11];  // for t+1, x+3, y-2, z;
 // temp_vec67[4] = temp_vec63[12];  // for t+1, x, y-1, z;
 // temp_vec67[5] = temp_vec63[13];  // for t+1, x+1, y-1, z;
 // temp_vec67[6] = temp_vec63[14];  // for t+1, x+2, y-1, z;
 // temp_vec67[7] = temp_vec63[15];  // for t+1, x+3, y-1, z;
 // temp_vec67[8] = temp_vec62[0];  // for t+1, x, y, z;
 // temp_vec67[9] = temp_vec62[1];  // for t+1, x+1, y, z;
 // temp_vec67[10] = temp_vec62[2];  // for t+1, x+2, y, z;
 // temp_vec67[11] = temp_vec62[3];  // for t+1, x+3, y, z;
 // temp_vec67[12] = temp_vec62[4];  // for t+1, x, y+1, z;
 // temp_vec67[13] = temp_vec62[5];  // for t+1, x+1, y+1, z;
 // temp_vec67[14] = temp_vec62[6];  // for t+1, x+2, y+1, z;
 // temp_vec67[15] = temp_vec62[7];  // for t+1, x+3, y+1, z;
 // Get 8 element(s) from temp_vec62 and 8 from temp_vec63.
 real_vec_align<8>(temp_vec67, temp_vec62, temp_vec63);

 // Read aligned vector block from vel_z at t+1, x, y, z.
 real_vec_t temp_vec68 = context.vel_z->readVecNorm(tv+(1/1), xv, yv, zv, __LINE__);

 // Read aligned vector block from vel_z at t+1, x, y, z-1.
 real_vec_t temp_vec69 = context.vel_z->readVecNorm(tv+(1/1), xv, yv, zv-(1/1), __LINE__);

 // Read aligned vector block from vel_z at t+1, x, y, z+1.
 real_vec_t temp_vec70 = context.vel_z->readVecNorm(tv+(1/1), xv, yv, zv+(1/1), __LINE__);

 // Read aligned vector block from vel_z at t+1, x, y, z-2.
 real_vec_t temp_vec71 = context.vel_z->readVecNorm(tv+(1/1), xv, yv, zv-(2/1), __LINE__);

 // temp_vec72 = (8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))).
 real_vec_t temp_vec72 = temp_vec26 / (temp_vec48 + temp_vec50 + temp_vec52 + temp_vec54 + temp_vec55 + temp_vec57 + temp_vec59 + temp_vec61);

 // temp_vec73 = (1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))).
 real_vec_t temp_vec73 = temp_vec43 + temp_vec45;

 // temp_vec74 = vel_y(t+1, x, y, z).
 real_vec_t temp_vec74 = temp_vec62;

 // temp_vec75 = (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z)).
 real_vec_t temp_vec75 = temp_vec74 - temp_vec64;

 // temp_vec76 = 1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z)).
 real_vec_t temp_vec76 = temp_vec40 * temp_vec75;

 // temp_vec77 = (1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))).
 real_vec_t temp_vec77 = temp_vec73 + temp_vec76;

 // temp_vec78 = -0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)).
 real_vec_t temp_vec78 = temp_vec44 * (temp_vec66 - temp_vec67);

 // temp_vec79 = (1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))).
 real_vec_t temp_vec79 = temp_vec77 + temp_vec78;

 // temp_vec80 = vel_z(t+1, x, y, z).
 real_vec_t temp_vec80 = temp_vec68;

 // temp_vec81 = (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1)).
 real_vec_t temp_vec81 = temp_vec80 - temp_vec69;

 // temp_vec82 = 1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1)).
 real_vec_t temp_vec82 = temp_vec40 * temp_vec81;

 // temp_vec83 = (1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))).
 real_vec_t temp_vec83 = temp_vec79 + temp_vec82;

 // temp_vec84 = -0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)).
 real_vec_t temp_vec84 = temp_vec44 * (temp_vec70 - temp_vec71);

 // temp_vec85 = (1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))).
 real_vec_t temp_vec85 = temp_vec83 + temp_vec84;

 // temp_vec86 = (8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))).
 real_vec_t temp_vec86 = temp_vec72 * temp_vec85;

 // temp_vec87 = (2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 real_vec_t temp_vec87 = temp_vec47 + temp_vec86;

 // temp_vec88 = (delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))).
 real_vec_t temp_vec88 = temp_vec4 * temp_vec87;

 // temp_vec89 = stress_xx(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))).
 real_vec_t temp_vec89 = temp_vec1 + temp_vec88;

 // Read aligned vector block from sponge at x, y, z.
 real_vec_t temp_vec90 = context.sponge->readVecNorm(xv, yv, zv, __LINE__);

 // temp_vec91 = sponge(x, y, z).
 real_vec_t temp_vec91 = temp_vec90;

 // temp_vec92 = (stress_xx(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))))) * sponge(x, y, z).
 real_vec_t temp_vec92 = temp_vec89 * temp_vec91;

 // temp_vec93 = ((stress_xx(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))))) * sponge(x, y, z)).
 real_vec_t temp_vec93 = temp_vec92;

 // Save result to stress_xx(t+1, x, y, z):
 
 // Write aligned vector block to stress_xx at t+1, x, y, z.
context.stress_xx->writeVecNorm(temp_vec93, tv+(1/1), xv, yv, zv, __LINE__);
;

 // Read aligned vector block from stress_yy at t, x, y, z.
 real_vec_t temp_vec94 = context.stress_yy->readVecNorm(tv, xv, yv, zv, __LINE__);

 // temp_vec95 = 2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))).
 real_vec_t temp_vec95 = temp_vec25 * temp_vec38;

 // temp_vec96 = (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))).
 real_vec_t temp_vec96 = temp_vec76 + temp_vec78;

 // temp_vec97 = 2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))).
 real_vec_t temp_vec97 = temp_vec95 * temp_vec96;

 // temp_vec98 = (2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 real_vec_t temp_vec98 = temp_vec97 + temp_vec86;

 // temp_vec99 = (delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))).
 real_vec_t temp_vec99 = temp_vec4 * temp_vec98;

 // temp_vec100 = stress_yy(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))).
 real_vec_t temp_vec100 = temp_vec94 + temp_vec99;

 // temp_vec101 = (stress_yy(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))))) * sponge(x, y, z).
 real_vec_t temp_vec101 = temp_vec100 * temp_vec91;

 // temp_vec102 = ((stress_yy(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))))) * sponge(x, y, z)).
 real_vec_t temp_vec102 = temp_vec101;

 // Save result to stress_yy(t+1, x, y, z):
 
 // Write aligned vector block to stress_yy at t+1, x, y, z.
context.stress_yy->writeVecNorm(temp_vec102, tv+(1/1), xv, yv, zv, __LINE__);
;

 // Read aligned vector block from stress_zz at t, x, y, z.
 real_vec_t temp_vec103 = context.stress_zz->readVecNorm(tv, xv, yv, zv, __LINE__);

 // temp_vec104 = 2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))).
 real_vec_t temp_vec104 = temp_vec25 * temp_vec38;

 // temp_vec105 = (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))).
 real_vec_t temp_vec105 = temp_vec82 + temp_vec84;

 // temp_vec106 = 2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))).
 real_vec_t temp_vec106 = temp_vec104 * temp_vec105;

 // temp_vec107 = (2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 real_vec_t temp_vec107 = temp_vec106 + temp_vec86;

 // temp_vec108 = (delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))).
 real_vec_t temp_vec108 = temp_vec4 * temp_vec107;

 // temp_vec109 = stress_zz(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))).
 real_vec_t temp_vec109 = temp_vec103 + temp_vec108;

 // temp_vec110 = (stress_zz(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))))) * sponge(x, y, z).
 real_vec_t temp_vec110 = temp_vec109 * temp_vec91;

 // temp_vec111 = ((stress_zz(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))))) * sponge(x, y, z)).
 real_vec_t temp_vec111 = temp_vec110;

 // Save result to stress_zz(t+1, x, y, z):
 
 // Write aligned vector block to stress_zz at t+1, x, y, z.
context.stress_zz->writeVecNorm(temp_vec111, tv+(1/1), xv, yv, zv, __LINE__);
;

 // Read aligned vector block from stress_xy at t, x, y, z.
 real_vec_t temp_vec112 = context.stress_xy->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from vel_x at t+1, x, y+4, z.
 real_vec_t temp_vec113 = context.vel_x->readVecNorm(tv+(1/1), xv, yv+(4/4), zv, __LINE__);

 // Construct unaligned vector block from vel_x at t+1, x, y+1, z.
 real_vec_t temp_vec114;
 // temp_vec114[0] = temp_vec19[4];  // for t+1, x, y+1, z;
 // temp_vec114[1] = temp_vec19[5];  // for t+1, x+1, y+1, z;
 // temp_vec114[2] = temp_vec19[6];  // for t+1, x+2, y+1, z;
 // temp_vec114[3] = temp_vec19[7];  // for t+1, x+3, y+1, z;
 // temp_vec114[4] = temp_vec19[8];  // for t+1, x, y+2, z;
 // temp_vec114[5] = temp_vec19[9];  // for t+1, x+1, y+2, z;
 // temp_vec114[6] = temp_vec19[10];  // for t+1, x+2, y+2, z;
 // temp_vec114[7] = temp_vec19[11];  // for t+1, x+3, y+2, z;
 // temp_vec114[8] = temp_vec19[12];  // for t+1, x, y+3, z;
 // temp_vec114[9] = temp_vec19[13];  // for t+1, x+1, y+3, z;
 // temp_vec114[10] = temp_vec19[14];  // for t+1, x+2, y+3, z;
 // temp_vec114[11] = temp_vec19[15];  // for t+1, x+3, y+3, z;
 // temp_vec114[12] = temp_vec113[0];  // for t+1, x, y+4, z;
 // temp_vec114[13] = temp_vec113[1];  // for t+1, x+1, y+4, z;
 // temp_vec114[14] = temp_vec113[2];  // for t+1, x+2, y+4, z;
 // temp_vec114[15] = temp_vec113[3];  // for t+1, x+3, y+4, z;
 // Get 4 element(s) from temp_vec113 and 12 from temp_vec19.
 real_vec_align<4>(temp_vec114, temp_vec113, temp_vec19);

 // Construct unaligned vector block from vel_x at t+1, x, y+2, z.
 real_vec_t temp_vec115;
 // temp_vec115[0] = temp_vec19[8];  // for t+1, x, y+2, z;
 // temp_vec115[1] = temp_vec19[9];  // for t+1, x+1, y+2, z;
 // temp_vec115[2] = temp_vec19[10];  // for t+1, x+2, y+2, z;
 // temp_vec115[3] = temp_vec19[11];  // for t+1, x+3, y+2, z;
 // temp_vec115[4] = temp_vec19[12];  // for t+1, x, y+3, z;
 // temp_vec115[5] = temp_vec19[13];  // for t+1, x+1, y+3, z;
 // temp_vec115[6] = temp_vec19[14];  // for t+1, x+2, y+3, z;
 // temp_vec115[7] = temp_vec19[15];  // for t+1, x+3, y+3, z;
 // temp_vec115[8] = temp_vec113[0];  // for t+1, x, y+4, z;
 // temp_vec115[9] = temp_vec113[1];  // for t+1, x+1, y+4, z;
 // temp_vec115[10] = temp_vec113[2];  // for t+1, x+2, y+4, z;
 // temp_vec115[11] = temp_vec113[3];  // for t+1, x+3, y+4, z;
 // temp_vec115[12] = temp_vec113[4];  // for t+1, x, y+5, z;
 // temp_vec115[13] = temp_vec113[5];  // for t+1, x+1, y+5, z;
 // temp_vec115[14] = temp_vec113[6];  // for t+1, x+2, y+5, z;
 // temp_vec115[15] = temp_vec113[7];  // for t+1, x+3, y+5, z;
 // Get 8 element(s) from temp_vec113 and 8 from temp_vec19.
 real_vec_align<8>(temp_vec115, temp_vec113, temp_vec19);

 // Read aligned vector block from vel_x at t+1, x, y-4, z.
 real_vec_t temp_vec116 = context.vel_x->readVecNorm(tv+(1/1), xv, yv-(4/4), zv, __LINE__);

 // Construct unaligned vector block from vel_x at t+1, x, y-1, z.
 real_vec_t temp_vec117;
 // temp_vec117[0] = temp_vec116[12];  // for t+1, x, y-1, z;
 // temp_vec117[1] = temp_vec116[13];  // for t+1, x+1, y-1, z;
 // temp_vec117[2] = temp_vec116[14];  // for t+1, x+2, y-1, z;
 // temp_vec117[3] = temp_vec116[15];  // for t+1, x+3, y-1, z;
 // temp_vec117[4] = temp_vec19[0];  // for t+1, x, y, z;
 // temp_vec117[5] = temp_vec19[1];  // for t+1, x+1, y, z;
 // temp_vec117[6] = temp_vec19[2];  // for t+1, x+2, y, z;
 // temp_vec117[7] = temp_vec19[3];  // for t+1, x+3, y, z;
 // temp_vec117[8] = temp_vec19[4];  // for t+1, x, y+1, z;
 // temp_vec117[9] = temp_vec19[5];  // for t+1, x+1, y+1, z;
 // temp_vec117[10] = temp_vec19[6];  // for t+1, x+2, y+1, z;
 // temp_vec117[11] = temp_vec19[7];  // for t+1, x+3, y+1, z;
 // temp_vec117[12] = temp_vec19[8];  // for t+1, x, y+2, z;
 // temp_vec117[13] = temp_vec19[9];  // for t+1, x+1, y+2, z;
 // temp_vec117[14] = temp_vec19[10];  // for t+1, x+2, y+2, z;
 // temp_vec117[15] = temp_vec19[11];  // for t+1, x+3, y+2, z;
 // Get 12 element(s) from temp_vec19 and 4 from temp_vec116.
 real_vec_align<12>(temp_vec117, temp_vec19, temp_vec116);

 // Read aligned vector block from vel_y at t+1, x-4, y, z.
 real_vec_t temp_vec118 = context.vel_y->readVecNorm(tv+(1/1), xv-(4/4), yv, zv, __LINE__);

 // Construct unaligned vector block from vel_y at t+1, x-1, y, z.
 real_vec_t temp_vec119;
 // temp_vec119[0] = temp_vec118[3];  // for t+1, x-1, y, z;
 // temp_vec119[1] = temp_vec62[0];  // for t+1, x, y, z;
 // temp_vec119[2] = temp_vec62[1];  // for t+1, x+1, y, z;
 // temp_vec119[3] = temp_vec62[2];  // for t+1, x+2, y, z;
 // temp_vec119[4] = temp_vec118[7];  // for t+1, x-1, y+1, z;
 // temp_vec119[5] = temp_vec62[4];  // for t+1, x, y+1, z;
 // temp_vec119[6] = temp_vec62[5];  // for t+1, x+1, y+1, z;
 // temp_vec119[7] = temp_vec62[6];  // for t+1, x+2, y+1, z;
 // temp_vec119[8] = temp_vec118[11];  // for t+1, x-1, y+2, z;
 // temp_vec119[9] = temp_vec62[8];  // for t+1, x, y+2, z;
 // temp_vec119[10] = temp_vec62[9];  // for t+1, x+1, y+2, z;
 // temp_vec119[11] = temp_vec62[10];  // for t+1, x+2, y+2, z;
 // temp_vec119[12] = temp_vec118[15];  // for t+1, x-1, y+3, z;
 // temp_vec119[13] = temp_vec62[12];  // for t+1, x, y+3, z;
 // temp_vec119[14] = temp_vec62[13];  // for t+1, x+1, y+3, z;
 // temp_vec119[15] = temp_vec62[14];  // for t+1, x+2, y+3, z;
 real_vec_permute2(temp_vec119, ctrl_A3_B0_B1_B2_A7_B4_B5_B6_A11_B8_B9_B10_A15_B12_B13_B14, temp_vec118, temp_vec62);

 // Read aligned vector block from vel_y at t+1, x+4, y, z.
 real_vec_t temp_vec120 = context.vel_y->readVecNorm(tv+(1/1), xv+(4/4), yv, zv, __LINE__);

 // Construct unaligned vector block from vel_y at t+1, x+1, y, z.
 real_vec_t temp_vec121;
 // temp_vec121[0] = temp_vec62[1];  // for t+1, x+1, y, z;
 // temp_vec121[1] = temp_vec62[2];  // for t+1, x+2, y, z;
 // temp_vec121[2] = temp_vec62[3];  // for t+1, x+3, y, z;
 // temp_vec121[3] = temp_vec120[0];  // for t+1, x+4, y, z;
 // temp_vec121[4] = temp_vec62[5];  // for t+1, x+1, y+1, z;
 // temp_vec121[5] = temp_vec62[6];  // for t+1, x+2, y+1, z;
 // temp_vec121[6] = temp_vec62[7];  // for t+1, x+3, y+1, z;
 // temp_vec121[7] = temp_vec120[4];  // for t+1, x+4, y+1, z;
 // temp_vec121[8] = temp_vec62[9];  // for t+1, x+1, y+2, z;
 // temp_vec121[9] = temp_vec62[10];  // for t+1, x+2, y+2, z;
 // temp_vec121[10] = temp_vec62[11];  // for t+1, x+3, y+2, z;
 // temp_vec121[11] = temp_vec120[8];  // for t+1, x+4, y+2, z;
 // temp_vec121[12] = temp_vec62[13];  // for t+1, x+1, y+3, z;
 // temp_vec121[13] = temp_vec62[14];  // for t+1, x+2, y+3, z;
 // temp_vec121[14] = temp_vec62[15];  // for t+1, x+3, y+3, z;
 // temp_vec121[15] = temp_vec120[12];  // for t+1, x+4, y+3, z;
 real_vec_permute2(temp_vec121, ctrl_A1_A2_A3_B0_A5_A6_A7_B4_A9_A10_A11_B8_A13_A14_A15_B12, temp_vec62, temp_vec120);

 // Construct unaligned vector block from vel_y at t+1, x-2, y, z.
 real_vec_t temp_vec122;
 // temp_vec122[0] = temp_vec118[2];  // for t+1, x-2, y, z;
 // temp_vec122[1] = temp_vec118[3];  // for t+1, x-1, y, z;
 // temp_vec122[2] = temp_vec62[0];  // for t+1, x, y, z;
 // temp_vec122[3] = temp_vec62[1];  // for t+1, x+1, y, z;
 // temp_vec122[4] = temp_vec118[6];  // for t+1, x-2, y+1, z;
 // temp_vec122[5] = temp_vec118[7];  // for t+1, x-1, y+1, z;
 // temp_vec122[6] = temp_vec62[4];  // for t+1, x, y+1, z;
 // temp_vec122[7] = temp_vec62[5];  // for t+1, x+1, y+1, z;
 // temp_vec122[8] = temp_vec118[10];  // for t+1, x-2, y+2, z;
 // temp_vec122[9] = temp_vec118[11];  // for t+1, x-1, y+2, z;
 // temp_vec122[10] = temp_vec62[8];  // for t+1, x, y+2, z;
 // temp_vec122[11] = temp_vec62[9];  // for t+1, x+1, y+2, z;
 // temp_vec122[12] = temp_vec118[14];  // for t+1, x-2, y+3, z;
 // temp_vec122[13] = temp_vec118[15];  // for t+1, x-1, y+3, z;
 // temp_vec122[14] = temp_vec62[12];  // for t+1, x, y+3, z;
 // temp_vec122[15] = temp_vec62[13];  // for t+1, x+1, y+3, z;
 real_vec_permute2(temp_vec122, ctrl_A2_A3_B0_B1_A6_A7_B4_B5_A10_A11_B8_B9_A14_A15_B12_B13, temp_vec118, temp_vec62);

 // temp_vec123 = mu(x, y, z) + mu(x, y, z-1).
 real_vec_t temp_vec123 = temp_vec27 + temp_vec33;

 // temp_vec124 = (2 / (mu(x, y, z) + mu(x, y, z-1))).
 real_vec_t temp_vec124 = temp_vec25 / temp_vec123;

 // temp_vec125 = (2 / (mu(x, y, z) + mu(x, y, z-1))) * delta_t().
 real_vec_t temp_vec125 = temp_vec124 * temp_vec2;

 // temp_vec126 = (((2 / (mu(x, y, z) + mu(x, y, z-1))) * delta_t) / h).
 real_vec_t temp_vec126 = temp_vec125 / temp_vec3;

 // temp_vec127 = (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z)).
 real_vec_t temp_vec127 = temp_vec114 - temp_vec41;

 // temp_vec128 = 1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z)).
 real_vec_t temp_vec128 = temp_vec40 * temp_vec127;

 // temp_vec129 = -0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z)).
 real_vec_t temp_vec129 = temp_vec44 * (temp_vec115 - temp_vec117);

 // temp_vec130 = (1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))).
 real_vec_t temp_vec130 = temp_vec128 + temp_vec129;

 // temp_vec131 = (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z)).
 real_vec_t temp_vec131 = temp_vec74 - temp_vec119;

 // temp_vec132 = 1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z)).
 real_vec_t temp_vec132 = temp_vec40 * temp_vec131;

 // temp_vec133 = (1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))).
 real_vec_t temp_vec133 = temp_vec130 + temp_vec132;

 // temp_vec134 = -0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z)).
 real_vec_t temp_vec134 = temp_vec44 * (temp_vec121 - temp_vec122);

 // temp_vec135 = (1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z))).
 real_vec_t temp_vec135 = temp_vec133 + temp_vec134;

 // temp_vec136 = (((2 / (mu(x, y, z) + mu(x, y, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z)))).
 real_vec_t temp_vec136 = temp_vec126 * temp_vec135;

 // temp_vec137 = stress_xy(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x, y, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z))))).
 real_vec_t temp_vec137 = temp_vec112 + temp_vec136;

 // temp_vec138 = (stress_xy(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x, y, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z)))))) * sponge(x, y, z).
 real_vec_t temp_vec138 = temp_vec137 * temp_vec91;

 // temp_vec139 = ((stress_xy(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x, y, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z)))))) * sponge(x, y, z)).
 real_vec_t temp_vec139 = temp_vec138;

 // Save result to stress_xy(t+1, x, y, z):
 
 // Write aligned vector block to stress_xy at t+1, x, y, z.
context.stress_xy->writeVecNorm(temp_vec139, tv+(1/1), xv, yv, zv, __LINE__);
;

 // Read aligned vector block from stress_xz at t, x, y, z.
 real_vec_t temp_vec140 = context.stress_xz->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from vel_x at t+1, x, y, z+1.
 real_vec_t temp_vec141 = context.vel_x->readVecNorm(tv+(1/1), xv, yv, zv+(1/1), __LINE__);

 // Read aligned vector block from vel_x at t+1, x, y, z+2.
 real_vec_t temp_vec142 = context.vel_x->readVecNorm(tv+(1/1), xv, yv, zv+(2/1), __LINE__);

 // Read aligned vector block from vel_x at t+1, x, y, z-1.
 real_vec_t temp_vec143 = context.vel_x->readVecNorm(tv+(1/1), xv, yv, zv-(1/1), __LINE__);

 // Read aligned vector block from vel_z at t+1, x-4, y, z.
 real_vec_t temp_vec144 = context.vel_z->readVecNorm(tv+(1/1), xv-(4/4), yv, zv, __LINE__);

 // Construct unaligned vector block from vel_z at t+1, x-1, y, z.
 real_vec_t temp_vec145;
 // temp_vec145[0] = temp_vec144[3];  // for t+1, x-1, y, z;
 // temp_vec145[1] = temp_vec68[0];  // for t+1, x, y, z;
 // temp_vec145[2] = temp_vec68[1];  // for t+1, x+1, y, z;
 // temp_vec145[3] = temp_vec68[2];  // for t+1, x+2, y, z;
 // temp_vec145[4] = temp_vec144[7];  // for t+1, x-1, y+1, z;
 // temp_vec145[5] = temp_vec68[4];  // for t+1, x, y+1, z;
 // temp_vec145[6] = temp_vec68[5];  // for t+1, x+1, y+1, z;
 // temp_vec145[7] = temp_vec68[6];  // for t+1, x+2, y+1, z;
 // temp_vec145[8] = temp_vec144[11];  // for t+1, x-1, y+2, z;
 // temp_vec145[9] = temp_vec68[8];  // for t+1, x, y+2, z;
 // temp_vec145[10] = temp_vec68[9];  // for t+1, x+1, y+2, z;
 // temp_vec145[11] = temp_vec68[10];  // for t+1, x+2, y+2, z;
 // temp_vec145[12] = temp_vec144[15];  // for t+1, x-1, y+3, z;
 // temp_vec145[13] = temp_vec68[12];  // for t+1, x, y+3, z;
 // temp_vec145[14] = temp_vec68[13];  // for t+1, x+1, y+3, z;
 // temp_vec145[15] = temp_vec68[14];  // for t+1, x+2, y+3, z;
 real_vec_permute2(temp_vec145, ctrl_A3_B0_B1_B2_A7_B4_B5_B6_A11_B8_B9_B10_A15_B12_B13_B14, temp_vec144, temp_vec68);

 // Read aligned vector block from vel_z at t+1, x+4, y, z.
 real_vec_t temp_vec146 = context.vel_z->readVecNorm(tv+(1/1), xv+(4/4), yv, zv, __LINE__);

 // Construct unaligned vector block from vel_z at t+1, x+1, y, z.
 real_vec_t temp_vec147;
 // temp_vec147[0] = temp_vec68[1];  // for t+1, x+1, y, z;
 // temp_vec147[1] = temp_vec68[2];  // for t+1, x+2, y, z;
 // temp_vec147[2] = temp_vec68[3];  // for t+1, x+3, y, z;
 // temp_vec147[3] = temp_vec146[0];  // for t+1, x+4, y, z;
 // temp_vec147[4] = temp_vec68[5];  // for t+1, x+1, y+1, z;
 // temp_vec147[5] = temp_vec68[6];  // for t+1, x+2, y+1, z;
 // temp_vec147[6] = temp_vec68[7];  // for t+1, x+3, y+1, z;
 // temp_vec147[7] = temp_vec146[4];  // for t+1, x+4, y+1, z;
 // temp_vec147[8] = temp_vec68[9];  // for t+1, x+1, y+2, z;
 // temp_vec147[9] = temp_vec68[10];  // for t+1, x+2, y+2, z;
 // temp_vec147[10] = temp_vec68[11];  // for t+1, x+3, y+2, z;
 // temp_vec147[11] = temp_vec146[8];  // for t+1, x+4, y+2, z;
 // temp_vec147[12] = temp_vec68[13];  // for t+1, x+1, y+3, z;
 // temp_vec147[13] = temp_vec68[14];  // for t+1, x+2, y+3, z;
 // temp_vec147[14] = temp_vec68[15];  // for t+1, x+3, y+3, z;
 // temp_vec147[15] = temp_vec146[12];  // for t+1, x+4, y+3, z;
 real_vec_permute2(temp_vec147, ctrl_A1_A2_A3_B0_A5_A6_A7_B4_A9_A10_A11_B8_A13_A14_A15_B12, temp_vec68, temp_vec146);

 // Construct unaligned vector block from vel_z at t+1, x-2, y, z.
 real_vec_t temp_vec148;
 // temp_vec148[0] = temp_vec144[2];  // for t+1, x-2, y, z;
 // temp_vec148[1] = temp_vec144[3];  // for t+1, x-1, y, z;
 // temp_vec148[2] = temp_vec68[0];  // for t+1, x, y, z;
 // temp_vec148[3] = temp_vec68[1];  // for t+1, x+1, y, z;
 // temp_vec148[4] = temp_vec144[6];  // for t+1, x-2, y+1, z;
 // temp_vec148[5] = temp_vec144[7];  // for t+1, x-1, y+1, z;
 // temp_vec148[6] = temp_vec68[4];  // for t+1, x, y+1, z;
 // temp_vec148[7] = temp_vec68[5];  // for t+1, x+1, y+1, z;
 // temp_vec148[8] = temp_vec144[10];  // for t+1, x-2, y+2, z;
 // temp_vec148[9] = temp_vec144[11];  // for t+1, x-1, y+2, z;
 // temp_vec148[10] = temp_vec68[8];  // for t+1, x, y+2, z;
 // temp_vec148[11] = temp_vec68[9];  // for t+1, x+1, y+2, z;
 // temp_vec148[12] = temp_vec144[14];  // for t+1, x-2, y+3, z;
 // temp_vec148[13] = temp_vec144[15];  // for t+1, x-1, y+3, z;
 // temp_vec148[14] = temp_vec68[12];  // for t+1, x, y+3, z;
 // temp_vec148[15] = temp_vec68[13];  // for t+1, x+1, y+3, z;
 real_vec_permute2(temp_vec148, ctrl_A2_A3_B0_B1_A6_A7_B4_B5_A10_A11_B8_B9_A14_A15_B12_B13, temp_vec144, temp_vec68);

 // temp_vec149 = mu(x, y, z) + mu(x, y-1, z).
 real_vec_t temp_vec149 = temp_vec27 + temp_vec30;

 // temp_vec150 = (2 / (mu(x, y, z) + mu(x, y-1, z))).
 real_vec_t temp_vec150 = temp_vec25 / temp_vec149;

 // temp_vec151 = (2 / (mu(x, y, z) + mu(x, y-1, z))) * delta_t().
 real_vec_t temp_vec151 = temp_vec150 * temp_vec2;

 // temp_vec152 = (((2 / (mu(x, y, z) + mu(x, y-1, z))) * delta_t) / h).
 real_vec_t temp_vec152 = temp_vec151 / temp_vec3;

 // temp_vec153 = (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z)).
 real_vec_t temp_vec153 = temp_vec141 - temp_vec41;

 // temp_vec154 = 1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z)).
 real_vec_t temp_vec154 = temp_vec40 * temp_vec153;

 // temp_vec155 = -0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1)).
 real_vec_t temp_vec155 = temp_vec44 * (temp_vec142 - temp_vec143);

 // temp_vec156 = (1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))).
 real_vec_t temp_vec156 = temp_vec154 + temp_vec155;

 // temp_vec157 = (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z)).
 real_vec_t temp_vec157 = temp_vec80 - temp_vec145;

 // temp_vec158 = 1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z)).
 real_vec_t temp_vec158 = temp_vec40 * temp_vec157;

 // temp_vec159 = (1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))).
 real_vec_t temp_vec159 = temp_vec156 + temp_vec158;

 // temp_vec160 = -0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z)).
 real_vec_t temp_vec160 = temp_vec44 * (temp_vec147 - temp_vec148);

 // temp_vec161 = (1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z))).
 real_vec_t temp_vec161 = temp_vec159 + temp_vec160;

 // temp_vec162 = (((2 / (mu(x, y, z) + mu(x, y-1, z))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z)))).
 real_vec_t temp_vec162 = temp_vec152 * temp_vec161;

 // temp_vec163 = stress_xz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x, y-1, z))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z))))).
 real_vec_t temp_vec163 = temp_vec140 + temp_vec162;

 // temp_vec164 = (stress_xz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x, y-1, z))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z)))))) * sponge(x, y, z).
 real_vec_t temp_vec164 = temp_vec163 * temp_vec91;

 // temp_vec165 = ((stress_xz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x, y-1, z))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z)))))) * sponge(x, y, z)).
 real_vec_t temp_vec165 = temp_vec164;

 // Save result to stress_xz(t+1, x, y, z):
 
 // Write aligned vector block to stress_xz at t+1, x, y, z.
context.stress_xz->writeVecNorm(temp_vec165, tv+(1/1), xv, yv, zv, __LINE__);
;

 // Read aligned vector block from stress_yz at t, x, y, z.
 real_vec_t temp_vec166 = context.stress_yz->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from vel_y at t+1, x, y, z+1.
 real_vec_t temp_vec167 = context.vel_y->readVecNorm(tv+(1/1), xv, yv, zv+(1/1), __LINE__);

 // Read aligned vector block from vel_y at t+1, x, y, z+2.
 real_vec_t temp_vec168 = context.vel_y->readVecNorm(tv+(1/1), xv, yv, zv+(2/1), __LINE__);

 // Read aligned vector block from vel_y at t+1, x, y, z-1.
 real_vec_t temp_vec169 = context.vel_y->readVecNorm(tv+(1/1), xv, yv, zv-(1/1), __LINE__);

 // Read aligned vector block from vel_z at t+1, x, y+4, z.
 real_vec_t temp_vec170 = context.vel_z->readVecNorm(tv+(1/1), xv, yv+(4/4), zv, __LINE__);

 // Construct unaligned vector block from vel_z at t+1, x, y+1, z.
 real_vec_t temp_vec171;
 // temp_vec171[0] = temp_vec68[4];  // for t+1, x, y+1, z;
 // temp_vec171[1] = temp_vec68[5];  // for t+1, x+1, y+1, z;
 // temp_vec171[2] = temp_vec68[6];  // for t+1, x+2, y+1, z;
 // temp_vec171[3] = temp_vec68[7];  // for t+1, x+3, y+1, z;
 // temp_vec171[4] = temp_vec68[8];  // for t+1, x, y+2, z;
 // temp_vec171[5] = temp_vec68[9];  // for t+1, x+1, y+2, z;
 // temp_vec171[6] = temp_vec68[10];  // for t+1, x+2, y+2, z;
 // temp_vec171[7] = temp_vec68[11];  // for t+1, x+3, y+2, z;
 // temp_vec171[8] = temp_vec68[12];  // for t+1, x, y+3, z;
 // temp_vec171[9] = temp_vec68[13];  // for t+1, x+1, y+3, z;
 // temp_vec171[10] = temp_vec68[14];  // for t+1, x+2, y+3, z;
 // temp_vec171[11] = temp_vec68[15];  // for t+1, x+3, y+3, z;
 // temp_vec171[12] = temp_vec170[0];  // for t+1, x, y+4, z;
 // temp_vec171[13] = temp_vec170[1];  // for t+1, x+1, y+4, z;
 // temp_vec171[14] = temp_vec170[2];  // for t+1, x+2, y+4, z;
 // temp_vec171[15] = temp_vec170[3];  // for t+1, x+3, y+4, z;
 // Get 4 element(s) from temp_vec170 and 12 from temp_vec68.
 real_vec_align<4>(temp_vec171, temp_vec170, temp_vec68);

 // Construct unaligned vector block from vel_z at t+1, x, y+2, z.
 real_vec_t temp_vec172;
 // temp_vec172[0] = temp_vec68[8];  // for t+1, x, y+2, z;
 // temp_vec172[1] = temp_vec68[9];  // for t+1, x+1, y+2, z;
 // temp_vec172[2] = temp_vec68[10];  // for t+1, x+2, y+2, z;
 // temp_vec172[3] = temp_vec68[11];  // for t+1, x+3, y+2, z;
 // temp_vec172[4] = temp_vec68[12];  // for t+1, x, y+3, z;
 // temp_vec172[5] = temp_vec68[13];  // for t+1, x+1, y+3, z;
 // temp_vec172[6] = temp_vec68[14];  // for t+1, x+2, y+3, z;
 // temp_vec172[7] = temp_vec68[15];  // for t+1, x+3, y+3, z;
 // temp_vec172[8] = temp_vec170[0];  // for t+1, x, y+4, z;
 // temp_vec172[9] = temp_vec170[1];  // for t+1, x+1, y+4, z;
 // temp_vec172[10] = temp_vec170[2];  // for t+1, x+2, y+4, z;
 // temp_vec172[11] = temp_vec170[3];  // for t+1, x+3, y+4, z;
 // temp_vec172[12] = temp_vec170[4];  // for t+1, x, y+5, z;
 // temp_vec172[13] = temp_vec170[5];  // for t+1, x+1, y+5, z;
 // temp_vec172[14] = temp_vec170[6];  // for t+1, x+2, y+5, z;
 // temp_vec172[15] = temp_vec170[7];  // for t+1, x+3, y+5, z;
 // Get 8 element(s) from temp_vec170 and 8 from temp_vec68.
 real_vec_align<8>(temp_vec172, temp_vec170, temp_vec68);

 // Read aligned vector block from vel_z at t+1, x, y-4, z.
 real_vec_t temp_vec173 = context.vel_z->readVecNorm(tv+(1/1), xv, yv-(4/4), zv, __LINE__);

 // Construct unaligned vector block from vel_z at t+1, x, y-1, z.
 real_vec_t temp_vec174;
 // temp_vec174[0] = temp_vec173[12];  // for t+1, x, y-1, z;
 // temp_vec174[1] = temp_vec173[13];  // for t+1, x+1, y-1, z;
 // temp_vec174[2] = temp_vec173[14];  // for t+1, x+2, y-1, z;
 // temp_vec174[3] = temp_vec173[15];  // for t+1, x+3, y-1, z;
 // temp_vec174[4] = temp_vec68[0];  // for t+1, x, y, z;
 // temp_vec174[5] = temp_vec68[1];  // for t+1, x+1, y, z;
 // temp_vec174[6] = temp_vec68[2];  // for t+1, x+2, y, z;
 // temp_vec174[7] = temp_vec68[3];  // for t+1, x+3, y, z;
 // temp_vec174[8] = temp_vec68[4];  // for t+1, x, y+1, z;
 // temp_vec174[9] = temp_vec68[5];  // for t+1, x+1, y+1, z;
 // temp_vec174[10] = temp_vec68[6];  // for t+1, x+2, y+1, z;
 // temp_vec174[11] = temp_vec68[7];  // for t+1, x+3, y+1, z;
 // temp_vec174[12] = temp_vec68[8];  // for t+1, x, y+2, z;
 // temp_vec174[13] = temp_vec68[9];  // for t+1, x+1, y+2, z;
 // temp_vec174[14] = temp_vec68[10];  // for t+1, x+2, y+2, z;
 // temp_vec174[15] = temp_vec68[11];  // for t+1, x+3, y+2, z;
 // Get 12 element(s) from temp_vec68 and 4 from temp_vec173.
 real_vec_align<12>(temp_vec174, temp_vec68, temp_vec173);

 // temp_vec175 = mu(x, y, z) + mu(x+1, y, z).
 real_vec_t temp_vec175 = temp_vec27 + temp_vec28;

 // temp_vec176 = (2 / (mu(x, y, z) + mu(x+1, y, z))).
 real_vec_t temp_vec176 = temp_vec25 / temp_vec175;

 // temp_vec177 = (2 / (mu(x, y, z) + mu(x+1, y, z))) * delta_t().
 real_vec_t temp_vec177 = temp_vec176 * temp_vec2;

 // temp_vec178 = (((2 / (mu(x, y, z) + mu(x+1, y, z))) * delta_t) / h).
 real_vec_t temp_vec178 = temp_vec177 / temp_vec3;

 // temp_vec179 = (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z)).
 real_vec_t temp_vec179 = temp_vec167 - temp_vec74;

 // temp_vec180 = 1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z)).
 real_vec_t temp_vec180 = temp_vec40 * temp_vec179;

 // temp_vec181 = -0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1)).
 real_vec_t temp_vec181 = temp_vec44 * (temp_vec168 - temp_vec169);

 // temp_vec182 = (1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))).
 real_vec_t temp_vec182 = temp_vec180 + temp_vec181;

 // temp_vec183 = (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z)).
 real_vec_t temp_vec183 = temp_vec171 - temp_vec80;

 // temp_vec184 = 1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z)).
 real_vec_t temp_vec184 = temp_vec40 * temp_vec183;

 // temp_vec185 = (1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))).
 real_vec_t temp_vec185 = temp_vec182 + temp_vec184;

 // temp_vec186 = -0.0416667 * (vel_z(t+1, x, y+2, z) - vel_z(t+1, x, y-1, z)).
 real_vec_t temp_vec186 = temp_vec44 * (temp_vec172 - temp_vec174);

 // temp_vec187 = (1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+2, z) - vel_z(t+1, x, y-1, z))).
 real_vec_t temp_vec187 = temp_vec185 + temp_vec186;

 // temp_vec188 = (((2 / (mu(x, y, z) + mu(x+1, y, z))) * delta_t) / h) * ((1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+2, z) - vel_z(t+1, x, y-1, z)))).
 real_vec_t temp_vec188 = temp_vec178 * temp_vec187;

 // temp_vec189 = stress_yz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z))) * delta_t) / h) * ((1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+2, z) - vel_z(t+1, x, y-1, z))))).
 real_vec_t temp_vec189 = temp_vec166 + temp_vec188;

 // temp_vec190 = (stress_yz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z))) * delta_t) / h) * ((1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+2, z) - vel_z(t+1, x, y-1, z)))))) * sponge(x, y, z).
 real_vec_t temp_vec190 = temp_vec189 * temp_vec91;

 // temp_vec191 = ((stress_yz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z))) * delta_t) / h) * ((1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + (1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+2, z) - vel_z(t+1, x, y-1, z)))))) * sponge(x, y, z)).
 real_vec_t temp_vec191 = temp_vec190;

 // Save result to stress_yz(t+1, x, y, z):
 
 // Write aligned vector block to stress_yz at t+1, x, y, z.
context.stress_yz->writeVecNorm(temp_vec191, tv+(1/1), xv, yv, zv, __LINE__);
;
} // vector calculation.

// Prefetches cache line(s) for entire stencil to L1 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=4 * y=4 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L1_vector(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned lambda at x, y-4, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv-(4/4), zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned lambda at x, y-4, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned lambda at x, y, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned lambda at x, y, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned lambda at x+4, y-4, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv-(4/4), zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned lambda at x+4, y-4, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned lambda at x+4, y, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned lambda at x+4, y, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x, y-4, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv-(4/4), zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x, y-4, z.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x, y, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x, y, z.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x+4, y-4, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv-(4/4), zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x+4, y-4, z.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x+4, y, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x+4, y, z.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned sponge at x, y, z.
 p = (const char*)context.sponge->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xx at t, x, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yy at t, x, y, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x-4, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y-4, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y, z-1.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y, z+1.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y, z+2.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y+4, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x+4, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x-4, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y-4, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y, z-1.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y, z+1.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y, z+2.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y+4, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x+4, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x-4, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y-4, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y, z-2.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv-(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y, z-1.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y, z+1.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y+4, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x+4, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);
}

// Prefetches cache line(s) for entire stencil to L2 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=4 * y=4 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L2_vector(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned lambda at x, y-4, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv-(4/4), zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned lambda at x, y-4, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned lambda at x, y, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned lambda at x, y, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned lambda at x+4, y-4, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv-(4/4), zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned lambda at x+4, y-4, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned lambda at x+4, y, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned lambda at x+4, y, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x, y-4, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv-(4/4), zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x, y-4, z.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x, y, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x, y, z.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x+4, y-4, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv-(4/4), zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x+4, y-4, z.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x+4, y, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x+4, y, z.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned sponge at x, y, z.
 p = (const char*)context.sponge->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xx at t, x, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yy at t, x, y, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x-4, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y-4, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y, z-1.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y, z+1.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y, z+2.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y+4, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x+4, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x-4, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y-4, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y, z-1.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y, z+1.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y, z+2.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y+4, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x+4, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x-4, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y-4, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y, z-2.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv-(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y, z-1.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y, z+1.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y+4, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x+4, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);
}

// Prefetches cache line(s) for leading edge of stencil in '+t' direction to L1 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=4 * y=4 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L1_vector_t(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned stress_xx at t, x, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yy at t, x, y, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x-4, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y-4, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y, z-1.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y, z+1.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y, z+2.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y+4, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x+4, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x-4, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y-4, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y, z-1.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y, z+1.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y, z+2.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y+4, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x+4, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x-4, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y-4, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y, z-2.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv-(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y, z-1.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y, z+1.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y+4, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x+4, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);
}

// Prefetches cache line(s) for leading edge of stencil in '+t' direction to L2 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=4 * y=4 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L2_vector_t(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned stress_xx at t, x, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yy at t, x, y, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x-4, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y-4, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y, z-1.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y, z+1.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y, z+2.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y+4, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x+4, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x-4, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y-4, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y, z-1.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y, z+1.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y, z+2.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y+4, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x+4, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x-4, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y-4, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y, z-2.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv-(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y, z-1.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y, z+1.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y+4, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x+4, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);
}

// Prefetches cache line(s) for leading edge of stencil in '+x' direction to L1 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=4 * y=4 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L1_vector_x(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned lambda at x+4, y-4, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv-(4/4), zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned lambda at x+4, y-4, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned lambda at x+4, y, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned lambda at x+4, y, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x+4, y-4, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv-(4/4), zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x+4, y-4, z.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x+4, y, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x+4, y, z.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned sponge at x, y, z.
 p = (const char*)context.sponge->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xx at t, x, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yy at t, x, y, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y-4, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y, z-1.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y, z+1.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y, z+2.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y+4, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x+4, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y-4, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y, z-1.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y, z+1.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y, z+2.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y+4, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x+4, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y-4, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y, z-2.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv-(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y, z-1.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y, z+1.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y+4, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x+4, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);
}

// Prefetches cache line(s) for leading edge of stencil in '+x' direction to L2 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=4 * y=4 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L2_vector_x(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned lambda at x+4, y-4, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv-(4/4), zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned lambda at x+4, y-4, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned lambda at x+4, y, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned lambda at x+4, y, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x+4, y-4, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv-(4/4), zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x+4, y-4, z.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x+4, y, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x+4, y, z.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned sponge at x, y, z.
 p = (const char*)context.sponge->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xx at t, x, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yy at t, x, y, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y-4, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y, z-1.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y, z+1.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y, z+2.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y+4, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x+4, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y-4, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y, z-1.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y, z+1.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y, z+2.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y+4, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x+4, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y-4, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y, z-2.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv-(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y, z-1.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y, z+1.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y+4, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x+4, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);
}

// Prefetches cache line(s) for leading edge of stencil in '+y' direction to L1 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=4 * y=4 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L1_vector_y(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned lambda at x, y, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned lambda at x, y, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned lambda at x+4, y, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned lambda at x+4, y, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x, y, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x, y, z.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x+4, y, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x+4, y, z.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned sponge at x, y, z.
 p = (const char*)context.sponge->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xx at t, x, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yy at t, x, y, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x-4, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y, z-1.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y, z+1.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y, z+2.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y+4, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x+4, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x-4, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y, z-1.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y, z+1.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y, z+2.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y+4, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x+4, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x-4, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y, z-2.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv-(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y, z-1.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y, z+1.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y+4, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x+4, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);
}

// Prefetches cache line(s) for leading edge of stencil in '+y' direction to L2 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=4 * y=4 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L2_vector_y(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned lambda at x, y, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned lambda at x, y, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned lambda at x+4, y, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned lambda at x+4, y, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x, y, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x, y, z.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x+4, y, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x+4, y, z.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned sponge at x, y, z.
 p = (const char*)context.sponge->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xx at t, x, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yy at t, x, y, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x-4, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y, z-1.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y, z+1.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y, z+2.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y+4, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x+4, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x-4, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y, z-1.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y, z+1.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y, z+2.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y+4, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x+4, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x-4, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y, z-2.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv-(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y, z-1.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y, z+1.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y+4, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x+4, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);
}

// Prefetches cache line(s) for leading edge of stencil in '+z' direction to L1 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=4 * y=4 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L1_vector_z(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned lambda at x, y-4, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned lambda at x, y, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned lambda at x+4, y-4, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned lambda at x+4, y, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x, y-4, z.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x, y, z.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x+4, y-4, z.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x+4, y, z.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned sponge at x, y, z.
 p = (const char*)context.sponge->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xx at t, x, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yy at t, x, y, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x-4, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y-4, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y, z+2.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y+4, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x+4, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x-4, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y-4, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y, z+2.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y+4, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x+4, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x-4, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y-4, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y, z+1.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y+4, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x+4, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);
}

// Prefetches cache line(s) for leading edge of stencil in '+z' direction to L2 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=4 * y=4 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L2_vector_z(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned lambda at x, y-4, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned lambda at x, y, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned lambda at x+4, y-4, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned lambda at x+4, y, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x, y-4, z.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x, y, z.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x+4, y-4, z.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x+4, y, z.
 p = (const char*)context.mu->getVecPtrNorm(xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned sponge at x, y, z.
 p = (const char*)context.sponge->getVecPtrNorm(xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xx at t, x, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yy at t, x, y, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x-4, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y-4, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y, z+2.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y+4, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x+4, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x-4, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y-4, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y, z+2.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y+4, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x+4, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x-4, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv-(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y-4, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv-(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y, z+1.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y+4, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv+(4/4), zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x+4, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv+(4/4), yv, zv, false);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);
}
};

////// Overall stencil-equations class //////
template <typename ContextClass>
struct StencilEquations_awp : public StencilEquations {

 // Stencils.
 StencilTemplate<Stencil_velocity,StencilContext_awp> stencil_velocity;
 StencilTemplate<Stencil_stress,StencilContext_awp> stencil_stress;

 StencilEquations_awp() {
name = "awp";
  stencils.push_back(&stencil_velocity);
  stencils.push_back(&stencil_stress);
 }
};
} // namespace yask.
