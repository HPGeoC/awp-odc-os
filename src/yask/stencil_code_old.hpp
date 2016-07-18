// Automatically generated code; do not edit.

////// Implementation of the 'awp' stencil //////

////// Overall stencil-context class //////
struct StencilContext_awp : public StencilContext {

 // Grids.
 Grid_TXYZ* vel_x;
 Grid_TXYZ* vel_y;
 Grid_TXYZ* vel_z;
 Grid_TXYZ* stress_xx;
 Grid_TXYZ* stress_yy;
 Grid_TXYZ* stress_zz;
 Grid_TXYZ* stress_xy;
 Grid_TXYZ* stress_xz;
 Grid_TXYZ* stress_yz;
 Grid_XYZ* lambda;
 Grid_XYZ* rho;
 Grid_XYZ* mu;
 Grid_XYZ* sponge;

 // Parameters.
 GenericGrid0d<Real>* delta_t;
 GenericGrid0d<Real>* h;

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

 void allocGrids() {
  gridPtrs.clear();
  eqGridPtrs.clear();
  vel_x = new Grid_TXYZ(dx, dy, dz, hx + px, hy + py, hz + pz, "vel_x");
  gridPtrs.push_back(vel_x);
  eqGridPtrs.push_back(vel_x);
  vel_y = new Grid_TXYZ(dx, dy, dz, hx + px, hy + py, hz + pz, "vel_y");
  gridPtrs.push_back(vel_y);
  eqGridPtrs.push_back(vel_y);
  vel_z = new Grid_TXYZ(dx, dy, dz, hx + px, hy + py, hz + pz, "vel_z");
  gridPtrs.push_back(vel_z);
  eqGridPtrs.push_back(vel_z);
  stress_xx = new Grid_TXYZ(dx, dy, dz, hx + px, hy + py, hz + pz, "stress_xx");
  gridPtrs.push_back(stress_xx);
  eqGridPtrs.push_back(stress_xx);
  stress_yy = new Grid_TXYZ(dx, dy, dz, hx + px, hy + py, hz + pz, "stress_yy");
  gridPtrs.push_back(stress_yy);
  eqGridPtrs.push_back(stress_yy);
  stress_zz = new Grid_TXYZ(dx, dy, dz, hx + px, hy + py, hz + pz, "stress_zz");
  gridPtrs.push_back(stress_zz);
  eqGridPtrs.push_back(stress_zz);
  stress_xy = new Grid_TXYZ(dx, dy, dz, hx + px, hy + py, hz + pz, "stress_xy");
  gridPtrs.push_back(stress_xy);
  eqGridPtrs.push_back(stress_xy);
  stress_xz = new Grid_TXYZ(dx, dy, dz, hx + px, hy + py, hz + pz, "stress_xz");
  gridPtrs.push_back(stress_xz);
  eqGridPtrs.push_back(stress_xz);
  stress_yz = new Grid_TXYZ(dx, dy, dz, hx + px, hy + py, hz + pz, "stress_yz");
  gridPtrs.push_back(stress_yz);
  eqGridPtrs.push_back(stress_yz);
  lambda = new Grid_XYZ(dx, dy, dz, px, py, pz, "lambda");
  gridPtrs.push_back(lambda);
  rho = new Grid_XYZ(dx, dy, dz, px, py, pz, "rho");
  gridPtrs.push_back(rho);
  mu = new Grid_XYZ(dx, dy, dz, px, py, pz, "mu");
  gridPtrs.push_back(mu);
  sponge = new Grid_XYZ(dx, dy, dz, px, py, pz, "sponge");
  gridPtrs.push_back(sponge);
 }

 void allocParams() {
  paramPtrs.clear();
  delta_t = new GenericGrid0d<Real>();
  paramPtrs.push_back(delta_t);
  h = new GenericGrid0d<Real>();
  paramPtrs.push_back(h);
 }
};

////// Stencil equation 'velocity' //////

struct Stencil_velocity {
 std::string name = "velocity";

 // 78 FP operation(s) per point:
 // vel_x(t+1, x, y, z) = ((vel_x(t, x, y, z) + ((delta_t / ((rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25 * h)) * ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))))) * sponge(x, y, z)).
 // vel_y(t+1, x, y, z) = ((vel_y(t, x, y, z) + ((delta_t / ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25 * h)) * ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))))) * sponge(x, y, z)).
 // vel_z(t+1, x, y, z) = ((vel_z(t, x, y, z) + ((delta_t / ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25 * h)) * ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))))) * sponge(x, y, z)).
 const int scalar_fp_ops = 78;
 // All grids updated by this equation.
 std::vector<RealvGridBase*> eqGridPtrs;
 void init(StencilContext_awp& context) {
  eqGridPtrs.clear();
  eqGridPtrs.push_back(context.vel_x);
  eqGridPtrs.push_back(context.vel_y);
  eqGridPtrs.push_back(context.vel_z);
 }

 // Calculate one scalar result relative to indices t, x, y, z.
 void calc_scalar(StencilContext_awp& context, idx_t t, idx_t x, idx_t y, idx_t z) {

 // temp1 = delta_t().
 Real temp1 = (*context.delta_t)();

 // temp2 = rho(x, y, z).
 Real temp2 = context.rho->readElem(x, y, z, __LINE__);

 // temp3 = rho(x, y-1, z).
 Real temp3 = context.rho->readElem(x, y-1, z, __LINE__);

 // temp4 = (rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)).
 Real temp4 = temp2 + temp3;

 // temp5 = rho(x, y, z-1).
 Real temp5 = context.rho->readElem(x, y, z-1, __LINE__);

 // temp6 = (rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)).
 Real temp6 = temp4 + temp5;

 // temp7 = (rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)).
 Real temp7 = temp6 + context.rho->readElem(x, y-1, z-1, __LINE__);

 // temp8 = 0.25.
 Real temp8 = 2.50000000000000000e-01;

 // temp9 = ((rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25 * h).
 Real temp9 = temp7 * temp8;

 // temp10 = h().
 Real temp10 = (*context.h)();

 // temp11 = ((rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25 * h).
 Real temp11 = temp9 * temp10;
 
 // temp12 = (delta_t / ((rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25 * h)).
 Real temp12 = temp1 / temp11;

 // temp13 = 1.125.
 Real temp13 = 1.12500000000000000e+00;

 // temp14 = (1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))).
 Real temp14 = temp13 * (context.stress_xx->readElem(t, x, y, z, __LINE__) - context.stress_xx->readElem(t, x-1, y, z, __LINE__));

 // temp15 = -0.0416667.
 Real temp15 = -4.16666666666666644e-02;

 // temp16 = (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))).
 Real temp16 = temp15 * (context.stress_xx->readElem(t, x+1, y, z, __LINE__) - context.stress_xx->readElem(t, x-2, y, z, __LINE__));

 // temp17 = ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))).
 Real temp17 = temp14 + temp16;

 // temp18 = stress_xy(t, x, y, z).
 Real temp18 = context.stress_xy->readElem(t, x, y, z, __LINE__);

 // temp19 = (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z)).
 Real temp19 = temp18 - context.stress_xy->readElem(t, x, y-1, z, __LINE__);

 // temp20 = (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))).
 Real temp20 = temp13 * temp19;

 // temp21 = ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))).
 Real temp21 = temp17 + temp20;

 // temp22 = (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))).
 Real temp22 = temp15 * (context.stress_xy->readElem(t, x, y+1, z, __LINE__) - context.stress_xy->readElem(t, x, y-2, z, __LINE__));

 // temp23 = ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))).
 Real temp23 = temp21 + temp22;

 // temp24 = stress_xz(t, x, y, z).
 Real temp24 = context.stress_xz->readElem(t, x, y, z, __LINE__);
 
 // temp25 = (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1)).
 Real temp25 = temp24 - context.stress_xz->readElem(t, x, y, z-1, __LINE__);

 // temp26 = (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))).
 Real temp26 = temp13 * temp25;

 // temp27 = ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))).
 Real temp27 = temp23 + temp26;

 
 // temp28 = (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2))).
 Real temp28 = temp15 * (context.stress_xz->readElem(t, x, y, z+1, __LINE__) - context.stress_xz->readElem(t, x, y, z-2, __LINE__));
 
 // temp29 = ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))).
 Real temp29 = temp27 + temp28;

 // temp30 = ((delta_t / ((rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25 * h)) * ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2))))).
 Real temp30 = temp12 * temp29; 
 
 // temp31 = (vel_x(t, x, y, z) + ((delta_t / ((rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25 * h)) * ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))))).
 Real temp31 = context.vel_x->readElem(t, x, y, z, __LINE__) + temp30;

 // temp32 = sponge(x, y, z).
 Real temp32 = context.sponge->readElem(x, y, z, __LINE__);

 // temp33 = ((vel_x(t, x, y, z) + ((delta_t / ((rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25 * h)) * ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))))) * sponge(x, y, z)).
 Real temp33 = temp31 * temp32;

 // temp34 = ((vel_x(t, x, y, z) + ((delta_t / ((rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25 * h)) * ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))))) * sponge(x, y, z)).
 Real temp34 = temp33;

 // Save result to vel_x(t+1, x, y, z):
 context.vel_x->writeElem(temp34, t+1, x, y, z, __LINE__);
 
 // temp35 = rho(x+1, y, z).
 Real temp35 = context.rho->readElem(x+1, y, z, __LINE__);

 // temp36 = (rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)).
 Real temp36 = temp2 + temp35;

 // temp37 = (rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)).
 Real temp37 = temp36 + temp5;

 // temp38 = (rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)).
 Real temp38 = temp37 + context.rho->readElem(x+1, y, z-1, __LINE__);

 // temp39 = ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25 * h).
 Real temp39 = temp38 * temp8;

 // temp40 = ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25 * h).
 Real temp40 = temp39 * temp10;

 // temp41 = (delta_t / ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25 * h)).
 Real temp41 = temp1 / temp40;

 // temp42 = (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z)).
 Real temp42 = context.stress_xy->readElem(t, x+1, y, z, __LINE__) - temp18;

 // temp43 = (1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))).
 Real temp43 = temp13 * temp42;

 // temp44 = (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))).
 Real temp44 = temp15 * (context.stress_xy->readElem(t, x+2, y, z, __LINE__) - context.stress_xy->readElem(t, x-1, y, z, __LINE__));

 // temp45 = ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))).
 Real temp45 = temp43 + temp44;

 // temp46 = (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))).
 Real temp46 = temp13 * (context.stress_yy->readElem(t, x, y+1, z, __LINE__) - context.stress_yy->readElem(t, x, y, z, __LINE__));

 // temp47 = ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))).
 Real temp47 = temp45 + temp46;

 // temp48 = (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))).
 Real temp48 = temp15 * (context.stress_yy->readElem(t, x, y+2, z, __LINE__) - context.stress_yy->readElem(t, x, y-1, z, __LINE__));

 // temp49 = ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))).
 Real temp49 = temp47 + temp48;

 // temp50 = stress_yz(t, x, y, z).
 Real temp50 = context.stress_yz->readElem(t, x, y, z, __LINE__);

 // temp51 = (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1)).
 Real temp51 = temp50 - context.stress_yz->readElem(t, x, y, z-1, __LINE__);

 // temp52 = (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))).
 Real temp52 = temp13 * temp51;

 // temp53 = ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))).
 Real temp53 = temp49 + temp52;

 // temp54 = (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2))).
 Real temp54 = temp15 * (context.stress_yz->readElem(t, x, y, z+1, __LINE__) - context.stress_yz->readElem(t, x, y, z-2, __LINE__));

 // temp55 = ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))).
 Real temp55 = temp53 + temp54;

 // temp56 = ((delta_t / ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25 * h)) * ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2))))).
 Real temp56 = temp41 * temp55;

 // temp57 = (vel_y(t, x, y, z) + ((delta_t / ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25 * h)) * ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))))).
 Real temp57 = context.vel_y->readElem(t, x, y, z, __LINE__) + temp56;

 // temp58 = ((vel_y(t, x, y, z) + ((delta_t / ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25 * h)) * ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))))) * sponge(x, y, z)).
 Real temp58 = temp57 * temp32;

 // temp59 = ((vel_y(t, x, y, z) + ((delta_t / ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25 * h)) * ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))))) * sponge(x, y, z)).
 Real temp59 = temp58;

 // Save result to vel_y(t+1, x, y, z):
 context.vel_y->writeElem(temp59, t+1, x, y, z, __LINE__);

 // temp60 = (rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)).
 Real temp60 = temp2 + temp35;

 // temp61 = (rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)).
 Real temp61 = temp60 + temp3;

 // temp62 = (rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)).
 Real temp62 = temp61 + context.rho->readElem(x+1, y-1, z, __LINE__);

 // temp63 = ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25 * h).
 Real temp63 = temp62 * temp8;

 // temp64 = ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25 * h).
 Real temp64 = temp63 * temp10;

 // temp65 = (delta_t / ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25 * h)).
 Real temp65 = temp1 / temp64;

 // temp66 = (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z)).
 Real temp66 = context.stress_xz->readElem(t, x+1, y, z, __LINE__) - temp24;

 // temp67 = (1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))).
 Real temp67 = temp13 * temp66;

 // temp68 = (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))).
 Real temp68 = temp15 * (context.stress_xz->readElem(t, x+2, y, z, __LINE__) - context.stress_xz->readElem(t, x-1, y, z, __LINE__));

 // temp69 = ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))).
 Real temp69 = temp67 + temp68;

 // temp70 = (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z)).
 Real temp70 = temp50 - context.stress_yz->readElem(t, x, y-1, z, __LINE__);

 // temp71 = (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))).
 Real temp71 = temp13 * temp70;

 // temp72 = ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))).
 Real temp72 = temp69 + temp71;

 // temp73 = (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))).
 Real temp73 = temp15 * (context.stress_yz->readElem(t, x, y+1, z, __LINE__) - context.stress_yz->readElem(t, x, y-2, z, __LINE__));

 // temp74 = ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))).
 Real temp74 = temp72 + temp73;

 // temp75 = (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))).
 Real temp75 = temp13 * (context.stress_zz->readElem(t, x, y, z+1, __LINE__) - context.stress_zz->readElem(t, x, y, z, __LINE__));

 // temp76 = ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))).
 Real temp76 = temp74 + temp75;

 // temp77 = (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1))).
 Real temp77 = temp15 * (context.stress_zz->readElem(t, x, y, z+2, __LINE__) - context.stress_zz->readElem(t, x, y, z-1, __LINE__));

 // temp78 = ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))).
 Real temp78 = temp76 + temp77;

 // temp79 = ((delta_t / ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25 * h)) * ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1))))).
 Real temp79 = temp65 * temp78;

 // temp80 = (vel_z(t, x, y, z) + ((delta_t / ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25 * h)) * ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))))).
 Real temp80 = context.vel_z->readElem(t, x, y, z, __LINE__) + temp79;

 // temp81 = ((vel_z(t, x, y, z) + ((delta_t / ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25 * h)) * ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))))) * sponge(x, y, z)).
 Real temp81 = temp80 * temp32;

 // temp82 = ((vel_z(t, x, y, z) + ((delta_t / ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25 * h)) * ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))))) * sponge(x, y, z)).
 Real temp82 = temp81;

 // Save result to vel_z(t+1, x, y, z):
 context.vel_z->writeElem(temp82, t+1, x, y, z, __LINE__);
} // scalar calculation.

 // Calculate 8 result(s) relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=8 * y=1 * z=1' vector(s).
 // Indices must be normalized, i.e., already divided by VLEN_*.
 // SIMD calculations use 44 vector block(s) created from 41 aligned vector-block(s).
 // There are 624 FP operation(s) per cluster.
 void calc_vector(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {

 // Un-normalized indices.
 idx_t t = tv;
 idx_t x = xv * 8;
 idx_t y = yv * 1;
 idx_t z = zv * 1;

 // Read aligned vector block from vel_x at t, x, y, z.
 realv temp_vec1 = context.vel_x->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from rho at x, y, z.
 realv temp_vec2 = context.rho->readVecNorm(xv, yv, zv, __LINE__);

 // Read aligned vector block from rho at x, y-1, z.
 realv temp_vec3 = context.rho->readVecNorm(xv, yv-(1/1), zv, __LINE__);

 // Read aligned vector block from rho at x, y, z-1.
 realv temp_vec4 = context.rho->readVecNorm(xv, yv, zv-(1/1), __LINE__);

 // Read aligned vector block from rho at x, y-1, z-1.
 realv temp_vec5 = context.rho->readVecNorm(xv, yv-(1/1), zv-(1/1), __LINE__);

 // Read aligned vector block from stress_xx at t, x, y, z.
 realv temp_vec6 = context.stress_xx->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from stress_xx at t, x-8, y, z.
 realv temp_vec7 = context.stress_xx->readVecNorm(tv, xv-(8/8), yv, zv, __LINE__);

 // Construct unaligned vector block from stress_xx at t, x-1, y, z.
 realv temp_vec8;
 temp_vec8[0] = temp_vec7[7];  // for t, x-1, y, z;
 temp_vec8[1] = temp_vec6[0];  // for t, x, y, z;
 temp_vec8[2] = temp_vec6[1];  // for t, x+1, y, z;
 temp_vec8[3] = temp_vec6[2];  // for t, x+2, y, z;
 temp_vec8[4] = temp_vec6[3];  // for t, x+3, y, z;
 temp_vec8[5] = temp_vec6[4];  // for t, x+4, y, z;
 temp_vec8[6] = temp_vec6[5];  // for t, x+5, y, z;
 temp_vec8[7] = temp_vec6[6];  // for t, x+6, y, z;

 // Read aligned vector block from stress_xx at t, x+8, y, z.
 realv temp_vec9 = context.stress_xx->readVecNorm(tv, xv+(8/8), yv, zv, __LINE__);

 // Construct unaligned vector block from stress_xx at t, x+1, y, z.
 realv temp_vec10;
 temp_vec10[0] = temp_vec6[1];  // for t, x+1, y, z;
 temp_vec10[1] = temp_vec6[2];  // for t, x+2, y, z;
 temp_vec10[2] = temp_vec6[3];  // for t, x+3, y, z;
 temp_vec10[3] = temp_vec6[4];  // for t, x+4, y, z;
 temp_vec10[4] = temp_vec6[5];  // for t, x+5, y, z;
 temp_vec10[5] = temp_vec6[6];  // for t, x+6, y, z;
 temp_vec10[6] = temp_vec6[7];  // for t, x+7, y, z;
 temp_vec10[7] = temp_vec9[0];  // for t, x+8, y, z;

 // Construct unaligned vector block from stress_xx at t, x-2, y, z.
 realv temp_vec11;
 temp_vec11[0] = temp_vec7[6];  // for t, x-2, y, z;
 temp_vec11[1] = temp_vec7[7];  // for t, x-1, y, z;
 temp_vec11[2] = temp_vec6[0];  // for t, x, y, z;
 temp_vec11[3] = temp_vec6[1];  // for t, x+1, y, z;
 temp_vec11[4] = temp_vec6[2];  // for t, x+2, y, z;
 temp_vec11[5] = temp_vec6[3];  // for t, x+3, y, z;
 temp_vec11[6] = temp_vec6[4];  // for t, x+4, y, z;
 temp_vec11[7] = temp_vec6[5];  // for t, x+5, y, z;

 // Read aligned vector block from stress_xy at t, x, y, z.
 realv temp_vec12 = context.stress_xy->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from stress_xy at t, x, y-1, z.
 realv temp_vec13 = context.stress_xy->readVecNorm(tv, xv, yv-(1/1), zv, __LINE__);

 // Read aligned vector block from stress_xy at t, x, y+1, z.
 realv temp_vec14 = context.stress_xy->readVecNorm(tv, xv, yv+(1/1), zv, __LINE__);

 // Read aligned vector block from stress_xy at t, x, y-2, z.
 realv temp_vec15 = context.stress_xy->readVecNorm(tv, xv, yv-(2/1), zv, __LINE__);

 // Read aligned vector block from stress_xz at t, x, y, z.
 realv temp_vec16 = context.stress_xz->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from stress_xz at t, x, y, z-1.
 realv temp_vec17 = context.stress_xz->readVecNorm(tv, xv, yv, zv-(1/1), __LINE__);

 // Read aligned vector block from stress_xz at t, x, y, z+1.
 realv temp_vec18 = context.stress_xz->readVecNorm(tv, xv, yv, zv+(1/1), __LINE__);

 // Read aligned vector block from stress_xz at t, x, y, z-2.
 realv temp_vec19 = context.stress_xz->readVecNorm(tv, xv, yv, zv-(2/1), __LINE__);

 // Read aligned vector block from sponge at x, y, z.
 realv temp_vec20 = context.sponge->readVecNorm(xv, yv, zv, __LINE__);

 // temp_vec21 = delta_t().
 realv temp_vec21 = (*context.delta_t)();

 // temp_vec22 = rho(x, y, z).
 realv temp_vec22 = temp_vec2;

 // temp_vec23 = rho(x, y-1, z).
 realv temp_vec23 = temp_vec3;

 // temp_vec24 = (rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)).
 realv temp_vec24 = temp_vec22 + temp_vec23;

 // temp_vec25 = rho(x, y, z-1).
 realv temp_vec25 = temp_vec4;

 // temp_vec26 = (rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)).
 realv temp_vec26 = temp_vec24 + temp_vec25;

 // temp_vec27 = (rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)).
 realv temp_vec27 = temp_vec26 + temp_vec5;

 // temp_vec28 = 0.25.
 realv temp_vec28 = 2.50000000000000000e-01;

 // temp_vec29 = ((rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25 * h).
 realv temp_vec29 = temp_vec27 * temp_vec28;

 // temp_vec30 = h().
 realv temp_vec30 = (*context.h)();

 // temp_vec31 = ((rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25 * h).
 realv temp_vec31 = temp_vec29 * temp_vec30;

 // temp_vec32 = (delta_t / ((rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25 * h)).
 realv temp_vec32 = temp_vec21 / temp_vec31;

 // temp_vec33 = 1.125.
 realv temp_vec33 = 1.12500000000000000e+00;

 // temp_vec34 = (1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))).
 realv temp_vec34 = temp_vec33 * (temp_vec6 - temp_vec8);

 // temp_vec35 = -0.0416667.
 realv temp_vec35 = -4.16666666666666644e-02;

 // temp_vec36 = (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))).
 realv temp_vec36 = temp_vec35 * (temp_vec10 - temp_vec11);

 // temp_vec37 = ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))).
 realv temp_vec37 = temp_vec34 + temp_vec36;

 // temp_vec38 = stress_xy(t, x, y, z).
 realv temp_vec38 = temp_vec12;

 // temp_vec39 = (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z)).
 realv temp_vec39 = temp_vec38 - temp_vec13;

 // temp_vec40 = (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))).
 realv temp_vec40 = temp_vec33 * temp_vec39;

 // temp_vec41 = ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))).
 realv temp_vec41 = temp_vec37 + temp_vec40;

 // temp_vec42 = (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))).
 realv temp_vec42 = temp_vec35 * (temp_vec14 - temp_vec15);

 // temp_vec43 = ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))).
 realv temp_vec43 = temp_vec41 + temp_vec42;

 // temp_vec44 = stress_xz(t, x, y, z).
 realv temp_vec44 = temp_vec16;

 // temp_vec45 = (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1)).
 realv temp_vec45 = temp_vec44 - temp_vec17;

 // temp_vec46 = (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))).
 realv temp_vec46 = temp_vec33 * temp_vec45;

 // temp_vec47 = ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))).
 realv temp_vec47 = temp_vec43 + temp_vec46;

 // temp_vec48 = (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2))).
 realv temp_vec48 = temp_vec35 * (temp_vec18 - temp_vec19);

 // temp_vec49 = ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))).
 realv temp_vec49 = temp_vec47 + temp_vec48;

 // temp_vec50 = ((delta_t / ((rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25 * h)) * ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2))))).
 realv temp_vec50 = temp_vec32 * temp_vec49;

 // temp_vec51 = (vel_x(t, x, y, z) + ((delta_t / ((rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25 * h)) * ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))))).
 realv temp_vec51 = temp_vec1 + temp_vec50;

 // temp_vec52 = sponge(x, y, z).
 realv temp_vec52 = temp_vec20;

 // temp_vec53 = ((vel_x(t, x, y, z) + ((delta_t / ((rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25 * h)) * ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))))) * sponge(x, y, z)).
 realv temp_vec53 = temp_vec51 * temp_vec52;

 // temp_vec54 = ((vel_x(t, x, y, z) + ((delta_t / ((rho(x, y, z) + rho(x, y-1, z) + rho(x, y, z-1) + rho(x, y-1, z-1)) * 0.25 * h)) * ((1.125 * (stress_xx(t, x, y, z) - stress_xx(t, x-1, y, z))) + (-0.0416667 * (stress_xx(t, x+1, y, z) - stress_xx(t, x-2, y, z))) + (1.125 * (stress_xy(t, x, y, z) - stress_xy(t, x, y-1, z))) + (-0.0416667 * (stress_xy(t, x, y+1, z) - stress_xy(t, x, y-2, z))) + (1.125 * (stress_xz(t, x, y, z) - stress_xz(t, x, y, z-1))) + (-0.0416667 * (stress_xz(t, x, y, z+1) - stress_xz(t, x, y, z-2)))))) * sponge(x, y, z)).
 realv temp_vec54 = temp_vec53;

 // Save result to vel_x(t+1, x, y, z):
 
 // Write aligned vector block to vel_x at t+1, x, y, z.
context.vel_x->writeVecNorm(temp_vec54, tv+(1/1), xv, yv, zv, __LINE__);
;

 // Read aligned vector block from vel_y at t, x, y, z.
 realv temp_vec55 = context.vel_y->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from rho at x+8, y, z.
 realv temp_vec56 = context.rho->readVecNorm(xv+(8/8), yv, zv, __LINE__);

 // Construct unaligned vector block from rho at x+1, y, z.
 realv temp_vec57;
 temp_vec57[0] = temp_vec2[1];  // for x+1, y, z;
 temp_vec57[1] = temp_vec2[2];  // for x+2, y, z;
 temp_vec57[2] = temp_vec2[3];  // for x+3, y, z;
 temp_vec57[3] = temp_vec2[4];  // for x+4, y, z;
 temp_vec57[4] = temp_vec2[5];  // for x+5, y, z;
 temp_vec57[5] = temp_vec2[6];  // for x+6, y, z;
 temp_vec57[6] = temp_vec2[7];  // for x+7, y, z;
 temp_vec57[7] = temp_vec56[0];  // for x+8, y, z;

 // Read aligned vector block from rho at x+8, y, z-1.
 realv temp_vec58 = context.rho->readVecNorm(xv+(8/8), yv, zv-(1/1), __LINE__);

 // Construct unaligned vector block from rho at x+1, y, z-1.
 realv temp_vec59;
 temp_vec59[0] = temp_vec4[1];  // for x+1, y, z-1;
 temp_vec59[1] = temp_vec4[2];  // for x+2, y, z-1;
 temp_vec59[2] = temp_vec4[3];  // for x+3, y, z-1;
 temp_vec59[3] = temp_vec4[4];  // for x+4, y, z-1;
 temp_vec59[4] = temp_vec4[5];  // for x+5, y, z-1;
 temp_vec59[5] = temp_vec4[6];  // for x+6, y, z-1;
 temp_vec59[6] = temp_vec4[7];  // for x+7, y, z-1;
 temp_vec59[7] = temp_vec58[0];  // for x+8, y, z-1;

 // Read aligned vector block from stress_xy at t, x+8, y, z.
 realv temp_vec60 = context.stress_xy->readVecNorm(tv, xv+(8/8), yv, zv, __LINE__);

 // Construct unaligned vector block from stress_xy at t, x+1, y, z.
 realv temp_vec61;
 temp_vec61[0] = temp_vec12[1];  // for t, x+1, y, z;
 temp_vec61[1] = temp_vec12[2];  // for t, x+2, y, z;
 temp_vec61[2] = temp_vec12[3];  // for t, x+3, y, z;
 temp_vec61[3] = temp_vec12[4];  // for t, x+4, y, z;
 temp_vec61[4] = temp_vec12[5];  // for t, x+5, y, z;
 temp_vec61[5] = temp_vec12[6];  // for t, x+6, y, z;
 temp_vec61[6] = temp_vec12[7];  // for t, x+7, y, z;
 temp_vec61[7] = temp_vec60[0];  // for t, x+8, y, z;

 // Construct unaligned vector block from stress_xy at t, x+2, y, z.
 realv temp_vec62;
 temp_vec62[0] = temp_vec12[2];  // for t, x+2, y, z;
 temp_vec62[1] = temp_vec12[3];  // for t, x+3, y, z;
 temp_vec62[2] = temp_vec12[4];  // for t, x+4, y, z;
 temp_vec62[3] = temp_vec12[5];  // for t, x+5, y, z;
 temp_vec62[4] = temp_vec12[6];  // for t, x+6, y, z;
 temp_vec62[5] = temp_vec12[7];  // for t, x+7, y, z;
 temp_vec62[6] = temp_vec60[0];  // for t, x+8, y, z;
 temp_vec62[7] = temp_vec60[1];  // for t, x+9, y, z;

 // Read aligned vector block from stress_xy at t, x-8, y, z.
 realv temp_vec63 = context.stress_xy->readVecNorm(tv, xv-(8/8), yv, zv, __LINE__);

 // Construct unaligned vector block from stress_xy at t, x-1, y, z.
 realv temp_vec64;
 temp_vec64[0] = temp_vec63[7];  // for t, x-1, y, z;
 temp_vec64[1] = temp_vec12[0];  // for t, x, y, z;
 temp_vec64[2] = temp_vec12[1];  // for t, x+1, y, z;
 temp_vec64[3] = temp_vec12[2];  // for t, x+2, y, z;
 temp_vec64[4] = temp_vec12[3];  // for t, x+3, y, z;
 temp_vec64[5] = temp_vec12[4];  // for t, x+4, y, z;
 temp_vec64[6] = temp_vec12[5];  // for t, x+5, y, z;
 temp_vec64[7] = temp_vec12[6];  // for t, x+6, y, z;

 // Read aligned vector block from stress_yy at t, x, y+1, z.
 realv temp_vec65 = context.stress_yy->readVecNorm(tv, xv, yv+(1/1), zv, __LINE__);

 // Read aligned vector block from stress_yy at t, x, y, z.
 realv temp_vec66 = context.stress_yy->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from stress_yy at t, x, y+2, z.
 realv temp_vec67 = context.stress_yy->readVecNorm(tv, xv, yv+(2/1), zv, __LINE__);

 // Read aligned vector block from stress_yy at t, x, y-1, z.
 realv temp_vec68 = context.stress_yy->readVecNorm(tv, xv, yv-(1/1), zv, __LINE__);

 // Read aligned vector block from stress_yz at t, x, y, z.
 realv temp_vec69 = context.stress_yz->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from stress_yz at t, x, y, z-1.
 realv temp_vec70 = context.stress_yz->readVecNorm(tv, xv, yv, zv-(1/1), __LINE__);

 // Read aligned vector block from stress_yz at t, x, y, z+1.
 realv temp_vec71 = context.stress_yz->readVecNorm(tv, xv, yv, zv+(1/1), __LINE__);

 // Read aligned vector block from stress_yz at t, x, y, z-2.
 realv temp_vec72 = context.stress_yz->readVecNorm(tv, xv, yv, zv-(2/1), __LINE__);

 // temp_vec73 = rho(x+1, y, z).
 realv temp_vec73 = temp_vec57;

 // temp_vec74 = (rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)).
 realv temp_vec74 = temp_vec22 + temp_vec73;

 // temp_vec75 = (rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)).
 realv temp_vec75 = temp_vec74 + temp_vec25;

 // temp_vec76 = (rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)).
 realv temp_vec76 = temp_vec75 + temp_vec59;

 // temp_vec77 = ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25 * h).
 realv temp_vec77 = temp_vec76 * temp_vec28;

 // temp_vec78 = ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25 * h).
 realv temp_vec78 = temp_vec77 * temp_vec30;

 // temp_vec79 = (delta_t / ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25 * h)).
 realv temp_vec79 = temp_vec21 / temp_vec78;

 // temp_vec80 = (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z)).
 realv temp_vec80 = temp_vec61 - temp_vec38;

 // temp_vec81 = (1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))).
 realv temp_vec81 = temp_vec33 * temp_vec80;

 // temp_vec82 = (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))).
 realv temp_vec82 = temp_vec35 * (temp_vec62 - temp_vec64);

 // temp_vec83 = ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))).
 realv temp_vec83 = temp_vec81 + temp_vec82;

 // temp_vec84 = (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))).
 realv temp_vec84 = temp_vec33 * (temp_vec65 - temp_vec66);

 // temp_vec85 = ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))).
 realv temp_vec85 = temp_vec83 + temp_vec84;

 // temp_vec86 = (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))).
 realv temp_vec86 = temp_vec35 * (temp_vec67 - temp_vec68);

 // temp_vec87 = ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))).
 realv temp_vec87 = temp_vec85 + temp_vec86;

 // temp_vec88 = stress_yz(t, x, y, z).
 realv temp_vec88 = temp_vec69;

 // temp_vec89 = (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1)).
 realv temp_vec89 = temp_vec88 - temp_vec70;

 // temp_vec90 = (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))).
 realv temp_vec90 = temp_vec33 * temp_vec89;

 // temp_vec91 = ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))).
 realv temp_vec91 = temp_vec87 + temp_vec90;

 // temp_vec92 = (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2))).
 realv temp_vec92 = temp_vec35 * (temp_vec71 - temp_vec72);

 // temp_vec93 = ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))).
 realv temp_vec93 = temp_vec91 + temp_vec92;

 // temp_vec94 = ((delta_t / ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25 * h)) * ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2))))).
 realv temp_vec94 = temp_vec79 * temp_vec93;

 // temp_vec95 = (vel_y(t, x, y, z) + ((delta_t / ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25 * h)) * ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))))).
 realv temp_vec95 = temp_vec55 + temp_vec94;

 // temp_vec96 = ((vel_y(t, x, y, z) + ((delta_t / ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25 * h)) * ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))))) * sponge(x, y, z)).
 realv temp_vec96 = temp_vec95 * temp_vec52;

 // temp_vec97 = ((vel_y(t, x, y, z) + ((delta_t / ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y, z-1) + rho(x+1, y, z-1)) * 0.25 * h)) * ((1.125 * (stress_xy(t, x+1, y, z) - stress_xy(t, x, y, z))) + (-0.0416667 * (stress_xy(t, x+2, y, z) - stress_xy(t, x-1, y, z))) + (1.125 * (stress_yy(t, x, y+1, z) - stress_yy(t, x, y, z))) + (-0.0416667 * (stress_yy(t, x, y+2, z) - stress_yy(t, x, y-1, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y, z-1))) + (-0.0416667 * (stress_yz(t, x, y, z+1) - stress_yz(t, x, y, z-2)))))) * sponge(x, y, z)).
 realv temp_vec97 = temp_vec96;

 // Save result to vel_y(t+1, x, y, z):
 
 // Write aligned vector block to vel_y at t+1, x, y, z.
context.vel_y->writeVecNorm(temp_vec97, tv+(1/1), xv, yv, zv, __LINE__);
;

 // Read aligned vector block from vel_z at t, x, y, z.
 realv temp_vec98 = context.vel_z->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from rho at x+8, y-1, z.
 realv temp_vec99 = context.rho->readVecNorm(xv+(8/8), yv-(1/1), zv, __LINE__);

 // Construct unaligned vector block from rho at x+1, y-1, z.
 realv temp_vec100;
 temp_vec100[0] = temp_vec3[1];  // for x+1, y-1, z;
 temp_vec100[1] = temp_vec3[2];  // for x+2, y-1, z;
 temp_vec100[2] = temp_vec3[3];  // for x+3, y-1, z;
 temp_vec100[3] = temp_vec3[4];  // for x+4, y-1, z;
 temp_vec100[4] = temp_vec3[5];  // for x+5, y-1, z;
 temp_vec100[5] = temp_vec3[6];  // for x+6, y-1, z;
 temp_vec100[6] = temp_vec3[7];  // for x+7, y-1, z;
 temp_vec100[7] = temp_vec99[0];  // for x+8, y-1, z;

 // Read aligned vector block from stress_xz at t, x+8, y, z.
 realv temp_vec101 = context.stress_xz->readVecNorm(tv, xv+(8/8), yv, zv, __LINE__);

 // Construct unaligned vector block from stress_xz at t, x+1, y, z.
 realv temp_vec102;
 temp_vec102[0] = temp_vec16[1];  // for t, x+1, y, z;
 temp_vec102[1] = temp_vec16[2];  // for t, x+2, y, z;
 temp_vec102[2] = temp_vec16[3];  // for t, x+3, y, z;
 temp_vec102[3] = temp_vec16[4];  // for t, x+4, y, z;
 temp_vec102[4] = temp_vec16[5];  // for t, x+5, y, z;
 temp_vec102[5] = temp_vec16[6];  // for t, x+6, y, z;
 temp_vec102[6] = temp_vec16[7];  // for t, x+7, y, z;
 temp_vec102[7] = temp_vec101[0];  // for t, x+8, y, z;

 // Construct unaligned vector block from stress_xz at t, x+2, y, z.
 realv temp_vec103;
 temp_vec103[0] = temp_vec16[2];  // for t, x+2, y, z;
 temp_vec103[1] = temp_vec16[3];  // for t, x+3, y, z;
 temp_vec103[2] = temp_vec16[4];  // for t, x+4, y, z;
 temp_vec103[3] = temp_vec16[5];  // for t, x+5, y, z;
 temp_vec103[4] = temp_vec16[6];  // for t, x+6, y, z;
 temp_vec103[5] = temp_vec16[7];  // for t, x+7, y, z;
 temp_vec103[6] = temp_vec101[0];  // for t, x+8, y, z;
 temp_vec103[7] = temp_vec101[1];  // for t, x+9, y, z;

 // Read aligned vector block from stress_xz at t, x-8, y, z.
 realv temp_vec104 = context.stress_xz->readVecNorm(tv, xv-(8/8), yv, zv, __LINE__);

 // Construct unaligned vector block from stress_xz at t, x-1, y, z.
 realv temp_vec105;
 temp_vec105[0] = temp_vec104[7];  // for t, x-1, y, z;
 temp_vec105[1] = temp_vec16[0];  // for t, x, y, z;
 temp_vec105[2] = temp_vec16[1];  // for t, x+1, y, z;
 temp_vec105[3] = temp_vec16[2];  // for t, x+2, y, z;
 temp_vec105[4] = temp_vec16[3];  // for t, x+3, y, z;
 temp_vec105[5] = temp_vec16[4];  // for t, x+4, y, z;
 temp_vec105[6] = temp_vec16[5];  // for t, x+5, y, z;
 temp_vec105[7] = temp_vec16[6];  // for t, x+6, y, z;

 // Read aligned vector block from stress_yz at t, x, y-1, z.
 realv temp_vec106 = context.stress_yz->readVecNorm(tv, xv, yv-(1/1), zv, __LINE__);

 // Read aligned vector block from stress_yz at t, x, y+1, z.
 realv temp_vec107 = context.stress_yz->readVecNorm(tv, xv, yv+(1/1), zv, __LINE__);

 // Read aligned vector block from stress_yz at t, x, y-2, z.
 realv temp_vec108 = context.stress_yz->readVecNorm(tv, xv, yv-(2/1), zv, __LINE__);

 // Read aligned vector block from stress_zz at t, x, y, z+1.
 realv temp_vec109 = context.stress_zz->readVecNorm(tv, xv, yv, zv+(1/1), __LINE__);

 // Read aligned vector block from stress_zz at t, x, y, z.
 realv temp_vec110 = context.stress_zz->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from stress_zz at t, x, y, z+2.
 realv temp_vec111 = context.stress_zz->readVecNorm(tv, xv, yv, zv+(2/1), __LINE__);

 // Read aligned vector block from stress_zz at t, x, y, z-1.
 realv temp_vec112 = context.stress_zz->readVecNorm(tv, xv, yv, zv-(1/1), __LINE__);

 // temp_vec113 = (rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)).
 realv temp_vec113 = temp_vec22 + temp_vec73;

 // temp_vec114 = (rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)).
 realv temp_vec114 = temp_vec113 + temp_vec23;

 // temp_vec115 = (rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)).
 realv temp_vec115 = temp_vec114 + temp_vec100;

 // temp_vec116 = ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25 * h).
 realv temp_vec116 = temp_vec115 * temp_vec28;

 // temp_vec117 = ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25 * h).
 realv temp_vec117 = temp_vec116 * temp_vec30;

 // temp_vec118 = (delta_t / ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25 * h)).
 realv temp_vec118 = temp_vec21 / temp_vec117;

 // temp_vec119 = (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z)).
 realv temp_vec119 = temp_vec102 - temp_vec44;

 // temp_vec120 = (1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))).
 realv temp_vec120 = temp_vec33 * temp_vec119;

 // temp_vec121 = (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))).
 realv temp_vec121 = temp_vec35 * (temp_vec103 - temp_vec105);

 // temp_vec122 = ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))).
 realv temp_vec122 = temp_vec120 + temp_vec121;

 // temp_vec123 = (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z)).
 realv temp_vec123 = temp_vec88 - temp_vec106;

 // temp_vec124 = (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))).
 realv temp_vec124 = temp_vec33 * temp_vec123;

 // temp_vec125 = ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))).
 realv temp_vec125 = temp_vec122 + temp_vec124;

 // temp_vec126 = (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))).
 realv temp_vec126 = temp_vec35 * (temp_vec107 - temp_vec108);

 // temp_vec127 = ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))).
 realv temp_vec127 = temp_vec125 + temp_vec126;

 // temp_vec128 = (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))).
 realv temp_vec128 = temp_vec33 * (temp_vec109 - temp_vec110);

 // temp_vec129 = ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))).
 realv temp_vec129 = temp_vec127 + temp_vec128;

 // temp_vec130 = (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1))).
 realv temp_vec130 = temp_vec35 * (temp_vec111 - temp_vec112);

 // temp_vec131 = ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))).
 realv temp_vec131 = temp_vec129 + temp_vec130;

 // temp_vec132 = ((delta_t / ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25 * h)) * ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1))))).
 realv temp_vec132 = temp_vec118 * temp_vec131;

 // temp_vec133 = (vel_z(t, x, y, z) + ((delta_t / ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25 * h)) * ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))))).
 realv temp_vec133 = temp_vec98 + temp_vec132;

 // temp_vec134 = ((vel_z(t, x, y, z) + ((delta_t / ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25 * h)) * ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))))) * sponge(x, y, z)).
 realv temp_vec134 = temp_vec133 * temp_vec52;

 // temp_vec135 = ((vel_z(t, x, y, z) + ((delta_t / ((rho(x, y, z) + rho(x+1, y, z) + rho(x, y-1, z) + rho(x+1, y-1, z)) * 0.25 * h)) * ((1.125 * (stress_xz(t, x+1, y, z) - stress_xz(t, x, y, z))) + (-0.0416667 * (stress_xz(t, x+2, y, z) - stress_xz(t, x-1, y, z))) + (1.125 * (stress_yz(t, x, y, z) - stress_yz(t, x, y-1, z))) + (-0.0416667 * (stress_yz(t, x, y+1, z) - stress_yz(t, x, y-2, z))) + (1.125 * (stress_zz(t, x, y, z+1) - stress_zz(t, x, y, z))) + (-0.0416667 * (stress_zz(t, x, y, z+2) - stress_zz(t, x, y, z-1)))))) * sponge(x, y, z)).
 realv temp_vec135 = temp_vec134;

 // Save result to vel_z(t+1, x, y, z):
 
 // Write aligned vector block to vel_z at t+1, x, y, z.
context.vel_z->writeVecNorm(temp_vec135, tv+(1/1), xv, yv, zv, __LINE__);
;
} // vector calculation.

// Prefetches cache line(s) for entire stencil to L1 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=8 * y=1 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L1_vector(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned rho at x, y-1, z-1.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv-(1/1), zv-(1/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned rho at x, y-1, z.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv-(1/1), zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned rho at x, y, z-1.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv, zv-(1/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned rho at x, y, z.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned rho at x+8, y-1, z.
 p = (const char*)context.rho->getVecPtrNorm(xv+(8/8), yv-(1/1), zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned rho at x+8, y, z-1.
 p = (const char*)context.rho->getVecPtrNorm(xv+(8/8), yv, zv-(1/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned rho at x+8, y, z.
 p = (const char*)context.rho->getVecPtrNorm(xv+(8/8), yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned sponge at x, y, z.
 p = (const char*)context.sponge->getVecPtrNorm(xv, yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xx at t, x-8, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv-(8/8), yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xx at t, x, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xx at t, x+8, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv+(8/8), yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x-8, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv-(8/8), yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x, y-2, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv-(2/1), zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x, y-1, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv-(1/1), zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x, y+1, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv+(1/1), zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x+8, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv+(8/8), yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x-8, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv-(8/8), yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z-2.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv-(2/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z-1.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv-(1/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z+1.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv+(1/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x+8, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv+(8/8), yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yy at t, x, y-1, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv-(1/1), zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yy at t, x, y, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yy at t, x, y+1, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv+(1/1), zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yy at t, x, y+2, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv+(2/1), zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y-2, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv-(2/1), zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y-1, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv-(1/1), zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z-2.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv-(2/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z-1.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv-(1/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z+1.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv+(1/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y+1, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv+(1/1), zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z-1.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv-(1/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z+1.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv+(1/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z+2.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv+(2/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t, x, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t, x, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t, x, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);
}

// Prefetches cache line(s) for entire stencil to L2 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=8 * y=1 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L2_vector(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned rho at x, y-1, z-1.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv-(1/1), zv-(1/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned rho at x, y-1, z.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv-(1/1), zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned rho at x, y, z-1.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv, zv-(1/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned rho at x, y, z.
 p = (const char*)context.rho->getVecPtrNorm(xv, yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned rho at x+8, y-1, z.
 p = (const char*)context.rho->getVecPtrNorm(xv+(8/8), yv-(1/1), zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned rho at x+8, y, z-1.
 p = (const char*)context.rho->getVecPtrNorm(xv+(8/8), yv, zv-(1/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned rho at x+8, y, z.
 p = (const char*)context.rho->getVecPtrNorm(xv+(8/8), yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned sponge at x, y, z.
 p = (const char*)context.sponge->getVecPtrNorm(xv, yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xx at t, x-8, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv-(8/8), yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xx at t, x, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xx at t, x+8, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv+(8/8), yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x-8, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv-(8/8), yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x, y-2, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv-(2/1), zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x, y-1, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv-(1/1), zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x, y+1, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv+(1/1), zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x+8, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv+(8/8), yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x-8, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv-(8/8), yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z-2.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv-(2/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z-1.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv-(1/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z+1.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv+(1/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x+8, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv+(8/8), yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yy at t, x, y-1, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv-(1/1), zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yy at t, x, y, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yy at t, x, y+1, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv+(1/1), zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yy at t, x, y+2, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv+(2/1), zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y-2, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv-(2/1), zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y-1, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv-(1/1), zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z-2.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv-(2/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z-1.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv-(1/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z+1.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv+(1/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y+1, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv+(1/1), zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z-1.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv-(1/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z+1.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv+(1/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z+2.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv+(2/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t, x, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t, x, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t, x, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);
}
};

////// Stencil equation 'stress' //////

struct Stencil_stress {
 std::string name = "stress";

 // 104 FP operation(s) per point:
 // stress_xx(t+1, x, y, z) = ((stress_xx(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))))) * sponge(x, y, z)).
 // stress_yy(t+1, x, y, z) = ((stress_yy(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))))) * sponge(x, y, z)).
 // stress_zz(t+1, x, y, z) = ((stress_zz(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))))) * sponge(x, y, z)).
 // stress_xy(t+1, x, y, z) = ((stress_xy(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z))))))) * sponge(x, y, z)).
 // stress_xz(t+1, x, y, z) = ((stress_xz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z))))))) * sponge(x, y, z)).
 // stress_yz(t+1, x, y, z) = ((stress_yz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y-1, z))))))) * sponge(x, y, z)).
 const int scalar_fp_ops = 104;
 // All grids updated by this equation.
 std::vector<RealvGridBase*> eqGridPtrs;
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
 Real temp1 = (*context.delta_t)();

 // temp2 = h().
 Real temp2 = (*context.h)();

 // temp3 = (delta_t / h).
 Real temp3 = temp1 / temp2;

 // temp4 = 2.
 Real temp4 = 2.00000000000000000e+00;

 // temp5 = 8.
 Real temp5 = 8.00000000000000000e+00;

 // temp6 = (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1)).
 Real temp6 = context.mu->readElem(x, y, z, __LINE__) + context.mu->readElem(x+1, y, z, __LINE__);

 // temp7 = (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1)).
 Real temp7 = temp6 + context.mu->readElem(x, y-1, z, __LINE__);

 // temp8 = (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1)).
 Real temp8 = temp7 + context.mu->readElem(x+1, y-1, z, __LINE__);

 // temp9 = (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1)).
 Real temp9 = temp8 + context.mu->readElem(x, y, z-1, __LINE__);

 // temp10 = (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1)).
 Real temp10 = temp9 + context.mu->readElem(x+1, y, z-1, __LINE__);

 // temp11 = (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1)).
 Real temp11 = temp10 + context.mu->readElem(x, y-1, z-1, __LINE__);

 // temp12 = (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1)).
 Real temp12 = temp11 + context.mu->readElem(x+1, y-1, z-1, __LINE__);

 // temp13 = (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))).
 Real temp13 = temp5 / temp12;

 // temp14 = (2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))).
 Real temp14 = temp4 * temp13;

 // temp15 = 1.125.
 Real temp15 = 1.12500000000000000e+00;

 // temp16 = vel_x(t+1, x, y, z).
 Real temp16 = context.vel_x->readElem(t+1, x, y, z, __LINE__);

 // temp17 = (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z)).
 Real temp17 = context.vel_x->readElem(t+1, x+1, y, z, __LINE__) - temp16;

 // temp18 = (1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))).
 Real temp18 = temp15 * temp17;

 // temp19 = -0.0416667.
 Real temp19 = -4.16666666666666644e-02;

 // temp20 = (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))).
 Real temp20 = temp19 * (context.vel_x->readElem(t+1, x+2, y, z, __LINE__) - context.vel_x->readElem(t+1, x-1, y, z, __LINE__));

 // temp21 = ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 Real temp21 = temp18 + temp20;

 // temp22 = vel_y(t+1, x, y, z).
 Real temp22 = context.vel_y->readElem(t+1, x, y, z, __LINE__);

 // temp23 = (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z)).
 Real temp23 = temp22 - context.vel_y->readElem(t+1, x, y-1, z, __LINE__);

 // temp24 = (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))).
 Real temp24 = temp15 * temp23;

 // temp25 = (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))).
 Real temp25 = temp19 * (context.vel_y->readElem(t+1, x, y+1, z, __LINE__) - context.vel_y->readElem(t+1, x, y-2, z, __LINE__));

 // temp26 = ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))).
 Real temp26 = temp24 + temp25;

 // temp27 = ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 Real temp27 = temp21 + temp26;

 // temp28 = vel_z(t+1, x, y, z).
 Real temp28 = context.vel_z->readElem(t+1, x, y, z, __LINE__);

 // temp29 = (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1)).
 Real temp29 = temp28 - context.vel_z->readElem(t+1, x, y, z-1, __LINE__);

 // temp30 = (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))).
 Real temp30 = temp15 * temp29;

 // temp31 = (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))).
 Real temp31 = temp19 * (context.vel_z->readElem(t+1, x, y, z+1, __LINE__) - context.vel_z->readElem(t+1, x, y, z-2, __LINE__));

 // temp32 = ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))).
 Real temp32 = temp30 + temp31;

 // temp33 = ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 Real temp33 = temp27 + temp32;

 // temp34 = ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 Real temp34 = temp33 + temp26;

 // temp35 = ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 Real temp35 = temp34 + temp32;

 // temp36 = ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 Real temp36 = temp35 + temp26;

 // temp37 = ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 Real temp37 = temp36 + temp32;

 // temp38 = (2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))).
 Real temp38 = temp14 * temp37;

 // temp39 = (8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))).
 Real temp39 = temp5 / (context.lambda->readElem(x, y, z, __LINE__) + context.lambda->readElem(x+1, y, z, __LINE__) + context.lambda->readElem(x, y-1, z, __LINE__) + context.lambda->readElem(x+1, y-1, z, __LINE__) + context.lambda->readElem(x, y, z-1, __LINE__) + context.lambda->readElem(x+1, y, z-1, __LINE__) + context.lambda->readElem(x, y-1, z-1, __LINE__) + context.lambda->readElem(x+1, y-1, z-1, __LINE__));

 // temp40 = ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))).
 Real temp40 = temp39 * temp37;

 // temp41 = ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))).
 Real temp41 = temp38 + temp40;

 // temp42 = ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))))).
 Real temp42 = temp3 * temp41;

 // temp43 = (stress_xx(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))))).
 Real temp43 = context.stress_xx->readElem(t, x, y, z, __LINE__) + temp42;

 // temp44 = sponge(x, y, z).
 Real temp44 = context.sponge->readElem(x, y, z, __LINE__);

 // temp45 = ((stress_xx(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))))) * sponge(x, y, z)).
 Real temp45 = temp43 * temp44;

 // temp46 = ((stress_xx(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))))) * sponge(x, y, z)).
 Real temp46 = temp45;

 // Save result to stress_xx(t+1, x, y, z):
 context.stress_xx->writeElem(temp46, t+1, x, y, z, __LINE__);

 // temp47 = (2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))).
 Real temp47 = temp4 * temp13;

 // temp48 = (2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))).
 Real temp48 = temp47 * temp26;

 // temp49 = ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))).
 Real temp49 = temp39 * temp37;

 // temp50 = ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))).
 Real temp50 = temp48 + temp49;

 // temp51 = ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))))).
 Real temp51 = temp3 * temp50;

 // temp52 = (stress_yy(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))))).
 Real temp52 = context.stress_yy->readElem(t, x, y, z, __LINE__) + temp51;

 // temp53 = ((stress_yy(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))))) * sponge(x, y, z)).
 Real temp53 = temp52 * temp44;

 // temp54 = ((stress_yy(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))))) * sponge(x, y, z)).
 Real temp54 = temp53;

 // Save result to stress_yy(t+1, x, y, z):
 context.stress_yy->writeElem(temp54, t+1, x, y, z, __LINE__);

 // temp55 = (2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 Real temp55 = temp4 * temp13;

 // temp56 = (2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 Real temp56 = temp55 * temp32;

 // temp57 = ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))).
 Real temp57 = temp39 * temp37;

 // temp58 = ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))).
 Real temp58 = temp56 + temp57;

 // temp59 = ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))))).
 Real temp59 = temp3 * temp58;

 // temp60 = (stress_zz(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))))).
 Real temp60 = context.stress_zz->readElem(t, x, y, z, __LINE__) + temp59;

 // temp61 = ((stress_zz(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))))) * sponge(x, y, z)).
 Real temp61 = temp60 * temp44;

 // temp62 = ((stress_zz(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))))) * sponge(x, y, z)).
 Real temp62 = temp61;

 // Save result to stress_zz(t+1, x, y, z):
 context.stress_zz->writeElem(temp62, t+1, x, y, z, __LINE__);

 // temp63 = (2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))).
 Real temp63 = temp4 / temp12;

 // temp64 = ((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t).
 Real temp64 = temp63 * temp1;

 // temp65 = (((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h).
 Real temp65 = temp64 / temp2;

 // temp66 = (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z)).
 Real temp66 = context.vel_x->readElem(t+1, x, y+1, z, __LINE__) - temp16;

 // temp67 = (1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))).
 Real temp67 = temp15 * temp66;

 // temp68 = (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))).
 Real temp68 = temp19 * (context.vel_x->readElem(t+1, x, y+2, z, __LINE__) - context.vel_x->readElem(t+1, x, y-1, z, __LINE__));

 // temp69 = ((1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z))))).
 Real temp69 = temp67 + temp68;

 // temp70 = (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z)).
 Real temp70 = temp22 - context.vel_y->readElem(t+1, x-1, y, z, __LINE__);

 // temp71 = (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))).
 Real temp71 = temp15 * temp70;

 // temp72 = (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z))).
 Real temp72 = temp19 * (context.vel_y->readElem(t+1, x+1, y, z, __LINE__) - context.vel_y->readElem(t+1, x-2, y, z, __LINE__));

 // temp73 = ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z)))).
 Real temp73 = temp71 + temp72;

 // temp74 = ((1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z))))).
 Real temp74 = temp69 + temp73;

 // temp75 = ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z)))))).
 Real temp75 = temp65 * temp74;

 // temp76 = (stress_xy(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z))))))).
 Real temp76 = context.stress_xy->readElem(t, x, y, z, __LINE__) + temp75;

 // temp77 = ((stress_xy(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z))))))) * sponge(x, y, z)).
 Real temp77 = temp76 * temp44;

 // temp78 = ((stress_xy(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z))))))) * sponge(x, y, z)).
 Real temp78 = temp77;

 // Save result to stress_xy(t+1, x, y, z):
 context.stress_xy->writeElem(temp78, t+1, x, y, z, __LINE__);

 // temp79 = (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z)).
 Real temp79 = context.vel_x->readElem(t+1, x, y, z+1, __LINE__) - temp16;

 // temp80 = (1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))).
 Real temp80 = temp15 * temp79;

 // temp81 = (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))).
 Real temp81 = temp19 * (context.vel_x->readElem(t+1, x, y, z+2, __LINE__) - context.vel_x->readElem(t+1, x, y, z-1, __LINE__));

 // temp82 = ((1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z))))).
 Real temp82 = temp80 + temp81;

 // temp83 = (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z)).
 Real temp83 = temp28 - context.vel_z->readElem(t+1, x-1, y, z, __LINE__);

 // temp84 = (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))).
 Real temp84 = temp15 * temp83;

 // temp85 = (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z))).
 Real temp85 = temp19 * (context.vel_z->readElem(t+1, x+1, y, z, __LINE__) - context.vel_z->readElem(t+1, x-2, y, z, __LINE__));

 // temp86 = ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z)))).
 Real temp86 = temp84 + temp85;

 // temp87 = ((1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z))))).
 Real temp87 = temp82 + temp86;

                if(x == 17 && y == 20 && z == 23)
                {
                  std::cout << "strain * (dh/0.5) " << temp87 << std::endl;
                }

 
 // temp88 = ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z)))))).
 Real temp88 = temp65 * temp87;

                if(x == 17 && y == 20 && z == 23)
                {
                  std::cout << "adding " << temp88 << std::endl;
                }

 
 // temp89 = (stress_xz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z))))))).
 Real temp89 = context.stress_xz->readElem(t, x, y, z, __LINE__) + temp88;

 // temp90 = ((stress_xz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z))))))) * sponge(x, y, z)).
 Real temp90 = temp89 * temp44;

 // temp91 = ((stress_xz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z))))))) * sponge(x, y, z)).
 Real temp91 = temp90;

 // Save result to stress_xz(t+1, x, y, z):
 context.stress_xz->writeElem(temp91, t+1, x, y, z, __LINE__);

 // temp92 = (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z)).
 Real temp92 = context.vel_y->readElem(t+1, x, y, z+1, __LINE__) - temp22;

 // temp93 = (1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))).
 Real temp93 = temp15 * temp92;

 // temp94 = (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))).
 Real temp94 = temp19 * (context.vel_y->readElem(t+1, x, y, z+2, __LINE__) - context.vel_y->readElem(t+1, x, y, z-1, __LINE__));

 // temp95 = ((1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y-1, z))))).
 Real temp95 = temp93 + temp94;

 // temp96 = vel_z(t+1, x, y+1, z).
 Real temp96 = context.vel_z->readElem(t+1, x, y+1, z, __LINE__);

 // temp97 = (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z)).
 Real temp97 = temp96 - temp28;

 // temp98 = (1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))).
 Real temp98 = temp15 * temp97;

 // temp99 = (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y-1, z)).
 Real temp99 = temp96 - context.vel_z->readElem(t+1, x, y-1, z, __LINE__);

 // temp100 = (-0.0416667 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y-1, z))).
 Real temp100 = temp19 * temp99;

 // temp101 = ((1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y-1, z)))).
 Real temp101 = temp98 + temp100;

 // temp102 = ((1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y-1, z))))).
 Real temp102 = temp95 + temp101;

 // temp103 = ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y-1, z)))))).
 Real temp103 = temp65 * temp102;

 // temp104 = (stress_yz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y-1, z))))))).
 Real temp104 = context.stress_yz->readElem(t, x, y, z, __LINE__) + temp103;

 // temp105 = ((stress_yz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y-1, z))))))) * sponge(x, y, z)).
 Real temp105 = temp104 * temp44;

 // temp106 = ((stress_yz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y-1, z))))))) * sponge(x, y, z)).
 Real temp106 = temp105;

 // Save result to stress_yz(t+1, x, y, z):
 context.stress_yz->writeElem(temp106, t+1, x, y, z, __LINE__);
} // scalar calculation.

 // Calculate 8 result(s) relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=8 * y=1 * z=1' vector(s).
 // Indices must be normalized, i.e., already divided by VLEN_*.
 // SIMD calculations use 52 vector block(s) created from 49 aligned vector-block(s).
 // There are 832 FP operation(s) per cluster.
 void calc_vector(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {

 // Un-normalized indices.
 idx_t t = tv;
 idx_t x = xv * 8;
 idx_t y = yv * 1;
 idx_t z = zv * 1;

 // Read aligned vector block from stress_xx at t, x, y, z.
 realv temp_vec1 = context.stress_xx->readVecNorm(tv, xv, yv, zv, __LINE__);

 // temp_vec2 = delta_t().
 realv temp_vec2 = (*context.delta_t)();

 // temp_vec3 = h().
 realv temp_vec3 = (*context.h)();

 // temp_vec4 = (delta_t / h).
 realv temp_vec4 = temp_vec2 / temp_vec3;

 // Read aligned vector block from mu at x, y, z.
 realv temp_vec5 = context.mu->readVecNorm(xv, yv, zv, __LINE__);

 // Read aligned vector block from mu at x+8, y, z.
 realv temp_vec6 = context.mu->readVecNorm(xv+(8/8), yv, zv, __LINE__);

 // Construct unaligned vector block from mu at x+1, y, z.
 realv temp_vec7;
 temp_vec7[0] = temp_vec5[1];  // for x+1, y, z;
 temp_vec7[1] = temp_vec5[2];  // for x+2, y, z;
 temp_vec7[2] = temp_vec5[3];  // for x+3, y, z;
 temp_vec7[3] = temp_vec5[4];  // for x+4, y, z;
 temp_vec7[4] = temp_vec5[5];  // for x+5, y, z;
 temp_vec7[5] = temp_vec5[6];  // for x+6, y, z;
 temp_vec7[6] = temp_vec5[7];  // for x+7, y, z;
 temp_vec7[7] = temp_vec6[0];  // for x+8, y, z;

 // Read aligned vector block from mu at x, y-1, z.
 realv temp_vec8 = context.mu->readVecNorm(xv, yv-(1/1), zv, __LINE__);

 // Read aligned vector block from mu at x+8, y-1, z.
 realv temp_vec9 = context.mu->readVecNorm(xv+(8/8), yv-(1/1), zv, __LINE__);

 // Construct unaligned vector block from mu at x+1, y-1, z.
 realv temp_vec10;
 temp_vec10[0] = temp_vec8[1];  // for x+1, y-1, z;
 temp_vec10[1] = temp_vec8[2];  // for x+2, y-1, z;
 temp_vec10[2] = temp_vec8[3];  // for x+3, y-1, z;
 temp_vec10[3] = temp_vec8[4];  // for x+4, y-1, z;
 temp_vec10[4] = temp_vec8[5];  // for x+5, y-1, z;
 temp_vec10[5] = temp_vec8[6];  // for x+6, y-1, z;
 temp_vec10[6] = temp_vec8[7];  // for x+7, y-1, z;
 temp_vec10[7] = temp_vec9[0];  // for x+8, y-1, z;

 // Read aligned vector block from mu at x, y, z-1.
 realv temp_vec11 = context.mu->readVecNorm(xv, yv, zv-(1/1), __LINE__);

 // Read aligned vector block from mu at x+8, y, z-1.
 realv temp_vec12 = context.mu->readVecNorm(xv+(8/8), yv, zv-(1/1), __LINE__);

 // Construct unaligned vector block from mu at x+1, y, z-1.
 realv temp_vec13;
 temp_vec13[0] = temp_vec11[1];  // for x+1, y, z-1;
 temp_vec13[1] = temp_vec11[2];  // for x+2, y, z-1;
 temp_vec13[2] = temp_vec11[3];  // for x+3, y, z-1;
 temp_vec13[3] = temp_vec11[4];  // for x+4, y, z-1;
 temp_vec13[4] = temp_vec11[5];  // for x+5, y, z-1;
 temp_vec13[5] = temp_vec11[6];  // for x+6, y, z-1;
 temp_vec13[6] = temp_vec11[7];  // for x+7, y, z-1;
 temp_vec13[7] = temp_vec12[0];  // for x+8, y, z-1;

 // Read aligned vector block from mu at x, y-1, z-1.
 realv temp_vec14 = context.mu->readVecNorm(xv, yv-(1/1), zv-(1/1), __LINE__);

 // Read aligned vector block from mu at x+8, y-1, z-1.
 realv temp_vec15 = context.mu->readVecNorm(xv+(8/8), yv-(1/1), zv-(1/1), __LINE__);

 // Construct unaligned vector block from mu at x+1, y-1, z-1.
 realv temp_vec16;
 temp_vec16[0] = temp_vec14[1];  // for x+1, y-1, z-1;
 temp_vec16[1] = temp_vec14[2];  // for x+2, y-1, z-1;
 temp_vec16[2] = temp_vec14[3];  // for x+3, y-1, z-1;
 temp_vec16[3] = temp_vec14[4];  // for x+4, y-1, z-1;
 temp_vec16[4] = temp_vec14[5];  // for x+5, y-1, z-1;
 temp_vec16[5] = temp_vec14[6];  // for x+6, y-1, z-1;
 temp_vec16[6] = temp_vec14[7];  // for x+7, y-1, z-1;
 temp_vec16[7] = temp_vec15[0];  // for x+8, y-1, z-1;

 // Read aligned vector block from vel_x at t+1, x, y, z.
 realv temp_vec17 = context.vel_x->readVecNorm(tv+(1/1), xv, yv, zv, __LINE__);

 // Read aligned vector block from vel_x at t+1, x+8, y, z.
 realv temp_vec18 = context.vel_x->readVecNorm(tv+(1/1), xv+(8/8), yv, zv, __LINE__);

 // Construct unaligned vector block from vel_x at t+1, x+1, y, z.
 realv temp_vec19;
 temp_vec19[0] = temp_vec17[1];  // for t+1, x+1, y, z;
 temp_vec19[1] = temp_vec17[2];  // for t+1, x+2, y, z;
 temp_vec19[2] = temp_vec17[3];  // for t+1, x+3, y, z;
 temp_vec19[3] = temp_vec17[4];  // for t+1, x+4, y, z;
 temp_vec19[4] = temp_vec17[5];  // for t+1, x+5, y, z;
 temp_vec19[5] = temp_vec17[6];  // for t+1, x+6, y, z;
 temp_vec19[6] = temp_vec17[7];  // for t+1, x+7, y, z;
 temp_vec19[7] = temp_vec18[0];  // for t+1, x+8, y, z;

 // Construct unaligned vector block from vel_x at t+1, x+2, y, z.
 realv temp_vec20;
 temp_vec20[0] = temp_vec17[2];  // for t+1, x+2, y, z;
 temp_vec20[1] = temp_vec17[3];  // for t+1, x+3, y, z;
 temp_vec20[2] = temp_vec17[4];  // for t+1, x+4, y, z;
 temp_vec20[3] = temp_vec17[5];  // for t+1, x+5, y, z;
 temp_vec20[4] = temp_vec17[6];  // for t+1, x+6, y, z;
 temp_vec20[5] = temp_vec17[7];  // for t+1, x+7, y, z;
 temp_vec20[6] = temp_vec18[0];  // for t+1, x+8, y, z;
 temp_vec20[7] = temp_vec18[1];  // for t+1, x+9, y, z;

 // Read aligned vector block from vel_x at t+1, x-8, y, z.
 realv temp_vec21 = context.vel_x->readVecNorm(tv+(1/1), xv-(8/8), yv, zv, __LINE__);

 // Construct unaligned vector block from vel_x at t+1, x-1, y, z.
 realv temp_vec22;
 temp_vec22[0] = temp_vec21[7];  // for t+1, x-1, y, z;
 temp_vec22[1] = temp_vec17[0];  // for t+1, x, y, z;
 temp_vec22[2] = temp_vec17[1];  // for t+1, x+1, y, z;
 temp_vec22[3] = temp_vec17[2];  // for t+1, x+2, y, z;
 temp_vec22[4] = temp_vec17[3];  // for t+1, x+3, y, z;
 temp_vec22[5] = temp_vec17[4];  // for t+1, x+4, y, z;
 temp_vec22[6] = temp_vec17[5];  // for t+1, x+5, y, z;
 temp_vec22[7] = temp_vec17[6];  // for t+1, x+6, y, z;

 // Read aligned vector block from vel_y at t+1, x, y, z.
 realv temp_vec23 = context.vel_y->readVecNorm(tv+(1/1), xv, yv, zv, __LINE__);

 // Read aligned vector block from vel_y at t+1, x, y-1, z.
 realv temp_vec24 = context.vel_y->readVecNorm(tv+(1/1), xv, yv-(1/1), zv, __LINE__);

 // Read aligned vector block from vel_y at t+1, x, y+1, z.
 realv temp_vec25 = context.vel_y->readVecNorm(tv+(1/1), xv, yv+(1/1), zv, __LINE__);

 // Read aligned vector block from vel_y at t+1, x, y-2, z.
 realv temp_vec26 = context.vel_y->readVecNorm(tv+(1/1), xv, yv-(2/1), zv, __LINE__);

 // Read aligned vector block from vel_z at t+1, x, y, z.
 realv temp_vec27 = context.vel_z->readVecNorm(tv+(1/1), xv, yv, zv, __LINE__);

 // Read aligned vector block from vel_z at t+1, x, y, z-1.
 realv temp_vec28 = context.vel_z->readVecNorm(tv+(1/1), xv, yv, zv-(1/1), __LINE__);

 // Read aligned vector block from vel_z at t+1, x, y, z+1.
 realv temp_vec29 = context.vel_z->readVecNorm(tv+(1/1), xv, yv, zv+(1/1), __LINE__);

 // Read aligned vector block from vel_z at t+1, x, y, z-2.
 realv temp_vec30 = context.vel_z->readVecNorm(tv+(1/1), xv, yv, zv-(2/1), __LINE__);

 // temp_vec31 = 2.
 realv temp_vec31 = 2.00000000000000000e+00;

 // temp_vec32 = 8.
 realv temp_vec32 = 8.00000000000000000e+00;

 // temp_vec33 = (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1)).
 realv temp_vec33 = temp_vec5 + temp_vec7;

 // temp_vec34 = (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1)).
 realv temp_vec34 = temp_vec33 + temp_vec8;

 // temp_vec35 = (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1)).
 realv temp_vec35 = temp_vec34 + temp_vec10;

 // temp_vec36 = (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1)).
 realv temp_vec36 = temp_vec35 + temp_vec11;

 // temp_vec37 = (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1)).
 realv temp_vec37 = temp_vec36 + temp_vec13;

 // temp_vec38 = (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1)).
 realv temp_vec38 = temp_vec37 + temp_vec14;

 // temp_vec39 = (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1)).
 realv temp_vec39 = temp_vec38 + temp_vec16;

 // temp_vec40 = (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))).
 realv temp_vec40 = temp_vec32 / temp_vec39;

 // temp_vec41 = (2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))).
 realv temp_vec41 = temp_vec31 * temp_vec40;

 // temp_vec42 = 1.125.
 realv temp_vec42 = 1.12500000000000000e+00;

 // temp_vec43 = vel_x(t+1, x, y, z).
 realv temp_vec43 = temp_vec17;

 // temp_vec44 = (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z)).
 realv temp_vec44 = temp_vec19 - temp_vec43;

 // temp_vec45 = (1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))).
 realv temp_vec45 = temp_vec42 * temp_vec44;

 // temp_vec46 = -0.0416667.
 realv temp_vec46 = -4.16666666666666644e-02;

 // temp_vec47 = (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))).
 realv temp_vec47 = temp_vec46 * (temp_vec20 - temp_vec22);

 // temp_vec48 = ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 realv temp_vec48 = temp_vec45 + temp_vec47;

 // temp_vec49 = vel_y(t+1, x, y, z).
 realv temp_vec49 = temp_vec23;

 // temp_vec50 = (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z)).
 realv temp_vec50 = temp_vec49 - temp_vec24;

 // temp_vec51 = (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))).
 realv temp_vec51 = temp_vec42 * temp_vec50;

 // temp_vec52 = (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))).
 realv temp_vec52 = temp_vec46 * (temp_vec25 - temp_vec26);

 // temp_vec53 = ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))).
 realv temp_vec53 = temp_vec51 + temp_vec52;

 // temp_vec54 = ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 realv temp_vec54 = temp_vec48 + temp_vec53;

 // temp_vec55 = vel_z(t+1, x, y, z).
 realv temp_vec55 = temp_vec27;

 // temp_vec56 = (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1)).
 realv temp_vec56 = temp_vec55 - temp_vec28;

 // temp_vec57 = (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))).
 realv temp_vec57 = temp_vec42 * temp_vec56;

 // temp_vec58 = (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))).
 realv temp_vec58 = temp_vec46 * (temp_vec29 - temp_vec30);

 // temp_vec59 = ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))).
 realv temp_vec59 = temp_vec57 + temp_vec58;

 // temp_vec60 = ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 realv temp_vec60 = temp_vec54 + temp_vec59;

 // temp_vec61 = ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 realv temp_vec61 = temp_vec60 + temp_vec53;

 // temp_vec62 = ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 realv temp_vec62 = temp_vec61 + temp_vec59;

 // temp_vec63 = ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 realv temp_vec63 = temp_vec62 + temp_vec53;

 // temp_vec64 = ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 realv temp_vec64 = temp_vec63 + temp_vec59;

 // temp_vec65 = (2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))).
 realv temp_vec65 = temp_vec41 * temp_vec64;

 // Read aligned vector block from lambda at x, y, z.
 realv temp_vec66 = context.lambda->readVecNorm(xv, yv, zv, __LINE__);

 // Read aligned vector block from lambda at x+8, y, z.
 realv temp_vec67 = context.lambda->readVecNorm(xv+(8/8), yv, zv, __LINE__);

 // Construct unaligned vector block from lambda at x+1, y, z.
 realv temp_vec68;
 temp_vec68[0] = temp_vec66[1];  // for x+1, y, z;
 temp_vec68[1] = temp_vec66[2];  // for x+2, y, z;
 temp_vec68[2] = temp_vec66[3];  // for x+3, y, z;
 temp_vec68[3] = temp_vec66[4];  // for x+4, y, z;
 temp_vec68[4] = temp_vec66[5];  // for x+5, y, z;
 temp_vec68[5] = temp_vec66[6];  // for x+6, y, z;
 temp_vec68[6] = temp_vec66[7];  // for x+7, y, z;
 temp_vec68[7] = temp_vec67[0];  // for x+8, y, z;

 // Read aligned vector block from lambda at x, y-1, z.
 realv temp_vec69 = context.lambda->readVecNorm(xv, yv-(1/1), zv, __LINE__);

 // Read aligned vector block from lambda at x+8, y-1, z.
 realv temp_vec70 = context.lambda->readVecNorm(xv+(8/8), yv-(1/1), zv, __LINE__);

 // Construct unaligned vector block from lambda at x+1, y-1, z.
 realv temp_vec71;
 temp_vec71[0] = temp_vec69[1];  // for x+1, y-1, z;
 temp_vec71[1] = temp_vec69[2];  // for x+2, y-1, z;
 temp_vec71[2] = temp_vec69[3];  // for x+3, y-1, z;
 temp_vec71[3] = temp_vec69[4];  // for x+4, y-1, z;
 temp_vec71[4] = temp_vec69[5];  // for x+5, y-1, z;
 temp_vec71[5] = temp_vec69[6];  // for x+6, y-1, z;
 temp_vec71[6] = temp_vec69[7];  // for x+7, y-1, z;
 temp_vec71[7] = temp_vec70[0];  // for x+8, y-1, z;

 // Read aligned vector block from lambda at x, y, z-1.
 realv temp_vec72 = context.lambda->readVecNorm(xv, yv, zv-(1/1), __LINE__);

 // Read aligned vector block from lambda at x+8, y, z-1.
 realv temp_vec73 = context.lambda->readVecNorm(xv+(8/8), yv, zv-(1/1), __LINE__);

 // Construct unaligned vector block from lambda at x+1, y, z-1.
 realv temp_vec74;
 temp_vec74[0] = temp_vec72[1];  // for x+1, y, z-1;
 temp_vec74[1] = temp_vec72[2];  // for x+2, y, z-1;
 temp_vec74[2] = temp_vec72[3];  // for x+3, y, z-1;
 temp_vec74[3] = temp_vec72[4];  // for x+4, y, z-1;
 temp_vec74[4] = temp_vec72[5];  // for x+5, y, z-1;
 temp_vec74[5] = temp_vec72[6];  // for x+6, y, z-1;
 temp_vec74[6] = temp_vec72[7];  // for x+7, y, z-1;
 temp_vec74[7] = temp_vec73[0];  // for x+8, y, z-1;

 // Read aligned vector block from lambda at x, y-1, z-1.
 realv temp_vec75 = context.lambda->readVecNorm(xv, yv-(1/1), zv-(1/1), __LINE__);

 // Read aligned vector block from lambda at x+8, y-1, z-1.
 realv temp_vec76 = context.lambda->readVecNorm(xv+(8/8), yv-(1/1), zv-(1/1), __LINE__);

 // Construct unaligned vector block from lambda at x+1, y-1, z-1.
 realv temp_vec77;
 temp_vec77[0] = temp_vec75[1];  // for x+1, y-1, z-1;
 temp_vec77[1] = temp_vec75[2];  // for x+2, y-1, z-1;
 temp_vec77[2] = temp_vec75[3];  // for x+3, y-1, z-1;
 temp_vec77[3] = temp_vec75[4];  // for x+4, y-1, z-1;
 temp_vec77[4] = temp_vec75[5];  // for x+5, y-1, z-1;
 temp_vec77[5] = temp_vec75[6];  // for x+6, y-1, z-1;
 temp_vec77[6] = temp_vec75[7];  // for x+7, y-1, z-1;
 temp_vec77[7] = temp_vec76[0];  // for x+8, y-1, z-1;

 // temp_vec78 = (8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))).
 realv temp_vec78 = temp_vec32 / (temp_vec66 + temp_vec68 + temp_vec69 + temp_vec71 + temp_vec72 + temp_vec74 + temp_vec75 + temp_vec77);

 // temp_vec79 = ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))).
 realv temp_vec79 = temp_vec78 * temp_vec64;

 // temp_vec80 = ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))).
 realv temp_vec80 = temp_vec65 + temp_vec79;

 // temp_vec81 = ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))))).
 realv temp_vec81 = temp_vec4 * temp_vec80;

 // temp_vec82 = (stress_xx(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))))).
 realv temp_vec82 = temp_vec1 + temp_vec81;

 // Read aligned vector block from sponge at x, y, z.
 realv temp_vec83 = context.sponge->readVecNorm(xv, yv, zv, __LINE__);

 // temp_vec84 = sponge(x, y, z).
 realv temp_vec84 = temp_vec83;

 // temp_vec85 = ((stress_xx(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))))) * sponge(x, y, z)).
 realv temp_vec85 = temp_vec82 * temp_vec84;

 // temp_vec86 = ((stress_xx(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))))) * sponge(x, y, z)).
 realv temp_vec86 = temp_vec85;

 // Save result to stress_xx(t+1, x, y, z):
 
 // Write aligned vector block to stress_xx at t+1, x, y, z.
context.stress_xx->writeVecNorm(temp_vec86, tv+(1/1), xv, yv, zv, __LINE__);
;

 // Read aligned vector block from stress_yy at t, x, y, z.
 realv temp_vec87 = context.stress_yy->readVecNorm(tv, xv, yv, zv, __LINE__);

 // temp_vec88 = (2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))).
 realv temp_vec88 = temp_vec31 * temp_vec40;

 // temp_vec89 = (2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))).
 realv temp_vec89 = temp_vec88 * temp_vec53;

 // temp_vec90 = ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))).
 realv temp_vec90 = temp_vec78 * temp_vec64;

 // temp_vec91 = ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))).
 realv temp_vec91 = temp_vec89 + temp_vec90;

 // temp_vec92 = ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))))).
 realv temp_vec92 = temp_vec4 * temp_vec91;

 // temp_vec93 = (stress_yy(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))))).
 realv temp_vec93 = temp_vec87 + temp_vec92;

 // temp_vec94 = ((stress_yy(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))))) * sponge(x, y, z)).
 realv temp_vec94 = temp_vec93 * temp_vec84;

 // temp_vec95 = ((stress_yy(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))))) * sponge(x, y, z)).
 realv temp_vec95 = temp_vec94;

 // Save result to stress_yy(t+1, x, y, z):
 
 // Write aligned vector block to stress_yy at t+1, x, y, z.
context.stress_yy->writeVecNorm(temp_vec95, tv+(1/1), xv, yv, zv, __LINE__);
;

 // Read aligned vector block from stress_zz at t, x, y, z.
 realv temp_vec96 = context.stress_zz->readVecNorm(tv, xv, yv, zv, __LINE__);

 // temp_vec97 = (2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 realv temp_vec97 = temp_vec31 * temp_vec40;

 // temp_vec98 = (2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))).
 realv temp_vec98 = temp_vec97 * temp_vec59;

 // temp_vec99 = ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))).
 realv temp_vec99 = temp_vec78 * temp_vec64;

 // temp_vec100 = ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))).
 realv temp_vec100 = temp_vec98 + temp_vec99;

 // temp_vec101 = ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))))))).
 realv temp_vec101 = temp_vec4 * temp_vec100;

 // temp_vec102 = (stress_zz(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))))).
 realv temp_vec102 = temp_vec96 + temp_vec101;

 // temp_vec103 = ((stress_zz(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))))) * sponge(x, y, z)).
 realv temp_vec103 = temp_vec102 * temp_vec84;

 // temp_vec104 = ((stress_zz(t, x, y, z) + ((delta_t / h) * ((2 * (8 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))) + ((8 / (lambda(x, y, z) + lambda(x+1, y, z) + lambda(x, y-1, z) + lambda(x+1, y-1, z) + lambda(x, y, z-1) + lambda(x+1, y, z-1) + lambda(x, y-1, z-1) + lambda(x+1, y-1, z-1))) * ((1.125 * (vel_x(t+1, x+1, y, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x+2, y, z) - vel_x(t+1, x-1, y, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2)))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x, y-1, z))) + (-0.0416667 * (vel_y(t+1, x, y+1, z) - vel_y(t+1, x, y-2, z)))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x, y, z-1))) + (-0.0416667 * (vel_z(t+1, x, y, z+1) - vel_z(t+1, x, y, z-2))))))))) * sponge(x, y, z)).
 realv temp_vec104 = temp_vec103;

 // Save result to stress_zz(t+1, x, y, z):
 
 // Write aligned vector block to stress_zz at t+1, x, y, z.
context.stress_zz->writeVecNorm(temp_vec104, tv+(1/1), xv, yv, zv, __LINE__);
;

 // Read aligned vector block from stress_xy at t, x, y, z.
 realv temp_vec105 = context.stress_xy->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from vel_x at t+1, x, y+1, z.
 realv temp_vec106 = context.vel_x->readVecNorm(tv+(1/1), xv, yv+(1/1), zv, __LINE__);

 // Read aligned vector block from vel_x at t+1, x, y+2, z.
 realv temp_vec107 = context.vel_x->readVecNorm(tv+(1/1), xv, yv+(2/1), zv, __LINE__);

 // Read aligned vector block from vel_x at t+1, x, y-1, z.
 realv temp_vec108 = context.vel_x->readVecNorm(tv+(1/1), xv, yv-(1/1), zv, __LINE__);

 // Read aligned vector block from vel_y at t+1, x-8, y, z.
 realv temp_vec109 = context.vel_y->readVecNorm(tv+(1/1), xv-(8/8), yv, zv, __LINE__);

 // Construct unaligned vector block from vel_y at t+1, x-1, y, z.
 realv temp_vec110;
 temp_vec110[0] = temp_vec109[7];  // for t+1, x-1, y, z;
 temp_vec110[1] = temp_vec23[0];  // for t+1, x, y, z;
 temp_vec110[2] = temp_vec23[1];  // for t+1, x+1, y, z;
 temp_vec110[3] = temp_vec23[2];  // for t+1, x+2, y, z;
 temp_vec110[4] = temp_vec23[3];  // for t+1, x+3, y, z;
 temp_vec110[5] = temp_vec23[4];  // for t+1, x+4, y, z;
 temp_vec110[6] = temp_vec23[5];  // for t+1, x+5, y, z;
 temp_vec110[7] = temp_vec23[6];  // for t+1, x+6, y, z;

 // Read aligned vector block from vel_y at t+1, x+8, y, z.
 realv temp_vec111 = context.vel_y->readVecNorm(tv+(1/1), xv+(8/8), yv, zv, __LINE__);

 // Construct unaligned vector block from vel_y at t+1, x+1, y, z.
 realv temp_vec112;
 temp_vec112[0] = temp_vec23[1];  // for t+1, x+1, y, z;
 temp_vec112[1] = temp_vec23[2];  // for t+1, x+2, y, z;
 temp_vec112[2] = temp_vec23[3];  // for t+1, x+3, y, z;
 temp_vec112[3] = temp_vec23[4];  // for t+1, x+4, y, z;
 temp_vec112[4] = temp_vec23[5];  // for t+1, x+5, y, z;
 temp_vec112[5] = temp_vec23[6];  // for t+1, x+6, y, z;
 temp_vec112[6] = temp_vec23[7];  // for t+1, x+7, y, z;
 temp_vec112[7] = temp_vec111[0];  // for t+1, x+8, y, z;

 // Construct unaligned vector block from vel_y at t+1, x-2, y, z.
 realv temp_vec113;
 temp_vec113[0] = temp_vec109[6];  // for t+1, x-2, y, z;
 temp_vec113[1] = temp_vec109[7];  // for t+1, x-1, y, z;
 temp_vec113[2] = temp_vec23[0];  // for t+1, x, y, z;
 temp_vec113[3] = temp_vec23[1];  // for t+1, x+1, y, z;
 temp_vec113[4] = temp_vec23[2];  // for t+1, x+2, y, z;
 temp_vec113[5] = temp_vec23[3];  // for t+1, x+3, y, z;
 temp_vec113[6] = temp_vec23[4];  // for t+1, x+4, y, z;
 temp_vec113[7] = temp_vec23[5];  // for t+1, x+5, y, z;

 // temp_vec114 = (2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))).
 realv temp_vec114 = temp_vec31 / temp_vec39;

 // temp_vec115 = ((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t).
 realv temp_vec115 = temp_vec114 * temp_vec2;

 // temp_vec116 = (((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h).
 realv temp_vec116 = temp_vec115 / temp_vec3;

 // temp_vec117 = (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z)).
 realv temp_vec117 = temp_vec106 - temp_vec43;

 // temp_vec118 = (1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))).
 realv temp_vec118 = temp_vec42 * temp_vec117;

 // temp_vec119 = (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))).
 realv temp_vec119 = temp_vec46 * (temp_vec107 - temp_vec108);

 // temp_vec120 = ((1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z))))).
 realv temp_vec120 = temp_vec118 + temp_vec119;

 // temp_vec121 = (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z)).
 realv temp_vec121 = temp_vec49 - temp_vec110;

 // temp_vec122 = (1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))).
 realv temp_vec122 = temp_vec42 * temp_vec121;

 // temp_vec123 = (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z))).
 realv temp_vec123 = temp_vec46 * (temp_vec112 - temp_vec113);

 // temp_vec124 = ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z)))).
 realv temp_vec124 = temp_vec122 + temp_vec123;

 // temp_vec125 = ((1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z))))).
 realv temp_vec125 = temp_vec120 + temp_vec124;

 // temp_vec126 = ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z)))))).
 realv temp_vec126 = temp_vec116 * temp_vec125;

 // temp_vec127 = (stress_xy(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z))))))).
 realv temp_vec127 = temp_vec105 + temp_vec126;

 // temp_vec128 = ((stress_xy(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z))))))) * sponge(x, y, z)).
 realv temp_vec128 = temp_vec127 * temp_vec84;

 // temp_vec129 = ((stress_xy(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y+1, z) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y+2, z) - vel_x(t+1, x, y-1, z))) + ((1.125 * (vel_y(t+1, x, y, z) - vel_y(t+1, x-1, y, z))) + (-0.0416667 * (vel_y(t+1, x+1, y, z) - vel_y(t+1, x-2, y, z))))))) * sponge(x, y, z)).
 realv temp_vec129 = temp_vec128;

 // Save result to stress_xy(t+1, x, y, z):
 
 // Write aligned vector block to stress_xy at t+1, x, y, z.
context.stress_xy->writeVecNorm(temp_vec129, tv+(1/1), xv, yv, zv, __LINE__);
;

 // Read aligned vector block from stress_xz at t, x, y, z.
 realv temp_vec130 = context.stress_xz->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from vel_x at t+1, x, y, z+1.
 realv temp_vec131 = context.vel_x->readVecNorm(tv+(1/1), xv, yv, zv+(1/1), __LINE__);

 // Read aligned vector block from vel_x at t+1, x, y, z+2.
 realv temp_vec132 = context.vel_x->readVecNorm(tv+(1/1), xv, yv, zv+(2/1), __LINE__);

 // Read aligned vector block from vel_x at t+1, x, y, z-1.
 realv temp_vec133 = context.vel_x->readVecNorm(tv+(1/1), xv, yv, zv-(1/1), __LINE__);

 // Read aligned vector block from vel_z at t+1, x-8, y, z.
 realv temp_vec134 = context.vel_z->readVecNorm(tv+(1/1), xv-(8/8), yv, zv, __LINE__);

 // Construct unaligned vector block from vel_z at t+1, x-1, y, z.
 realv temp_vec135;
 temp_vec135[0] = temp_vec134[7];  // for t+1, x-1, y, z;
 temp_vec135[1] = temp_vec27[0];  // for t+1, x, y, z;
 temp_vec135[2] = temp_vec27[1];  // for t+1, x+1, y, z;
 temp_vec135[3] = temp_vec27[2];  // for t+1, x+2, y, z;
 temp_vec135[4] = temp_vec27[3];  // for t+1, x+3, y, z;
 temp_vec135[5] = temp_vec27[4];  // for t+1, x+4, y, z;
 temp_vec135[6] = temp_vec27[5];  // for t+1, x+5, y, z;
 temp_vec135[7] = temp_vec27[6];  // for t+1, x+6, y, z;

 // Read aligned vector block from vel_z at t+1, x+8, y, z.
 realv temp_vec136 = context.vel_z->readVecNorm(tv+(1/1), xv+(8/8), yv, zv, __LINE__);

 // Construct unaligned vector block from vel_z at t+1, x+1, y, z.
 realv temp_vec137;
 temp_vec137[0] = temp_vec27[1];  // for t+1, x+1, y, z;
 temp_vec137[1] = temp_vec27[2];  // for t+1, x+2, y, z;
 temp_vec137[2] = temp_vec27[3];  // for t+1, x+3, y, z;
 temp_vec137[3] = temp_vec27[4];  // for t+1, x+4, y, z;
 temp_vec137[4] = temp_vec27[5];  // for t+1, x+5, y, z;
 temp_vec137[5] = temp_vec27[6];  // for t+1, x+6, y, z;
 temp_vec137[6] = temp_vec27[7];  // for t+1, x+7, y, z;
 temp_vec137[7] = temp_vec136[0];  // for t+1, x+8, y, z;

 // Construct unaligned vector block from vel_z at t+1, x-2, y, z.
 realv temp_vec138;
 temp_vec138[0] = temp_vec134[6];  // for t+1, x-2, y, z;
 temp_vec138[1] = temp_vec134[7];  // for t+1, x-1, y, z;
 temp_vec138[2] = temp_vec27[0];  // for t+1, x, y, z;
 temp_vec138[3] = temp_vec27[1];  // for t+1, x+1, y, z;
 temp_vec138[4] = temp_vec27[2];  // for t+1, x+2, y, z;
 temp_vec138[5] = temp_vec27[3];  // for t+1, x+3, y, z;
 temp_vec138[6] = temp_vec27[4];  // for t+1, x+4, y, z;
 temp_vec138[7] = temp_vec27[5];  // for t+1, x+5, y, z;

 // temp_vec139 = (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z)).
 realv temp_vec139 = temp_vec131 - temp_vec43;

 // temp_vec140 = (1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))).
 realv temp_vec140 = temp_vec42 * temp_vec139;

 // temp_vec141 = (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))).
 realv temp_vec141 = temp_vec46 * (temp_vec132 - temp_vec133);

 // temp_vec142 = ((1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z))))).
 realv temp_vec142 = temp_vec140 + temp_vec141;

 // temp_vec143 = (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z)).
 realv temp_vec143 = temp_vec55 - temp_vec135;

 // temp_vec144 = (1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))).
 realv temp_vec144 = temp_vec42 * temp_vec143;

 // temp_vec145 = (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z))).
 realv temp_vec145 = temp_vec46 * (temp_vec137 - temp_vec138);

 // temp_vec146 = ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z)))).
 realv temp_vec146 = temp_vec144 + temp_vec145;

 // temp_vec147 = ((1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z))))).
 realv temp_vec147 = temp_vec142 + temp_vec146;

 // temp_vec148 = ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z)))))).
 realv temp_vec148 = temp_vec116 * temp_vec147;

 // temp_vec149 = (stress_xz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z))))))).
 realv temp_vec149 = temp_vec130 + temp_vec148;

 // temp_vec150 = ((stress_xz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z))))))) * sponge(x, y, z)).
 realv temp_vec150 = temp_vec149 * temp_vec84;

 // temp_vec151 = ((stress_xz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_x(t+1, x, y, z+1) - vel_x(t+1, x, y, z))) + (-0.0416667 * (vel_x(t+1, x, y, z+2) - vel_x(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y, z) - vel_z(t+1, x-1, y, z))) + (-0.0416667 * (vel_z(t+1, x+1, y, z) - vel_z(t+1, x-2, y, z))))))) * sponge(x, y, z)).
 realv temp_vec151 = temp_vec150;

 // Save result to stress_xz(t+1, x, y, z):
 
 // Write aligned vector block to stress_xz at t+1, x, y, z.
context.stress_xz->writeVecNorm(temp_vec151, tv+(1/1), xv, yv, zv, __LINE__);
;

 // Read aligned vector block from stress_yz at t, x, y, z.
 realv temp_vec152 = context.stress_yz->readVecNorm(tv, xv, yv, zv, __LINE__);

 // Read aligned vector block from vel_y at t+1, x, y, z+1.
 realv temp_vec153 = context.vel_y->readVecNorm(tv+(1/1), xv, yv, zv+(1/1), __LINE__);

 // Read aligned vector block from vel_y at t+1, x, y, z+2.
 realv temp_vec154 = context.vel_y->readVecNorm(tv+(1/1), xv, yv, zv+(2/1), __LINE__);

 // Read aligned vector block from vel_y at t+1, x, y, z-1.
 realv temp_vec155 = context.vel_y->readVecNorm(tv+(1/1), xv, yv, zv-(1/1), __LINE__);

 // Read aligned vector block from vel_z at t+1, x, y+1, z.
 realv temp_vec156 = context.vel_z->readVecNorm(tv+(1/1), xv, yv+(1/1), zv, __LINE__);

 // Read aligned vector block from vel_z at t+1, x, y-1, z.
 realv temp_vec157 = context.vel_z->readVecNorm(tv+(1/1), xv, yv-(1/1), zv, __LINE__);

 // temp_vec158 = (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z)).
 realv temp_vec158 = temp_vec153 - temp_vec49;

 // temp_vec159 = (1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))).
 realv temp_vec159 = temp_vec42 * temp_vec158;

 // temp_vec160 = (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))).
 realv temp_vec160 = temp_vec46 * (temp_vec154 - temp_vec155);

 // temp_vec161 = ((1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y-1, z))))).
 realv temp_vec161 = temp_vec159 + temp_vec160;

 // temp_vec162 = vel_z(t+1, x, y+1, z).
 realv temp_vec162 = temp_vec156;

 // temp_vec163 = (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z)).
 realv temp_vec163 = temp_vec162 - temp_vec55;

 // temp_vec164 = (1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))).
 realv temp_vec164 = temp_vec42 * temp_vec163;

 // temp_vec165 = (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y-1, z)).
 realv temp_vec165 = temp_vec162 - temp_vec157;

 // temp_vec166 = (-0.0416667 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y-1, z))).
 realv temp_vec166 = temp_vec46 * temp_vec165;

 // temp_vec167 = ((1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y-1, z)))).
 realv temp_vec167 = temp_vec164 + temp_vec166;

 // temp_vec168 = ((1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y-1, z))))).
 realv temp_vec168 = temp_vec161 + temp_vec167;

 // temp_vec169 = ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y-1, z)))))).
 realv temp_vec169 = temp_vec116 * temp_vec168;

 // temp_vec170 = (stress_yz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y-1, z))))))).
 realv temp_vec170 = temp_vec152 + temp_vec169;

 // temp_vec171 = ((stress_yz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y-1, z))))))) * sponge(x, y, z)).
 realv temp_vec171 = temp_vec170 * temp_vec84;

 // temp_vec172 = ((stress_yz(t, x, y, z) + ((((2 / (mu(x, y, z) + mu(x+1, y, z) + mu(x, y-1, z) + mu(x+1, y-1, z) + mu(x, y, z-1) + mu(x+1, y, z-1) + mu(x, y-1, z-1) + mu(x+1, y-1, z-1))) * delta_t) / h) * ((1.125 * (vel_y(t+1, x, y, z+1) - vel_y(t+1, x, y, z))) + (-0.0416667 * (vel_y(t+1, x, y, z+2) - vel_y(t+1, x, y, z-1))) + ((1.125 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y, z))) + (-0.0416667 * (vel_z(t+1, x, y+1, z) - vel_z(t+1, x, y-1, z))))))) * sponge(x, y, z)).
 realv temp_vec172 = temp_vec171;

 // Save result to stress_yz(t+1, x, y, z):
 
 // Write aligned vector block to stress_yz at t+1, x, y, z.
context.stress_yz->writeVecNorm(temp_vec172, tv+(1/1), xv, yv, zv, __LINE__);
;
} // vector calculation.

// Prefetches cache line(s) for entire stencil to L1 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=8 * y=1 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L1_vector(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned lambda at x, y-1, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv-(1/1), zv-(1/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned lambda at x, y-1, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv-(1/1), zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned lambda at x, y, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv, zv-(1/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned lambda at x, y, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned lambda at x+8, y-1, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(8/8), yv-(1/1), zv-(1/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned lambda at x+8, y-1, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(8/8), yv-(1/1), zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned lambda at x+8, y, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(8/8), yv, zv-(1/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned lambda at x+8, y, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(8/8), yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x, y-1, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv-(1/1), zv-(1/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x, y-1, z.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv-(1/1), zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x, y, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv, zv-(1/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x, y, z.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x+8, y-1, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv+(8/8), yv-(1/1), zv-(1/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x+8, y-1, z.
 p = (const char*)context.mu->getVecPtrNorm(xv+(8/8), yv-(1/1), zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x+8, y, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv+(8/8), yv, zv-(1/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned mu at x+8, y, z.
 p = (const char*)context.mu->getVecPtrNorm(xv+(8/8), yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned sponge at x, y, z.
 p = (const char*)context.sponge->getVecPtrNorm(xv, yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xx at t, x, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xy at t, x, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_xz at t, x, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yy at t, x, y, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_yz at t, x, y, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned stress_zz at t, x, y, z.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x-8, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv-(8/8), yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y-1, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv-(1/1), zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y, z-1.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y, z+1.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y, z+2.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y+1, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv+(1/1), zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x, y+2, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv+(2/1), zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_x at t+1, x+8, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv+(8/8), yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x-8, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv-(8/8), yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y-2, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv-(2/1), zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y-1, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv-(1/1), zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y, z-1.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y, z+1.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y, z+2.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x, y+1, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv+(1/1), zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_y at t+1, x+8, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv+(8/8), yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x-8, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv-(8/8), yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y-1, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv-(1/1), zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y, z-2.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv-(2/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y, z-1.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y, z+1.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x, y+1, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv+(1/1), zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);

 // Aligned vel_z at t+1, x+8, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv+(8/8), yv, zv, __LINE__);
 MCP(p, L1, __LINE__);
 _mm_prefetch(p, L1);
}

// Prefetches cache line(s) for entire stencil to L2 cache relative to indices t, x, y, z in a 'x=1 * y=1 * z=1' cluster of 'x=8 * y=1 * z=1' vector(s).
// Indices must be normalized, i.e., already divided by VLEN_*.
 void prefetch_L2_vector(StencilContext_awp& context, idx_t tv, idx_t xv, idx_t yv, idx_t zv) {
 const char* p = 0;

 // Aligned lambda at x, y-1, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv-(1/1), zv-(1/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned lambda at x, y-1, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv-(1/1), zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned lambda at x, y, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv, zv-(1/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned lambda at x, y, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv, yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned lambda at x+8, y-1, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(8/8), yv-(1/1), zv-(1/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned lambda at x+8, y-1, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(8/8), yv-(1/1), zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned lambda at x+8, y, z-1.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(8/8), yv, zv-(1/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned lambda at x+8, y, z.
 p = (const char*)context.lambda->getVecPtrNorm(xv+(8/8), yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x, y-1, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv-(1/1), zv-(1/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x, y-1, z.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv-(1/1), zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x, y, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv, zv-(1/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x, y, z.
 p = (const char*)context.mu->getVecPtrNorm(xv, yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x+8, y-1, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv+(8/8), yv-(1/1), zv-(1/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x+8, y-1, z.
 p = (const char*)context.mu->getVecPtrNorm(xv+(8/8), yv-(1/1), zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x+8, y, z-1.
 p = (const char*)context.mu->getVecPtrNorm(xv+(8/8), yv, zv-(1/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned mu at x+8, y, z.
 p = (const char*)context.mu->getVecPtrNorm(xv+(8/8), yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned sponge at x, y, z.
 p = (const char*)context.sponge->getVecPtrNorm(xv, yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xx at t, x, y, z.
 p = (const char*)context.stress_xx->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xy at t, x, y, z.
 p = (const char*)context.stress_xy->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_xz at t, x, y, z.
 p = (const char*)context.stress_xz->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yy at t, x, y, z.
 p = (const char*)context.stress_yy->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_yz at t, x, y, z.
 p = (const char*)context.stress_yz->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned stress_zz at t, x, y, z.
 p = (const char*)context.stress_zz->getVecPtrNorm(tv, xv, yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x-8, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv-(8/8), yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y-1, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv-(1/1), zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y, z-1.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y, z+1.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y, z+2.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y+1, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv+(1/1), zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x, y+2, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv, yv+(2/1), zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_x at t+1, x+8, y, z.
 p = (const char*)context.vel_x->getVecPtrNorm(tv+(1/1), xv+(8/8), yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x-8, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv-(8/8), yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y-2, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv-(2/1), zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y-1, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv-(1/1), zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y, z-1.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y, z+1.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y, z+2.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv, zv+(2/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x, y+1, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv, yv+(1/1), zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_y at t+1, x+8, y, z.
 p = (const char*)context.vel_y->getVecPtrNorm(tv+(1/1), xv+(8/8), yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x-8, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv-(8/8), yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y-1, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv-(1/1), zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y, z-2.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv-(2/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y, z-1.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv-(1/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y, z+1.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv, zv+(1/1), __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x, y+1, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv, yv+(1/1), zv, __LINE__);
 MCP(p, L2, __LINE__);
 _mm_prefetch(p, L2);

 // Aligned vel_z at t+1, x+8, y, z.
 p = (const char*)context.vel_z->getVecPtrNorm(tv+(1/1), xv+(8/8), yv, zv, __LINE__);
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
