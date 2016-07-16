/**
   @section LICENSE
 
   Copyright (c) 2013-2016, Regents of the University of California
   All rights reserved.
 
   Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
   1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
   2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
*/

#ifndef Mesh_hpp
#define Mesh_hpp

#include "parallel/Mpi.hpp"

#include <cstdio>

#include "constants.hpp"
#include "data/SoA.hpp"
#include "io/OptionParser.h"


namespace odc
{
  namespace data
  {
    class Mesh;
  }
}


class odc::data::Mesh
{
    
public:
    
  bool m_usingAnelastic = false;
    
  real m_vse[2], m_vpe[2], m_dde[2];
    
  // static sized data
  Grid1D m_coeff;
    
  // material properties
  Grid3D m_density, m_mu, m_lam, m_lam_mu;
    
  // anelastic coefficients
  Grid3D m_qp, m_qs, m_tau1, m_tau2, m_weights;
  Grid3Dww m_weight_index;
    
  Mesh() {};

  void initialize(odc::io::OptionParser i_options, int_pt nx, int_pt ny, int_pt nz,
                  int_pt bdry_size, bool anelastic, Grid1D i_inputBuffer,
                  int_pt globalX, int_pt globalY, int_pt globalZ);
  
  // initialize and store mesh parameters using input from command line options
  Mesh(odc::io::OptionParser i_options, odc::data::SoA i_data);
    
  // cleanup
  void finalize();
    
    
private:
    
  void weights_sub(Grid3D weights,Grid1D coeff, float ex, float fac);
  void inimesh(int MEDIASTART, Grid3D d1, Grid3D mu, Grid3D lam, Grid3D qp, Grid3D qs, float *taumax, float *taumin,
               Grid3D tau, Grid3D weights,Grid1D coeff,
               int nvar, float FP,  float FAC, float Q0, float EX, int nxt, int nyt, int nzt, int PX, int PY, int NX, int NY,
               int NZ, int *coords, MPI_Comm MCW, int IDYNA, int NVE, int SoCalQ, char *INVEL,
               float *vse, float *vpe, float *dde);

  void new_inimesh(int MEDIASTART,
                   real *d1,
                   real *mu,
                   real *lam,
                   real *qp,
                   real *qs,
                   int_pt i_strideX,
                   int_pt i_strideY,
                   int_pt i_strideZ,
                   float *taumax,
                   float *taumin,
                   Grid3D tau,
                   Grid3D weights,
                   Grid1D coeff,
                   int nvar,
                   float FP,
                   float FAC,
                   float Q0,
                   float EX,
                   int nxt,
                   int nyt,
                   int nzt,
                   int NX,
                   int NY,
                   int NZ,
                   int IDYNA,
                   int NVE,
                   int SoCalQ,
                   real *i_inputBuffer,
                   int_pt i_inputSizeX,
                   int_pt i_inputSizeY,
                   int_pt i_inputSizeZ);
    
  void tausub( Grid3D tau, float taumin,float taumax);
    
  void init_texture(int nxt,  int nyt,  int nzt,  Grid3D tau1,  Grid3D tau2,  Grid3D vx1,  Grid3D vx2,
                    Grid3D weights, Grid3Dww ww,Grid3D wwo,
                    int xls,  int xre,  int yls,  int yre);

  void new_init_texture(Grid3D tau1,  Grid3D tau2,  Grid3D vx1,  Grid3D vx2, Grid3D weights, Grid3Dww ww,Grid3D wwo,
                        int_pt startX,  int_pt endX,  int_pt startY,  int_pt endY, int_pt startZ, int_pt endZ,
                        int_pt globalStartX, int_pt globalStartY, int_pt globalStartZ, int_pt sizeZ);

   
    
  void set_boundaries(Grid3D density, Grid3D mu, Grid3D lam, Grid3D qp, Grid3D qs, bool anelastic,
                      int_pt bdry_width, int_pt nx, int_pt ny, int_pt nz);
    
    
    
    
};

#endif /* Mesh_hpp */
