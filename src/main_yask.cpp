/**
 @author Gautam Wilkins

 @section DESCRIPTION
 Main file to validate integration with YASK vector folding library (this is for testing only)

 @section LICENSE
 Copyright (c) 2015-2016, Regents of the University of California
 All rights reserved.

 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "parallel/Mpi.hpp"

#include <iostream>

#include "io/OptionParser.h"
#include "io/Sources.hpp"
#include "io/OutputWriter.hpp"

#include "data/SoA.hpp"
#include "data/common.hpp"
#include "data/Mesh.hpp"
#include "data/Cerjan.hpp"

#include "kernels/cpu_vanilla.h"

#include "constants.hpp"

#include "data/Grid.hpp"

// define statics
int odc::parallel::Mpi::m_rank;
int odc::parallel::Mpi::m_size;

void applyYASKStencil(STENCIL_CONTEXT context, StencilBase *stencil, real t) {

    for(int_pt ix = 0; ix < context.dx; ix++) {

        for(int_pt iy = 0; iy < context.dy; iy++) {

            for(int_pt iz = 0; iz < context.dz; iz++) {

                // Evaluate the reference scalar code.
                stencil->calc_scalar(context, t, 0, ix, iy, iz);
            }
        }
    }

}

void writeAllGridsToYASK(odc::data::SoA data, odc::data::Mesh mesh,
                             odc::data::Cerjan cerjan, STENCIL_CONTEXT context) {

    int_pt nx = data.m_numXGridPoints;
    int_pt ny = data.m_numYGridPoints;
    int_pt nz = data.m_numZGridPoints;

    odc::data::WriteToYASKGrid(data.m_velocityX, context->vel_x, 0, 0, 0, nx, ny, nz);
    odc::data::WriteToYASKGrid(data.m_velocityY, context->vel_y, 0, 0, 0, nx, ny, nz);
    odc::data::WriteToYASKGrid(data.m_velocityZ, context->vel_z, 0, 0, 0, nx, ny, nz);

    odc::data::WriteToYASKGrid(data.m_stressXX, context->stress_xx, 0, 0, 0, nx, ny, nz);
    odc::data::WriteToYASKGrid(data.m_stressXY, context->stress_xy, 0, 0, 0, nx, ny, nz);
    odc::data::WriteToYASKGrid(data.m_stressXZ, context->stress_xz, 0, 0, 0, nx, ny, nz);
    odc::data::WriteToYASKGrid(data.m_stressYY, context->stress_yy, 0, 0, 0, nx, ny, nz);
    odc::data::WriteToYASKGrid(data.m_stressYZ, context->stress_yz, 0, 0, 0, nx, ny, nz);
    odc::data::WriteToYASKGrid(data.m_stressZZ, context->stress_zz, 0, 0, 0, nx, ny, nz);

    odc::data::WriteToYASKGrid(mesh.m_lam, context->lambda, -1, -1, -1, nx+2, ny+2, nz+2);
    odc::data::WriteToYASKGrid(mesh.m_mu, context->mu, -1, -1, -1, nx+2, ny+2, nz+2);
    odc::data::WriteToYASKGrid(mesh.m_density, context->rho, -1, -1, -1, nx+2, ny+2, nz+2);

    // Convert cerjan into grid
    Grid3D cerjanGrid = Alloc3D(nx, ny, nz);
    for (int_pt i=0; i<nx; i++) {
        for (int_pt j=0; j<ny; j++) {
            for (int_pt k=0; k<nz; k++) {
                cerjanGrid[i][j][k] = cerjan.m_spongeCoeffX[i]*cerjan.m_spongeCoeffY[j]*cerjan.m_spongeCoeffZ[k];
            }
        }
    }


    odc::data::WriteToYASKGrid(cerjanGrid, context->sponge, 0, 0, 0, nx, ny, nz);

    Delloc3D(cerjanGrid);

}


void writeVelocityGridsToYASK(odc::data::SoA data, STENCIL_CONTEXT context) {

    int_pt nx = data.m_numXGridPoints;
    int_pt ny = data.m_numYGridPoints;
    int_pt nz = data.m_numZGridPoints;

    // Go to nz+1 for free surface grid points
    odc::data::WriteToYASKGrid(data.m_velocityX, context->vel_x, 0, 0, 0, nx, ny, nz+1);
    odc::data::WriteToYASKGrid(data.m_velocityY, context->vel_y, 0, 0, 0, nx, ny, nz+1);
    odc::data::WriteToYASKGrid(data.m_velocityZ, context->vel_z, 0, 0, 0, nx, ny, nz+1);

}

void writeStressGridsToYASK(odc::data::SoA data, STENCIL_CONTEXT context) {


    int_pt nx = data.m_numXGridPoints;
    int_pt ny = data.m_numYGridPoints;
    int_pt nz = data.m_numZGridPoints;

    // Go to nz+2 for free surface grid points
    odc::data::WriteToYASKGrid(data.m_stressXX, context->stress_xx, 0, 0, 0, nx, ny, nz+2);
    odc::data::WriteToYASKGrid(data.m_stressXY, context->stress_xy, 0, 0, 0, nx, ny, nz+2);
    odc::data::WriteToYASKGrid(data.m_stressXZ, context->stress_xz, 0, 0, 0, nx, ny, nz+2);
    odc::data::WriteToYASKGrid(data.m_stressYY, context->stress_yy, 0, 0, 0, nx, ny, nz+2);
    odc::data::WriteToYASKGrid(data.m_stressYZ, context->stress_yz, 0, 0, 0, nx, ny, nz+2);
    odc::data::WriteToYASKGrid(data.m_stressZZ, context->stress_zz, 0, 0, 0, nx, ny, nz+2);

}

void copyVelocityGridsFromYASK(odc::data::SoA data, STENCIL_CONTEXT context) {

    int_pt nx = data.m_numXGridPoints;
    int_pt ny = data.m_numYGridPoints;
    int_pt nz = data.m_numZGridPoints;

    odc::data::CopyFromYASKGrid(data.m_velocityX, context->vel_x, 0, 0, 0, nx, ny, nz+1);
    odc::data::CopyFromYASKGrid(data.m_velocityY, context->vel_y, 0, 0, 0, nx, ny, nz+1);
    odc::data::CopyFromYASKGrid(data.m_velocityZ, context->vel_z, 0, 0, 0, nx, ny, nz+1);

}

void copyStressGridsFromYASK(odc::data::SoA data, STENCIL_CONTEXT context) {


    int_pt nx = data.m_numXGridPoints;
    int_pt ny = data.m_numYGridPoints;
    int_pt nz = data.m_numZGridPoints;

    odc::data::CopyFromYASKGrid(data.m_stressXX, context->stress_xx, 0, 0, 0, nx, ny, nz+2);
    odc::data::CopyFromYASKGrid(data.m_stressXY, context->stress_xy, 0, 0, 0, nx, ny, nz+2);
    odc::data::CopyFromYASKGrid(data.m_stressXZ, context->stress_xz, 0, 0, 0, nx, ny, nz+2);
    odc::data::CopyFromYASKGrid(data.m_stressYY, context->stress_yy, 0, 0, 0, nx, ny, nz+2);
    odc::data::CopyFromYASKGrid(data.m_stressYZ, context->stress_yz, 0, 0, 0, nx, ny, nz+2);
    odc::data::CopyFromYASKGrid(data.m_stressZZ, context->stress_zz, 0, 0, 0, nx, ny, nz+2);

}




int main( int i_argc, char *i_argv[] ) {
    // TODO: Log
    std::cout << "welcome to AWP-ODC" << std::endl;



    // TODO: Log
    std::cout << "starting MPI" << std::endl;
    // Fire up MPI, dummy call for non-mpi compilation
    odc::parallel::Mpi::initialize( i_argc, i_argv );


    // parse options
    odc::io::OptionParser l_options( i_argc, i_argv );


    // TODO: Log
    std::cout << "setting up data structures" << std::endl;

    // initialize data layout
    odc::data::SoA l_data;
    l_data.initialize(l_options.m_nX, l_options.m_nY, l_options.m_nZ);
    // allocate memory
    // TODO: Log
    std::cout << "allocating memory: " << l_data.getSize() << " GiB" << std::endl;
    l_data.allocate();

    // set up mesh data structure
    odc::data::Mesh l_mesh(l_options, l_data);


    // set up checkpoint writer
    odc::io::CheckpointWriter l_checkpoint(l_options.m_chkFile, l_options.m_nD,
                                           l_options.m_nTiSkp, l_data.m_numZGridPoints);

    l_checkpoint.writeInitialStats(l_options.m_nTiSkp, l_options.m_dT, l_options.m_dH,
                                   l_data.m_numXGridPoints, l_data.m_numYGridPoints,
                                   l_data.m_numZGridPoints, l_options.m_numTimesteps,
                                   l_options.m_arbc, l_options.m_nPc, l_options.m_nVe,
                                   l_options.m_fac, l_options.m_q0, l_options.m_ex, l_options.m_fp,
                                   l_mesh.m_vse, l_mesh.m_vpe, l_mesh.m_dde);


    // set up absorbing boundary condition data structure
    odc::data::Cerjan l_cerjan(l_options, l_data);

    // set up output writer
    std::cout << "initialized output writer: " << std::endl;
    odc::io::OutputWriter l_output(l_options);

    // initialize sources
    // TODO: Log
    std::cout << "initializing sources" << std::endl;

    // TODO: Number of nodes (nx, ny, nz) should be aware of MPI partitioning.
    odc::io::Sources l_sources(l_options.m_iFault,
                               l_options.m_nSrc,
                               l_options.m_readStep,
                               l_options.m_nSt,
                               l_options.m_nZ,
                               l_options.m_nX, l_options.m_nY, l_options.m_nZ,
                               l_options.m_inSrc,
                               l_options.m_inSrcI2 );


    // Source function timestep
    int_pt source_step = 0;

    // If one or more source fault nodes are owned by this process then call "addsrc" to update the stress tensor values
    std::cout << "Add initial rupture source" << std::endl;
    l_sources.addsrc(source_step, l_options.m_dH, l_options.m_dT, l_options.m_nSt,
                     l_options.m_readStep, 3, l_data.m_stressXX, l_data.m_stressYY,
                     l_data.m_stressZZ, l_data.m_stressXY, l_data.m_stressYZ, l_data.m_stressXZ);
    source_step++;

    // Calculate strides
    int_pt strideX = (l_data.m_numYGridPoints+2*odc::constants::boundary)*(l_data.m_numZGridPoints+2*odc::constants::boundary);
    int_pt strideY = l_data.m_numZGridPoints+2*odc::constants::boundary;
    int_pt strideZ = 1;

    int_pt lamMuStrideY = (1 + 2*odc::constants::boundary);
    int_pt lamMuStrideX = (l_data.m_numYGridPoints+2*odc::constants::boundary) * lamMuStrideY;


    // Perform initialization for YASK grids
    STENCIL_EQUATIONS stencils;
    STENCIL_CONTEXT context;

    context.dn = 1;
    context.dx = l_data.m_numXGridPoints;
    context.dy = l_data.m_numYGridPoints;
    context.dz = l_data.m_numZGridPoints;

    context.allocGrids();
    context.allocParams();

    // Copy data from Grid3D structures into YASK structures
    context->h = l_options.m_dH;
    context->delta_t = l_options.m_dT;
    writeAllGridsToYASK(l_data, l_mesh, l_cerjan, context);

    for (auto stencil : stencils.stencils) {
        stencil->init(context);
    }

    StencilBase *velocityStencil = stencils.stencils[0];
    StencilBase *stressStencil = stencils.stencils[1];


    //  Main Loop Starts
    for (int_pt tstep=1; tstep<=l_options.m_numTimesteps; tstep++) {
        real time = (tstep-1)*l_options.m_dT;
        std::cout << "Beginning  timestep: " << tstep << std::endl;

        // Velocity update
        applyYASKStencil(context, velocityStencil, time);
        copyVelocityGridsFromYASK(l_data, context);


        // Free surface velocity update
        update_free_surface_boundary_velocity(&l_data.m_velocityX[0][0][0], &l_data.m_velocityY[0][0][0], &l_data.m_velocityZ[0][0][0], strideX, strideY, strideZ,
                                              l_data.m_numXGridPoints, l_data.m_numYGridPoints, l_data.m_numZGridPoints, &l_mesh.m_lam_mu[0][0][0], lamMuStrideX, lamMuStrideY);
        writeVelocityGridsToYASK(l_data, context);


        // Stress update
        applyYASKStencil(context, stressStencil, time);
        copyStressGridsFromYASK(l_data, context);


        // Free surface stress update
        update_free_surface_boundary_stress(&l_data.m_stressZZ[0][0][0], &l_data.m_stressXZ[0][0][0], &l_data.m_stressYZ[0][0][0],
                                            strideX, strideY, strideZ, l_data.m_numXGridPoints, l_data.m_numYGridPoints, l_data.m_numZGridPoints);


        // Fault source update
        if (tstep < l_options.m_nSt) {


            update_stress_from_fault_sources(source_step, l_options.m_readStep, 3,
                                             &l_sources.m_ptpSrc[0], l_sources.m_nPsrc,
                                             l_data.m_numXGridPoints, l_data.m_numYGridPoints, l_data.m_numZGridPoints,
                                             strideX, strideY, strideZ,
                                             &l_sources.m_ptAxx[0], &l_sources.
                            m_ptAxy[0], &l_sources.m_ptAxz[0], &l_sources.m_ptAyy[0], &l_sources.m_ptAyz[0],
                                             &l_sources.m_ptAzz[0], &l_data.m_stressXX[0][0][0], &l_data.m_stressXY[0][0][0],
                                             &l_data.m_stressXZ[0][0][0], &l_data.m_stressYY[0][0][0], &l_data.m_stressYZ[0][0][0],
                                             &l_data.m_stressZZ[0][0][0], l_options.m_dT, l_options.m_dH);

            source_step++;
        }

        // Write data back to YASK for next timestep
        writeStressGridsToYASK(l_data, context);


        l_output.update(tstep, l_data.m_velocityX, l_data.m_velocityY, l_data.m_velocityZ,
                        l_data.m_numZGridPoints);

        l_checkpoint.writeUpdatedStats(tstep, l_data.m_velocityX, l_data.m_velocityY,
                                       l_data.m_velocityZ);

    }

    // release memory
    // TODO: log
    std::cout << "releasing memory" << std::endl;
    l_data.finalize();
    l_checkpoint.finalize();
    l_output.finalize();
    l_cerjan.finalize();
    l_mesh.finalize();

    // close mpi
    // TODO: log
    std::cout << "closing mpi" << std::endl;
    odc::parallel::Mpi::finalize();


}




