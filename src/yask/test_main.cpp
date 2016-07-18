/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2016, Intel Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

*****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "stencil.hpp"

// Set MODEL_CACHE to 1 or 2 to model that cache level
// and create a global cache object here.
#ifdef MODEL_CACHE
Cache cache(MODEL_CACHE);
#endif

// Fix bsize, if needed, to fit into rsize and be a multiple of mult.
// Return number of blocks.
idx_t findNumSubsets(idx_t& bsize, const string& bname,
                    idx_t rsize, const string& rname,
                    idx_t mult, string dim) {
    if (bsize < 1) bsize = rsize; // 0 => use full size.
    if (bsize > rsize) bsize = rsize;
    bsize = ROUND_UP(bsize, mult);
    idx_t nblks = (rsize + bsize - 1) / bsize;
    idx_t rem = rsize % bsize;
    idx_t nfull_blks = rem ? (nblks - 1) : nblks;

    cout << "#  '" << dim << "' dimension: dividing " << rname << " of size " <<
        rsize << " into " << nfull_blks << " " << bname << "(s) of size " << bsize;
    if (rem)
        cout << " plus 1 remainder " << bname << " of size " << rem;
    cout << "." << endl;
    return nblks;
}
idx_t findNumBlocks(idx_t& bsize, idx_t rsize, idx_t mult, string dim) {
    return findNumSubsets(bsize, "block", rsize, "region", mult, dim);
}
idx_t findNumRegions(idx_t& rsize, idx_t dsize, idx_t mult, string dim) {
    return findNumSubsets(rsize, "region", dsize, "rank", mult, dim);
}

// Parse command-line args, run kernel, run validation if requested.
int main(int argc, char** argv)
{
  int nx = 128, ny = 128, nz = 64;
  

    idx_t dt = 60;     // number of time-steps per trial, over which performance is averaged.
    idx_t dn = 1, dx = nx, dy = ny, dz = nz;
    idx_t rt = 1;                         // wavefront time steps.
    idx_t rn = 0, rx = 0, ry = 0, rz = 0;  // region sizes (0 => use rank size).
    idx_t bt = 1;                          // temporal block size.
    idx_t bn = 1, bx = DEF_BLOCK_SIZE, by = DEF_BLOCK_SIZE, bz = DEF_BLOCK_SIZE;  // size of cache blocks.
    idx_t pn = 0, px = 0, py = 0, pz = 0; // padding.
    bool validate = false;


    // Round up vars as needed.
    dt = roundUp(dt, CPTS_T, "# rank size in t (time steps)");
    dn = roundUp(dn, CPTS_N, "# rank size in n");
    dx = roundUp(dx, CPTS_X, "# rank size in x");
    dy = roundUp(dy, CPTS_Y, "# rank size in y");
    dz = roundUp(dz, CPTS_Z, "# rank size in z");

    // Determine num regions based on region sizes.
    // Also fix up region sizes as needed.
    cout << "#\n# Regions:" << endl;
    idx_t nrt = findNumRegions(rt, dt, CPTS_T, "t");
    idx_t nrn = findNumRegions(rn, dn, CPTS_N, "n");
    idx_t nrx = findNumRegions(rx, dx, CPTS_X, "x");
    idx_t nry = findNumRegions(ry, dy, CPTS_Y, "y");
    idx_t nrz = findNumRegions(rz, dz, CPTS_Z, "z");
    idx_t nr = nrn * nrx * nry * nrz;
    cout << "# num-regions = " << nr << endl;

    // Determine num blocks based on block sizes.
    // Also fix up block sizes as needed.
    cout << "#\n# Blocks:" << endl;
    idx_t nbt = findNumBlocks(bt, rt, CPTS_T, "t");
    idx_t nbn = findNumBlocks(bn, rn, CPTS_N, "n");
    idx_t nbx = findNumBlocks(bx, rx, CPTS_X, "x");
    idx_t nby = findNumBlocks(by, ry, CPTS_Y, "y");
    idx_t nbz = findNumBlocks(bz, rz, CPTS_Z, "z");
    idx_t nb = nbn * nbx * nby * nbz;
    cout << "# num-blocks-per-region = " << nb << endl;

    // Round up padding as needed.
    pn = roundUp(pn, VLEN_N, "# extra padding in n");
    px = roundUp(px, VLEN_X, "# extra padding in x");
    py = roundUp(py, VLEN_Y, "# extra padding in y");
    pz = roundUp(pz, VLEN_Z, "# extra padding in z");

    // Round up halos as needed.
    if (STENCIL_ORDER % 2) {
        cerr << "error: stencil-order not even." << endl;
        exit(1);
    }
    idx_t halo_size = STENCIL_ORDER/2; // TODO: make dim-specific.
    idx_t hn = 0;                      // TODO: support N halo.
    idx_t hx = ROUND_UP(halo_size, VLEN_X);
    idx_t hy = ROUND_UP(halo_size, VLEN_Y);
    idx_t hz = ROUND_UP(halo_size, VLEN_Z);

    std::cout << "Halo sizes: " << hx << ' ' << hy << ' ' << hz << std::endl;
    
    printf("\n# Sizes in points per grid (t*n*x*y*z):\n");
    printf("# vector-size = %d*%d*%d*%d*%d\n", VLEN_T, VLEN_N, VLEN_X, VLEN_Y, VLEN_Z);
    printf("# cluster-size = %d*%d*%d*%d*%d\n", CPTS_T, CPTS_N, CPTS_X, CPTS_Y, CPTS_Z);
    printf("# block-size = %ld*%ld*%ld*%ld*%ld\n", bt, bn, bx, by, bz);
    printf("# region-size = %ld*%ld*%ld*%ld*%ld\n", rt, rn, rx, ry, rz);
    printf("# rank-size = %ld*%ld*%ld*%ld*%ld\n", dt, dn, dx, dy, dz);
//    printf(" overall-size = %ld*%ld*%ld*%ld*%ld\n", dt, dn, dx * num_ranks, dy, dz);
    cout << "# \n# Other settings:\n";
    printf("# stencil-order = %d\n", STENCIL_ORDER); // really just used for halo size.
    printf("# stencil-shape = " STENCIL_NAME "\n");
    printf("# time-dim-size = %d\n", TIME_DIM_SIZE);
    printf("# vector-len = %d\n", VLEN);
    printf("# padding = %ld+%ld+%ld+%ld\n", pn, px, py, pz);
    printf("# halos = %ld+%ld+%ld+%ld\n", hn, hx, hy, hz);
    printf("# manual-L1-prefetch-distance = %d\n", PFDL1);
    printf("# manual-L2-prefetch-distance = %d\n", PFDL2);

    // Stencil functions.
    idx_t scalar_fp_ops = 0;
    STENCIL_EQUATIONS stencils;
    idx_t num_stencils = stencils.stencils.size();
    cout << endl << "# Num stencil equations = " << num_stencils << ":" << endl;
    for (auto stencil : stencils.stencils) {
        idx_t fpos = stencil->get_scalar_fp_ops();
        cout << "#  '" << stencil->get_name() << "': " <<
            fpos << " FP ops per point." << endl;
        scalar_fp_ops += fpos;
    }

    cout << endl;
    const idx_t numpts = dn*dx*dy*dz;
    const idx_t rank_numpts = dt * numpts;
    const idx_t tot_numpts = rank_numpts * 1;//num_ranks;
    cout << "# Points to calculate per time step: " << printWithMultiplier(numpts) <<
        " (" << dn << " * " << dx << " * " << dy << " * " << dz << ")" << endl;
    cout << "# Points to calculate per rank: " << printWithMultiplier(rank_numpts) << endl;
    cout << "# Points to calculate overall: " << printWithMultiplier(rank_numpts) << endl;
    const idx_t numFpOps = numpts * scalar_fp_ops;
    const idx_t rank_numFpOps = dt * numpts * scalar_fp_ops;
    const idx_t tot_numFpOps = rank_numFpOps * 1;//num_ranks;
    cout << "# FP ops (est) per point: " << scalar_fp_ops << endl;
    cout << "# FP ops (est) per rank: " << printWithMultiplier(rank_numFpOps) << endl;
    cout << "# FP ops (est) overall: " << printWithMultiplier(tot_numFpOps) << endl;
 

    // Context for evaluating results.
    STENCIL_CONTEXT context;
    context.num_ranks = 1;//num_ranks;
    context.my_rank = 0; //my_rank; PPP is this ok?
    context.comm = 0;
    
    // Save sizes in context struct.
    // - dt not used for allocation; set later.
    context.dn = dn;
    context.dx = dx;
    context.dy = dy;
    context.dz = dz;
    
    context.rt = rt;
    context.rn = rn;
    context.rx = rx;
    context.ry = ry;
    context.rz = rz;

    context.bt = bt;
    context.bn = bn;
    context.bx = bx;
    context.by = by;
    context.bz = bz;

    context.pn = pn;
    context.px = px;
    context.py = py;
    context.pz = pz;

    context.hn = hn;
    context.hx = hx;
    context.hy = hy;
    context.hz = hz;
    
    // Alloc mem.
    cout << endl;
    cout << "# Allocating grids..." << endl;
    context.allocGrids();
    cout << "# Allocating parameters..." << endl;
    context.allocParams();
    cout << "# Allocating buffers..." << endl;
    context.setupMPI();    


    (*(context.h))() = 10.;
    (*(context.delta_t))() = 0.0025;
    
    
    for(int x=0; x<nx; x++)
    {
      for(int y=0; y<ny; y++)
      {
        for(int z=0; z<nz; z++)
        {
          *(context.vel_x->getElemPtr(0,x,y,z,false)) = 0.;
          *(context.vel_y->getElemPtr(0,x,y,z,false)) = 0.;
          *(context.vel_z->getElemPtr(0,x,y,z,false)) = 0.;


          double stress = 0.;
          if(x >= 23 && x <= 28 && y >= 23 && y <= 28 && z >= 23 && z <= 28)
          {
            stress = fabs(x-25)*fabs(y-26)*fabs(z-25.5)*10000;
          }
          
          *(context.stress_xx->getElemPtr(0,x,y,z,false)) = stress;
          *(context.stress_yy->getElemPtr(0,x,y,z,false)) = stress;
          *(context.stress_zz->getElemPtr(0,x,y,z,false)) = stress;

          
          *(context.stress_xy->getElemPtr(0,x,y,z,false)) = 0.;
          *(context.stress_xz->getElemPtr(0,x,y,z,false)) = 0.;
          *(context.stress_yz->getElemPtr(0,x,y,z,false)) = 0.;
          
          *(context.lambda->getElemPtr(x,y,z,false)) = 64801900800;
          *(context.rho->getElemPtr(x,y,z,false)) = 2700;
          *(context.mu->getElemPtr(x,y,z,false)) = 32398099200;
          *(context.sponge->getElemPtr(x,y,z,false)) = 1.;          
          
        }
      }
    }
    
    

    context.dt = TIME_DIM_SIZE;
    for(int T=0; T<2000; T++)
    {
        stencils.calc_rank_opt(context);

        for(int t=1; t<=TIME_DIM_SIZE; t++)
        {
#ifndef GLOBAL_STATS        
          cout << t + T * context.dt << ' ' << *(context.vel_y->getElemPtr(t,23,22,23,false)) << endl;
#else
          if(!((t + T * context.dt) % 20))
          {
            double total = 0.;
            for(int x=0; x<nx; x++)
            {
              for(int y=0; y<ny; y++)
              {
                for(int z=0; z<nz; z++)
                {
                  total += fabs(*(context.vel_x->getElemPtr(t,x,y,z,false)));
                  total += fabs(*(context.vel_y->getElemPtr(t,x,y,z,false)));
                  total += fabs(*(context.vel_z->getElemPtr(t,x,y,z,false)));
                }
              }
            }
            cout << t + T * context.dt << ' ' << total << endl;
          }
#endif
        }
    }

    
    return 0;
}
