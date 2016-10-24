/**
 @author Josh Tobin (rjtobin AT ucsd.edu)
 
 @section DESCRIPTION
 Patch decomposition data structure.
 
 @section LICENSE
 Copyright (c) 2015-2016, Regents of the University of California
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#if !defined(PATCHDECOMP_H)
#define PATCHDECOMP_H

#include "io/OptionParser.h"
#include "Patch.hpp"

class PatchDecomp
{
public:
  /*
   * Grid-of-patches is the 3d grid that stores the spatial location of each patch.
   * Eg. if there are exactly two patches, split in the x direction, then the two
   * grid-of-patches coordinates are (0,0,0) and (1,0,0).
   * Additionally each patch has a unique patch_id.
   */
  //! map patch id to which patch it is in the 3d grid-of-patches
  int_pt* m_idToGridX;
  int_pt* m_idToGridY;
  int_pt* m_idToGridZ;

  //! map coordinate in 3d grid of patches into a patch_id
  int*** m_coordToId; // TODO(Josh): should be int_pt for larger runs

  //! map patch_id to the corresponding Patch object
  Patch* m_patches;

  //! store the intended size of each patch.  Note that patches at the
  //! end might be smaller.
  int_pt m_patchXSize, m_patchYSize, m_patchZSize;

  //! number of patches in each direction
  int_pt m_numXPatches, m_numYPatches, m_numZPatches;

  //! total number of patches
  int_pt m_numPatches;

  //! total number of grid points across all patches, in each dimension
  int_pt m_numXGridPoints,m_numYGridPoints,m_numZGridPoints;

  //! total number of grid points in this node
  int_pt m_numGridPoints;

  //! the width of the patch overlap region.  Same in each dimension
  int_pt m_overlapSize;

  /**
   * Allocated the patches and determine dimensions.
   *
   * @param i_options arguments from command line.
   * @param xSize size of computational domain in x direction
   * @param ySize size of computational domain in y direction
   * @param zSize size of computational domain in z direction
   * @param xPatchSize size of patch in x direction.
   * @param yPatchSize size of patch in y direction.
   * @param zPatchSize size of patch in z direction.
   * @param overlapSize size of patch overlap region in each dimension
   **/  
  void initialize(odc::io::OptionParser i_options, int_pt xSize, int_pt ySize, int_pt zSize,
                  int_pt xPatchSize, int_pt yPatchSize, int_pt zPatchSize,
                  int_pt overlapSize);

  /**
   * Deallocate patches.
   **/    
  void finalize();

  /**
   * Synchronize all patches.
   *
   * @param allGrids synchronize static arrays (eg. true during initializtion, default false)
   **/      
  void synchronize(bool allGrids=false);

  /**
   * Maps coordinates of a point in the domain to the corresponding patch_id.
   *
   * @param x x-coordinate 
   * @param y y-coordinate 
   * @param z z-coordinate 
   **/        
  int    globalToPatch(int_pt x, int_pt y, int_pt z);

  /**
   * Maps coordinates of a point in the domain to the x-coordinate of that point in
   * its corresponding patch.
   *
   * @param x x-coordinate 
   * @param y y-coordinate 
   * @param z z-coordinate 
   **/          
  int_pt globalToLocalX(int_pt x, int_pt y, int_pt z);

  /**
   * Maps coordinates of a point in the domain to the y-coordinate of that point in
   * its corresponding patch.
   *
   * @param x x-coordinate 
   * @param y y-coordinate 
   * @param z z-coordinate 
   **/            
  int_pt globalToLocalY(int_pt x, int_pt y, int_pt z);

  /**
   * Maps coordinates of a point in the domain to the z-coordinate of that point in
   * its corresponding patch.
   *
   * @param x x-coordinate 
   * @param y y-coordinate 
   * @param z z-coordinate 
   **/            
  int_pt globalToLocalZ(int_pt x, int_pt y, int_pt z);

  /**
   * Maps coordinates of a point in its corresponding patch and a patch_id to
   * the corresponding (node) global x-coordinate
   *
   * @param x x-coordinate 
   * @param y y-coordinate 
   * @param z z-coordinate 
   **/              
  int_pt localToGlobalX(int_pt i_ptch, int_pt x, int_pt y, int_pt z);

  /**
   * Maps coordinates of a point in its corresponding patch and a patch_id to
   * the corresponding (node) global y-coordinate
   *
   * @param x x-coordinate 
   * @param y y-coordinate 
   * @param z z-coordinate 
   **/                
  int_pt localToGlobalY(int_pt i_ptch, int_pt x, int_pt y, int_pt z);

  /**
   * Maps coordinates of a point in its corresponding patch and a patch_id to
   * the corresponding (node) global z-coordinate
   *
   * @param x x-coordinate 
   * @param y y-coordinate 
   * @param z z-coordinate 
   **/              
  int_pt localToGlobalZ(int_pt i_ptch, int_pt x, int_pt y, int_pt z);

  /**
   * Maps global coordinates to the x velocity of that point,
   * at a given time index (not used for AWP-vanilla, just YASK)
   *
   * @param x x-coordinate 
   * @param y y-coordinate 
   * @param z z-coordinate
   * @param i_timestep desired time index 
   **/                
  real   getVelX(int_pt i_x, int_pt i_y, int_pt i_z, int_pt i_timestep);

  /**
   * Maps patch & local coordinates to the x velocity of that point,
   * at a given time index (not used for AWP-vanilla, just YASK)
   *
   * @param i_ptch id of patch containing this point 
   * @param i_locX x-coordinate 
   * @param i_locY y-coordinate 
   * @param i_locZ z-coordinate
   * @param i_timestep desired time index 
   **/                
  real   getVelX(int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep);

  /**
   * Maps global coordinates to the y velocity of that point,
   * at a given time index (not used for AWP-vanilla, just YASK)
   *
   * @param x x-coordinate 
   * @param y y-coordinate 
   * @param z z-coordinate
   * @param i_timestep desired time index 
   **/                  
  real   getVelY(int_pt i_x, int_pt i_y, int_pt i_z, int_pt i_timestep);

  /**
   * Maps patch & local coordinates to the y velocity of that point,
   * at a given time index (not used for AWP-vanilla, just YASK)
   *
   * @param i_ptch id of patch containing this point 
   * @param i_locX x-coordinate 
   * @param i_locY y-coordinate 
   * @param i_locZ z-coordinate
   * @param i_timestep desired time index 
   **/                
  real   getVelY(int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep);

  /**
   * Maps global coordinates to the z velocity of that point,
   * at a given time index (not used for AWP-vanilla, just YASK)
   *
   * @param x x-coordinate 
   * @param y y-coordinate 
   * @param z z-coordinate
   * @param i_timestep desired time index 
   **/                  
  real   getVelZ(int_pt i_x, int_pt i_y, int_pt i_z, int_pt i_timestep);

  /**
   * Maps patch & local coordinates to the z velocity of that point,
   * at a given time index (not used for AWP-vanilla, just YASK)
   *
   * @param i_ptch id of patch containing this point 
   * @param i_locX x-coordinate 
   * @param i_locY y-coordinate 
   * @param i_locZ z-coordinate
   * @param i_timestep desired time index 
   **/                
  real   getVelZ(int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep);

  /**
   * Sets x velocity of the point at given patchId and local coords,
   * at a given time index (not used for AWP-vanilla, just YASK)
   *
   * @param i_vel new velocity value 
   * @param i_ptch id of patch containing this point 
   * @param i_locX x-coordinate 
   * @param i_locY y-coordinate 
   * @param i_locZ z-coordinate
   * @param i_timestep desired time index 
   **/                
  void   setVelX(real i_vel, int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep);

  /**
   * Sets y velocity of the point at given patchId and local coords,
   * at a given time index (not used for AWP-vanilla, just YASK)
   *
   * @param i_vel new velocity value 
   * @param i_ptch id of patch containing this point 
   * @param i_locX x-coordinate 
   * @param i_locY y-coordinate 
   * @param i_locZ z-coordinate
   * @param i_timestep desired time index 
   **/                
  void   setVelY(real i_vel, int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep);

  /**
   * Sets z velocity of the point at given patchId and local coords,
   * at a given time index (not used for AWP-vanilla, just YASK)
   *
   * @param i_vel new velocity value 
   * @param i_ptch id of patch containing this point 
   * @param i_locX x-coordinate 
   * @param i_locY y-coordinate 
   * @param i_locZ z-coordinate
   * @param i_timestep desired time index 
   **/                
  void   setVelZ(real i_vel, int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep);

  /**
   * Maps patch & local coordinates to the xx stress of that point,
   * at a given time index (not used for AWP-vanilla, just YASK)
   *
   * @param i_ptch id of patch containing this point 
   * @param i_locX x-coordinate 
   * @param i_locY y-coordinate 
   * @param i_locZ z-coordinate
   * @param i_timestep desired time index 
   **/                
  real   getStressXX(int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep);
  
  /**
   * Maps patch & local coordinates to the xy stress of that point,
   * at a given time index (not used for AWP-vanilla, just YASK)
   *
   * @param i_ptch id of patch containing this point 
   * @param i_locX x-coordinate 
   * @param i_locY y-coordinate 
   * @param i_locZ z-coordinate
   * @param i_timestep desired time index 
   **/                
  real   getStressXY(int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep);
  
  /**
   * Maps patch & local coordinates to the xz stress of that point,
   * at a given time index (not used for AWP-vanilla, just YASK)
   *
   * @param i_ptch id of patch containing this point 
   * @param i_locX x-coordinate 
   * @param i_locY y-coordinate 
   * @param i_locZ z-coordinate
   * @param i_timestep desired time index 
   **/                
  real   getStressXZ(int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep);
  
  /**
   * Maps patch & local coordinates to the yy stress of that point,
   * at a given time index (not used for AWP-vanilla, just YASK)
   *
   * @param i_ptch id of patch containing this point 
   * @param i_locX x-coordinate 
   * @param i_locY y-coordinate 
   * @param i_locZ z-coordinate
   * @param i_timestep desired time index 
   **/                
  real   getStressYY(int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep);
  
  /**
   * Maps patch & local coordinates to the yz stress of that point,
   * at a given time index (not used for AWP-vanilla, just YASK)
   *
   * @param i_ptch id of patch containing this point 
   * @param i_locX x-coordinate 
   * @param i_locY y-coordinate 
   * @param i_locZ z-coordinate
   * @param i_timestep desired time index 
   **/                
  real   getStressYZ(int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep);
  
  /**
   * Maps patch & local coordinates to the zz stress of that point,
   * at a given time index (not used for AWP-vanilla, just YASK)
   *
   * @param i_ptch id of patch containing this point 
   * @param i_locX x-coordinate 
   * @param i_locY y-coordinate 
   * @param i_locZ z-coordinate
   * @param i_timestep desired time index 
   **/                
  real   getStressZZ(int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep);

  /**
   * Sets xx stress of the point at given patchId and local coords,
   * at a given time index (not used for AWP-vanilla, just YASK)
   *
   * @param i_stress new velocity value 
   * @param i_ptch id of patch containing this point 
   * @param i_locX x-coordinate 
   * @param i_locY y-coordinate 
   * @param i_locZ z-coordinate
   * @param i_timestep desired time index 
   **/                
  void   setStressXX(real i_stress, int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep);

  /**
   * Sets xy stress of the point at given patchId and local coords,
   * at a given time index (not used for AWP-vanilla, just YASK)
   *
   * @param i_stress new velocity value 
   * @param i_ptch id of patch containing this point 
   * @param i_locX x-coordinate 
   * @param i_locY y-coordinate 
   * @param i_locZ z-coordinate
   * @param i_timestep desired time index 
   **/                
  void   setStressXY(real i_stress, int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep);

  /**
   * Sets xz stress of the point at given patchId and local coords,
   * at a given time index (not used for AWP-vanilla, just YASK)
   *
   * @param i_stress new velocity value 
   * @param i_ptch id of patch containing this point 
   * @param i_locX x-coordinate 
   * @param i_locY y-coordinate 
   * @param i_locZ z-coordinate
   * @param i_timestep desired time index 
   **/                
  void   setStressXZ(real i_stress, int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep);

   /**
   * Sets yy stress of the point at given patchId and local coords,
   * at a given time index (not used for AWP-vanilla, just YASK)
   *
   * @param i_stress new velocity value 
   * @param i_ptch id of patch containing this point 
   * @param i_locX x-coordinate 
   * @param i_locY y-coordinate 
   * @param i_locZ z-coordinate
   * @param i_timestep desired time index 
   **/                
  void   setStressYY(real i_stress, int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep);

  /**
   * Sets yz stress of the point at given patchId and local coords,
   * at a given time index (not used for AWP-vanilla, just YASK)
   *
   * @param i_stress new velocity value 
   * @param i_ptch id of patch containing this point 
   * @param i_locX x-coordinate 
   * @param i_locY y-coordinate 
   * @param i_locZ z-coordinate
   * @param i_timestep desired time index 
   **/                
  void   setStressYZ(real i_stress, int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep);

  /**
   * Sets zz stress of the point at given patchId and local coords,
   * at a given time index (not used for AWP-vanilla, just YASK)
   *
   * @param i_stress new velocity value 
   * @param i_ptch id of patch containing this point 
   * @param i_locX x-coordinate 
   * @param i_locY y-coordinate 
   * @param i_locZ z-coordinate
   * @param i_timestep desired time index 
   **/                
  void   setStressZZ(real i_stress, int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep);

  /**
   * Perform MPI synchronization of velocity grids in a given direction.
   *
   * @param i_dir direction of boundaryto synchronize: 0 = x, 1 = y, 2 = z
   * @param i_timestep timestep to copy from
   **/                  
  void velMpiSynchronize(int i_dir, int_pt i_timestep);

  /**
   * Perform MPI synchronization of stress grids in a given direction.
   *
   * @param i_dir direction of boundaryto synchronize: 0 = x, 1 = y, 2 = z
   * @param i_timestep timestep to copy from
   **/                  
  void stressMpiSynchronize(int i_dir, int_pt i_timestep);  

 /**
   * Copy velocities to 1D buffers, ordered as required by OutputWriter.
   *
   * @param o_bufferX buffer for x velocities
   * @param o_bufferY buffer for y velocities
   * @param o_bufferZ buffer for z velocities
   * @param i_firstX first x-coordinate to copy
   * @param i_lastX last x-coordinate to copy
   * @param i_skipX number of x-coordinates to skip each iteration   
   * @param i_firstY first y-coordinate to copy
   * @param i_lastY last y-coordinate to copy
   * @param i_skipY number of y-coordinates to skip each iteration   
   * @param i_firstZ first z-coordinate to copy
   * @param i_lastZ last z-coordinate to copy
   * @param i_skipZ number of z-coordinates to skip each iteration   
   * @param i_timestep desired time index 
   **/                  
  void copyVelToBuffer(real* o_bufferX, real* o_bufferY, real* o_bufferZ,
                       int_pt i_firstX, int_pt i_lastX, int_pt i_skipX,
                       int_pt i_firstY, int_pt i_lastY, int_pt i_skipY,
                       int_pt i_firstZ, int_pt i_lastZ, int_pt i_skipZ,
                       int_pt i_timestep);

  /**
   * Copy velocity boundary to 1D buffers for MPI (DEPRECATED).
   *
   * @param o_buffer buffer for velocities
   * @param i_dirX x direction of boundary
   * @param i_dirY y direction of boundary
   * @param i_dirZ z direction of boundary
   * @param i_timestep timestep to copy from
   **/                  
  void copyVelBoundaryToBuffer(real* o_buffer, int i_dirX, int i_dirY,
			       int i_dirZ, int_pt i_timestep);

  /**
   * Copy portion of velocity boundary to 1D buffers for MPI.
   * Note that the start and end values for the dimension orthogonal
   * to the slice to be copied are _ignored_
   *
   * @param o_buffer buffer for velocities
   * @param i_dirX x direction of boundary
   * @param i_dirY y direction of boundary
   * @param i_dirZ z direction of boundary
   * @param i_startX first x val to copy
   * @param i_startY first y val to copy
   * @param i_startZ first z val to copy
   * @param i_endX last x val to copy
   * @param i_endY last y val to copy
   * @param i_endZ last z val to copy
   * @param i_timestep timestep to copy from
   **/                  
  void copyVelBoundaryToBuffer(real* o_buffer, int i_dirX, int i_dirY,
        	               int i_dirZ, int_pt i_startX, int_pt i_startY, int_pt i_startZ,
			       int_pt i_endX, int_pt i_endY, int_pt i_endZ, int_pt i_timestep);

  
  /**
   * Copy velocity boundary from 1D buffers from MPI (DEPRECATED).
   *
   * @param o_buffer buffer for velocities
   * @param i_dirX x direction of boundary
   * @param i_dirY y direction of boundary
   * @param i_dirZ z direction of boundary
   * @param i_timestep timestep to copy to
   **/                  
  void copyVelBoundaryFromBuffer(real* o_buffer, int i_dirX, int i_dirY,
			       int i_dirZ, int_pt i_timestep);

  /**
   * Copy portion of velocity boundary from 1D buffers for MPI.
   * Note that the start and end values for the dimension orthogonal
   * to the slice to be copied are _ignored_
   *
   * @param o_buffer buffer for velocities
   * @param i_dirX x direction of boundary
   * @param i_dirY y direction of boundary
   * @param i_dirZ z direction of boundary
   * @param i_startX first x val to copy
   * @param i_startY first y val to copy
   * @param i_startZ first z val to copy
   * @param i_endX last x val to copy
   * @param i_endY last y val to copy
   * @param i_endZ last z val to copy
   * @param i_timestep timestep to copy from
   **/                  
  void copyVelBoundaryFromBuffer(real* o_buffer, int i_dirX, int i_dirY,
        	               int i_dirZ, int_pt i_startX, int_pt i_startY, int_pt i_startZ,
			       int_pt i_endX, int_pt i_endY, int_pt i_endZ, int_pt i_timestep);

  /**
   * Copy stress boundary to 1D buffers for MPI (DEPRECATED).
   *
   * @param o_buffer buffer for stress
   * @param i_dirX x direction of boundary
   * @param i_dirY y direction of boundary
   * @param i_dirZ z direction of boundary
   * @param i_timestep timestep to copy from
   **/                  
  void copyStressBoundaryToBuffer(real* o_buffer, int i_dirX, int i_dirY,
			       int i_dirZ, int_pt i_timestep);

  /**
   * Copy portion of stress boundary to 1D buffers for MPI.
   * Note that the start and end values for the dimension orthogonal
   * to the slice to be copied are _ignored_
   *
   * @param o_buffer buffer for stress
   * @param i_dirX x direction of boundary
   * @param i_dirY y direction of boundary
   * @param i_dirZ z direction of boundary
   * @param i_startX first x val to copy
   * @param i_startY first y val to copy
   * @param i_startZ first z val to copy
   * @param i_endX last x val to copy
   * @param i_endY last y val to copy
   * @param i_endZ last z val to copy
   * @param i_timestep timestep to copy from
   **/                  
  void copyStressBoundaryToBuffer(real* o_buffer, int i_dirX, int i_dirY,
        	               int i_dirZ, int_pt i_startX, int_pt i_startY, int_pt i_startZ,
			       int_pt i_endX, int_pt i_endY, int_pt i_endZ, int_pt i_timestep);
  
  /**
   * Copy stress boundary from 1D buffers from MPI (DEPRECATED).
   *
   * @param o_buffer buffer for stress
   * @param i_dirX x direction of boundary
   * @param i_dirY y direction of boundary
   * @param i_dirZ z direction of boundary
   * @param i_timestep timestep to copy to
   **/                  
  void copyStressBoundaryFromBuffer(real* o_buffer, int i_dirX, int i_dirY,
			       int i_dirZ, int_pt i_timestep);

  /**
   * Copy portion of stress boundary from 1D buffers for MPI.
   * Note that the start and end values for the dimension orthogonal
   * to the slice to be copied are _ignored_
   *
   * @param o_buffer buffer for stress
   * @param i_dirX x direction of boundary
   * @param i_dirY y direction of boundary
   * @param i_dirZ z direction of boundary
   * @param i_startX first x val to copy
   * @param i_startY first y val to copy
   * @param i_startZ first z val to copy
   * @param i_endX last x val to copy
   * @param i_endY last y val to copy
   * @param i_endZ last z val to copy
   * @param i_timestep timestep to copy from
   **/                  
  void copyStressBoundaryFromBuffer(real* o_buffer, int i_dirX, int i_dirY,
        	               int i_dirZ, int_pt i_startX, int_pt i_startY, int_pt i_startZ,
			       int_pt i_endX, int_pt i_endY, int_pt i_endZ, int_pt i_timestep);

  /**
   * Gets max/min tmpvs observed during mesh initialization.
   *
   * @param max return max val (otherwise return min)
   **/                  
  real getVse(bool max);
  /**
   * Gets max/min tmpvp observed during mesh initialization.
   *
   * @param max return max val (otherwise return min)
   **/                    
  real getVpe(bool max);
  /**
   * Gets max/min tmpdd observed during mesh initialization.
   *
   * @param max return max val (otherwise return min)
   **/                    
  real getDde(bool max);
  
};

#endif
