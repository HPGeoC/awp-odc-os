/**
 @author Josh Tobin (rjtobin AT ucsd.edu)
 @author Rajdeep Konwar (rkonwar AT ucsd.edu)
 
 @section DESCRIPTION
 Output writer.

 @section LICENSE
 Copyright (c) 2015-2018, Regents of the University of California
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "OutputWriter.hpp"
#include <cstdio>
#include <cstring>

#ifdef AWP_USE_MPI
void odc::io::OutputWriter::calcRecordingPoints( int *rec_nbgx, int *rec_nedx,
                                                 int *rec_nbgy, int *rec_nedy, int *rec_nbgz, int *rec_nedz,
                                                 int *rec_nxt, int *rec_nyt, int *rec_nzt, MPI_Offset *displacement,
                                                 long int nxt, long int nyt, long int nzt, int rec_NX, int rec_NY, int rec_NZ,
                                                 int NBGX, int NEDX, int NSKPX, int NBGY, int NEDY, int NSKPY,
                                                 int NBGZ, int NEDZ, int NSKPZ, int *coord ) {
  *displacement = 0;

  if( NBGX > nxt * (coord[0] + 1) )
    *rec_nxt  = 0;
  else if( NEDX < nxt * coord[0] + 1 )
    *rec_nxt  = 0;
  else {
    if( nxt * coord[0] >= NBGX ) {
      *rec_nbgx     = (nxt * coord[0] + NBGX - 1) % NSKPX;
      *displacement += (nxt * coord[0] - NBGX) / NSKPX + 1;
    } else
      *rec_nbgx     = NBGX - nxt * coord[0] - 1;  //! since rec_nbgx is 0-based

    if( nxt * (coord[0] + 1) <= NEDX )
      *rec_nedx     = (nxt * (coord[0] + 1) + NBGX - 1) % NSKPX - NSKPX + nxt;
    else
      *rec_nedx     = NEDX - nxt * coord[0] - 1;

    *rec_nxt  = (*rec_nedx - *rec_nbgx) / NSKPX + 1;
  }

  if( NBGY > nyt * (coord[1] + 1) )
    *rec_nyt  = 0;
  else if( NEDY < nyt * coord[1] + 1 )
    *rec_nyt  = 0;
  else {
    if( nyt * coord[1] >= NBGY ) {
      *rec_nbgy     = (nyt * coord[1] + NBGY - 1) % NSKPY;
      *displacement += ((nyt * coord[1] - NBGY) / NSKPY + 1) * rec_NX;
    } else
      *rec_nbgy     = NBGY - nyt * coord[1] - 1;  //! since rec_nbgy is 0-based

    if( nyt * (coord[1] + 1) <= NEDY )
      *rec_nedy     = (nyt * (coord[1] + 1) + NBGY - 1) % NSKPY - NSKPY + nyt;
    else
      *rec_nedy     = NEDY - nyt * coord[1] - 1;

    *rec_nyt  = (*rec_nedy - *rec_nbgy) / NSKPY + 1;
  }

  if( NBGZ > nzt )
    *rec_nzt  = 0;
  else {
    *rec_nbgz = NBGZ - 1;  //! since rec_nbgz is 0-based
    *rec_nedz = NEDZ - 1;
    *rec_nzt  = (*rec_nedz - *rec_nbgz) / NSKPZ + 1;
  }

  if( *rec_nxt == 0 || *rec_nyt == 0 || *rec_nzt == 0 ) {
    *rec_nxt = 0;
    *rec_nyt = 0;
    *rec_nzt = 0;
  }

  //! displacement assumes NPZ=1!
  *displacement *= sizeof( float );
  return;
}
#endif

// TODO(Josh):  Output writer assumes at least some node from each rank is an
//              output point;  should remove this requirement
odc::io::OutputWriter::OutputWriter( odc::io::OptionParser& i_options ) {
  m_firstXNodeToRecord  = i_options.m_nBgX;
  m_firstYNodeToRecord  = i_options.m_nBgY;
  m_firstZNodeToRecord  = i_options.m_nBgZ;

  m_lastXNodeToRecord   = i_options.m_nEdX;
  m_lastYNodeToRecord   = i_options.m_nEdY;
  m_lastZNodeToRecord   = i_options.m_nEdZ;

  m_numXNodesToSkip     = i_options.m_nSkpX;
  m_numYNodesToSkip     = i_options.m_nSkpY;
  m_numZNodesToSkip     = i_options.m_nSkpZ;

  //! If last node is set to -1 then record all nodes
  if( m_lastXNodeToRecord == -1 )
    m_lastXNodeToRecord = i_options.m_nX;

  if( m_lastYNodeToRecord == -1 )
    m_lastYNodeToRecord = i_options.m_nY;

  if( m_lastZNodeToRecord == -1 )
    m_lastZNodeToRecord = i_options.m_nZ;

  //! Make sure that last node to record at is actually a record point.
  //! For example 1:3:9 will record (1, 4, 7), so change it to 1:4:7
  m_lastXNodeToRecord -= (m_lastXNodeToRecord - m_firstXNodeToRecord) % m_numXNodesToSkip;
  m_lastYNodeToRecord -= (m_lastYNodeToRecord - m_firstYNodeToRecord) % m_numYNodesToSkip;
  m_lastZNodeToRecord -= (m_lastZNodeToRecord - m_firstZNodeToRecord) % m_numZNodesToSkip;

  //! Calculate total number of nodes to record
  m_numXNodesToRecord = (m_lastXNodeToRecord - m_firstXNodeToRecord) / m_numXNodesToSkip + 1;
  m_numYNodesToRecord = (m_lastYNodeToRecord - m_firstYNodeToRecord) / m_numYNodesToSkip + 1;
  m_numZNodesToRecord = (m_lastZNodeToRecord - m_firstZNodeToRecord) / m_numZNodesToSkip + 1;

  m_numGlobalXNodesToRecord = m_numXNodesToRecord;
  m_numGlobalYNodesToRecord = m_numYNodesToRecord;
  m_numGlobalZNodesToRecord = m_numZNodesToRecord;

  int_pt globalFirstX = m_firstXNodeToRecord;
  int_pt globalFirstY = m_firstYNodeToRecord;
  int_pt globalFirstZ = m_firstZNodeToRecord;

  while( m_firstXNodeToRecord < odc::parallel::Mpi::m_startX + 1 )
    m_firstXNodeToRecord += m_numXNodesToSkip;
  while( m_firstYNodeToRecord < odc::parallel::Mpi::m_startY + 1 )
    m_firstYNodeToRecord += m_numYNodesToSkip;
  while( m_firstZNodeToRecord < odc::parallel::Mpi::m_startZ + 1 )
    m_firstZNodeToRecord += m_numZNodesToSkip;

  m_lastXNodeToRecord = odc::parallel::Mpi::m_endX;
  m_lastYNodeToRecord = odc::parallel::Mpi::m_endY;
  m_lastZNodeToRecord = odc::parallel::Mpi::m_endZ;

  m_lastXNodeToRecord -= (m_lastXNodeToRecord - m_firstXNodeToRecord) % m_numXNodesToSkip;
  m_lastYNodeToRecord -= (m_lastYNodeToRecord - m_firstYNodeToRecord) % m_numYNodesToSkip;
  m_lastZNodeToRecord -= (m_lastZNodeToRecord - m_firstZNodeToRecord) % m_numZNodesToSkip;

  m_numXNodesToRecord = (m_lastXNodeToRecord - m_firstXNodeToRecord) / m_numXNodesToSkip + 1;
  m_numYNodesToRecord = (m_lastYNodeToRecord - m_firstYNodeToRecord) / m_numYNodesToSkip + 1;
  m_numZNodesToRecord = (m_lastZNodeToRecord - m_firstZNodeToRecord) / m_numZNodesToSkip + 1;

  int_pt strideX = sizeof( real );
  int_pt strideY = m_numGlobalXNodesToRecord * strideX;
  int_pt strideZ = m_numGlobalYNodesToRecord * strideY;

#ifdef AWP_USE_MPI
  m_displacement =  ((m_firstXNodeToRecord - globalFirstX) / m_numXNodesToSkip) * strideX;
  m_displacement += ((m_firstYNodeToRecord - globalFirstY) / m_numYNodesToSkip) * strideY;
  m_displacement += ((m_firstZNodeToRecord - globalFirstZ) / m_numZNodesToSkip) * strideZ;
#endif

  //! Switch from 1-based indexing to 0-based indexing
  m_firstXNodeToRecord--;
  m_firstYNodeToRecord--;
  m_firstZNodeToRecord--;

  m_lastXNodeToRecord--;
  m_lastYNodeToRecord--;
  m_lastZNodeToRecord--;

  //! Create MPI filetype to store output data
  int_pt maxNX_NY_NZ_WS = (m_numXNodesToRecord > m_numYNodesToRecord ? m_numXNodesToRecord : m_numYNodesToRecord);
  maxNX_NY_NZ_WS        = (maxNX_NY_NZ_WS > m_numZNodesToRecord ? maxNX_NY_NZ_WS : m_numZNodesToRecord);
  maxNX_NY_NZ_WS        = (maxNX_NY_NZ_WS > i_options.m_writeStep ? maxNX_NY_NZ_WS : i_options.m_writeStep);

#ifdef AWP_USE_MPI
  //! "dispArray" will store the block offsets when making the 3D grid to store the file output values
  //! "ones" stores the number of elements in each block (always one element per block)
  int ones[maxNX_NY_NZ_WS];
  MPI_Aint dispArray[maxNX_NY_NZ_WS];
  for( int_pt i = 0; i < maxNX_NY_NZ_WS; i++ )
    ones[i] = 1;

  //! Makes filetype an array of m_numXNodesToRecord floats.
  int err = MPI_Type_contiguous( m_numXNodesToRecord, AWP_MPI_REAL, &m_filetype );
  err     = MPI_Type_commit( &m_filetype );
  for( int_pt i = 0; i < m_numYNodesToRecord; i++ ) {
    dispArray[i] = sizeof( real );
    dispArray[i] = dispArray[i] * m_numGlobalXNodesToRecord * i;
  }

  //! Makes filetype an array of rec_nyt arrays of rec_nxt floats
  err = MPI_Type_create_hindexed( m_numYNodesToRecord, ones, dispArray, m_filetype, &m_filetype );
  err = MPI_Type_commit( &m_filetype );
  for( int_pt i = 0; i < m_numZNodesToRecord; i++ ) {
    dispArray[i] = sizeof( real );
    dispArray[i] = dispArray[i] * m_numGlobalYNodesToRecord * m_numGlobalXNodesToRecord * i;
  }

  //! Makes filetype an array of rec_nzt arrays of rec_nyt arrays of rec_nxt floats.
  //! Then filetype will be large enough to hold a single (x,y,z) grid
  err = MPI_Type_create_hindexed( m_numZNodesToRecord, ones, dispArray, m_filetype, &m_filetype );
  err = MPI_Type_commit( &m_filetype );
  for( int_pt i = 0; i < i_options.m_writeStep; i++ ) {
    dispArray[i] = sizeof( real );
    dispArray[i] = dispArray[i] * m_numGlobalZNodesToRecord * m_numGlobalYNodesToRecord * m_numGlobalXNodesToRecord * i;
  }

  //! Makes writeStep copies of the filetype grid
  err = MPI_Type_create_hindexed( i_options.m_writeStep, ones, dispArray, m_filetype, &m_filetype );

  //! Commit "filetype" after making sure it has enough space to hold all of the (x,y,z) nodes
  err = MPI_Type_commit( &m_filetype );
  //MPI_Type_size(m_filetype, &tmpSize);
#endif

  m_numTimestepsToSkip  = i_options.m_nTiSkp;
  m_writeStep           = i_options.m_writeStep;

  //! Allocate write buffers
  m_numGridPointsToRecord = m_numXNodesToRecord * m_numYNodesToRecord * m_numZNodesToRecord;

  m_velocityXWriteBuffer = (real *)odc::data::common::allocate( m_numGridPointsToRecord * m_writeStep * sizeof( real ), ALIGNMENT );
  m_velocityYWriteBuffer = (real *)odc::data::common::allocate( m_numGridPointsToRecord * m_writeStep * sizeof( real ), ALIGNMENT );
  m_velocityZWriteBuffer = (real *)odc::data::common::allocate( m_numGridPointsToRecord * m_writeStep * sizeof( real ), ALIGNMENT );

  memcpy( m_outputFolder, i_options.m_out, sizeof( i_options.m_out ) );

  snprintf( m_filenamebasex, sizeof( m_filenamebasex ), "%s/SX", m_outputFolder );
  snprintf( m_filenamebasey, sizeof( m_filenamebasey ), "%s/SY", m_outputFolder );
  snprintf( m_filenamebasez, sizeof( m_filenamebasez ), "%s/SZ", m_outputFolder );

  m_firstXNodeToRecord -= odc::parallel::Mpi::m_startX;
  m_firstYNodeToRecord -= odc::parallel::Mpi::m_startY;
  m_firstZNodeToRecord -= odc::parallel::Mpi::m_startZ;

  m_lastXNodeToRecord -= odc::parallel::Mpi::m_startX;
  m_lastYNodeToRecord -= odc::parallel::Mpi::m_startY;
  m_lastZNodeToRecord -= odc::parallel::Mpi::m_startZ;
}

void odc::io::OutputWriter::update( int_pt i_timestep, PatchDecomp& i_ptchDec ) {
  if( i_timestep % m_numTimestepsToSkip == 0 ) {
    int bufInd  = (i_timestep / m_numTimestepsToSkip + m_writeStep - 1) % m_writeStep;
    bufInd      *= m_numGridPointsToRecord;

    i_ptchDec.copyVelToBuffer( m_velocityXWriteBuffer + bufInd, m_velocityYWriteBuffer + bufInd,
                               m_velocityZWriteBuffer + bufInd,
                               m_firstXNodeToRecord, m_lastXNodeToRecord, m_numXNodesToSkip,
                               m_firstYNodeToRecord, m_lastYNodeToRecord, m_numYNodesToSkip,
                               m_firstZNodeToRecord, m_lastZNodeToRecord, m_numZNodesToSkip, i_timestep + 1 );

    if( (i_timestep / m_numTimestepsToSkip) % m_writeStep == 0 ) {
      if( odc::parallel::Mpi::m_rank == 0 )
        std::cout << "Writing to file" << std::endl;

      char datarep[] = "native";
      char filename[AWP_PATH_MAX];
#ifdef AWP_USE_MPI
      MPI_File file;

      MPI_Offset displacement = m_displacement;
      MPI_Status filestatus;

      //! Write x velocity data
      snprintf( filename, sizeof( filename ), "%s%07" AWP_PT_FORMAT_STRING, m_filenamebasex, i_timestep );
      int err = MPI_File_open( MPI_COMM_WORLD, filename,
                               MPI_MODE_CREATE|MPI_MODE_WRONLY,
                               MPI_INFO_NULL, &file );

      err = MPI_File_set_view( file, displacement, AWP_MPI_REAL, m_filetype, datarep, MPI_INFO_NULL );
      err = MPI_File_write_all( file, m_velocityXWriteBuffer, m_numGridPointsToRecord * m_writeStep, AWP_MPI_REAL, &filestatus );
      err = MPI_File_close( &file );

      //! Write y velocity data
      snprintf( filename, sizeof( filename ), "%s%07" AWP_PT_FORMAT_STRING, m_filenamebasey, i_timestep );
      err = MPI_File_open( MPI_COMM_WORLD, filename,
                           MPI_MODE_CREATE|MPI_MODE_WRONLY,
                           MPI_INFO_NULL, &file );

      err = MPI_File_set_view( file, displacement, AWP_MPI_REAL, m_filetype, datarep, MPI_INFO_NULL );
      err = MPI_File_write_all( file, m_velocityYWriteBuffer, m_numGridPointsToRecord * m_writeStep, AWP_MPI_REAL, &filestatus );
      err = MPI_File_close( &file );

      //! Write z velocity data
      snprintf( filename, sizeof( filename ), "%s%07" AWP_PT_FORMAT_STRING, m_filenamebasez, i_timestep );
      err = MPI_File_open( MPI_COMM_WORLD, filename,
                           MPI_MODE_CREATE|MPI_MODE_WRONLY,
                           MPI_INFO_NULL, &file );

      err = MPI_File_set_view( file, displacement, AWP_MPI_REAL, m_filetype, datarep, MPI_INFO_NULL );
      err = MPI_File_write_all( file, m_velocityZWriteBuffer, m_numGridPointsToRecord*m_writeStep, AWP_MPI_REAL, &filestatus );
      err = MPI_File_close( &file );
#endif
    }
  }
}

void odc::io::OutputWriter::finalize() {
  odc::data::common::release( m_velocityXWriteBuffer );
  odc::data::common::release( m_velocityYWriteBuffer );
  odc::data::common::release( m_velocityZWriteBuffer );
}

//! ReceiverWriter class constructor
//! TODO(Raj): Get m_buffSkip from script file instead of defining it here?
odc::io::ReceiverWriter::ReceiverWriter( char *inputFileName, char *outputFileName, real deltaH, int_pt numZGridPoints ):
                                    m_buffSkip(10), m_receiverInputFilePtr(nullptr), m_receiverOutputFilePtr(nullptr) {
  int_pt  i = 0;                    //! receiver counter
  int_pt  k;                        //! loop counter
  real    rcvrX, rcvrY, rcvrZ;      //! receiver coordinates
  int_pt  gridX, gridY, gridZ;      //! receiver grid points
  char    *token;                   //! token read from receiver input file
  char    line[1024];               //! line read from receiver input file

  if( inputFileName[0] == '\0' ) {
    m_numberOfReceivers = 0;
    return;
  }

  //! Copy the receiver input file name set in launch script
  std::strncpy( m_receiverInputFileName, inputFileName, sizeof( m_receiverInputFileName ) );
  m_receiverInputFileName[sizeof(m_receiverInputFileName)] = '\0';

  if( !m_receiverInputFilePtr ) {
    //! Open the receiver input-file in read-mode
    m_receiverInputFilePtr = std::fopen(m_receiverInputFileName, "r");

    if( !m_receiverInputFilePtr ) {
      std::cerr << "Error opening receiver-list " << m_receiverInputFileName << std::endl;
      return;
    }
  }

  //! Get the total no. of receivers (first line in input file)
  if ( !fgets( line, sizeof( line ), m_receiverInputFilePtr ) ) {
    std::cerr << "Cannot read file " << m_receiverInputFileName << std::endl;
    fclose( m_receiverInputFilePtr );
    m_receiverInputFilePtr = nullptr;
    return;
  }
  m_numberOfReceivers = atoi( line );

  if( m_numberOfReceivers < 0 || m_numberOfReceivers >= 1.8e+19 ) {
    std::cerr << "Unhandled receiver-list length passed in file!" << std::endl;
    fclose( m_receiverInputFilePtr );
    m_receiverInputFilePtr = nullptr;
    return;
  }

  //! Dynamic memory allocation
  m_ownedByThisRank   = new bool [m_numberOfReceivers];
  m_arrayGridX        = new int_pt [m_numberOfReceivers];
  m_arrayGridY        = new int_pt [m_numberOfReceivers];
  m_arrayGridZ        = new int_pt [m_numberOfReceivers];
  m_buffCount         = new int_pt [m_numberOfReceivers];
  for( k = 0; k < m_numberOfReceivers; k++ )
    m_buffCount[k] = 0;

  m_buffTimestep      = new int_pt* [m_numberOfReceivers];
  for( k = 0; k < m_numberOfReceivers; k++ )
    m_buffTimestep[k] = new int_pt [m_buffSkip];

  m_buffVelX          = new real* [m_numberOfReceivers];
  for( k = 0; k < m_numberOfReceivers; k++ )
    m_buffVelX[k] = new real [m_buffSkip];

  m_buffVelY          = new real* [m_numberOfReceivers];
  for( k = 0; k < m_numberOfReceivers; k++ )
    m_buffVelY[k] = new real [m_buffSkip];

  m_buffVelZ          = new real* [m_numberOfReceivers];
  for( k = 0; k < m_numberOfReceivers; k++ )
    m_buffVelZ[k] = new real [m_buffSkip];

  m_receiverOutputLogs = new char* [m_numberOfReceivers];
  for( k = 0; k < m_numberOfReceivers; k++ )
    m_receiverOutputLogs[k] = new char [AWP_PATH_MAX];

  while( fgets( line, sizeof( line ), m_receiverInputFilePtr ) ) {
    //! Skip empty line
    if( line[0] == '\n' )
      continue;

    //! Get token words every line between spaces and convert to real coordinates
    token   = strtok( line, " " );
    rcvrX   = atof( token );
    token   = strtok( NULL, " " );
    rcvrY   = atof( token );
    token   = strtok( NULL, " " );
    rcvrZ   = atof( token );

    //! Convert real coordinates to integral grid points in the mesh
    gridX   = (int_pt)(rcvrX / deltaH + 0.5);
    gridY   = (int_pt)(rcvrY / deltaH + 0.5);

    /** Receiver z-coordinate is specified starting from Earth surface whereas internally,
        front lower left corner is considered origin, thus the following transformation */
    gridZ   = numZGridPoints - 1 - (int_pt)(rcvrZ / deltaH + 0.5);

    //! Determine if the current MPI contains the receiver
    m_ownedByThisRank[i] = odc::parallel::Mpi::isInThisRank( gridX, gridY, gridZ );

    //! Subtracting the MPI starting coordinates to get local coordinates in that MPI
    m_arrayGridX[i] = gridX - odc::parallel::Mpi::m_startX;
    m_arrayGridY[i] = gridY - odc::parallel::Mpi::m_startY;
    m_arrayGridZ[i] = gridZ - odc::parallel::Mpi::m_startZ;

    //! Construct name of receiver output log based on receiver, for ex: receiverOutput_13.log and store in an array
    std::strncpy( m_receiverOutputFileName, outputFileName, sizeof( m_receiverOutputFileName ) );
    m_receiverOutputFileName[sizeof(m_receiverOutputFileName)] = '\0';
    sprintf( m_receiverOutputLogs[i], "%s_%" AWP_PT_FORMAT_STRING ".csv", m_receiverOutputFileName, i + 1 );

    //! If the current MPI owns the grid point, we create a corresponding file
    if( m_ownedByThisRank[i] ) {
      //! Open file in truncate mode, i.e. discard previous run values and close file
      m_receiverOutputFilePtr = std::fopen( m_receiverOutputLogs[i], "w" );
      if( !m_receiverOutputFilePtr )
        continue;

      fclose( m_receiverOutputFilePtr );
    }

    m_receiverOutputFilePtr = nullptr;
    i++;
  }

  fclose( m_receiverInputFilePtr );
  m_receiverInputFilePtr = nullptr;
}

//! writeReceiverOutputFiles function
void odc::io::ReceiverWriter::writeReceiverOutputFiles( int_pt currentTimeStep, int_pt numTimestepsToSkip, PatchDecomp& i_ptchDec ) {
  for( int_pt i = 0; i < m_numberOfReceivers; i++ ) {
    //! Only consider time-steps which are a multiple of timesteps-to-skip
    if( m_ownedByThisRank[i] && (currentTimeStep % numTimestepsToSkip == 0) ) {

      //! Only write output to file every buffer-skip steps (by default 10)
      if( (currentTimeStep / numTimestepsToSkip) % m_buffSkip == 0 ) {
        if( !m_receiverOutputFilePtr ) {

          //! Open receiver output log in append mode
          m_receiverOutputFilePtr = std::fopen( m_receiverOutputLogs[i], "a+" );

          if( !m_receiverOutputFilePtr ) {
            std::cerr << "Error opening receiver output log file!" << std::endl;
            continue;
          }
        }

        //! Append into array the velocity values at the buffer-skip step (by default 10th)
        m_buffTimestep[i][m_buffSkip-1] = currentTimeStep;
        m_buffVelX[i][m_buffSkip-1]     = i_ptchDec.getVelX( m_arrayGridX[i], m_arrayGridY[i], m_arrayGridZ[i], currentTimeStep );
        m_buffVelY[i][m_buffSkip-1]     = i_ptchDec.getVelY( m_arrayGridX[i], m_arrayGridY[i], m_arrayGridZ[i], currentTimeStep );
        m_buffVelZ[i][m_buffSkip-1]     = i_ptchDec.getVelZ( m_arrayGridX[i], m_arrayGridY[i], m_arrayGridZ[i], currentTimeStep );

        //! Write to file in a csv format
        for( int_pt j = 0; j < m_buffSkip; j++ )
          fprintf( m_receiverOutputFilePtr, "%" AWP_PT_FORMAT_STRING ",%e,%e,%e\n",
                   m_buffTimestep[i][j], m_buffVelX[i][j], m_buffVelY[i][j], m_buffVelZ[i][j] );

        //! After every disk-write, flush and close file pointer and reset buffer counter to zero
        m_buffCount[i] = 0;
        fflush( m_receiverOutputFilePtr );
        fclose( m_receiverOutputFilePtr );
        m_receiverOutputFilePtr = nullptr;
      } else {
        //! Store intermediate (till buffer is filled) velocity values in arrays
        m_buffTimestep[i][m_buffCount[i]] = currentTimeStep;
        m_buffVelX[i][m_buffCount[i]]     = i_ptchDec.getVelX( m_arrayGridX[i], m_arrayGridY[i], m_arrayGridZ[i], currentTimeStep );
        m_buffVelY[i][m_buffCount[i]]     = i_ptchDec.getVelY( m_arrayGridX[i], m_arrayGridY[i], m_arrayGridZ[i], currentTimeStep );
        m_buffVelZ[i][m_buffCount[i]]     = i_ptchDec.getVelZ( m_arrayGridX[i], m_arrayGridY[i], m_arrayGridZ[i], currentTimeStep );
        m_buffCount[i]++;
      }
    }
  }
}

//! ReceiverWriter class cleanup function
void odc::io::ReceiverWriter::finalize() {
  int_pt k;     //! loop counter

  if( m_receiverInputFilePtr ) {
    fclose( m_receiverInputFilePtr );
    m_receiverInputFilePtr = nullptr;
  }

  if( m_receiverOutputFilePtr ) {
    fclose( m_receiverOutputFilePtr );
    m_receiverOutputFilePtr = nullptr;
  }

  //! Memory deallocation
  delete[] m_arrayGridX;
  delete[] m_arrayGridY;
  delete[] m_arrayGridZ;
  delete[] m_ownedByThisRank;
  delete[] m_buffCount;

  for( k = 0; k < m_numberOfReceivers; k++ )
    delete[] m_buffTimestep[k];
  delete[] m_buffTimestep;

  for( k = 0; k < m_numberOfReceivers; k++ )
    delete[] m_buffVelX[k];
  delete[] m_buffVelX;

  for( k = 0; k < m_numberOfReceivers; k++ )
    delete[] m_buffVelY[k];
  delete[] m_buffVelY;

  for( k = 0; k < m_numberOfReceivers; k++ )
    delete[] m_buffVelZ[k];
  delete[] m_buffVelZ;

  for( k = 0; k < m_numberOfReceivers; k++ )
    delete[] m_receiverOutputLogs[k];
  delete[] m_receiverOutputLogs;
}

odc::io::CheckpointWriter::CheckpointWriter( char *chkfile, int_pt nd, int_pt timestepsToSkip, int_pt numZGridPoints ) {
  if( (chkfile != NULL && chkfile[0] == '\0') || chkfile == nullptr ) {
    std::cerr << "\nWarning: no checkfile specified in input parameters!\n";
    m_ownedByThisRank = false;
    return;
  }

  m_nd = nd;
  m_numTimestepsToSkip = timestepsToSkip;
  m_numZGridPoints = numZGridPoints;
  std::strncpy( m_checkPointFileName, chkfile, sizeof( m_checkPointFileName ) );
  m_checkPointFileName[sizeof(m_checkPointFileName)] = '\0';

  m_rcdX = m_nd;
  m_rcdY = m_nd;
  m_rcdZ = m_numZGridPoints - 1 - m_nd;

  m_ownedByThisRank = odc::parallel::Mpi::isInThisRank( m_rcdX, m_rcdY, m_rcdZ );
}

void odc::io::CheckpointWriter::writeUpdatedStats( int_pt currentTimeStep, PatchDecomp& i_ptchDec ) {
  if( m_ownedByThisRank && currentTimeStep % m_numTimestepsToSkip == 0 ) {
    if( !m_checkPointFile )
      m_checkPointFile = std::fopen( m_checkPointFileName, "a+" );

// TODO(Josh): This update seems pointless?  Is there a reason?
#ifdef YASK
    currentTimeStep++;
#endif
    fprintf( m_checkPointFile,"%" AWP_PT_FORMAT_STRING " :\t%e\t%e\t%e\n",
#ifdef YASK
             currentTimeStep - 1,
#else
             currentTimeStep,
#endif
             i_ptchDec.getVelX( m_rcdX, m_rcdY, m_rcdZ, currentTimeStep ),
             i_ptchDec.getVelY( m_rcdX, m_rcdY, m_rcdZ, currentTimeStep ),
             i_ptchDec.getVelZ( m_rcdX, m_rcdY, m_rcdZ, currentTimeStep ) );

    fflush( m_checkPointFile );
  }
}

void odc::io::CheckpointWriter::writeInitialStats( int_pt ntiskp, real dt, real dh, int_pt nxt, int_pt nyt, int_pt nzt,
                                                   int_pt nt, real arbc, int_pt npc, int_pt nve, real fac, real q0, real ex, real fp,
                                                   real vse_min, real vse_max, real vpe_min, real vpe_max,
                                                   real dde_min, real dde_max ) {
  if( !m_ownedByThisRank )
    return;

  FILE *fchk;
  fchk = std::fopen( m_checkPointFileName, "w" );
  fprintf( fchk, "STABILITY CRITERIA .5 > CMAX*DT/DX:\t%f\n", vpe_max * dt / dh );

  if( vpe_max * dt / dh >= 0.5 )
    fprintf( fchk, "!! WARNING: STABILITY CRITERIA NOT MET, MAY DIVERGE !!\n" );

  fprintf( fchk, "# OF X,Y,Z NODES PER PROC:\t%" AWP_PT_FORMAT_STRING ", %" AWP_PT_FORMAT_STRING ", %" AWP_PT_FORMAT_STRING "\n", nxt, nyt, nzt );
  fprintf( fchk, "# OF TIME STEPS:\t%" AWP_PT_FORMAT_STRING "\n", nt );
  fprintf( fchk, "DISCRETIZATION IN SPACE:\t%f\n", dh );
  fprintf( fchk, "DISCRETIZATION IN TIME:\t%f\n", dt );
  fprintf( fchk, "PML REFLECTION COEFFICIENT:\t%f\n", arbc );
  fprintf( fchk, "HIGHEST P-VELOCITY ENCOUNTERED:\t%f\n", vpe_max );
  fprintf( fchk, "LOWEST P-VELOCITY ENCOUNTERED:\t%f\n", vpe_min );
  fprintf( fchk, "HIGHEST S-VELOCITY ENCOUNTERED:\t%f\n", vse_max );
  fprintf( fchk, "LOWEST S-VELOCITY ENCOUNTERED:\t%f\n", vse_min );
  fprintf( fchk, "HIGHEST DENSITY ENCOUNTERED:\t%f\n", dde_max );
  fprintf( fchk, "LOWEST  DENSITY ENCOUNTERED:\t%f\n", dde_min );
  fprintf( fchk, "SKIP OF SEISMOGRAMS IN TIME (LOOP COUNTER):\t%" AWP_PT_FORMAT_STRING "\n", ntiskp );
  fprintf( fchk, "ABC CONDITION, PML=1 OR CERJAN=0:\t%" AWP_PT_FORMAT_STRING "\n", npc );
  fprintf( fchk, "FD SCHEME, VISCO=1 OR ELASTIC=0:\t%" AWP_PT_FORMAT_STRING "\n", nve );
  fprintf( fchk, "Q, FAC,Q0,EX,FP:\t%f, %f, %f, %f\n", fac, q0, ex, fp );
  fclose( fchk );
}

void odc::io::CheckpointWriter::finalize() {
  if( m_checkPointFile ) {
    fclose( m_checkPointFile );
    m_checkPointFile = nullptr;
  }
}
