/**
@section LICENSE
Copyright (c) 2013-2017, Regents of the University of California, San Diego State University
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include"pmcl3d.h"
#define MPIRANKX        100000
#define MPIRANKY         50000
#define MPIRANKYLDFAC   200000

void update_bound_y_H(float* u1,   float* v1, float* w1, float* f_u1,      float* f_v1,      float* f_w1, float* b_u1, float* b_v1,
                      float* b_w1, int nxt,   int nzt,   cudaStream_t St1, cudaStream_t St2, int rank_f,  int rank_b);

void update_yldfac_H(float *yldfac, 
    float *buf_L, float *buf_R, float *buf_F, float *buf_B,
    float *buf_FL,float *buf_FR,float *buf_BL,float *buf_BR,
    cudaStream_t St, int nxt, int nyt, int nzt);

void mediaswap(Grid3D d1, Grid3D mu,     Grid3D lam,    Grid3D qp,     Grid3D qs, 
               int rank,  int x_rank_L,  int x_rank_R,  int y_rank_F,  int y_rank_B,
               int nxt,   int nyt,       int nzt,       MPI_Comm MCW)
{
	int i, j, k, idx, idy, idz;
	int media_count_x, media_count_y;
	int media_size_x, media_size_y;
        MPI_Request  request_x[4], request_y[4];
        MPI_Status   status_x[4],  status_y[4];
	Grid1D mediaL_S=NULL, mediaR_S=NULL, mediaF_S=NULL, mediaB_S=NULL;
        Grid1D mediaL_R=NULL, mediaR_R=NULL, mediaF_R=NULL, mediaB_R=NULL;

	if(x_rank_L<0 && x_rank_R<0 && y_rank_F<0 && y_rank_B<0)
		return;
	
	if(y_rank_F>=0 || y_rank_B>=0)
	{
		mediaF_S      = Alloc1D(5*ngsl*(nxt+2)*(nzt+2));
		mediaB_S      = Alloc1D(5*ngsl*(nxt+2)*(nzt+2));
                mediaF_R      = Alloc1D(5*ngsl*(nxt+2)*(nzt+2));
                mediaB_R      = Alloc1D(5*ngsl*(nxt+2)*(nzt+2));
                media_size_y  = 5*(ngsl)*(nxt+2)*(nzt+2);
		media_count_y = 0;

                PostRecvMsg_Y(mediaF_R, mediaB_R, MCW, request_y, &media_count_y, media_size_y, y_rank_F, y_rank_B);
		
		if(y_rank_F>=0)
		{
	        	for(i=1+ngsl;i<nxt+3+ngsl;i++)
        	  	  for(j=2+ngsl;j<2+ngsl2;j++)
            	    	    for(k=align-1;k<nzt+align+1;k++)
			    {
            			idx = i-1-ngsl;
            			idy = (j-2-ngsl)*5;
	            		idz = k-align+1;
        	    		mediaF_S[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz] = d1[i][j][k];
            			idy++;
            			mediaF_S[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz] = mu[i][j][k];
            			idy++;
            			mediaF_S[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz] = lam[i][j][k];
            			idy++;
            			mediaF_S[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz] = qp[i][j][k];
            			idy++;
            			mediaF_S[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz] = qs[i][j][k];            
            	    	    }
		}

		if(y_rank_B>=0)
		{
        		for(i=1+ngsl;i<nxt+3+ngsl;i++)
          	  	  for(j=nyt+2;j<nyt+2+ngsl;j++)
            	    	    for(k=align-1;k<nzt+align+1;k++)
			    {
                		idx = i-1-ngsl;
	                	idy = (j-nyt-2)*5;
        	        	idz = k-align+1;
                		mediaB_S[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz] = d1[i][j][k];
                		idy++;
                		mediaB_S[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz] = mu[i][j][k];
                		idy++;
                		mediaB_S[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz] = lam[i][j][k];
                		idy++;
                		mediaB_S[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz] = qp[i][j][k];
                		idy++;
                		mediaB_S[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz] = qs[i][j][k];
            	    	     }
		}

       		PostSendMsg_Y(mediaF_S, mediaB_S, MCW, request_y, &media_count_y, media_size_y, y_rank_F, y_rank_B, rank, Both);
                MPI_Waitall(media_count_y, request_y, status_y);

		if(y_rank_F>=0)
		{
                	for(i=1+ngsl;i<nxt+3+ngsl;i++)
                  	  for(j=2;j<2+ngsl;j++)
                    	    for(k=align-1;k<nzt+align+1;k++)
		    	    {
                        	idx = i-1-ngsl;
                        	idy = (j-2)*5;
                        	idz = k-align+1;
                        	d1[i][j][k]  = mediaF_R[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz];
                        	idy++;
                        	mu[i][j][k]  = mediaF_R[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz];
                        	idy++;
                        	lam[i][j][k] = mediaF_R[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz];
                        	idy++;
                        	qp[i][j][k]  = mediaF_R[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz];
                        	idy++;
                        	qs[i][j][k]  = mediaF_R[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz];
                    	    }
		}
		
		if(y_rank_B>=0)
		{
                        for(i=1+ngsl;i<nxt+3+ngsl;i++)
                          for(j=nyt+2+ngsl;j<nyt+2+ngsl2;j++)
                            for(k=align-1;k<nzt+align+1;k++)
                            {   
                                idx = i-1-ngsl;
                                idy = (j-nyt-2-ngsl)*5;
                                idz = k-align+1;
                                d1[i][j][k]  = mediaB_R[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz];
                                idy++;
                                mu[i][j][k]  = mediaB_R[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz];
                                idy++;
                                lam[i][j][k] = mediaB_R[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz];
                                idy++;
                                qp[i][j][k]  = mediaB_R[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz];
                                idy++;
                                qs[i][j][k]  = mediaB_R[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz];
                            }
		}
		
		Delloc1D(mediaF_S);
		Delloc1D(mediaB_S);
		Delloc1D(mediaF_R);
		Delloc1D(mediaB_R);		
	}

	if(x_rank_L>=0 || x_rank_R>=0)
	{
                mediaL_S      = Alloc1D(5*ngsl*(nyt+ngsl2)*(nzt+2));
                mediaR_S      = Alloc1D(5*ngsl*(nyt+ngsl2)*(nzt+2));
                mediaL_R      = Alloc1D(5*ngsl*(nyt+ngsl2)*(nzt+2));
                mediaR_R      = Alloc1D(5*ngsl*(nyt+ngsl2)*(nzt+2));
                media_size_x  = 5*(ngsl)*(nyt+ngsl2)*(nzt+2);
                media_count_x = 0;

		PostRecvMsg_X(mediaL_R, mediaR_R, MCW, request_x, &media_count_x, media_size_x, x_rank_L, x_rank_R);
                if(x_rank_L>=0)
                {
                        for(i=2+ngsl;i<2+ngsl2;i++)
                          for(j=2;j<nyt+2+ngsl2;j++)
                            for(k=align-1;k<nzt+align+1;k++)
                            {
                                idx = (i-2-ngsl)*5;
                                idy = j-2;
                                idz = k-align+1;
                                mediaL_S[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz] = d1[i][j][k];
                                idx++;
                                mediaL_S[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz] = mu[i][j][k];
                                idx++;
                                mediaL_S[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz] = lam[i][j][k];
                                idx++;
                                mediaL_S[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz] = qp[i][j][k];
                                idx++;
                                mediaL_S[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz] = qs[i][j][k];
                            }
                }

                if(x_rank_R>=0)
                {
                        for(i=nxt+2;i<nxt+2+ngsl;i++)
                          for(j=2;j<nyt+2+ngsl2;j++)
                            for(k=align-1;k<nzt+align+1;k++)
                            {
                                idx = (i-nxt-2)*5;
                                idy = j-2;
                                idz = k-align+1;
                                mediaR_S[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz] = d1[i][j][k];
                                idx++;
                                mediaR_S[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz] = mu[i][j][k];
                                idx++;
                                mediaR_S[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz] = lam[i][j][k];
                                idx++;
                                mediaR_S[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz] = qp[i][j][k];
                                idx++;
                                mediaR_S[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz] = qs[i][j][k];
                            }
                }

	        PostSendMsg_X(mediaL_S, mediaR_S, MCW, request_x, &media_count_x, media_size_x, x_rank_L, x_rank_R, rank, Both);
         	MPI_Waitall(media_count_x, request_x, status_x);

                if(x_rank_L>=0)
                {
                        for(i=2;i<2+ngsl;i++)
                          for(j=2;j<nyt+2+ngsl2;j++)
                            for(k=align-1;k<nzt+align+1;k++)
                            {
                                idx = (i-2)*5;
                                idy = j-2;
                                idz = k-align+1;
                                d1[i][j][k]  = mediaL_R[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                mu[i][j][k]  = mediaL_R[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                lam[i][j][k] = mediaL_R[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                qp[i][j][k]  = mediaL_R[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                qs[i][j][k]  = mediaL_R[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz];
                            }
                }

                if(x_rank_R>=0)
                {
                        for(i=nxt+2+ngsl;i<nxt+2+ngsl2;i++)
                          for(j=2;j<nyt+2+ngsl2;j++)
                            for(k=align-1;k<nzt+align+1;k++)
                            {
                                idx = (i-nxt-2-ngsl)*5;
                                idy = j-2;
                                idz = k-align+1;
                                d1[i][j][k]  = mediaR_R[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                mu[i][j][k]  = mediaR_R[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                lam[i][j][k] = mediaR_R[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                qp[i][j][k]  = mediaR_R[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                qs[i][j][k]  = mediaR_R[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz];
                            }
                }

                Delloc1D(mediaL_S);
                Delloc1D(mediaR_S);
                Delloc1D(mediaL_R);
                Delloc1D(mediaR_R);
	}

	return;
}

void PostRecvMsg_X(float* RL_M, float* RR_M, MPI_Comm MCW, MPI_Request* request, int* count, int msg_size, int rank_L, int rank_R)
{
	int temp_count = 0;
       
	if(rank_L>=0){
		MPI_Irecv(RL_M, msg_size, MPI_FLOAT, rank_L, MPIRANKX+rank_L, MCW, &request[temp_count]);
		++temp_count;
	}

	if(rank_R>=0){
		MPI_Irecv(RR_M, msg_size, MPI_FLOAT, rank_R, MPIRANKX+rank_R, MCW, &request[temp_count]);
		++temp_count;
	}

	*count = temp_count;
	return;
}

void PostSendMsg_X(float* SL_M, float* SR_M, MPI_Comm MCW, MPI_Request* request, int* count, int msg_size,
                   int rank_L,  int rank_R,  int rank,     int flag)
{
        int temp_count, flag_L=-1, flag_R=-1;
        temp_count = *count;
        if(rank<0)
                return;

        if(flag==Both || flag==Left)  flag_L=1;
        if(flag==Both || flag==Right) flag_R=1;

        if(rank_L>=0 && flag_L==1){
                MPI_Isend(SL_M, msg_size, MPI_FLOAT, rank_L, MPIRANKX+rank, MCW, &request[temp_count]);
                ++temp_count;
        }

        if(rank_R>=0 && flag_R==1){
                MPI_Isend(SR_M, msg_size, MPI_FLOAT, rank_R, MPIRANKX+rank, MCW, &request[temp_count]);
                ++temp_count;
        }

        *count = temp_count;
        return;
}

void PostRecvMsg_Y(float* RF_M, float* RB_M, MPI_Comm MCW, MPI_Request* request, int* count, int msg_size, int rank_F, int rank_B)
{
        int temp_count = 0;

        if(rank_F>=0){
                MPI_Irecv(RF_M, msg_size, MPI_FLOAT, rank_F, MPIRANKY+rank_F, MCW, &request[temp_count]);
                ++temp_count;
        }

        if(rank_B>=0){
                MPI_Irecv(RB_M, msg_size, MPI_FLOAT, rank_B, MPIRANKY+rank_B, MCW, &request[temp_count]);
                ++temp_count;
        }

        *count = temp_count;
        return;
}

void PostSendMsg_Y(float* SF_M, float* SB_M, MPI_Comm MCW, MPI_Request* request, int* count, int msg_size,
                   int rank_F,  int rank_B,  int rank,     int flag)
{
        int temp_count, flag_F=-1, flag_B=-1;
        temp_count = *count;
        if(rank<0)
                return;

        if(flag==Both || flag==Front)  flag_F=1;
        if(flag==Both || flag==Back)   flag_B=1;

        if(rank_F>=0 && flag_F==1){
                MPI_Isend(SF_M, msg_size, MPI_FLOAT, rank_F, MPIRANKY+rank, MCW, &request[temp_count]);
                ++temp_count;
        }

        if(rank_B>=0 && flag_B==1){
                MPI_Isend(SB_M, msg_size, MPI_FLOAT, rank_B, MPIRANKY+rank, MCW, &request[temp_count]);
                ++temp_count;
        }

        *count = temp_count;
        return;
}

// MPI receive and send functions for plasticity variable yldfac
void PostRecvMsg_yldfac(float *RL_M, float *RR_M, float *RF_M, float *RB_M, 
      float *RFL_M, float *RFR_M, float *RBL_M, float *RBR_M,
      MPI_Comm MCW, MPI_Request *request, int *count, 
      int msg_size_x, int msg_size_y,
      int rank_L,   int rank_R,   int rank_F,   int rank_B,
      int rank_FL,   int rank_FR,   int rank_BL,   int rank_BR){

        int temp_count = 0;

        if(rank_F>=0){
                MPI_Irecv(RF_M, msg_size_y, MPI_FLOAT, rank_F, MPIRANKYLDFAC+rank_F, MCW, &request[temp_count]);
                ++temp_count;
        }

        if(rank_B>=0){
                MPI_Irecv(RB_M, msg_size_y, MPI_FLOAT, rank_B, MPIRANKYLDFAC+rank_B, MCW, &request[temp_count]);
                ++temp_count;
        }

        if(rank_L>=0){
                MPI_Irecv(RL_M, msg_size_x, MPI_FLOAT, rank_L, MPIRANKYLDFAC+rank_L, MCW, &request[temp_count]);
                ++temp_count;
        }

        if(rank_R>=0){
                MPI_Irecv(RR_M, msg_size_x, MPI_FLOAT, rank_R, MPIRANKYLDFAC+rank_R, MCW, &request[temp_count]);
                ++temp_count;
        }

        // 4 corners
        if(rank_FL>=0){
                MPI_Irecv(RFL_M, 1, MPI_FLOAT, rank_FL, MPIRANKYLDFAC+rank_FL, MCW, &request[temp_count]);
                ++temp_count;
        }
        if(rank_FR>=0){
                MPI_Irecv(RFR_M, 1, MPI_FLOAT, rank_FR, MPIRANKYLDFAC+rank_FR, MCW, &request[temp_count]);
                ++temp_count;
        }
        if(rank_BL>=0){
                MPI_Irecv(RBL_M, 1, MPI_FLOAT, rank_BL, MPIRANKYLDFAC+rank_BL, MCW, &request[temp_count]);
                ++temp_count;
        }
        if(rank_BR>=0){
                MPI_Irecv(RBR_M, 1, MPI_FLOAT, rank_BR, MPIRANKYLDFAC+rank_BR, MCW, &request[temp_count]);
                ++temp_count;
        }

        *count = temp_count;
        return;
}

void PostSendMsg_yldfac(float *SL_M, float *SR_M, float *SF_M, float *SB_M, 
      float *SFL_M, float *SFR_M, float *SBL_M, float *SBR_M,
      MPI_Comm MCW, MPI_Request *request, int *count, 
      int msg_size_x, int msg_size_y, int rank,
      int rank_L,   int rank_R,   int rank_F,   int rank_B,
      int rank_FL,  int rank_FR,  int rank_BL,  int rank_BR){

        int temp_count;
        temp_count = *count;
        if(rank<0)
                return;
        
        if(rank_F>=0){
                MPI_Isend(SF_M, msg_size_y, MPI_FLOAT, rank_F, MPIRANKYLDFAC+rank, MCW, &request[temp_count]);
                ++temp_count;
        }

        if(rank_B>=0){
                MPI_Isend(SB_M, msg_size_y, MPI_FLOAT, rank_B, MPIRANKYLDFAC+rank, MCW, &request[temp_count]);
                ++temp_count;
        }

        if(rank_L>=0){
                MPI_Isend(SL_M, msg_size_x, MPI_FLOAT, rank_L, MPIRANKYLDFAC+rank, MCW, &request[temp_count]);
                ++temp_count;
        }

        if(rank_R>=0){
                MPI_Isend(SR_M, msg_size_x, MPI_FLOAT, rank_R, MPIRANKYLDFAC+rank, MCW, &request[temp_count]);
                ++temp_count;
        }

        // 4 corners
        if(rank_FL>=0){
                MPI_Isend(SFL_M, 1, MPI_FLOAT, rank_FL, MPIRANKYLDFAC+rank, MCW, &request[temp_count]);
                ++temp_count;
        }
        if(rank_FR>=0){
                MPI_Isend(SFR_M, 1, MPI_FLOAT, rank_FR, MPIRANKYLDFAC+rank, MCW, &request[temp_count]);
                ++temp_count;
        }
        if(rank_BL>=0){
                MPI_Isend(SBL_M, 1, MPI_FLOAT, rank_BL, MPIRANKYLDFAC+rank, MCW, &request[temp_count]);
                ++temp_count;
        }
        if(rank_BR>=0){
                MPI_Isend(SBR_M, 1, MPI_FLOAT, rank_BR, MPIRANKYLDFAC+rank, MCW, &request[temp_count]);
                ++temp_count;
        }

        *count = temp_count;
return;
}

void Cpy2Device_source(int npsrc, int READ_STEP,
      int index_offset,
      Grid1D taxx, Grid1D tayy, Grid1D tazz,
      Grid1D taxz, Grid1D tayz, Grid1D taxy,
      float *d_taxx, float *d_tayy, float *d_tazz,
      float *d_taxz, float *d_tayz, float *d_taxy){

      long int num_bytes;
      cudaError_t cerr;
       num_bytes = sizeof(float)*npsrc*READ_STEP;
       cerr=cudaMemcpy(d_taxx,taxx+index_offset,num_bytes,cudaMemcpyHostToDevice);
       if(cerr!=cudaSuccess) printf("CUDA ERROR: Cpy2Device: %s\n",cudaGetErrorString(cerr));
       cerr=cudaMemcpy(d_tayy,tayy+index_offset,num_bytes,cudaMemcpyHostToDevice);
       if(cerr!=cudaSuccess) printf("CUDA ERROR: Cpy2Device: %s\n",cudaGetErrorString(cerr));
       cerr=cudaMemcpy(d_tazz,tazz+index_offset,num_bytes,cudaMemcpyHostToDevice);
       if(cerr!=cudaSuccess) printf("CUDA ERROR: Cpy2Device: %s\n",cudaGetErrorString(cerr));
       cerr=cudaMemcpy(d_taxz,taxz+index_offset,num_bytes,cudaMemcpyHostToDevice);
       if(cerr!=cudaSuccess) printf("CUDA ERROR: Cpy2Device: %s\n",cudaGetErrorString(cerr));
       cerr=cudaMemcpy(d_tayz,tayz+index_offset,num_bytes,cudaMemcpyHostToDevice);
       if(cerr!=cudaSuccess) printf("CUDA ERROR: Cpy2Device: %s\n",cudaGetErrorString(cerr));
       cerr=cudaMemcpy(d_taxy,taxy+index_offset,num_bytes,cudaMemcpyHostToDevice);
       if(cerr!=cudaSuccess) printf("CUDA ERROR: Cpy2Device: %s\n",cudaGetErrorString(cerr));

return;
}

void Cpy2Host_VX(float* u1, float* v1, float* w1, float* h_m, int nxt, int nyt, int nzt, cudaStream_t St, int rank, int flag)
{
	int d_offset=0, h_offset=0, msg_size=0;
        if(rank<0 || flag<1 || flag>2)
	        return;

	if(flag==Left)	d_offset = (2+ngsl)*(nyt+4+ngsl2)*(nzt+2*align); 
	if(flag==Right)	d_offset = (nxt+2)*(nyt+4+ngsl2)*(nzt+2*align);
	
        h_offset = (ngsl)*(nyt+4+ngsl2)*(nzt+2*align);
        msg_size = sizeof(float)*(ngsl)*(nyt+4+ngsl2)*(nzt+2*align);
        cudaMemcpyAsync(h_m,            u1+d_offset, msg_size, cudaMemcpyDeviceToHost, St);
        cudaMemcpyAsync(h_m+h_offset,   v1+d_offset, msg_size, cudaMemcpyDeviceToHost, St);
        cudaMemcpyAsync(h_m+h_offset*2, w1+d_offset, msg_size, cudaMemcpyDeviceToHost, St);
	return;
}

void Cpy2Host_VY(float* s_u1, float* s_v1, float* s_w1, float* h_m, int nxt, int nzt, cudaStream_t St, int rank)
{
        int h_offset, msg_size;
        if(rank<0)
                return;

        h_offset = (ngsl)*(nxt+4+ngsl2)*(nzt+2*align);
        msg_size = sizeof(float)*(ngsl)*(nxt+4+ngsl2)*(nzt+2*align);
        cudaMemcpyAsync(h_m,            s_u1, msg_size, cudaMemcpyDeviceToHost, St);
        cudaMemcpyAsync(h_m+h_offset,   s_v1, msg_size, cudaMemcpyDeviceToHost, St);
        cudaMemcpyAsync(h_m+h_offset*2, s_w1, msg_size, cudaMemcpyDeviceToHost, St);
        return;
}

void Cpy2Device_VX(float* u1, float* v1, float* w1,        float* L_m,       float* R_m, int nxt,
                   int nyt,   int nzt,   cudaStream_t St1, cudaStream_t St2, int rank_L, int rank_R)
{
        int d_offset, h_offset, msg_size;

        h_offset = (ngsl)*(nyt+4+ngsl2)*(nzt+2*align);
        msg_size = sizeof(float)*(ngsl)*(nyt+4+ngsl2)*(nzt+2*align);

        if(rank_L>=0){
		d_offset = 2*(nyt+4+ngsl2)*(nzt+2*align);
                cudaMemcpyAsync(u1+d_offset, L_m,            msg_size, cudaMemcpyHostToDevice, St1);
                cudaMemcpyAsync(v1+d_offset, L_m+h_offset,   msg_size, cudaMemcpyHostToDevice, St1);
                cudaMemcpyAsync(w1+d_offset, L_m+h_offset*2, msg_size, cudaMemcpyHostToDevice, St1);
	}

        if(rank_R>=0){
		d_offset = (nxt+ngsl+2)*(nyt+4+ngsl2)*(nzt+2*align);
        	cudaMemcpyAsync(u1+d_offset, R_m,            msg_size, cudaMemcpyHostToDevice, St2);
        	cudaMemcpyAsync(v1+d_offset, R_m+h_offset,   msg_size, cudaMemcpyHostToDevice, St2);
        	cudaMemcpyAsync(w1+d_offset, R_m+h_offset*2, msg_size, cudaMemcpyHostToDevice, St2);
	}
        return;
}

void Cpy2Device_VY(float* u1,   float *v1,  float *w1,  float* f_u1, float* f_v1, float* f_w1, float* b_u1,      float* b_v1, 
                   float* b_w1, float* F_m, float* B_m, int nxt,     int nyt,     int nzt,     cudaStream_t St1, cudaStream_t St2, 
                   int rank_F,  int rank_B)
{
        int h_offset, msg_size;

        h_offset = (ngsl)*(nxt+4+ngsl2)*(nzt+2*align);
        msg_size = sizeof(float)*(ngsl)*(nxt+4+ngsl2)*(nzt+2*align);
        if(rank_F>=0){
                cudaMemcpyAsync(f_u1, F_m,            msg_size, cudaMemcpyHostToDevice, St1);
                cudaMemcpyAsync(f_v1, F_m+h_offset,   msg_size, cudaMemcpyHostToDevice, St1);
                cudaMemcpyAsync(f_w1, F_m+h_offset*2, msg_size, cudaMemcpyHostToDevice, St1);
        }

        if(rank_B>=0){
                cudaMemcpyAsync(b_u1, B_m,            msg_size, cudaMemcpyHostToDevice, St2);
                cudaMemcpyAsync(b_v1, B_m+h_offset,   msg_size, cudaMemcpyHostToDevice, St2);
                cudaMemcpyAsync(b_w1, B_m+h_offset*2, msg_size, cudaMemcpyHostToDevice, St2);
        }

        update_bound_y_H(u1, v1, w1, f_u1, f_v1, f_w1, b_u1, b_v1, b_w1, nxt, nzt, St1, St2, rank_F, rank_B);
        return;
}

// NVE=3: Copy yldfac to host. We have 4 neighbors and 4 corners
//
//   FL  F  FR
//    L  X  R
//   BL  B  BR
//
void Cpy2Host_yldfac(float *d_L, float *d_R, float *d_F, float *d_B,
      float *d_FL, float *d_FR, float *d_BL, float *d_BR, 
      float *SL, float *SR, float *SF, float *SB, 
      float *SFL, float *SFR, float *SBL, float *SBR,
      cudaStream_t St, int rank_L, int rank_R, int rank_F, int rank_B, 
      int nxt, int nyt){

  int msize = sizeof(float);
  int msize_x = sizeof(float)*nyt;
  int msize_y = sizeof(float)*nxt;

  if(rank_F>=0)
    cudaMemcpyAsync(SF, d_F, msize_y, cudaMemcpyDeviceToHost, St);
  if(rank_B>=0)
    cudaMemcpyAsync(SF, d_B, msize_y, cudaMemcpyDeviceToHost, St);
  if(rank_L>=0)
    cudaMemcpyAsync(SL, d_L, msize_x, cudaMemcpyDeviceToHost, St);
  if(rank_R>=0)
    cudaMemcpyAsync(SR, d_R, msize_x, cudaMemcpyDeviceToHost, St);
  if(rank_F>=0 && rank_L>=0)
    cudaMemcpyAsync(SFL, d_FL, msize, cudaMemcpyDeviceToHost, St);
  if(rank_F>=0 && rank_R>=0)
    cudaMemcpyAsync(SFR, d_FR, msize, cudaMemcpyDeviceToHost, St);
  if(rank_B>=0 && rank_L>=0)
    cudaMemcpyAsync(SBL, d_BL, msize, cudaMemcpyDeviceToHost, St);
  if(rank_B>=0 && rank_R>=0)
    cudaMemcpyAsync(SBR, d_BR, msize, cudaMemcpyDeviceToHost, St);

return;
}

void Cpy2Device_yldfac(float *d_yldfac,
      float *d_L, float *d_R, float *d_F, float *d_B,
      float *d_FL, float *d_FR, float *d_BL, float *d_BR, 
      float *RL, float *RR, float *RF, float *RB, 
      float *RFL, float *RFR, float *RBL, float *RBR,
      cudaStream_t St, int rank_L, int rank_R, int rank_F, int rank_B, 
      int nxt, int nyt, int nzt){

  int msize = sizeof(float);
  int msize_x = sizeof(float)*nyt;
  int msize_y = sizeof(float)*nxt;

  if(rank_F>=0)
    cudaMemcpyAsync(d_F, RF, msize_y, cudaMemcpyHostToDevice, St);
  if(rank_B>=0)
    cudaMemcpyAsync(d_F, RB, msize_y, cudaMemcpyHostToDevice, St);
  if(rank_L>=0)
    cudaMemcpyAsync(d_L, RL, msize_x, cudaMemcpyHostToDevice, St);
  if(rank_R>=0)
    cudaMemcpyAsync(d_R, RR, msize_x, cudaMemcpyHostToDevice, St);
  if(rank_F>=0 && rank_L>=0)
    cudaMemcpyAsync(d_FL, RFL, msize, cudaMemcpyHostToDevice, St);
  if(rank_F>=0 && rank_R>=0)
    cudaMemcpyAsync(d_FR, RFR, msize, cudaMemcpyHostToDevice, St);
  if(rank_B>=0 && rank_L>=0)
    cudaMemcpyAsync(d_BL, RBL, msize, cudaMemcpyHostToDevice, St);
  if(rank_B>=0 && rank_R>=0)
    cudaMemcpyAsync(d_BR, RBR, msize, cudaMemcpyHostToDevice, St);

  update_yldfac_H(d_yldfac, d_L, d_R, d_F, d_B, d_FL, d_FR, d_BL, d_BR, St,
      nxt, nyt, nzt);

return;
}
