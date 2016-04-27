/**
@section LICENSE
Copyright (c) 2013-2016, Regents of the University of California
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include"pmcl3d.h"
#define MPIRANKX 100000
#define MPIRANKY  50000

void update_bound_y_H(float* u1,   float* v1, float* w1, float* f_u1,      float* f_v1,      float* f_w1, float* b_u1, float* b_v1,
                      float* b_w1, int nxt,   int nzt,   cudaStream_t St1, cudaStream_t St2, int rank_f,  int rank_b);

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
		mediaF_S      = Alloc1D(5*4*loop*(nxt+2)*(nzt+2));
		mediaB_S      = Alloc1D(5*4*loop*(nxt+2)*(nzt+2));
                mediaF_R      = Alloc1D(5*4*loop*(nxt+2)*(nzt+2));
                mediaB_R      = Alloc1D(5*4*loop*(nxt+2)*(nzt+2));
                media_size_y  = 5*(4*loop)*(nxt+2)*(nzt+2);
		media_count_y = 0;

                PostRecvMsg_Y(mediaF_R, mediaB_R, MCW, request_y, &media_count_y, media_size_y, y_rank_F, y_rank_B);

		if(y_rank_F>=0)
		{
	        	for(i=1+4*loop;i<nxt+3+4*loop;i++)
        	  	  for(j=2+4*loop;j<2+8*loop;j++)
            	    	    for(k=align-1;k<nzt+align+1;k++)
			    {
            			idx = i-1-4*loop;
            			idy = (j-2-4*loop)*5;
	            		idz = k-align+1;
        	    		mediaF_S[idx*5*(4*loop)*(nzt+2)+idy*(nzt+2)+idz] = d1[i][j][k];
            			idy++;
            			mediaF_S[idx*5*(4*loop)*(nzt+2)+idy*(nzt+2)+idz] = mu[i][j][k];
            			idy++;
            			mediaF_S[idx*5*(4*loop)*(nzt+2)+idy*(nzt+2)+idz] = lam[i][j][k];
            			idy++;
            			mediaF_S[idx*5*(4*loop)*(nzt+2)+idy*(nzt+2)+idz] = qp[i][j][k];
            			idy++;
            			mediaF_S[idx*5*(4*loop)*(nzt+2)+idy*(nzt+2)+idz] = qs[i][j][k];
            	    	    }
		}

		if(y_rank_B>=0)
		{
        		for(i=1+4*loop;i<nxt+3+4*loop;i++)
          	  	  for(j=nyt+2;j<nyt+2+4*loop;j++)
            	    	    for(k=align-1;k<nzt+align+1;k++)
			    {
                		idx = i-1-4*loop;
	                	idy = (j-nyt-2)*5;
        	        	idz = k-align+1;
                		mediaB_S[idx*5*(4*loop)*(nzt+2)+idy*(nzt+2)+idz] = d1[i][j][k];
                		idy++;
                		mediaB_S[idx*5*(4*loop)*(nzt+2)+idy*(nzt+2)+idz] = mu[i][j][k];
                		idy++;
                		mediaB_S[idx*5*(4*loop)*(nzt+2)+idy*(nzt+2)+idz] = lam[i][j][k];
                		idy++;
                		mediaB_S[idx*5*(4*loop)*(nzt+2)+idy*(nzt+2)+idz] = qp[i][j][k];
                		idy++;
                		mediaB_S[idx*5*(4*loop)*(nzt+2)+idy*(nzt+2)+idz] = qs[i][j][k];
            	    	     }
		}

       		PostSendMsg_Y(mediaF_S, mediaB_S, MCW, request_y, &media_count_y, media_size_y, y_rank_F, y_rank_B, rank, Both);
                MPI_Waitall(media_count_y, request_y, status_y);

		if(y_rank_F>=0)
		{
                	for(i=1+4*loop;i<nxt+3+4*loop;i++)
                  	  for(j=2;j<2+4*loop;j++)
                    	    for(k=align-1;k<nzt+align+1;k++)
		    	    {
                        	idx = i-1-4*loop;
                        	idy = (j-2)*5;
                        	idz = k-align+1;
                        	d1[i][j][k]  = mediaF_R[idx*5*(4*loop)*(nzt+2)+idy*(nzt+2)+idz];
                        	idy++;
                        	mu[i][j][k]  = mediaF_R[idx*5*(4*loop)*(nzt+2)+idy*(nzt+2)+idz];
                        	idy++;
                        	lam[i][j][k] = mediaF_R[idx*5*(4*loop)*(nzt+2)+idy*(nzt+2)+idz];
                        	idy++;
                        	qp[i][j][k]  = mediaF_R[idx*5*(4*loop)*(nzt+2)+idy*(nzt+2)+idz];
                        	idy++;
                        	qs[i][j][k]  = mediaF_R[idx*5*(4*loop)*(nzt+2)+idy*(nzt+2)+idz];
                    	    }
		}

		if(y_rank_B>=0)
		{
                        for(i=1+4*loop;i<nxt+3+4*loop;i++)
                          for(j=nyt+2+4*loop;j<nyt+2+8*loop;j++)
                            for(k=align-1;k<nzt+align+1;k++)
                            {
                                idx = i-1-4*loop;
                                idy = (j-nyt-2-4*loop)*5;
                                idz = k-align+1;
                                d1[i][j][k]  = mediaB_R[idx*5*(4*loop)*(nzt+2)+idy*(nzt+2)+idz];
                                idy++;
                                mu[i][j][k]  = mediaB_R[idx*5*(4*loop)*(nzt+2)+idy*(nzt+2)+idz];
                                idy++;
                                lam[i][j][k] = mediaB_R[idx*5*(4*loop)*(nzt+2)+idy*(nzt+2)+idz];
                                idy++;
                                qp[i][j][k]  = mediaB_R[idx*5*(4*loop)*(nzt+2)+idy*(nzt+2)+idz];
                                idy++;
                                qs[i][j][k]  = mediaB_R[idx*5*(4*loop)*(nzt+2)+idy*(nzt+2)+idz];
                            }
		}

		Delloc1D(mediaF_S);
		Delloc1D(mediaB_S);
		Delloc1D(mediaF_R);
		Delloc1D(mediaB_R);
	}

	if(x_rank_L>=0 || x_rank_R>=0)
	{
                mediaL_S      = Alloc1D(5*4*loop*(nyt+8*loop)*(nzt+2));
                mediaR_S      = Alloc1D(5*4*loop*(nyt+8*loop)*(nzt+2));
                mediaL_R      = Alloc1D(5*4*loop*(nyt+8*loop)*(nzt+2));
                mediaR_R      = Alloc1D(5*4*loop*(nyt+8*loop)*(nzt+2));
                media_size_x  = 5*(4*loop)*(nyt+8*loop)*(nzt+2);
                media_count_x = 0;

		PostRecvMsg_X(mediaL_R, mediaR_R, MCW, request_x, &media_count_x, media_size_x, x_rank_L, x_rank_R);
                if(x_rank_L>=0)
                {
                        for(i=2+4*loop;i<2+8*loop;i++)
                          for(j=2;j<nyt+2+8*loop;j++)
                            for(k=align-1;k<nzt+align+1;k++)
                            {
                                idx = (i-2-4*loop)*5;
                                idy = j-2;
                                idz = k-align+1;
                                mediaL_S[idx*(nyt+8*loop)*(nzt+2)+idy*(nzt+2)+idz] = d1[i][j][k];
                                idx++;
                                mediaL_S[idx*(nyt+8*loop)*(nzt+2)+idy*(nzt+2)+idz] = mu[i][j][k];
                                idx++;
                                mediaL_S[idx*(nyt+8*loop)*(nzt+2)+idy*(nzt+2)+idz] = lam[i][j][k];
                                idx++;
                                mediaL_S[idx*(nyt+8*loop)*(nzt+2)+idy*(nzt+2)+idz] = qp[i][j][k];
                                idx++;
                                mediaL_S[idx*(nyt+8*loop)*(nzt+2)+idy*(nzt+2)+idz] = qs[i][j][k];
                            }
                }

                if(x_rank_R>=0)
                {
                        for(i=nxt+2;i<nxt+2+4*loop;i++)
                          for(j=2;j<nyt+2+8*loop;j++)
                            for(k=align-1;k<nzt+align+1;k++)
                            {
                                idx = (i-nxt-2)*5;
                                idy = j-2;
                                idz = k-align+1;
                                mediaR_S[idx*(nyt+8*loop)*(nzt+2)+idy*(nzt+2)+idz] = d1[i][j][k];
                                idx++;
                                mediaR_S[idx*(nyt+8*loop)*(nzt+2)+idy*(nzt+2)+idz] = mu[i][j][k];
                                idx++;
                                mediaR_S[idx*(nyt+8*loop)*(nzt+2)+idy*(nzt+2)+idz] = lam[i][j][k];
                                idx++;
                                mediaR_S[idx*(nyt+8*loop)*(nzt+2)+idy*(nzt+2)+idz] = qp[i][j][k];
                                idx++;
                                mediaR_S[idx*(nyt+8*loop)*(nzt+2)+idy*(nzt+2)+idz] = qs[i][j][k];
                            }
                }

	        PostSendMsg_X(mediaL_S, mediaR_S, MCW, request_x, &media_count_x, media_size_x, x_rank_L, x_rank_R, rank, Both);
         	MPI_Waitall(media_count_x, request_x, status_x);

                if(x_rank_L>=0)
                {
                        for(i=2;i<2+4*loop;i++)
                          for(j=2;j<nyt+2+8*loop;j++)
                            for(k=align-1;k<nzt+align+1;k++)
                            {
                                idx = (i-2)*5;
                                idy = j-2;
                                idz = k-align+1;
                                d1[i][j][k]  = mediaL_R[idx*(nyt+8*loop)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                mu[i][j][k]  = mediaL_R[idx*(nyt+8*loop)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                lam[i][j][k] = mediaL_R[idx*(nyt+8*loop)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                qp[i][j][k]  = mediaL_R[idx*(nyt+8*loop)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                qs[i][j][k]  = mediaL_R[idx*(nyt+8*loop)*(nzt+2)+idy*(nzt+2)+idz];
                            }
                }

                if(x_rank_R>=0)
                {
                        for(i=nxt+2+4*loop;i<nxt+2+8*loop;i++)
                          for(j=2;j<nyt+2+8*loop;j++)
                            for(k=align-1;k<nzt+align+1;k++)
                            {
                                idx = (i-nxt-2-4*loop)*5;
                                idy = j-2;
                                idz = k-align+1;
                                d1[i][j][k]  = mediaR_R[idx*(nyt+8*loop)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                mu[i][j][k]  = mediaR_R[idx*(nyt+8*loop)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                lam[i][j][k] = mediaR_R[idx*(nyt+8*loop)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                qp[i][j][k]  = mediaR_R[idx*(nyt+8*loop)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                qs[i][j][k]  = mediaR_R[idx*(nyt+8*loop)*(nzt+2)+idy*(nzt+2)+idz];
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

	if(flag==Left)	d_offset = (2+4*loop)*(nyt+4+8*loop)*(nzt+2*align);
	if(flag==Right)	d_offset = (nxt+2)*(nyt+4+8*loop)*(nzt+2*align);

        h_offset = (4*loop)*(nyt+4+8*loop)*(nzt+2*align);
        msg_size = sizeof(float)*(4*loop)*(nyt+4+8*loop)*(nzt+2*align);
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

        h_offset = (4*loop)*(nxt+4+8*loop)*(nzt+2*align);
        msg_size = sizeof(float)*(4*loop)*(nxt+4+8*loop)*(nzt+2*align);
        cudaMemcpyAsync(h_m,            s_u1, msg_size, cudaMemcpyDeviceToHost, St);
        cudaMemcpyAsync(h_m+h_offset,   s_v1, msg_size, cudaMemcpyDeviceToHost, St);
        cudaMemcpyAsync(h_m+h_offset*2, s_w1, msg_size, cudaMemcpyDeviceToHost, St);
        return;
}

void Cpy2Device_VX(float* u1, float* v1, float* w1,        float* L_m,       float* R_m, int nxt,
                   int nyt,   int nzt,   cudaStream_t St1, cudaStream_t St2, int rank_L, int rank_R)
{
        int d_offset, h_offset, msg_size;

        h_offset = (4*loop)*(nyt+4+8*loop)*(nzt+2*align);
        msg_size = sizeof(float)*(4*loop)*(nyt+4+8*loop)*(nzt+2*align);

        if(rank_L>=0){
		d_offset = 2*(nyt+4+8*loop)*(nzt+2*align);
                cudaMemcpyAsync(u1+d_offset, L_m,            msg_size, cudaMemcpyHostToDevice, St1);
                cudaMemcpyAsync(v1+d_offset, L_m+h_offset,   msg_size, cudaMemcpyHostToDevice, St1);
                cudaMemcpyAsync(w1+d_offset, L_m+h_offset*2, msg_size, cudaMemcpyHostToDevice, St1);
	}

        if(rank_R>=0){
		d_offset = (nxt+4*loop+2)*(nyt+4+8*loop)*(nzt+2*align);
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

        h_offset = (4*loop)*(nxt+4+8*loop)*(nzt+2*align);
        msg_size = sizeof(float)*(4*loop)*(nxt+4+8*loop)*(nzt+2*align);
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
