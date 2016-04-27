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
#include <math.h>
#include "pmcl3d.h"

void inimesh(int MEDIASTART, Grid3D d1, Grid3D mu, Grid3D lam, Grid3D qp, Grid3D qs, float *taumax, float *taumin,
             int nvar, float FP,  float FL, float FH, int nxt, int nyt, int nzt, int PX, int PY, int NX, int NY,
             int NZ, int *coords, MPI_Comm MCW, int IDYNA, int NVE, int SoCalQ, char *INVEL,
             float *vse, float *vpe, float *dde)
{
  int merr;
  int rank;
  int i,j,k,err;
  float vp,vs,dd,pi;
  int   rmtype[3], rptype[3], roffset[3];
  MPI_Datatype readtype;
  MPI_Status   filestatus;
  MPI_File     fh;

  pi      = 4.*atan(1.);
  if(MEDIASTART==0)
  {
    *taumax = 1./(2*pi*FL);
    *taumin = 1./(2*pi*FH);
    if(IDYNA==1)
    {
       vp=6000.0;
       vs=3464.0;
       dd=2670.0;
    }
    else
    {
       vp=4800.0;
       vs=2800.0;
       dd=2500.0;
    }

    for(i=0;i<nxt+4+8*loop;i++)
      for(j=0;j<nyt+4+8*loop;j++)
        for(k=0;k<nzt+2*align;k++)
        {
           lam[i][j][k]=1./(dd*(vp*vp - 2.*vs*vs));
           mu[i][j][k]=1./(dd*vs*vs);
           d1[i][j][k]=dd;
        }
  }
  else
  {
      Grid3D tmpvp=NULL, tmpvs=NULL, tmpdd=NULL;
      Grid3D tmppq=NULL, tmpsq=NULL;
      int var_offset;

      tmpvp = Alloc3D(nxt, nyt, nzt);
      tmpvs = Alloc3D(nxt, nyt, nzt);
      tmpdd = Alloc3D(nxt, nyt, nzt);
      for(i=0;i<nxt;i++)
        for(j=0;j<nyt;j++)
          for(k=0;k<nzt;k++)
          {
             tmpvp[i][j][k]=0.0f;
             tmpvs[i][j][k]=0.0f;
             tmpdd[i][j][k]=0.0f;
          }

      if(NVE==1)
      {
        tmppq = Alloc3D(nxt, nyt, nzt);
        tmpsq = Alloc3D(nxt, nyt, nzt);
        for(i=0;i<nxt;i++)
          for(j=0;j<nyt;j++)
            for(k=0;k<nzt;k++)
            {
               tmppq[i][j][k]=0.0f;
               tmpsq[i][j][k]=0.0f;
            }
      }

      if(nvar==8)
      {
          var_offset=3;
      }
      else if(nvar==5)
      {
          var_offset=0;
      }
      else
      {
          var_offset=0;
      }

      if(MEDIASTART>=1 && MEDIASTART<=3)
      {
          char filename[40];
          if(MEDIASTART<3) sprintf(filename,INVEL);
          else if(MEDIASTART==3){
            MPI_Comm_rank(MCW,&rank);
            sprintf(filename,"input_rst/mediapart/media%07d.bin",rank);
            if(rank%100==0) printf("Rank=%d, reading file=%s\n",rank,filename);
          }
          Grid1D tmpta = Alloc1D(nvar*nxt*nyt*nzt);
          if(MEDIASTART==3 || (PX==1 && PY==1))
          {
             FILE   *file;
             file = fopen(filename,"rb");
             if(!file)
             {
                printf("can't open file %s", filename);
                return;
             }
             if(!fread(tmpta,sizeof(float),nvar*nxt*nyt*nzt,file))
             {
                printf("can't read file %s", filename);
                return;
             }
             //printf("%d) 0-0-0,1-10-3=%f, %f\n",rank,tmpta[0],tmpta[1+10*nxt+3*nxt*nyt]);
          }
          else{
		    rmtype[0]  = NZ;
    		    rmtype[1]  = NY;
    		    rmtype[2]  = NX*nvar;
    		    rptype[0]  = nzt;
    		    rptype[1]  = nyt;
    		    rptype[2]  = nxt*nvar;
    		    roffset[0] = 0;
    		    roffset[1] = nyt*coords[1];
    		    roffset[2] = nxt*coords[0]*nvar;
    		    err = MPI_Type_create_subarray(3, rmtype, rptype, roffset, MPI_ORDER_C, MPI_FLOAT, &readtype);
    		    err = MPI_Type_commit(&readtype);
                    err = MPI_File_open(MCW,filename,MPI_MODE_RDONLY,MPI_INFO_NULL,&fh);
                    err = MPI_File_set_view(fh, 0, MPI_FLOAT, readtype, "native", MPI_INFO_NULL);
                    err = MPI_File_read_all(fh, tmpta, nvar*nxt*nyt*nzt, MPI_FLOAT, &filestatus);
                    err = MPI_File_close(&fh);
          }
          for(k=0;k<nzt;k++)
            for(j=0;j<nyt;j++)
              for(i=0;i<nxt;i++){
              	tmpvp[i][j][k]=tmpta[(k*nyt*nxt+j*nxt+i)*nvar+var_offset];
              	tmpvs[i][j][k]=tmpta[(k*nyt*nxt+j*nxt+i)*nvar+var_offset+1];
               	tmpdd[i][j][k]=tmpta[(k*nyt*nxt+j*nxt+i)*nvar+var_offset+2];
              	if(nvar>3){
                	tmppq[i][j][k]=tmpta[(k*nyt*nxt+j*nxt+i)*nvar+var_offset+3];
                	tmpsq[i][j][k]=tmpta[(k*nyt*nxt+j*nxt+i)*nvar+var_offset+4];
                }
                /*if(tmpvp[i][j][k]!=tmpvp[i][j][k] ||
                    tmpvs[i][j][k]!=tmpvs[i][j][k] ||
                    tmpdd[i][j][k]!=tmpdd[i][j][k]){
                      printf("%d) tmpvp,vs,dd is NAN!\n");
                      MPI_Abort(MPI_COMM_WORLD,1);
                }*/
         }
         //printf("%d) vp,vs,dd[0^3]=%f,%f,%f\n",rank,tmpvp[0][0][0],
         //   tmpvs[0][0][0], tmpdd[0][0][0]);
         Delloc1D(tmpta);
      }

      if(nvar==3 && NVE==1)
      {
        for(i=0;i<nxt;i++)
          for(j=0;j<nyt;j++){
            for(k=0;k<nzt;k++){
                 tmpsq[i][j][k]=0.05*tmpvs[i][j][k];
                 tmppq[i][j][k]=2.0*tmpsq[i][j][k];
            }
          }
      }

      float w0=0.0f, ww1=0.0f, w2=0.0f, tmp1=0.0f, tmp2=0.0f;
      float qpinv=0.0f, qsinv=0.0f, vpvs=0.0f;
      if(NVE==1)
      {
         w0=2*pi*FP;
         ww1=2*pi*FL;
         w2=2*pi*FH;
         *taumax=1./ww1;
         *taumin=1./w2;
         tmp1=2./pi*(log((*taumax)/(*taumin)));
         tmp2=2./pi*log(w0*(*taumin));
      }

      vse[0] = 1.0e10;
      vpe[0] = 1.0e10;
      dde[0] = 1.0e10;
      vse[1] = -1.0e10;
      vpe[1] = -1.0e10;
      dde[1] = -1.0e10;
      for(i=0;i<nxt;i++)
        for(j=0;j<nyt;j++)
          for(k=0;k<nzt;k++)
          {
             tmpvs[i][j][k] = tmpvs[i][j][k]*(1+ ( log(w2/w0) )/(pi*tmpsq[i][j][k]) );
             tmpvp[i][j][k] = tmpvp[i][j][k]*(1+ ( log(w2/w0) )/(pi*tmppq[i][j][k]) );
             if (SoCalQ==1)
             {
                vpvs=tmpvp[i][j][k]/tmpvs[i][j][k];
                if (vpvs<1.45)  tmpvs[i][j][k]=tmpvp[i][j][k]/1.45;
             }
             //if(tmpvs[i][j][k]<400.0)
             if(tmpvs[i][j][k]<200.0)
             {
                //tmpvs[i][j][k]=400.0;
                //tmpvp[i][j][k]=1200.0;
                tmpvs[i][j][k]=200.0;
                tmpvp[i][j][k]=600.0;
             }
             if(tmpvp[i][j][k]>6500.0){
                tmpvs[i][j][k]=3752.0;
                tmpvp[i][j][k]=6500.0;
             }
             if(tmpdd[i][j][k]<1700.0) tmpdd[i][j][k]=1700.0;
             mu[i+2+4*loop][j+2+4*loop][(nzt+align-1) - k]  = 1./(tmpdd[i][j][k]*tmpvs[i][j][k]*tmpvs[i][j][k]);
             lam[i+2+4*loop][j+2+4*loop][(nzt+align-1) - k] = 1./(tmpdd[i][j][k]*(tmpvp[i][j][k]*tmpvp[i][j][k]
                                                                              -2.*tmpvs[i][j][k]*tmpvs[i][j][k]));
             d1[i+2+4*loop][j+2+4*loop][(nzt+align-1) - k]  = tmpdd[i][j][k];
             if(NVE==1)
             {
                if(tmppq[i][j][k]<=0.0)
                {
                   qpinv=0.0;
                   qsinv=0.0;
                }
                else
                {
                   qpinv=1./tmppq[i][j][k];
                   qsinv=1./tmpsq[i][j][k];
                }
                tmppq[i][j][k]=tmp1*qpinv/(1.0-tmp2*qpinv);
                tmpsq[i][j][k]=tmp1*qsinv/(1.0-tmp2*qsinv);
                qp[i+2+4*loop][j+2+4*loop][(nzt+align-1) - k] = tmppq[i][j][k];
                qs[i+2+4*loop][j+2+4*loop][(nzt+align-1) - k] = tmpsq[i][j][k];
             }
             if(tmpvs[i][j][k]<vse[0]) vse[0] = tmpvs[i][j][k];
             if(tmpvs[i][j][k]>vse[1]) vse[1] = tmpvs[i][j][k];
             if(tmpvp[i][j][k]<vpe[0]) vpe[0] = tmpvp[i][j][k];
             if(tmpvp[i][j][k]>vpe[1]) vpe[1] = tmpvp[i][j][k];
             if(tmpdd[i][j][k]<dde[0]) dde[0] = tmpdd[i][j][k];
             if(tmpdd[i][j][k]>dde[1]) dde[1] = tmpdd[i][j][k];
          }
      Delloc3D(tmpvp);
      Delloc3D(tmpvs);
      Delloc3D(tmpdd);
      if(NVE==1){
         Delloc3D(tmppq);
         Delloc3D(tmpsq);
      }

      //5 Planes (except upper XY-plane)
      for(j=2+4*loop;j<nyt+2+4*loop;j++)
        for(k=align;k<nzt+align;k++){
          lam[1+4*loop][j][k]     = lam[2+4*loop][j][k];
          lam[nxt+2+4*loop][j][k] = lam[nxt+1+4*loop][j][k];
          mu[1+4*loop][j][k]      = mu[2+4*loop][j][k];
          mu[nxt+2+4*loop][j][k]  = mu[nxt+1+4*loop][j][k];
          d1[1+4*loop][j][k]      = d1[2+4*loop][j][k];
          d1[nxt+2+4*loop][j][k]  = d1[nxt+1+4*loop][j][k];
        }

      for(i=2+4*loop;i<nxt+2+4*loop;i++)
        for(k=align;k<nzt+align;k++){
          lam[i][1+4*loop][k]     = lam[i][2+4*loop][k];
          lam[i][nyt+2+4*loop][k] = lam[i][nyt+1+4*loop][k];
          mu[i][1+4*loop][k]      = mu[i][2+4*loop][k];
          mu[i][nyt+2+4*loop][k]  = mu[i][nyt+1+4*loop][k];
          d1[i][1+4*loop][k]      = d1[i][2+4*loop][k];
          d1[i][nyt+2+4*loop][k]  = d1[i][nyt+1+4*loop][k];
        }

      for(i=2+4*loop;i<nxt+2+4*loop;i++)
        for(j=2+4*loop;j<nyt+2+4*loop;j++){
          lam[i][j][align-1]   = lam[i][j][align];
          mu[i][j][align-1]    = mu[i][j][align];
          d1[i][j][align-1]    = d1[i][j][align];
        }

      //12 border lines
      for(i=2+4*loop;i<nxt+2+4*loop;i++){
        lam[i][1+4*loop][align-1]          = lam[i][2+4*loop][align];
        mu[i][1+4*loop][align-1]           = mu[i][2+4*loop][align];
        d1[i][1+4*loop][align-1]           = d1[i][2+4*loop][align];
        lam[i][nyt+2+4*loop][align-1]      = lam[i][nyt+1+4*loop][align];
        mu[i][nyt+2+4*loop][align-1]       = mu[i][nyt+1+4*loop][align];
        d1[i][nyt+2+4*loop][align-1]       = d1[i][nyt+1+4*loop][align];
        lam[i][1+4*loop][nzt+align]        = lam[i][2+4*loop][nzt+align-1];
        mu[i][1+4*loop][nzt+align]         = mu[i][2+4*loop][nzt+align-1];
        d1[i][1+4*loop][nzt+align]         = d1[i][2+4*loop][nzt+align-1];
        lam[i][nyt+2+4*loop][nzt+align]    = lam[i][nyt+1+4*loop][nzt+align-1];
        mu[i][nyt+2+4*loop][nzt+align]     = mu[i][nyt+1+4*loop][nzt+align-1];
        d1[i][nyt+2+4*loop][nzt+align]     = d1[i][nyt+1+4*loop][nzt+align-1];
      }

      for(j=2+4*loop;j<nyt+2+4*loop;j++){
        lam[1+4*loop][j][align-1]          = lam[2+4*loop][j][align];
        mu[1+4*loop][j][align-1]           = mu[2+4*loop][j][align];
        d1[1+4*loop][j][align-1]           = d1[2+4*loop][j][align];
        lam[nxt+2+4*loop][j][align-1]      = lam[nxt+1+4*loop][j][align];
        mu[nxt+2+4*loop][j][align-1]       = mu[nxt+1+4*loop][j][align];
        d1[nxt+2+4*loop][j][align-1]       = d1[nxt+1+4*loop][j][align];
        lam[1+4*loop][j][nzt+align]        = lam[2+4*loop][j][nzt+align-1];
        mu[1+4*loop][j][nzt+align]         = mu[2+4*loop][j][nzt+align-1];
        d1[1+4*loop][j][nzt+align]         = d1[2+4*loop][j][nzt+align-1];
        lam[nxt+2+4*loop][j][nzt+align]    = lam[nxt+1+4*loop][j][nzt+align-1];
        mu[nxt+2+4*loop][j][nzt+align]     = mu[nxt+1+4*loop][j][nzt+align-1];
        d1[nxt+2+4*loop][j][nzt+align]     = d1[nxt+1+4*loop][j][nzt+align-1];
      }

      for(k=align;k<nzt+align;k++){
        lam[1+4*loop][1+4*loop][k]         = lam[2+4*loop][2+4*loop][k];
        mu[1+4*loop][1+4*loop][k]          = mu[2+4*loop][2+4*loop][k];
        d1[1+4*loop][1+4*loop][k]          = d1[2+4*loop][2+4*loop][k];
        lam[nxt+2+4*loop][1+4*loop][k]     = lam[nxt+1+4*loop][2+4*loop][k];
        mu[nxt+2+4*loop][1+4*loop][k]      = mu[nxt+1+4*loop][2+4*loop][k];
        d1[nxt+2+4*loop][1+4*loop][k]      = d1[nxt+1+4*loop][2+4*loop][k];
        lam[1+4*loop][nyt+2+4*loop][k]     = lam[2+4*loop][nyt+1+4*loop][k];
        mu[1+4*loop][nyt+2+4*loop][k]      = mu[2+4*loop][nyt+1+4*loop][k];
        d1[1+4*loop][nyt+2+4*loop][k]      = d1[2+4*loop][nyt+1+4*loop][k];
        lam[nxt+2+4*loop][nyt+2+4*loop][k] = lam[nxt+1+4*loop][nyt+1+4*loop][k];
        mu[nxt+2+4*loop][nyt+2+4*loop][k]  = mu[nxt+1+4*loop][nyt+1+4*loop][k];
        d1[nxt+2+4*loop][nyt+2+4*loop][k]  = d1[nxt+1+4*loop][nyt+1+4*loop][k];
      }

      //8 Corners
      lam[1+4*loop][1+4*loop][align-1]           = lam[2+4*loop][2+4*loop][align];
      mu[1+4*loop][1+4*loop][align-1]            = mu[2+4*loop][2+4*loop][align];
      d1[1+4*loop][1+4*loop][align-1]            = d1[2+4*loop][2+4*loop][align];
      lam[nxt+2+4*loop][1+4*loop][align-1]       = lam[nxt+1+4*loop][2+4*loop][align];
      mu[nxt+2+4*loop][1+4*loop][align-1]        = mu[nxt+1+4*loop][2+4*loop][align];
      d1[nxt+2+4*loop][1+4*loop][align-1]        = d1[nxt+1+4*loop][2+4*loop][align];
      lam[1+4*loop][nyt+2+4*loop][align-1]       = lam[2+4*loop][nyt+1+4*loop][align];
      mu[1+4*loop][nyt+2+4*loop][align-1]        = mu[2+4*loop][nyt+1+4*loop][align];
      d1[1+4*loop][nyt+2+4*loop][align-1]        = d1[2+4*loop][nyt+1+4*loop][align];
      lam[1+4*loop][1+4*loop][nzt+align]         = lam[2+4*loop][2+4*loop][nzt+align-1];
      mu[1+4*loop][1+4*loop][nzt+align]          = mu[2+4*loop][2+4*loop][nzt+align-1];
      d1[1+4*loop][1+4*loop][nzt+align]          = d1[2+4*loop][2+4*loop][nzt+align-1];
      lam[nxt+2+4*loop][1+4*loop][nzt+align]     = lam[nxt+1+4*loop][2+4*loop][nzt+align-1];
      mu[nxt+2+4*loop][1+4*loop][nzt+align]      = mu[nxt+1+4*loop][2+4*loop][nzt+align-1];
      d1[nxt+2+4*loop][1+4*loop][nzt+align]      = d1[nxt+1+4*loop][2+4*loop][nzt+align-1];
      lam[nxt+2+4*loop][nyt+2+4*loop][align-1]   = lam[nxt+1+4*loop][nyt+1+4*loop][align];
      mu[nxt+2+4*loop][nyt+2+4*loop][align-1]    = mu[nxt+1+4*loop][nyt+1+4*loop][align];
      d1[nxt+2+4*loop][nyt+2+4*loop][align-1]    = d1[nxt+1+4*loop][nyt+1+4*loop][align];
      lam[1+4*loop][nyt+2+4*loop][nzt+align]     = lam[2+4*loop][nyt+1+4*loop][nzt+align-1];
      mu[1+4*loop][nyt+2+4*loop][nzt+align]      = mu[2+4*loop][nyt+1+4*loop][nzt+align-1];
      d1[1+4*loop][nyt+2+4*loop][nzt+align]      = d1[2+4*loop][nyt+1+4*loop][nzt+align-1];
      lam[nxt+2+4*loop][nyt+2+4*loop][nzt+align] = lam[nxt+1+4*loop][nyt+1+4*loop][nzt+align-1];
      mu[nxt+2+4*loop][nyt+2+4*loop][nzt+align]  = mu[nxt+1+4*loop][nyt+1+4*loop][nzt+align-1];
      d1[nxt+2+4*loop][nyt+2+4*loop][nzt+align]  = d1[nxt+1+4*loop][nyt+1+4*loop][nzt+align-1];

      k = nzt+align;
      for(i=2+4*loop;i<nxt+2+4*loop;i++)
        for(j=2+4*loop;j<nyt+2+4*loop;j++){
           d1[i][j][k]   = d1[i][j][k-1];
           mu[i][j][k]   = mu[i][j][k-1];
           lam[i][j][k]  = lam[i][j][k-1];
           if(NVE==1){
             qp[i][j][k] = qp[i][j][k-1];
             qs[i][j][k] = qs[i][j][k-1];
           }
        }

      float tmpvse[2],tmpvpe[2],tmpdde[2];
      merr = MPI_Allreduce(vse,tmpvse,2,MPI_FLOAT,MPI_MAX,MCW);
      merr = MPI_Allreduce(vpe,tmpvpe,2,MPI_FLOAT,MPI_MAX,MCW);
      merr = MPI_Allreduce(dde,tmpdde,2,MPI_FLOAT,MPI_MAX,MCW);
      vse[1] = tmpvse[1];
      vpe[1] = tmpvpe[1];
      dde[1] = tmpdde[1];
      merr = MPI_Allreduce(vse,tmpvse,2,MPI_FLOAT,MPI_MIN,MCW);
      merr = MPI_Allreduce(vpe,tmpvpe,2,MPI_FLOAT,MPI_MIN,MCW);
      merr = MPI_Allreduce(dde,tmpdde,2,MPI_FLOAT,MPI_MIN,MCW);
      vse[0] = tmpvse[0];
      vpe[0] = tmpvpe[0];
      dde[0] = tmpdde[0];
  }
  return;
}

void tausub( Grid3D tau, float taumin,float taumax)
{
  int idx, idy, idz;
  float tautem[2][2][2];
  float tmp;

  tautem[0][0][0]=1.0;
  tautem[1][0][0]=6.0;
  tautem[0][1][0]=7.0;
  tautem[1][1][0]=4.0;
  tautem[0][0][1]=8.0;
  tautem[1][0][1]=3.0;
  tautem[0][1][1]=2.0;
  tautem[1][1][1]=5.0;

  for(idx=0;idx<2;idx++)
    for(idy=0;idy<2;idy++)
      for(idz=0;idz<2;idz++)
      {
         tmp = tautem[idx][idy][idz];
         tmp = (tmp-0.5)/8.0;
         tmp = 2.0*tmp - 1.0;

         tau[idx][idy][idz] = exp(0.5*(log(taumax*taumin) + log(taumax/taumin)*tmp));
      }

  return;
}


void init_texture(int nxt,  int nyt,  int nzt,  Grid3D tau1,  Grid3D tau2,  Grid3D vx1,  Grid3D vx2,
		  int xls,  int xre,  int yls,  int yre)
{
   int i, j, k, itx, ity, itz;
   itx = 0;
   ity = 0;
   itz = (nzt-1)%2;
   for(i=xls;i<=xre;i++)
   {
       itx = 1 - itx;
       for(j=yls;j<=yre;j++)
       {
          ity = 1 - ity;
          for(k=align;k<nzt+align;k++)
          {
                itz           = 1 - itz;
                vx1[i][j][k]  = tau1[itx][ity][itz];
                vx2[i][j][k]  = tau2[itx][ity][itz];
           }
       }
   }
   return;
}
