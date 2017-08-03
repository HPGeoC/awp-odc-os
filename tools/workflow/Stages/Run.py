##
# @author Josh Tobin (rjtobin AT ucsd.edu)
#
# @section LICENSE
# Copyright (c) 2016, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##

#/-------------------------------Run.py--------------------------------------\
#|                                                                           |
#| Add the run part of the workflow shell script.  First checks which        |
#| which scheduler is requested, then outputs the required commands to run   |
#| or schedule the workflow, for each benchmark.                             |
#|                                                                           |
#\---------------------------------------------------------------------------/

import os, sys, ConfigParser

def schedule(scheduler, output_file, wf_cfg, ds_cfg, bench_name):

    wf_name = wf_cfg.get("workflow", "name")
    ds_name = wf_cfg.get(bench_name, "dataset")
    dir_name = "%s_%s" % (wf_name, bench_name)

    dim_x = int(wf_cfg.get(bench_name, "x"))
    dim_y = int(wf_cfg.get(bench_name, "y"))

    nx    = int(ds_cfg.get(ds_name, "x"))
    ny    = int(ds_cfg.get(ds_name, "y"))
    nz    = int(ds_cfg.get(ds_name, "z"))

    tmax         =   wf_cfg.get(bench_name, "tmax")
    dh           =   ds_cfg.get(ds_name, "dh")
    dt           =   wf_cfg.get(bench_name, "dt")
    nvar         =   ds_cfg.get(ds_name, "nvar")
    nsrc         =   ds_cfg.get(ds_name, "num_src_terms")
    nst          =   ds_cfg.get(ds_name, "num_src_steps")
    ifault       =   ds_cfg.get(ds_name, "ifault")
    media        =   ds_cfg.get(ds_name, "media")
    fac          =   ds_cfg.get(ds_name, "fac")
    q0           =   ds_cfg.get(ds_name, "q0")
    ex           =   ds_cfg.get(ds_name, "ex")
    fp           =   ds_cfg.get(ds_name, "fp")
    read_step    =   nst
    write_step   =   wf_cfg.get(bench_name, "write_step")
    nskpx        =   wf_cfg.get(bench_name, "skip_x")
    nskpy        =   wf_cfg.get(bench_name, "skip_y")
    nedz         =   wf_cfg.get(bench_name, "end_z")
    ntiskp       =   wf_cfg.get(bench_name, "skip_t")
    log_file     =   "$OUTPUT_DIR/logs/%s.log" % dir_name
    outdir       =   "$OUTPUT_DIR/raw/%s" % dir_name
    insrc        =   "$INPUT_DIR/%s" % ds_cfg.get(ds_name, "insrc")
    invel        =   "$INPUT_DIR/%s" % ds_cfg.get(ds_name, "invel")


    num_ranks = dim_x * dim_y
    exe = "$OUTPUT_DIR/bin/%s" % wf_name
    if scheduler == "bash": 
        output_file.write("""
mpiexec -n %d %s -X       %d \\
                 -Y       %d \\
                 -Z       %d \\
                 -x       %d \\
                 -y       %d \\
            --TMAX        %s \\
            --DH          %s \\
            --DT          %s \\
            --NVAR        %s \\
            --NSRC        %s \\
            --NST         %s \\
            --IFAULT      %s \\
            --MEDIASTART  %s \\
            --FAC         %s \\
            --Q0          %s \\
            --EX          %s \\
            --FP          %s \\
            --READ_STEP   %s \\
            --WRITE_STEP  %s \\
            --NSKPX       %s \\
            --NSKPY       %s \\
            --NEDZ        %s \\
            --NTISKP      %s \\
                 -c       %s \\
                 -o       %s \\
            --INSRC       %s \\
            --INVEL       %s 
    """ % (num_ranks, exe, nx, ny, nz, dim_x, dim_y, tmax, dh, dt, nvar, nsrc, nst, ifault, media, fac, q0, ex, fp, read_step, write_step, nskpx, nskpy, nedz, ntiskp, log_file, outdir, insrc, invel))
    elif scheduler == "slurm": 
        job_name = "awp_%s" % dir_name
        output_file.write("""
sbatch --job-name="%s" --partition=gpu --gres=gpu:4 \\
       --output="$OUTPUT_DIR/logs/slurm.%%j.%%N.out" \\
       --nodes=1                                     \\
       --ntasks-per-node=%d                          \\
       --export=ALL                                  \\
       -t 00:10:00                                   \\
$TOOL_DIR/scripts/submit.slurm                      \\
       --startdir  `pwd`                    \\
       --exe        %s                      \\
       --npernode   %d                      \\
                 -X       %d \\
                 -Y       %d \\
                 -Z       %d \\
                 -x       %d \\
                 -y       %d \\
            --TMAX        %s \\
            --DH          %s \\
            --DT          %s \\
            --NVAR        %s \\
            --NSRC        %s \\
            --NST         %s \\
            --IFAULT      %s \\
            --MEDIASTART  %s \\
            --FAC         %s \\
            --Q0          %s \\
            --EX          %s \\
            --FP          %s \\
            --READ_STEP   %s \\
            --WRITE_STEP  %s \\
            --NSKPX       %s \\
            --NSKPY       %s \\
            --NEDZ        %s \\
            --NTISKP      %s \\
                 -c       %s \\
                 -o       %s \\
            --INSRC       %s \\
            --INVEL       %s 
    """ % (job_name, num_ranks, exe, num_ranks, nx, ny, nz, dim_x, dim_y, tmax, dh, dt, nvar, nsrc, nst, ifault, media, fac, q0, ex, fp, read_step, write_step, nskpx, nskpy, nedz, ntiskp, log_file, outdir, insrc, invel))

def write(output_file, wf_cfg, ds_cfg):
    name = wf_cfg.get("workflow", "code")
    scheduler = wf_cfg.get("workflow", "scheduler") 
    output_file.write("""
# (4) Run code

if [ ${RUN} != "0" ]; then
    """)

    for section in wf_cfg.sections():
        if section != "workflow":    
            bench_name = section
            if scheduler == "none":
                schedule("bash", output_file, wf_cfg, ds_cfg, bench_name)
            elif scheduler == "slurm":
                schedule("slurm", output_file, wf_cfg, ds_cfg, bench_name)

    # close up the run section
    output_file.write("""
fi
    """)

