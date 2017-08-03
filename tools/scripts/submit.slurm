#!/bin/sh
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

# Submit an AWP-ODC-OS slurm job, designed for comet.sdsc.edu
# Takes many commandline arguments (to match those of awp-odc-os)
# Meant for automated use via workflow code (too many arguments 
# too be useful for humans) -- Josh

# awp-odc takes many command line options in long argument form,
# to avoid using the non-portable 'getopts', we first must map
# longform arguments to shortform arguments 

for arg in "$@"; do
  shift
  case "$arg" in
    "--startdir")     set -- "$@" "-D" ;;
    "--exe")          set -- "$@" "-E" ;;
    "--npernode")     set -- "$@" "-P" ;; 
    "--TMAX")         set -- "$@" "-t" ;;
    "--DH")           set -- "$@" "-h" ;;
    "--DT")           set -- "$@" "-T" ;;
    "--NVAR")         set -- "$@" "-n" ;;
    "--NSRC")         set -- "$@" "-N" ;;
    "--NST")          set -- "$@" "-s" ;;
    "--IFAULT")       set -- "$@" "-f" ;;
    "--MEDIASTART")   set -- "$@" "-m" ;;
    "--FAC")          set -- "$@" "-F" ;;
    "--Q0")           set -- "$@" "-q" ;;
    "--EX")           set -- "$@" "-e" ;;
    "--FP")           set -- "$@" "-p" ;;
    "--READ_STEP")    set -- "$@" "-r" ;;
    "--WRITE_STEP")   set -- "$@" "-w" ;;
    "--NSKPX")        set -- "$@" "-k" ;;
    "--NSKPY")        set -- "$@" "-K" ;;
    "--NEDZ")         set -- "$@" "-W" ;;
    "--NTISKP")       set -- "$@" "-I" ;;
    "--INSRC")        set -- "$@" "-S" ;;
    "--INVEL")        set -- "$@" "-V" ;;
    *)                set -- "$@" "$arg"
  esac
done


echo $@

while getopts "D:X:Y:Z:x:y:c:o:E:P:t:h:T:n:N:s:f:m:F:q:e:p:r:w:k:K:W:I:S:V:" opt; do
  case "$opt" in 
    D) STARTDIR=$OPTARG
     ;;  
    X) XSIZE=$OPTARG
     ;;  
    Y) YSIZE=$OPTARG
     ;;  
    Z) ZSIZE=$OPTARG
     ;;  
    x) XRANKS=$OPTARG
     ;;  
    y) YRANKS=$OPTARG
     ;;  
    c) LOGFILE=$OPTARG
     ;;  
    o) OUTDIR=$OPTARG
     ;;  
    E) CMDNAME=$OPTARG
     ;;  
    D) STARTDIR=$OPTARG
     ;;  
    P) NPERNODE=$OPTARG
     ;;  
    t) TMAX=$OPTARG
     ;;  
    h) DH=$OPTARG
     ;;  
    T) DT=$OPTARG
     ;;  
    n) NVAR=$OPTARG
     ;;  
    N) NSRC=$OPTARG
     ;;  
    s) NST=$OPTARG
     ;;  
    f) IFAULT=$OPTARG
     ;;  
    m) MEDIASTART=$OPTARG
     ;;  
    F) FAC=$OPTARG
     ;;  
    q) Q0=$OPTARG
     ;;  
    e) EX=$OPTARG
     ;;  
    p) FP=$OPTARG
     ;; 
    r) READ_STEP=$OPTARG
     ;;  
    w) WRITE_STEP=$OPTARG
     ;;  
    k) NSKPX=$OPTARG
     ;;  
    K) NSKPY=$OPTARG
     ;;  
    W) NEDZ=$OPTARG
     ;;  
    I) NTISKP=$OPTARG
     ;; 
    S) INSRC=$OPTARG
     ;;  
    V) INVEL=$OPTARG
     ;;  

  esac
done

set -o verbose

module load gnutools/2.69
module load mvapich2_ib/2.1
module load cuda/7.0

cd $STARTDIR 

ibrun -v --npernode $NPERNODE -bp compact -bl socket $CMDNAME  -X            $XSIZE             \
                             -Y             $YSIZE             \
                             -Z             $ZSIZE             \
                             -x             $XRANKS             \
                             -y             $YRANKS             \
                            --TMAX          $TMAX           \
                            --DH            $DH           \
                            --DT            $DT        \
                            --NVAR          $NVAR             \
                            --NSRC          $NSRC             \
                            --NST           $NST             \
                            --IFAULT        $IFAULT             \
                            --MEDIASTART    $MEDIASTART             \
                            --FAC           $FAC          \
                            --Q0            $Q0           \
                            --EX            $EX           \
                            --FP            $FP            \
                            --READ_STEP     $READ_STEP             \
                            --WRITE_STEP    $WRITE_STEP             \
                            --NSKPX         $NSKPX             \
                            --NSKPY         $NSKPY             \
                            --NEDZ          $NEDZ             \
                            --NTISKP        $NTISKP             \
                             -c             $LOGFILE         \
                             -o             $OUTDIR        \
                            --INSRC         $INSRC  \
                            --INVEL         $INVEL  

