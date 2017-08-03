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

#/----------------------------Header.py--------------------------------------\
#|                                                                           |
#| Create the initial header of the workflow shell script.  This handles     |
#| input argument reading, as well as initial directory structure            |
#| initialization.                                                           |
#|                                                                           |
#| TODO:                                                                     |
#|  1) Set a default value for INPUT_DIR er al                               |
#|                                                                           |
#\---------------------------------------------------------------------------/

import os, sys, ConfigParser

def write(output_file, wf_cfg, ds_cfg):
    code_name = wf_cfg.get("workflow", "code")
    wf_name = wf_cfg.get("workflow", "name") 
    
    output_file.write("""#!/bin/sh

# (1) Handle input options

SLURM=0
BUILD=0
RUN=0
ANALYSIS=0
UPLOAD=0

while getopts "sbrai:o:w:u:" opt; do
  case "$opt" in 
    s) SLURM=1
     ;;
    b) BUILD=1
     ;;
    r) RUN=1
     ;;
    a) ANALYSIS=1
     ;;
    i) INPUT_DIR=$OPTARG
     ;;  
    o) OUTPUT_DIR=$OPTARG
     ;;  
    w) WORK_DIR=$OPTARG
     ;; 
    u) UPLOAD=$OPTARG
     ;; 
  esac
done

# (2) Setup variables

if [ -z "$TMPDIR" ]; then TMPDIR=/tmp/; fi

WFNAME=%s
TMPCODEDIR=$TMPDIR/code_$(date +"%%y_%%m_%%d-%%H_%%M_%%S")
SRC_DIR=$WORK_DIR/src/%s
TOOL_DIR=$WORK_DIR/tools
    """ % (wf_name, code_name))

