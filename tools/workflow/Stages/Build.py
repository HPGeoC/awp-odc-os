##
# @author Josh Tobin (rjtobin AT ucsd.edu)
#
# @section LICENSE
# Copyright (c) 2016, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modificati$
#
# 1. Redistributions of source code must retain the above copyright notice, thi$
#
# 2. Redistributions in binary form must reproduce the above copyright notice, $
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" A$
##

#/------------------------------Build.py-------------------------------------\
#|                                                                           |
#| Add the build part of the workflow shell script.  First copies the source |
#| to a temporary directory, builds by calling build.sh, and copies the      |
#| binaries back to the output directory.                                    |
#|                                                                           |
#\---------------------------------------------------------------------------/

import os, sys, ConfigParser

def write(output_file, wf_cfg, ds_cfg):
    wf_name = wf_cfg.get("workflow", "name") 
 
    parallel_mode = wf_cfg.get("workflow", "parallel")

    # copy code to tmpdir, build it, copy binary to workdir

    output_file.write("""
# (3) Build code

if [ ${BUILD} != "0" ]; then

# Setup all necessary directories
  if [ ! -d  "$TMPCODEDIR" ]; then
    mkdir $TMPCODEDIR
  fi

  if [ ! -d  "$OUTPUT_DIR" ]; then
    mkdir $OUTPUT_DIR
  fi

  if [ ! -d  "$OUTPUT_DIR/bin" ]; then
    mkdir $OUTPUT_DIR/bin
  fi  

  if [ ! -d  "$OUTPUT_DIR/raw" ]; then
    mkdir $OUTPUT_DIR/raw
  fi  

  if [ ! -d  "$OUTPUT_DIR/logs" ]; then
    mkdir $OUTPUT_DIR/logs
  fi  

  if [ ! -d  "$OUTPUT_DIR/report" ]; then
    mkdir $OUTPUT_DIR/report
  fi  

# Copy across and compile
  cp -R $SRC_DIR/* $TMPCODEDIR
  cp $TOOL_DIR/scripts/build.sh $TMPCODEDIR
  cd $TMPCODEDIR
  ./build.sh -o bin/a.out -p %s
  cd -

# Copy binaries back to output directory
  cp $TMPCODEDIR/bin/a.out $OUTPUT_DIR/bin/%s


    """ % (parallel_mode, wf_name))

    # create directories for each benchmark
  
    for section in wf_cfg.sections():
        if section != "workflow":
            bench_name = section
            dir_name = "%s_%s" % (wf_name, bench_name)
            output_file.write("""
  if [ ! -d  "$OUTPUT_DIR/raw/%s" ]; then
    mkdir $OUTPUT_DIR/raw/%s
  fi  
            """ % (dir_name, dir_name))


    # close up the build section
    output_file.write("""
fi
    """)
