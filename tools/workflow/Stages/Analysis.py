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

#/----------------------------Analysis.py------------------------------------\
#|                                                                           |
#| Add the analysis part of the workflow shell script.  First creates the    |
#| configuration file, then invokes the python plot script.                  |
#|                                                                           |
#\---------------------------------------------------------------------------/


import os, sys, ConfigParser

def echo(s):
    print s

def write(output_file, wf_cfg, ds_cfg, base_dir):
    wf_name = wf_cfg.get("workflow", "name")
    output_file.write("""
# (5) Analyse runs

if [ ${ANALYSIS} != "0" ]; then
  
  TMPCFG=$OUTPUT_DIR/raw/tmp.cfg
""")

    # To get list of datasets, remove "workflow" from list 
    # of sections in workflow config file
    ds_list = wf_cfg.sections()
    ds_list.remove("workflow")

    cfg_string = ""

    for section in ds_list:
        bench_name = section
        ds_name = wf_cfg.get(section, "dataset")
        rec_list_str = wf_cfg.get(bench_name, "receivers")
        dir_name = "%s_%s" % (wf_name, bench_name)

        log_file = "$OUTPUT_DIR/logs/%s.log" % dir_name
        output_file.write("""
  GFLOPS_%s="`grep GFLOPS %s | sed 's/GFLOPS://g'`" 
  TPT_%s="`grep TPT %s    | sed 's/TPT://g'`"
        """ % (bench_name, log_file, bench_name, log_file))

        dim_x = int(ds_cfg.get(ds_name, "x"))
        dim_y = int(ds_cfg.get(ds_name, "y"))
        dim_z = int(ds_cfg.get(ds_name, "z"))

        tmax = float(wf_cfg.get(bench_name, "tmax"))
        dt = float(wf_cfg.get(bench_name, "dt"))
        timestep = int(wf_cfg.get(bench_name, "skip_t"))
        num = (tmax / dt) / timestep

        cfg_string += "[%s]\n" % bench_name
        cfg_string += "input = $OUTPUT_DIR/raw/%s\n" % dir_name
        cfg_string += "dim_x = %d\n" % dim_x
        cfg_string += "dim_y = %d\n" % dim_y
        cfg_string += "dim_z = %d\n" % dim_z
        cfg_string += "timestep = %d\n" % timestep
        cfg_string += "num = %d\n" % num
        cfg_string += "gflops = $GFLOPS_%s\n" % bench_name
        cfg_string += "tpt = $TPT_%s\n" % bench_name


        # Turn the recevier list from a string into a list
        rec_list = map(lambda s:s.strip(), rec_list_str.split(','))
 
        rec_path = base_dir + "/receivers.cfg" 
        rec_cfg = ConfigParser.ConfigParser()
        rec_cfg.read(rec_path)
        
        rec_out = ""

        for rec in rec_list:
            rec_x = int(rec_cfg.get(rec, "x"))
            rec_y = int(rec_cfg.get(rec, "y"))
            rec_z = int(rec_cfg.get(rec, "z"))

#            output_file.write("""
#  python $TOOL_DIR/plot.py $OUTPUT_DIR/report/%s_%s.png $OUTPUT_DIR/raw/%s/SX %d %d %d %d %d %d %d %d 
#            """ % (dir_name, rec, dir_name, dim_x, dim_y, dim_z, rec_x, rec_y, rec_z, timestep, num)) 
            
            rec_out += "(%d,%d,%d); " % (rec_x, rec_y, rec_z)
        cfg_string += "rec = %s\n\n" % rec_out

    
    output_file.write("""
  echo "%s" > $TMPCFG
  if [ ${SLURM} != "0" ]; then
    sbatch --output="$OUTPUT_DIR/logs/plot.%%j.%%N.out" \
         $TOOL_DIR/scripts/plot.slurm -d `pwd` -s $TOOL_DIR/plot/plot.py \
                                      -o $OUTPUT_DIR/report -i $TMPCFG
  else
    python $TOOL_DIR/plot.py $OUTPUT_DIR/report $TMPCFG
  fi
fi

if [ ${UPLOAD} != "0" ]; then
  LAST_DIR=`pwd`
  cd $OUTPUT_DIR/report
  echo "put *" | sftp $UPLOAD
  cd $LAST_DIR
fi   

    """ % cfg_string)
