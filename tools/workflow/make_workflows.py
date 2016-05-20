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

#/-------------------------make_workflows.py---------------------------------\
#|                                                                           |
#| Create shell scripts to execute every workflow, specified through config  |
#| files.  It takes one argument, the directory containing the config files. |
#| The shell scripts are created in that same directory.                     |
#|                                                                           |
#| TODO:                                                                     |
#|  1) Use os module for proper paths                                        |
#|  2) Add some error checking                                               |
#|  3) Add logging                                                           |
#|  4) Make all function args be the same                                    |
#|  5) Documentation                                                         |
#|                                                                           |
#\---------------------------------------------------------------------------/


import os, sys, ConfigParser, time
import Stages.Header   as Header
import Stages.Build    as Build
import Stages.Run      as Run
import Stages.Analysis as Analysis


#------------------------------ESTABLISH VARIABLES---------------------------#

# 'ds' stands for datasets
ds_file_name = "datasets.cfg"

# 'wf' stands for workflows
wf_dir = "workflows"

if len(sys.argv) < 2:
    print "Usage: " + sys.argv[0] + " <config file directory>"
    exit(0)

base_dir = sys.argv[1]


#-----------------------------OPEN CONFIG FILES------------------------------#


ds_cfg = ConfigParser.ConfigParser()
ds_cfg.read(base_dir + "/" + ds_file_name)

wf_dir_list = os.listdir(base_dir + "/" + wf_dir)
wf_files = [f for f in wf_dir_list if os.path.isfile(base_dir + "/" + wf_dir + "/" + f)]

#--------------------------------MAIN LOOP-----------------------------------#

for wf_filename in wf_files:

    print "Processing workflow name: %s" % wf_filename
 
    # open config file and output file for this workflow

    wf_path = base_dir + "/" + wf_dir + "/" + wf_filename 
    wf_cfg = ConfigParser.ConfigParser()
    wf_cfg.read(wf_path)

    wf_name = wf_cfg.get("workflow", "code") 
    output_filename = wf_cfg.get("workflow", "output")
    output_file = open(base_dir + "/" + output_filename, "w")

    # create header

    Header.write(output_file, wf_cfg, ds_cfg)

    # create build

    Build.write(output_file, wf_cfg, ds_cfg)

    # create run

    Run.write(output_file, wf_cfg, ds_cfg)
 
    # create analyse

    Analysis.write(output_file, wf_cfg, ds_cfg, base_dir)

    # set executable permissions 
 
    os.chmod(base_dir + "/" + output_filename, 0o755)  



