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

#/------------------------------plot.py--------------------------------------\
#|                                                                           |
#| Creates a html report based on seismic receivers at different locations,  |
#| using binary input from awp-odc-os.                                       |
#|                                                                           |
#| The report is a single html file with content generated via Javascript.   |
#| The template report file is stored in Template.html.  The dynamic         |
#| content is generated via a chunk of Javascript generated here.            |
#|                                                                           |
#| TODO:                                                                     |
#|  1) Improve performance (do not repeatedly open same file)                |
#|  2) Add workflow names to report                                          |
#|  3) Append to the same report when dedaling with multiple workflows       |
#|  4) Documentation                                                         |
#|                                                                           |
#\---------------------------------------------------------------------------/

import ConfigParser
import struct, sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.interpolate import interp1d

if len(sys.argv) < 3:
    print "Usage: %s <output directory> <configuration file>" % sys.argv[0]
    exit(1)

outdir      = sys.argv[1]
config_path = sys.argv[2]

plt_cfg = ConfigParser.ConfigParser()
plt_cfg.read(config_path)

js_string = "" # stores the Javascript we will insert into template
bench_num = 1

for sec in plt_cfg.sections():

#   (1) Establish variables

    dx = int(plt_cfg.get(sec, "dim_x"))
    dy = int(plt_cfg.get(sec, "dim_y"))
    dz = int(plt_cfg.get(sec, "dim_z"))

    gflops = plt_cfg.get(sec, "gflops");
    time_per_step = plt_cfg.get(sec, "tpt");

    timestep = int(plt_cfg.get(sec, "timestep"))
    num = int(plt_cfg.get(sec, "num"))

    input = plt_cfg.get(sec, "input")

    rec_string = plt_cfg.get(sec, 'rec')
    rec_list = map(lambda s:s.strip(), rec_string.split(';'))
     
    rec_num = 1

    indices = np.linspace(timestep,timestep*num,num,endpoint=True)
 

#   (2) Compute max, min and average for every file in this set
 
    #ci = 0
    for prefix in ['X', 'Y', 'Z']:
        ci = 0
        min_vals = list(indices)
        max_vals = list(indices)
        avg_vals = list(indices)

        for a in indices:
            avg_val_x = 0
            min_val_x = 0 # XXX: should initialize to first element instead
            max_val_x = 0
            i = int(a)
            file_suffix = "{:07d}".format(i)
            filename = "%s/S%s%s" % (input, prefix, file_suffix)
            file = open(filename, 'rb')
            #fdata = file.read(4 * dx * dy * dz)
            #val = struct.unpack('f', file.read(4 * dx * dy * dz))

            for z in range(dz):
                for y in range(dy):
                    fdata = file.read(4 * dx)
                    x_list = list(struct.unpack("%df" % dx, fdata))
                    for x_val in x_list:
                        avg_val_x = avg_val_x + abs(x_val)
                        if x_val > max_val_x:
                            max_val_x = x_val
                        if x_val < min_val_x:
                            min_val_x = x_val
            avg_val_x = avg_val_x / dx
            avg_val_x = avg_val_x / dy
            avg_val_x = avg_val_x / dz

            min_vals[ci] = min_val_x
            max_vals[ci] = max_val_x
            avg_vals[ci] = avg_val_x
            ci = ci + 1

            print "stats for %s:" % file_suffix
            print avg_val_x
            print min_val_x
            print max_val_x

        f = interp1d(indices, min_vals, kind='cubic')
        xnew = list(indices)
        plt.plot(xnew, f(xnew), '-')
        plt.xlabel("time")
        plt.ylabel(r"min $v_%s$" % prefix)
        plt.savefig("%s/%s_min_%s.png" %(outdir, sec, prefix))
        plt.clf()

        f = interp1d(indices, max_vals, kind='cubic')
        xnew = list(indices)
        plt.plot(xnew, f(xnew), '-')
        plt.xlabel("time")
        plt.ylabel(r"max $v_%s$" % prefix)
        plt.savefig("%s/%s_max_%s.png" %(outdir, sec, prefix))
        plt.clf()

        f = interp1d(indices, avg_vals, kind='cubic')
        xnew = list(indices)
        plt.plot(xnew, f(xnew), '-')
        plt.xlabel("time")
        plt.ylabel(r"avg $v_%s$" % prefix)
        plt.savefig("%s/%s_avg_%s.png" %(outdir, sec, prefix))
        plt.clf()


#   (3) For each recorder, go through files and form the plot

    for rec in rec_list:
        if len(rec) == 0:
          continue
        coords = rec[1:-1].split(',')
        x = int(coords[0].strip())
        y = int(coords[1].strip())
        z = int(coords[2].strip())


        for cd in ['X', 'Y', 'Z']:
            vals = list(indices)
            ci = 0

            for a in indices:
                i = int(a) # Rounding! XXX
                file_suffix = "{:07d}".format(i)
                filename = "%s/S%s%s" % (input, cd, file_suffix)
                file = open(filename, 'rb')
                index = z * dx * dy + y * dx + x
                file.seek(4 * index)
                val = struct.unpack('f', file.read(4))
                vals[ci] = val[0]
                ci = ci + 1


            f = interp1d(indices, vals, kind='cubic')

            xnew = list(indices)
            plt.plot(xnew, f(xnew), '-')
            plt.xlabel("time")
            plt.ylabel(r"$v_%s$" % cd)
            plt.savefig("%s/%s_rc%d%s.png" %(outdir, sec, rec_num, cd))
            plt.clf()

        js_string += "Rec%d_%d = " % (bench_num, rec_num)
        js_string += "{x: %d, y: %d, z: %d, " % (x,y,z)
        js_string += "image: \"%s_rc%d\"};\n" % (sec, rec_num)

        rec_num = rec_num + 1
  
  
    rec_list_string = ""
    for x in range(1,rec_num):
        if x != 1:
            rec_list_string += ", "
        rec_list_string += "Rec%d_%d" % (bench_num, x)
    rec_list_string = "[%s]" % rec_list_string
    js_string += 'B%d = {name: "%s", rec: %s, gflops: "%s", tpt: "%s"};\n' % (bench_num, sec, rec_list_string, gflops, time_per_step) 

    bench_num = bench_num + 1


# Now that we have finished making plots, start building the JavaScript

bench_list_string = ""
for x in range(1,bench_num):
    if x != 1:
        bench_list_string += ", "
    bench_list_string += "B%d" % x
bench_list_string = "[%s]" % bench_list_string

js_string += "Sections.push(%s);\n" % bench_list_string 
js_string += "SectionNames.push(\"Workflow 1\");\n" # XXX fix this


this_dir = os.path.dirname(os.path.realpath(__file__))
template_path = this_dir + "/Template.html"

template = open(template_path, "r")

content = ""

insert_text = "//INSERTHERE"


for line in template:
    if insert_text in line:
        content += line.replace(insert_text, js_string) 
    else:
        content += line

out_file = open(outdir + "/Report.html", "w")
out_file.write(content)





