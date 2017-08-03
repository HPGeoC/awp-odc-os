##
# @author Alexander Breuer (anbreuer AT ucsd.edu)
# @author Josh Tobin (rjtobin AT ucsd.edu)
#
# @section DESCRIPTION
# Creates a XDMF container file for AWP's raw binary output.
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

# Generates the XDMF header.
# $1: Nodes in x-, y- and z-direction, space separated.
# $2: Origin. x-, y- and z-direction, space separated).
# $3: Mesh width. dx, dy, dz, space separated.
# $4: Output times of the grid files (seconds), space separated.
generate_header() {
sed "s/TMPL_NODES_XYZ/${1}/g" template_header.xdmf |
sed "s/TMPL_ORIGIN_XYZ/${2}/g" |
sed "s/TMPL_D_XYZ/${3}/g" |
sed "s/TMPL_PLOT_TIMES/${4}/g"
}

# Usage info
show_help() {
cat << EOF
Usage: ${0##*/} [-h] [-x NODES_X -y NODES_Y -z NODES_Z -d DELTA_H -t TIME_STEP -n NOUTPUTS -f OUTPUT_FREQUENCY -o XDMF_OUTPUT]
Builds the code.
     -h                  display this help and exit
     -x NODES_X          Number of nodes in x-direction. Remark: This refers to nodes in the output file, not the simulation.
     -y NODES_Y          Number of nodes in y-direction. Remark: This refers to nodes in the output file, not the simulation.
     -z NODES_Z          Number of nodes in z-direction. Remark: This refers to nodes in the output file, not the simulation.
     -d DELTA_H          Mesh width.
     -t TIME_STEP        Time step (seconds).
     -n NOUTPUTS         Number of time steps which produced output.
     -f OUTPUT_FREQUENCY Output frequency, e.g. 20 means output is plotted every 20 time steps.
     -o XDMF_OUTPUT      Output file where the generated XDMF goes.
EOF
}

# Generates the grids in of a output time.
# $1: Name of the time step.
# $2: Nodes in x-, y- and z-direction, space separated.
# $3: Name of the binary for x-vel.
# $4: Name of the binary for y-vel.
# $5: name of the binary for z-vel.
generate_grid() {
sed "s/TMPL_TS_NAME/${1}/g" template_grids.xdmf |
sed "s/TMPL_NODES_XYZ/${2}/g" |
sed "s/TMPL_BINARY_X/${3}/g" |
sed "s/TMPL_BINARY_Y/${4}/g" |
sed "s/TMPL_BINARY_Z/${5}/g"
}

# parse command line arguments
NODES_X=NOT_SET
NODES_Y=NOT_SET
NODES_Z=NOT_SET
DELTA_H=NOT_SET
TIME_STEP=NOT_SET
NOUTPUTS=NOT_SET
OUTPUT_FREQUENCY=NOT_SET
XDMF_OUTPUT=NOT_SET

OPTIND=1
while getopts "hx:y:z:d:t:f:o:n:" opt; do
    case "$opt" in
        h)
            show_help
            exit 0
            ;;
        x) NODES_X=$OPTARG
            ;;
        y) NODES_Y=$OPTARG
            ;;
        z) NODES_Z=$OPTARG
            ;;
        d) DELTA_H=$OPTARG
            ;;
        t) TIME_STEP=$OPTARG
            ;;
        n) NOUTPUTS=$OPTARG
            ;;
        f) OUTPUT_FREQUENCY=$OPTARG
            ;;
        o) XDMF_OUTPUT=$OPTARG
            ;;
        '?')
            show_help >&2
            exit 1
            ;;
    esac
done
shift "$((OPTIND-1))" # Shift off the options and optional --.

echo "Generating XDMF wrapper. It goes to ${XDMF_OUTPUT}!"

# generate file
touch ${XDMF_OUTPUT}


# time steps at which output was written
OUT_TIMES=""

# simulation time at which output was written
OUT_STEPS=""

# generate series of outputs
for output in $(seq 1 1 ${NOUTPUTS})
do
  OUT_STEPS="${OUT_STEPS} $(echo "$output*${OUTPUT_FREQUENCY}" | bc)"
  OUT_TIMES="${OUT_TIMES} $(echo "$output*${TIME_STEP}*${OUTPUT_FREQUENCY}" | bc)"
done

# generate the header
generate_header "${NODES_Z} ${NODES_Y} ${NODES_X}" "0 0 0" "${DELTA_H} ${DELTA_H} ${DELTA_H}" "${OUT_TIMES}" > ${XDMF_OUTPUT}

# generate the grids
for plot in ${OUT_STEPS}
do
  # generate leading zeros
  plot_with_lz="$(printf "%07d" ${plot})"
  generate_grid "TS_${plot}" "${NODES_Z} ${NODES_Y} ${NODES_X}" "SX${plot_with_lz}" "SY${plot_with_lz}" "SZ${plot_with_lz}" >> ${XDMF_OUTPUT}
done

# close file
cat template_footer.xdmf >> ${XDMF_OUTPUT}

echo "Done! Copy ${XDMF_OUTPUT} to the location of your output files SX*, SY* and SZ*."
