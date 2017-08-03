## Workflow process example on Comet

1. To begin, copy the example directory structure to your desired working directory,
 and cd into it.

  ```
cp -r /oasis/projects/nsf/sds143/rjt/sample awp
cd awp
  ```
2. Let's check out a new copy of AWP.

  ```
cd work/src
git clone --recursive https://github.com/HPGeoC/awp-odc-os.git
  ```

3. We will need to make a new configuration file to correspond to 
  this new copy of the source.  We will base it off the existing configuration file.

  ```
cd ../../cfg/workflows
cp wf1.cfg wf_new.cfg
  ```

  The configuration file consists of a `workflow` section and a section 
  for each benchmark (these benchmark sections can be named anything, and 
  these names will be used in the report to designate the different workflows).
  Edit the workflow section as follows:

  ```
name = wf_new
code = awp-odc-os
output = wf_new.sh
scheduler = slurm
parallel = mpi_cuda
  ```

  The `name` field is simply a description for our benefit.  The `code` 
  field specifies the subdirectory of `src/` containing our code.  The
  `output` field is the name of the shell script to create.  The
  `scheduler` is either `slurm` or `none` (in the latter case, the 
  awp code is invoked directly in the generated shell script, instead 
  of via `sbatch`).  Finally `parallel` is either `mpi_cuda` or `mpi_omp`
  for GPU and CPU versions respectively. 
4. Now let's generate our workflow scripts.

  ```
cd ../../work/tools/workflow
python make_workflows.py ../../../cfg
  ```

  Our shell scripts will be in the cfg directory.

  ```
cd ../../../
cp cfg/wf_new.sh .
  ```

5. Let's use the script to setup the directory structure and build the code

  ```
mkdir output
./wf_new.sh -b -o output -w work -i work
  ```

  Now let's schedule a run of our two benchmarks.

  ```
./wf_new.sh -r -o output -w work -i work
  ```

  When these jobs have completed, we can generate an analysis report.

  ```
./wf_new.sh -a -s -o output -w work -i input
  ```

  Note the `-s` flag, this schedules the analysis code to run on a compute
  node.  If it is absent, python will be called directly (and be promptly
  killed on Comet for excessive memory usage).  
  The report will be saved in `output/report` (the raw output as well as 
  run logs are stored in other subdirectories).  
  Finally, to push the report to an easy-to-access location, use

  ```
./wf_new.sh -u username@server:/path/ -o output -w work -i work
  ```




