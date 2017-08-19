##
# @author Alexande Breuer (anbreuer AT ucsd.edu)
# @author Rajdeep Konwar (rkonwar AT ucsd.edu)
#
# @section DESCRIPTION
# SCons build file of AWP.
#
# @section LICENSE
# Copyright (c) 2016-2017, Regents of the University of California
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

import os
import SCons
import subprocess
import warnings

# configuration
vars = Variables()

vars.AddVariables(
  EnumVariable( 'parallelization',
                'parallelization',
                'mpi_omp',
                 allowed_values = ( 'mpi_cuda', 'mpi_omp', 'mpi_omp_yask' )
              ),
  EnumVariable( 'cpu_arch',
                'CPU architecture to compile for',
                'host',
                 allowed_values = ( 'host', 'snb', 'hsw', 'skx', 'knl' )
              ),
  EnumVariable( 'mode',
                'compile mode',
                'release',
                 allowed_values = ( 'release', 'debug', 'asan' )
              ),
  BoolVariable( 'cov',
                'enable code coverage',
                 False ),
)

vars.AddVariables(
  PathVariable( 'cudaToolkitDir', 'directory of CUDA toolkit', None ),
)

# include environment
env = Environment( variables = vars )

# generate help message
Help( vars.GenerateHelpText(env) )

# print welcome message
print( 'Running build script of AWP-ODC-OS.' )

# configuration
conf = Configure( env )

# setup environment
env['ENV'] = os.environ

# forward compiler
if 'CC' in env['ENV'].keys():
  env['CC']   = env['ENV']['CC']
if 'CXX' in env['ENV'].keys():
  env['CXX']  = env['ENV']['CXX']

# forward flags
if 'CFLAGS' in env['ENV'].keys():
  env['CFLAGS']     = env['ENV']['CFLAGS']
if 'CXXFLAGS' in env['ENV'].keys():
  env['CXXFLAGS']   = env['ENV']['CXXFLAGS']
if 'LINKFLAGS' in env['ENV'].keys():
  env['LINKFLAGS']  = env['ENV']['LINKFLAGS']

# detect compilers
try:
  compVer = subprocess.check_output( [env['CXX'], "--version"] )
except:
  compVer = 'unknown'

if 'clang' in compVer:
  compilers = 'clang'
elif 'g++' in compVer:
  compilers = 'gnu'
elif 'icpc' in compVer:
  compilers = 'intel'
else:
  compilers = 'gnu'
  print '  no compiler detected'
env['compilers']  = compilers
print '  using', compilers, 'as compiler suite'

# set compiler and CPU architecture
if env['cpu_arch'] == 'host':
  if compilers == 'intel':
    env.Append( CPPFLAGS = ['-xHost'] )
  elif compilers == 'gnu':
    env.Append( CPPFLAGS = ['-march=native'] )
elif env['cpu_arch'] == 'snb':
  env.Append( CPPFLAGS = ['-mavx'] )
elif env['cpu_arch'] == 'hsw':
  env.Append( CPPFLAGS = ['-mavx2'] )
elif env['cpu_arch'] == 'skx':
  if compilers == 'gnu':
    env.Append( CPPFLAGS = ['-mavx512f', '-mavx512cd'] )
  elif compilers == 'intel':
    env.Append( CPPFLAGS = ['-xCOMMON-AVX512'] )
elif env['cpu_arch'] == 'knl':
  if compilers == 'gnu':
    env.Append( CPPFLAGS = ['-mavx512f', '-mavx512cd', '-mavx512er', '-mavx512pf'] )
  elif compilers == 'intel':
    env.Append( CPPFLAGS = ['-xMIC-AVX512'] )
#  else:
#    env.Append( CPPFLAGS = ['-xHost'] )

# set openmp flags
if 'omp' in env['parallelization']:
  if compilers == 'intel':
    env.Append( CPPFLAGS  = ['-qopenmp'] )
    env.Append( LINKFLAGS = ['-qopenmp'] )
    env.Append( LINKFLAGS = ['-qopenmp-link=static'] )
  elif compilers == 'gnu':
    env.Append( CPPFLAGS  = ['-fopenmp'] )
    env.Append( LINKFLAGS = ['-fopenmp'] )
  elif compilers == 'clang':
    env.Append( CPPFLAGS  = ['-fopenmp'] )
    env.Append( LINKFLAGS = ['-fopenmp'] )
    # get the path of libiomp5 from clang
    try:
      rpath = subprocess.check_output( [env['CXX'], "--print-file-name=libiomp5.so"] )
      rpath = os.path.dirname( rpath )
      env.Append( LINKFLAGS = [ '-Wl,-rpath='+rpath ] )
    except:
      print '  failed getting dir of libiomp5.so, not setting rpath'

# fix clang's confusion with old libstdc++ at runtime (no proper automatic rpath)
if compilers == 'clang':
  try:
    rpath = subprocess.check_output( [env['CXX'], "--print-file-name=libstdc++.so"] )
    rpath = os.path.dirname( rpath )
    env.Append( LINKFLAGS = [ '-Wl,-rpath='+rpath ] )
  except:
    print '  failed getting dir of libstdc++.so, not setting rpath'

# add cuda support if requested
if env['parallelization'] in ['cuda', 'mpi_cuda' ]:
  if 'cudaToolkitDir' not in env:
    print( '*** cudaToolkitDir not set; defaulting to /usr/local/cuda' )
    env['cudaToolkitDir'] = '/usr/local/cuda'

  env.Tool( 'nvcc', toolpath = ['tools/scons'] )
  env.Append( CPPPATH = [ env['cudaToolkitDir'] + '/include' ] )
  env.Append( LIBPATH = [ env['cudaToolkitDir'] + '/lib64' ] )
  env.Append( LIBS    = ['cudart'] )

# set alignment and flags
env.Append( CPPDEFINES = ['ALIGNMENT=64'] )

if env['parallelization'] in ['cuda']:
  env.Append( CPPDEFINES = ['USE_CUDA'] )
elif env['parallelization'] in ['mpi_cuda']:
  env.Append( CPPDEFINES = ['USE_CUDA'] )
  env.Append( CPPDEFINES = ['USE_MPI'] )
elif env['parallelization'] in ['mpi_omp']:
  env.Append( CPPDEFINES = ['AWP_USE_MPI'] )
elif env['parallelization'] in ['mpi_omp_yask']:
  env.Append( CPPDEFINES = ['AWP_USE_MPI'] )
  env.Append( CPPDEFINES = ['YASK',
                            'REAL_BYTES=4',
                            'LAYOUT_3D=Layout_123',
                            'LAYOUT_4D=Layout_1234',
                            'ARCH_HOST',
                            'NO_STORE_INTRINSICS',
                            'USE_RCP28'] )
  if env['cpu_arch'] == 'knl':
    env.Append( CPPDEFINES = ['USE_INTRIN512'] )
  else:
    env.Append( CPPDEFINES = ['USE_INTRIN256'] )
  if not conf.CheckLibWithHeader( 'numa', 'numa.h', 'cxx' ):
    print( 'Did not find libnuma.a or numa.lib, exiting!' )
    Exit( 1 )

# add current path to seach path
env.Append( CPPPATH = [Dir('#.').path, Dir('#./src')] )

# enable c++11
env.Append( CXXFLAGS="-std=c++11" )

# set optimization mode
if 'debug' in env['mode']:
  env.Append( CXXFLAGS = ['-g', '-O0'] )
elif 'release' in env['mode']:
  env.Append( CXXFLAGS = ['-O3'] )
  if compilers == 'gnu':
    env.Append( CXXFLAGS = '-ftree-vectorize' )

# add sanitizers
if 'asan' in  env['mode']:
  env.Append( CXXFLAGS  = ['-g', '-fsanitize=address', '-O1', '-fno-omit-frame-pointer'] )
  env.Append( LINKFLAGS = ['-g', '-fsanitize=address', '-O1'] )

# enable code coverage, if requested
if env['cov'] == True:
  env.Append( CXXFLAGS  = ['-coverage', '-fno-inline', '-fno-inline-small-functions', '-fno-default-inline'] )
  env.Append( LINKFLAGS = ['-coverage'] )

# add math library for gcc
env.Append( LIBS=['m'] )

VariantDir( 'bin', 'src' )

Export( 'env' )
SConscript( 'bin/SConscript' )
