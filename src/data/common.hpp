/**
 @author Alexander Breuer (anbreuer AT ucsd.edu)
 
 @section DESCRIPTION
 Shared functions for data.
 
 @section LICENSE
 Copyright (c) 2015-2016, Regents of the University of California
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef COMMON_HPP
#define COMMON_HPP

#include <cstdlib>
#include <iostream>
#include "constants.hpp"

namespace odc {
  namespace data {
    class common;
  }
}

class odc::data::common {
  public:
    /**
     * Allocates aligned memory of the given size.
     *
     * @param i_size size in bytes.
     * @param i_alignment alignment of the base pointer.
     **/
    static void* allocate( size_t i_size,
                           size_t i_alignment ) {

      void* l_ptrBuffer;
      bool l_error = (posix_memalign( &l_ptrBuffer, i_alignment, i_size ) != 0);

      if( l_error ) {
        // TODO: Log
        std::cout << "The malloc failed (bytes: " << i_size
                  << ", alignment: " << i_alignment << ")."
                  << std::endl;
      }

      return l_ptrBuffer;
    }
    
    /**
     Initalizes memory block of @c real's to a set value
     
     @param location        A pointer to the beginning of the memory block
     @param toValue         The value to which all @c real's in the memory block should be set
     @param length          The number of @c real's in the block
     */
    static void set_mem(real *location, real toValue, int_pt length) {
        for (int_pt i=0; i<length; i++) {
            location[i] = toValue;
        }
    }
    
    static void release( void *io_memory ) {
      free(io_memory);
    }
};

#endif
