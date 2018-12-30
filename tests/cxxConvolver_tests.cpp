#if !defined(_WIN32)
  #define BOOST_TEST_DYN_LINK
#endif

#define BOOST_TEST_MAIN

#include <ctime>
#include <cmath>
#include <iostream>
#include <boost/test/unit_test.hpp>
#include "../src/cxxConvolver.hpp"


BOOST_AUTO_TEST_CASE(convolver_getters) {
 CxxConvolver cc;
 cc.initializePartitioning(11, 13, 11200, 523);
 BOOST_CHECK_EQUAL(cc.getNumOutputs(), 13);
 BOOST_CHECK_EQUAL(cc.getNumInputs(), 11);
 BOOST_CHECK_EQUAL(cc.getFilterLen(), 11200);
 BOOST_CHECK_EQUAL(cc.getBufferLen(), 523);

 BOOST_CHECK_EQUAL(cc.getTransformLen(), 523*2-1);

 unsigned int num_filter_parts = ceil(11200.f/523.f);
 BOOST_CHECK_EQUAL(cc.getNumFilterParts(), num_filter_parts);
 BOOST_CHECK_EQUAL(cc.getNumFilters(), 11*13);

 // Number of float values in single filter container in time domain
 BOOST_CHECK_EQUAL(cc.getFilterContainerSizeT(), num_filter_parts*(523*2-1));

 // Freq domain the values are complex, hence container size is N/2+1
 BOOST_CHECK_EQUAL(cc.getFilterContainerSizeF(), num_filter_parts*((523*2-1)/2+1));
 cc.destroyPartitioning();
}


BOOST_AUTO_TEST_CASE(convolver_filter_assign_long_simple) {
 CxxConvolver cc;
 cc.initializePartitioning(5, 7, 10, 5);

 unsigned int test_len = 20;
 unsigned int count = 0;

 std::vector< std::vector<float> > filters;
 for(unsigned int i = 0; i < cc.getNumFilters(); i++) {
   filters.push_back(std::vector<float>());
   for(unsigned int j = 0; j < test_len; j++ ) {
     filters.at(i).push_back((float)count);
     count++;
   }
 }

 count = 0;
 for(unsigned int i = 0; i < cc.getNumInputs(); i++) {
   for(unsigned int j = 0; j < cc.getNumOutputs(); j++) {
     cc.updateFilter(i, j, filters.at(count));
     count++;
   }
 }

 count=0;
 for(unsigned int i = 0; i < cc.getNumInputs(); i++) {
   for(unsigned int j = 0; j < cc.getNumOutputs(); j++) {
     float* cur_filter = cc.getTimeFilterAt(i, j);
     for(unsigned int k = 0; k<cc.getFilterContainerSizeT(); k++) {
       unsigned int idx = k%cc.getTransformLen();
       unsigned int f_idx = k/cc.getTransformLen();
       if(idx<cc.getBufferLen()) {
         BOOST_CHECK_EQUAL(cur_filter[idx+f_idx*cc.getTransformLen()], (float)count);
         count++;
       }
       else{
         BOOST_CHECK_EQUAL(cur_filter[idx+f_idx*cc.getTransformLen()], 0.f);
       }
     }
     count+=test_len-cc.getBufferLen()*cc.getNumFilterParts();
   }
 }

 cc.destroyPartitioning();
}

BOOST_AUTO_TEST_CASE(convolver_filter_assign_long) {
 CxxConvolver cc;
 cc.initializePartitioning(2, 4, 2117, 513);

 unsigned int test_len = 5000;
 unsigned int count = 0;

 std::vector< std::vector<float> > filters;
 for(unsigned int i = 0; i < cc.getNumFilters(); i++) {
   filters.push_back(std::vector<float>());
   for(unsigned int j = 0; j < test_len; j++ ) {
     filters.at(i).push_back((float)count);
     count++;
   }
 }

 count = 0;
 for(unsigned int i = 0; i < cc.getNumInputs(); i++) {
   for(unsigned int j = 0; j < cc.getNumOutputs(); j++) {
     cc.updateFilter(i, j, filters.at(count));
     count++;
   }
 }

 count=0;
 for(unsigned int i = 0; i < cc.getNumInputs(); i++) {
   for(unsigned int j = 0; j < cc.getNumOutputs(); j++) {
     float* cur_filter = cc.getTimeFilterAt(i, j);
     for(unsigned int k = 0; k<cc.getFilterContainerSizeT(); k++) {
       unsigned int idx = k%cc.getTransformLen();
       unsigned int f_idx = k/cc.getTransformLen();
       if(idx<cc.getBufferLen()) {
         BOOST_CHECK_EQUAL(cur_filter[idx+f_idx*cc.getTransformLen()], (float)count);
         count++;
       }
       else{
         BOOST_CHECK_EQUAL(cur_filter[idx+f_idx*cc.getTransformLen()], 0.f);
       }
     }
     count+=test_len-cc.getBufferLen()*cc.getNumFilterParts();
   }
 }

 cc.destroyPartitioning();
}

BOOST_AUTO_TEST_CASE(convolver_fft_ifft) {
 CxxConvolver cc;
 unsigned int filter_len = 6889;
 unsigned int buffer_len = 1882;
 unsigned int transform_len = buffer_len*2-1;

 cc.initializePartitioning(1, 1, filter_len, buffer_len);

 float* t_input = (float*)calloc(transform_len, sizeof(float));
 float* t_output = (float*)calloc(transform_len, sizeof(float));
 fftwf_complex* f_output = (fftwf_complex*)calloc(transform_len/2+1, sizeof(fftwf_complex));

 t_input[0] = 1.f;

 cc.fft(t_input, f_output);

 for(int i = 0; i < transform_len/2+1; i++) {
   BOOST_CHECK_EQUAL((float)f_output[i][0], 1.0f);
   BOOST_CHECK_EQUAL((float)f_output[i][1], 0.0f);
 }

 cc.ifft(f_output, t_output);

 for (int i = 0; i<transform_len; i++) {
   if(i==0)
     BOOST_CHECK_EQUAL((float)t_output[i]/(float)transform_len, 1.0f);
   else {
     BOOST_CHECK((bool)(abs(t_output[i]/(float)transform_len)<1.0e-7f));
   }
 }

 for (int i = 0; i<transform_len; i++)
   if (i<buffer_len)
     t_input[i] = ((float)rand()/(float)RAND_MAX)*2.0f-1.0f;
   else
     t_input[i] = 0.f;

 cc.fft(t_input, f_output);

 // Test if the real to complex dump random crap in the trunctaion part
 for (int i = 0; i<transform_len; i++)
   if (i>=buffer_len)
     BOOST_CHECK_EQUAL(t_input[i], 0.f);

 free(t_input);
 free(t_output);
 free(f_output);
 cc.destroyPartitioning();
}


BOOST_AUTO_TEST_CASE(add_buffer) {
 CxxConvolver cc;
 unsigned int filter_len = 24000;
 unsigned int buffer_len = 512;
 unsigned int transform_len = buffer_len*2-1;
 unsigned int num_inputs = 2;
 unsigned int num_outputs = 6;

 cc.initializePartitioning(num_inputs, num_outputs, filter_len, buffer_len);

 unsigned int num_filter_parts = cc.getNumFilterParts();

 std::vector< std::vector<float> > buffers;

 // Initialize generic buffer sequence, long enough to fill the whole fdl,
 // in this case 2 buffers longer
 unsigned int test_container_len = buffer_len*(num_filter_parts+2);

 for (int i = 0; i<num_inputs; i++) {
   BOOST_CHECK_EQUAL(cc.getFdlPosition(i), 0);

   std::vector<float> input;
   input.assign(test_container_len+buffer_len, 0.f);

   // Assign value input*1e5 + buffer_index*1e3 + buffer_sample_index
   for (int j = 0; j<test_container_len; j++)
     input.at(j) = 1e5*i+j/buffer_len*1e3f;//+j%buffer_len;

   buffers.push_back(input);
 }

 // Test adding buffers to the convolver
 float* cur_timedomain = (float*)calloc(cc.getTransformLen(), sizeof(float));

 // Go through the generic buffers and add them one by one
 // Test on the fly that the partition index works, and that
 // the frequency domain version is correct at right position
 for (int part = 0; part<cc.getNumFilterParts()+2; part++) {

   for (int i = 0; i<num_inputs; i++) {
     cc.addNewBuffer(i, &(buffers.at(i)[part*buffer_len]));
     for(int k = 0; k<buffer_len;k++)
       BOOST_CHECK_EQUAL(cc.getInBuffers(i)[k], buffers.at(i)[part*buffer_len+k]);
   }

   // Add new buffer increments the fdl_position counter for each input
   for (int i = 0; i<num_inputs; i++) {
     BOOST_CHECK_EQUAL(cc.getFdlPosition(i), MOD((part+1), cc.getNumFilterParts()));

     // Added buffer in frequency domain (with time domain padding)
     fftwf_complex* cur = cc.getFdlAt(i, part);
     // ifft
     cc.ifft(cur, cur_timedomain);

     // Check that input index is correct
     BOOST_CHECK(cur_timedomain[0]/transform_len-buffers.at(i)[part*buffer_len]<1e-1);
     // Seems like with very large numbers (>1e5) the error is aroud 20 dB

     // std::cout<<cur_timedomain[0]/transform_len<<" , "<<1e5*i<<", "<<<<std::endl;
     // std::cout<<cur_timedomain[0]/transform_len-buffers.at(i)[part*buffer_len]<<std::endl;
   }
 }

 free(cur_timedomain);
 cc.destroyPartitioning();
}

BOOST_AUTO_TEST_CASE(add_buffer_more_parts) {
  CxxConvolver cc;
  unsigned int filter_len = 45001;
  unsigned int buffer_len = 1289;
  unsigned int transform_len = buffer_len*2-1;
  unsigned int num_inputs = 5;
  unsigned int num_outputs = 11;

  cc.initializePartitioning(num_inputs, num_outputs, filter_len, buffer_len);

  unsigned int num_filter_parts = cc.getNumFilterParts();

  std::vector< std::vector<float> > buffers;

  // Initialize generic buffer sequence, long enough to fill the whole fdl,

  unsigned int num_extra_buffers=10;
  unsigned int test_container_len = buffer_len*(num_filter_parts+num_extra_buffers);

  for (int i = 0; i<num_inputs; i++) {
    BOOST_CHECK_EQUAL(cc.getFdlPosition(i), 0);

    std::vector<float> input;
    input.assign(test_container_len+buffer_len, 0.f);

    // Assign value input*1e5 + buffer_index*1e3 + buffer_sample_index
    for (int j = 0; j<test_container_len; j++)
      input.at(j) = (1e5*i+j/buffer_len*1e3f)*1e-5;//+j%buffer_len;

    for(int i = 0; i<cc.getBufferLen()/2-1; i++)
      input.insert(input.begin(), 0.f);

    buffers.push_back(input);
  }

  // Test adding buffers to the convolver
  float* cur_timedomain = (float*)calloc(cc.getTransformLen(), sizeof(float));

  // Go through the generic buffers and add them one by one
  // Test on the fly that the partition index works, and that
  // the frequency domain version is correct at right position
  for (int part = 0; part < cc.getNumFilterParts()+num_extra_buffers; part++) {

    for (int i = 0; i<num_inputs; i++) {
      cc.addNewBuffer(i, &(buffers.at(i)[part*buffer_len]));

      for(int k = 0; k<buffer_len;k++)
        BOOST_CHECK_EQUAL(cc.getInBuffers(i)[k], buffers.at(i)[part*buffer_len+k]);
    }

    // Adding a new buffer increments the fdl_position counter for each input
    for (int i = 0; i<num_inputs; i++) {
      BOOST_CHECK_EQUAL(cc.getFdlPosition(i), MOD((part+1), cc.getNumFilterParts()));

      // Added buffer (with time domain padding) in frequency domain
      fftwf_complex* cur = cc.getFdlAt(i, cc.getFdlPosition(i));
      // ifft
      cc.ifft(cur, cur_timedomain);

      // Check that input index is correct
      BOOST_CHECK(std::abs(cur_timedomain[0]/transform_len-buffers.at(i)[part*buffer_len])<1e-4);

    }
  }

  free(cur_timedomain);
  cc.destroyPartitioning();
}

// BOOST_AUTO_TEST_CASE(process_input_output) {
//   CxxConvolver cc;
//   std::ifstream file("reference_data.json");
//   json j;
//   j<<file;
//
//   std::vector< float > input = j["input"].get< std::vector<float> >();
//   std::vector< float > output = j["output"].get< std::vector<float> >();
//   std::vector< float > filters = j["filters"].get< std::vector<float> >();
//
//   unsigned int num_inputs = 1;
//   unsigned int num_outputs = j["num_outputs"].get< unsigned int >();
//   unsigned int buffer_len = j["buffer_len"].get< unsigned int >();
//   unsigned int filter_len = j["filter_len"].get< unsigned int >();
//   unsigned int num_buffers = j["num_buffers"].get< unsigned int >();
//
//   cc.initializePartitioning(num_inputs, num_outputs, filter_len, buffer_len);
//   std::vector<float> output_buffers(buffer_len*num_buffers*num_outputs);
//
//   BOOST_CHECK_EQUAL(input.size(), num_buffers*buffer_len);
//   for(int i = 0; i < num_outputs; i++) {
//     float* cur_filt = &(filters[i*filter_len]);
//     cc.updateFilter(0, i, cur_filt, filter_len);
//   }
//
//
//   for(int i = 0; i < cc.getBufferLen()-1; i++)
//     input.insert(input.begin(), 0.f);
//
//   unsigned int temp_num = num_buffers;
//
//   for(int i = 0; i < temp_num; i++) {
//     cc.addNewBuffer(0, &(input[i*buffer_len]));
//     BOOST_CHECK_EQUAL(cc.getFdlPosition(0), MOD((i+1), cc.getNumFilterParts()));
//
//     for(int j = 0; j < num_outputs; j++) {
//       cc.processInputOutput(0, j);
//       float* cur = cc.getOutputAt(j);
//       unsigned int out_buf_idx = j*buffer_len+i*buffer_len*num_outputs;
//       memcpy(&(output_buffers[out_buf_idx]), cur+cc.getBufferLen()-1, buffer_len*sizeof(float));
//     }
//   }
//
//   file.close();
//   cc.destroyPartitioning();
// }

BOOST_AUTO_TEST_CASE(swap_filters) {
  CxxConvolver cc;
  unsigned int filter_len = 45001;
  unsigned int buffer_len = 1289;
  unsigned int transform_len = buffer_len*2-1;
  unsigned int num_inputs = 5;
  unsigned int num_outputs = 11;

  cc.initializePartitioning(num_inputs, num_outputs, filter_len, buffer_len);

  std::vector<float> filters_1(num_inputs*num_outputs*filter_len);
  std::vector<float> filters_2(num_inputs*num_outputs*filter_len);
  int count = 0;
  for(int i = 0; i < num_inputs; i++) {
    for(int j = 0; j < num_outputs; j++) {
      int f_address = (j+num_outputs*i)*filter_len;
      for(int k = 0; k < filter_len; k++) {
        filters_1.at(f_address+k) = count;
        filters_2.at(f_address+k) = -count;
        count++;
      }
    }
  }

  for(int i = 0; i < num_inputs; i++) {
    for(int j = 0; j < num_outputs; j++) {
      int f_address = (j+num_outputs*i)*filter_len;
      cc.updateFilter(i, j, &(filters_1[f_address]), filter_len);
    }
  }

  int swapped_input = 2;
  int swapped_output = 3;
  int swapped_part = 4;
  int swapped_address = (swapped_output+swapped_input*num_outputs)*filter_len;
  cc.swapPart(swapped_input, swapped_output,
              &(filters_2[swapped_address]), filter_len, swapped_part);

  float* cc_ptr = cc.getTimeSwapFilterAt(swapped_input, swapped_output);
  int f_address = (swapped_output+num_outputs*swapped_input)*filter_len;

  for(int i = 0; i < cc.getBufferLen(); i++) {
    BOOST_CHECK_EQUAL(cc_ptr[i+transform_len*(swapped_part-1)],
                      0.f);
    BOOST_CHECK_EQUAL(cc_ptr[i+transform_len*(swapped_part+1)],
                      0.f);
    BOOST_CHECK_EQUAL(cc_ptr[i+transform_len*swapped_part],
                      filters_2[f_address+buffer_len*swapped_part+i]);
  }

  cc.setAllSwap(true);
  for(int i = 0; i < num_inputs; i++)
    for(int j = 0; j < num_outputs; j++)
      BOOST_CHECK_EQUAL(cc.isSwapping(i, j), true);

  int num_filter_parts = cc.getNumFilterParts();

  // Swap more filter parts that there is in the class
  for(int count = 0; count < 50; count++) {
    if(count < num_filter_parts) {
      for(int i = 0; i < num_inputs; i++) {
        for(int j = 0; j < num_outputs; j++) {
          BOOST_CHECK_EQUAL(cc.getSwapPartPosition(i, j), count);
          BOOST_CHECK_EQUAL(cc.isSwapping(i, j), true);
          BOOST_CHECK_EQUAL(cc.getNumSwapping(), num_outputs*num_inputs);
        }
      }
    }
    else {
      for(int i = 0; i < num_inputs; i++) {
        for(int j = 0; j < num_outputs; j++) {
          BOOST_CHECK_EQUAL(cc.isSwapping(i, j), false);
        }
      }
    }

    for(int i = 0; i < num_inputs; i++) {
      for(int j = 0; j < num_outputs; j++) {
        int f_address = (j+num_outputs*i)*filter_len;
        cc.execPartSwap(i,j, &filters_2[f_address], filter_len);
      }
    }
  }

  // Check swapped filter
  for(int i = 0; i < num_inputs; i++) {
    for(int j = 0; j < num_outputs; j++) {
      float* cc_ptr = cc.getTimeSwapFilterAt(i, j);
      int f_address = (j+num_outputs*i)*filter_len;
      for(int k = 0; k < cc.getBufferLen(); k++)
        BOOST_CHECK_EQUAL(cc_ptr[k], filters_2[f_address+k]);
    }
  }
  cc.destroyPartitioning();
}
