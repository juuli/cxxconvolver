#include <iostream>
#include <boost/thread.hpp>
#include "../include/threadpool.hpp"
#include "cxxConvolver.hpp"

void updateFunc(CxxConvolver* cc, unsigned int output, unsigned int num_outputs,
                unsigned int buffer_idx, float* output_buffers) {
  unsigned int buffer_len = cc->getBufferLen();
  cc->processOutput(output);
  float* cur = cc->getOutputAt(output);
  unsigned int out_buf_idx = output*buffer_len+buffer_idx*buffer_len*num_outputs;
  memcpy(&(output_buffers[out_buf_idx]),
         cur+cc->getBufferLen()-1,
         buffer_len*sizeof(float));
}

int main() {
  std::cout<<"In Main"<<std::endl;

  CxxConvolver cc;
  std::ifstream file("io_init_data.json");
  json j;
  j<<file;
  file.close();

  std::vector< float > input = j["input"].get< std::vector<float> >();
  std::vector< float > output = j["output"].get< std::vector<float> >();
  std::vector< float > filters = j["filters"].get< std::vector<float> >();

  unsigned int num_inputs = j["num_inputs"].get< unsigned int >();
  unsigned int num_outputs = j["num_outputs"].get< unsigned int >();
  unsigned int buffer_len = j["buffer_len"].get< unsigned int >();
  unsigned int filter_len = j["filter_len"].get< unsigned int >();
  unsigned int num_buffers = j["num_buffers"].get< unsigned int >();

  boost::threadpool::pool threadpool(num_outputs);

  cc.initializePartitioning(num_inputs, num_outputs, filter_len, buffer_len);

  std::vector<float> output_buffers(buffer_len*num_buffers*num_outputs);
  std::vector< std::vector<float> > input_signals(num_inputs);
  unsigned int signal_len = buffer_len*num_buffers;

  for(int j = 0; j < num_inputs; j++) {
    for(int i = 0; i < num_outputs; i++) {
      float* cur_filt = &(filters[i*filter_len+j*num_outputs*filter_len]);
      cc.updateFilter(j, i, cur_filt, filter_len);
    }
  }

  for(int i = 0; i < num_inputs; i++) {
    std::vector<float> f;
    f.assign(input.begin()+i*signal_len, input.begin()+(i+1)*signal_len);
    for(int j = 0; j < cc.getBufferLen()-1; j++)
      f.insert(f.begin(), 0.f);
    input_signals.at(i) = f;
  }

  unsigned int temp_num = num_buffers;

  for(int i = 0; i < temp_num; i++) {
    for(int k = 0; k < num_inputs; k++) {
      float* cur = &((input_signals.at(k))[0]);
      cc.addNewBuffer(k, cur+i*buffer_len);
    }

    for(int j = 0; j < num_outputs; j++) {
      threadpool.schedule(boost::bind(updateFunc, &cc, j, num_outputs, i, &(output_buffers[0])));
    }

    threadpool.wait();
  }

  // writeJSON<float>("test_out.json", output_buffers);
  // std::cout<<"Done"<<std::endl;
  // file.close();
  cc.destroyPartitioning();
}
