////////////////////////////////////////////////////////////////////////////////
//
// This file is a part of the CxxConvolver streaming convolution engine
// library. It is released under the MIT License. You should have
// received a copy of the MIT License along with CxxConvolver.  If not, see
// http://www.opensource.org/licenses/mit-license.php
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// For details, see the LICENSE file
//
// (C) 2018 Jukka Saarelma
//
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <boost/thread.hpp>
#include "cxxConvolver.hpp"
#include "../include/threadpool.hpp"
#include "../include/portaudio.h"


std::vector< std::vector<float> > filter_cont;
int cur_filters = 0;
int next_filters = 0;

bool paCheck(PaError err, std::string loc) {
  if(err != 0) {
    printf("PortAudio error num: %d, message: %s, from: %s" ,err, Pa_GetErrorText(err), loc.c_str());
    return false;
  }
  return true;
}

void printDeviceList() {
  paCheck(Pa_Initialize(), "printDeviceList() - init");
  PaDeviceInfo *device_info = (PaDeviceInfo*)NULL;

  for(int i = 0; i < Pa_GetDeviceCount(); i++) {
      device_info = (PaDeviceInfo*)Pa_GetDeviceInfo(i);
      printf("_______________________________________________\n");
      printf( "Index: %d - Name = %s\n",i, device_info->name );
      printf( "Host API  = %s\n",  Pa_GetHostApiInfo( device_info->hostApi )->name );
      printf( "Max inputs = %d\n", device_info->maxInputChannels  );
      printf( "Max outputs = %d\n", device_info->maxOutputChannels  );
      printf( "Default low input latency   = %8.4f\n", device_info->defaultLowInputLatency);
      printf( "Default low output latency  = %8.4f\n", device_info->defaultLowOutputLatency);
      printf( "Default high input latency  = %8.4f\n", device_info->defaultHighInputLatency);
      printf( "Default high output latency = %8.4f\n", device_info->defaultHighOutputLatency);
      printf( "Default sample rate         = %8.2f\n", device_info->defaultSampleRate );
  }
}

typedef struct callback_struct_t{
  int num_inputs;
  int num_outputs;
  CxxConvolver* cc;
  boost::threadpool::pool* pool;
} CallbackStruct;

void updateFunc(CxxConvolver* cc, unsigned int output) {
  int filter_len = cc->getFilterLen();
  int num_outputs = cc->getNumOutputs();

  for(int i = 0; i < cc->getNumInputs(); i++) {
    bool swap = cc->isSwapping(i, output);
    if(swap) {
      int filter_address = (output+i*num_outputs)*filter_len;
      float* filter = &(filter_cont.at(next_filters)[filter_address]);
      cc->execPartSwap(i, output, filter, filter_len);
    }
  }

  if(cc->getNumSwapping()==0)
    cur_filters = next_filters;

  cc->processOutput(output);
}
static int playRecCallback(const void *inputBuffer,
                           void *outputBuffer,
                           unsigned long frames_per_buffer,
                           const PaStreamCallbackTimeInfo* timeInfo,
                           PaStreamCallbackFlags statusFlags,
                           void *user_data ) {

  float *out = (float*)outputBuffer;
  const float *in = (const float*)inputBuffer;

  CallbackStruct* ud = (CallbackStruct*)user_data;
  ud->cc->addInputFrame(in);

  for(int j = 0; j < ud->cc->getNumOutputs(); j++)
    ud->pool->schedule(boost::bind(updateFunc, ud->cc, j));

  ud->pool->wait();
  ud->cc->getOutputFrame(out);

  return paContinue;
}

int main() {
  std::cout<<"In Main"<<std::endl;

  CxxConvolver cc;

  std::ifstream file("playback_data_brir_240.json");
  json j;
  j<<file;
  file.close();

  std::vector< float > filters = j["filters"].get< std::vector<float> >();
  filter_cont.push_back(filters);


  std::ifstream file2("playback_data_brir_60.json");
  json j2;
  j2<<file2;
  file2.close();

  std::vector< float > filters2 = j2["filters"].get< std::vector<float> >();
  filter_cont.push_back(filters2);


  std::ifstream file3("playback_data_brir_120.json");
  json j3;
  j3<<file3;
  file3.close();

  std::vector< float > filters3= j3["filters"].get< std::vector<float> >();
  filter_cont.push_back(filters3);

  unsigned int num_inputs = j["num_inputs"].get< unsigned int >();
  unsigned int num_outputs = j["num_outputs"].get< unsigned int >();
  unsigned int buffer_len = j["buffer_len"].get< unsigned int >();
  unsigned int filter_len = j["filter_len"].get< unsigned int >();
  unsigned int fs = j["fs"].get< unsigned int >();

  std::cout<<"Outputs: "<<num_outputs<<std::endl;
  std::cout<<"Inputs: "<<num_inputs<<std::endl;
  boost::threadpool::pool threadpool(num_outputs);

  cc.initializePartitioning(num_inputs, num_outputs, filter_len, buffer_len);

  for(int j = 0; j < num_inputs; j++) {
    for(int i = 0; i < num_outputs; i++) {
      int filter_address = (i+j*num_outputs)*filter_len;
      float* cur_filt = &(filters[filter_address]);
      cc.updateFilter(j, i, cur_filt, filter_len);
    }
  }

  CallbackStruct cbs;
  cbs.cc = &cc;
  cbs.pool = &threadpool;
  cbs.num_outputs = num_outputs;
  cbs.num_inputs = num_inputs;

  PaStream* stream;
  PaStreamParameters input_params, output_params;
  printDeviceList();
  paCheck(Pa_Initialize(), "main init");

  int device_idx = 6; // 3 M-Audio, 6 is RME
  const PaDeviceInfo* device_info = Pa_GetDeviceInfo(device_idx);


  input_params.device = device_idx;
  input_params.channelCount = (int)num_inputs;
  input_params.sampleFormat = paFloat32;
  input_params.hostApiSpecificStreamInfo = NULL;
  input_params.suggestedLatency = device_info->defaultLowInputLatency;

  output_params.device = device_idx;
  output_params.channelCount = (int)num_outputs;
  output_params.sampleFormat = paFloat32;
  output_params.hostApiSpecificStreamInfo = NULL;
  output_params.suggestedLatency = device_info->defaultLowOutputLatency;

  paCheck(Pa_IsFormatSupported(&input_params, &output_params, fs), "main");

  paCheck(Pa_OpenStream(&stream,
                        &input_params,
                        &output_params,
                        fs,
                        buffer_len,
                        0,
                        playRecCallback,
                        &cbs), "MeasurementBlock::playRec");

  paCheck(Pa_StartStream(stream), "MeasurementBlock::playRec");

  while(true) {
    char c;
    std::cin>>c;
    int i = atoi(&c);
    if(i < 0 || i >= filter_cont.size())
      std::cout<<"Input 0 - "<<filter_cont.size()-1<< " to swap filters or 'x' to close"<<std::endl;
    if(c=='x')
      break;
    if(i >= 0 && i < filter_cont.size()) {
      next_filters = i;
      if(next_filters!=cur_filters) {
        std::cout<<"Switching to filters "<<i<<std::endl;
        cc.setAllSwap();
      }
    }
  }

  paCheck(Pa_CloseStream(stream), "MeasurementBlock::playRec");
  paCheck(Pa_Terminate(), "MeasurementBlock::playRec");

  std::cout<<"Done"<<std::endl;

  cc.destroyPartitioning();
}
