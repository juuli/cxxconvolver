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

#include "cxxConvolver.hpp"

// C interface
extern "C" {
  DLLEXPORT void* initialize(unsigned int num_inputs,
                             unsigned int num_outputs,
                             unsigned int filter_len,
                             unsigned int buffer_len) {
    CxxConvolver* cc = new CxxConvolver();
    cc->initializePartitioning(num_inputs, num_outputs, filter_len, buffer_len);
    return (void*)cc;
  }

  DLLEXPORT void addInputFrame(void* cc_, const float* data) {
    CxxConvolver* cc = (CxxConvolver*)cc_;
    cc->addInputFrame(data);
  }

  DLLEXPORT void destroy(void* cc_) {
    CxxConvolver* cc = (CxxConvolver*)cc_;
    cc->destroyPartitioning();
    delete cc;
  }

  DLLEXPORT void processOutput(void* cc_, unsigned int output) {
    CxxConvolver* cc = (CxxConvolver*)cc_;
    cc->processOutput(output);
  }

  DLLEXPORT void getOutputFrame(void* cc_, float* data) {
    CxxConvolver* cc = (CxxConvolver*)cc_;
    cc->getOutputFrame(data);
  }

  DLLEXPORT void setAllSwap(void* cc_) {
    CxxConvolver* cc = (CxxConvolver*)cc_;
    cc->setAllSwap();
  }

  DLLEXPORT void updateFilter(void* cc_, unsigned int in, unsigned int out,
                              float* filter, unsigned int filter_len) {
    CxxConvolver* cc = (CxxConvolver*)cc_;
    cc->updateFilter(in, out, filter, filter_len);
  }

  DLLEXPORT void execPartSwap(void* cc_, unsigned int in, unsigned int out,
                              float* filter, unsigned int filter_len) {
    CxxConvolver* cc = (CxxConvolver*)cc_;
    cc->execPartSwap(in, out, filter, filter_len);
  }

  DLLEXPORT int getNumSwapping(void* cc_) {
    CxxConvolver* cc = (CxxConvolver*)cc_;
    return cc->getNumSwapping();
  }
  DLLEXPORT int isSwapping(void* cc_, unsigned int in, unsigned int out) {
    CxxConvolver* cc = (CxxConvolver*)cc_;
    return (int)cc->isSwapping(in, out);
  }

}

void CxxConvolver::fft(float* input, fftwf_complex* output) {
  fftwf_execute_dft_r2c(this->fft_, input, output);
}

void CxxConvolver::ifft(fftwf_complex* input, float* output) {
  fftwf_execute_dft_c2r(this->ifft_, input, output);
}

void CxxConvolver::initializePartitioning(unsigned int num_inputs,
                                          unsigned int num_outputs,
                                          unsigned int filter_len,
                                          unsigned int buffer_len) {
  this->num_inputs_ = num_inputs;
  this->num_outputs_ = num_outputs;

  this->filter_len_ = filter_len;
  this->buffer_len_ = buffer_len;
  this->transform_len_ = buffer_len*2-1;
  this->len_scale_ = 1.0/this->transform_len_;
  this->complex_transform_len_ = this->transform_len_/2+1;
  this->num_filters_ = num_inputs*num_outputs;
  this->num_filter_parts_ = ceil(roundf(10.0*((float)filter_len/(float)buffer_len))/10.0f);

  std::cout<<"------------------"<<std::endl;
  std::cout<<"Init"<<std::endl;
  std::cout<<"Filter len: "<<filter_len<<std::endl;
  std::cout<<"Buffer len: "<<buffer_len<<std::endl;
  std::cout<<"Num filter parts: "<<this->num_filter_parts_<<std::endl;
  std::cout<<"------------------"<<std::endl;

  // Initialize filter partitioning for each input - output combination

  // Filter container for each input output contains
  // [transform_len x num_filter_parts] elements
  // Frequency domain container has only real specturm, therefore
  // size in complex numbers is [transformlen/2-1 x num_filter_parts]

  for(unsigned int i = 0; i < this->num_inputs_; i++) {
    time_domain_filters_.push_back(std::vector< std::vector<float> >());
    freq_domain_filters_.push_back(std::vector< fftwf_complex* >());
    time_domain_swap_filters_.push_back(std::vector< std::vector<float> >());
    freq_domain_swap_filters_.push_back(std::vector< fftwf_complex* >());

    for(unsigned int j = 0; j < this->num_outputs_; j++) {
      time_domain_filters_.at(i).push_back(std::vector<float>(getFilterContainerSizeT(), 0.f));
      time_domain_swap_filters_.at(i).push_back(std::vector<float>(getFilterContainerSizeT(), 0.f));
      fftwf_complex* f_filter = (fftwf_complex*)calloc(getFilterContainerSizeF(),
                                                       sizeof(fftwf_complex));

      this->freq_domain_filters_.at(i).push_back(f_filter);

      fftwf_complex* f_swap_filter = (fftwf_complex*)calloc(getFilterContainerSizeF(),
                                                            sizeof(fftwf_complex));
      this->freq_domain_swap_filters_.at(i).push_back(f_swap_filter);
    }
  }

  // Default plan pointers are the first filters in the containers
  // fftw_execute_dft_r2c( const fftw_plan p, double *in, fftw_complex *out)
  // is used to use the plan with different data
  this->fft_ = fftwf_plan_dft_r2c_1d(this->transform_len_,
                                     getTimeFilterAt(0,0),
                                     getFreqFilterAt(0,0),
                                     NULL);

  this->ifft_ = fftwf_plan_dft_c2r_1d(this->transform_len_,
                                      getFreqFilterAt(0, 0),
                                      getTimeFilterAt(0, 0),
                                      NULL);

  // Input buffer allocations, we need num_filter_parts*transform_len
  // containers for each input
  // Lenght of the complex container is transform_len/2+1 as we only save
  // half spectrum due to real to complex transform

  this->fdl_positions_.assign(num_inputs, 0);
  this->in_dl_positions_.assign(num_inputs, this->buffer_len_-1);

  // Filter swap controlled for each in-out combination separately
  for(int i = 0; i < this->num_inputs_; i++) {
    this->swap_part_positions_.push_back(std::vector< int >());
    this->swapping_.push_back(std::vector<bool>());
    this->swap_part_positions_.at(i).assign(num_outputs, 0);
    this->swapping_.at(i).assign(num_outputs, false);
  }

  for (unsigned int i = 0; i < this->num_inputs_; i++) {
    this->input_fdl_.push_back(std::vector< fftwf_complex* >());

    for (unsigned int j = 0; j < this->num_filter_parts_; j++) {
      fftwf_complex* fdl = (fftwf_complex*)calloc(this->transform_len_/2+1,
                                                  sizeof(fftwf_complex));

      this->input_fdl_.at(i).push_back(fdl);
    }
  }

  // Allocate buffers for the real signal values, at least for now
  for(int i = 0; i < this->num_inputs_; i++)  {
    in_dl_.push_back(std::vector<float>(this->transform_len_));
    in_buffers_padded_.push_back(std::vector<float>(this->transform_len_));
  }

  for (int i = 0; i < this->num_outputs_; i++) {
    out_buffers_padded_.push_back(std::vector<float>(this->transform_len_));
    fftwf_complex* n_buf = (fftwf_complex*)calloc(this->transform_len_/2+1,
                                                  sizeof(fftwf_complex));

    f_out_buffers_padded_.push_back(n_buf);
  }
}

void CxxConvolver::destroyPartitioning() {
  for(unsigned int i = 0; i < this->num_inputs_; i++) {
    for(unsigned int j = 0; j < this->num_outputs_; j++) {
      free(this->getFreqFilterAt(i,j));
      free(this->getFreqSwapFilterAt(i,j));
    }
  }

  for (unsigned int i = 0; i<this->num_inputs_; i++) {
    for (unsigned int j = 0; j<this->num_filter_parts_; j++) {
      free(this->getFdlAt(i, j));
    }
  }

  for(unsigned int i = 0; i < this->num_outputs_; i++)
    free(this->f_out_buffers_padded_.at(i));

  fftwf_destroy_plan(this->fft_);
  fftwf_destroy_plan(this->ifft_);
  fftwf_cleanup();
}

void CxxConvolver::updateFilter(unsigned int in, unsigned int out,
                                std::vector<float>& filter) {
  this->updateFilter(in, out, &(filter[0]), filter.size());
}

void CxxConvolver::updateFilter(unsigned int in, unsigned int out,
                                float* filter, unsigned int filter_len) {
  float* cur_filter = getTimeFilterAt(in, out);
  fftwf_complex* cur_ffilter = getFreqFilterAt(in, out);

  memset(cur_filter, 0, sizeof(float)*this->getFilterContainerSizeT());

  // If filter is too short, pad the end, if its too long, truncate
  unsigned int num_parts = filter_len/this->getBufferLen();
  unsigned int reminder = filter_len - num_parts*this->getBufferLen();

  // Stride for partitioned filter is transform length, and for the input
  // filter buffer_len. Padding therefore is trasnform_len-buffer_len
  for(unsigned int i = 0; i < this->num_filter_parts_; i++) {
    // Copy all the full parts, and finally the reminder
    if(i<num_parts) {
      memcpy(cur_filter+i*this->transform_len_,
             &filter[i*this->getBufferLen()],
             this->getBufferLen()*sizeof(float));


    } else {
      memcpy(cur_filter+i*this->transform_len_,
             &filter[i*this->getBufferLen()],
             reminder*sizeof(float));
    }
    this->fft(cur_filter+i*this->transform_len_,
              cur_ffilter+i*this->complex_transform_len_);
  }
}

void CxxConvolver::swapPart(unsigned int in, unsigned int out,
                            float* filter, unsigned int filter_len,
                            unsigned int part) {
  float* cur_filter = getTimeSwapFilterAt(in, out);
  fftwf_complex* cur_ffilter = getFreqSwapFilterAt(in, out);
  fftwf_complex* dest_ffilter = getFreqFilterAt(in, out);
  memset(cur_filter+part*this->transform_len_, 0,
         sizeof(float)*this->transform_len_);

  // If filter is too short, pad the end, if its too long, truncate
  unsigned int num_parts = filter_len/this->getBufferLen();
  unsigned int reminder = filter_len - num_parts*this->getBufferLen();

  if(part<num_parts) {
    memcpy(cur_filter+part*this->transform_len_,
           &filter[part*this->getBufferLen()],
           this->getBufferLen()*sizeof(float));
  } else {
    memcpy(cur_filter+part*this->transform_len_,
           &filter[part*this->getBufferLen()],
           reminder*sizeof(float));
  }
  this->fft(cur_filter+part*this->transform_len_,
            cur_ffilter+part*this->complex_transform_len_);

  memcpy(dest_ffilter+part*this->complex_transform_len_,
         cur_ffilter+part*this->complex_transform_len_,
         sizeof(fftwf_complex)*this->complex_transform_len_);
}

void CxxConvolver::execPartSwap(unsigned int in, unsigned int out,
                                float* filter, unsigned int filter_len) {
  if(!this->swapping_[in][out])
    return;

  // Get current part and swap filter part for given input
  int part = this->swap_part_positions_[in][out];
  this->swapPart(in, out, filter, filter_len, part);

  // Increment swap position counter, currently one part per swap
  this->swap_part_positions_[in][out]++;

  // Whole filter swapped, switch off, reset position counter
  if(this->swap_part_positions_[in][out]>=this->num_filter_parts_) {
    this->swapping_[in][out]=false;
    this->swap_part_positions_[in][out] = 0;
  }
}

void CxxConvolver::addNewBuffer(unsigned int input, float* data) {
  // Increment the filter position and take modulo in the begining so
  // we point at the current FDL in the end
  this->fdl_positions_.at(input) = (this->fdl_positions_.at(input)+1)%
                                    this->num_filter_parts_;

  // Copy time-domain data to structure, TRANSFROM LEN, need padding in the
  // beginning of the singlal (transfer_len/2-1)
  memcpy((void*)this->getInBuffers(input),
         data,
         this->transform_len_*sizeof(float));

  // Grab the fdl position and fft the input to fdl
  fftwf_complex* cur = this->getFdlAt(input, this->getFdlPosition(input));
  this->fft(this->getInBuffers(input), cur);
}

void CxxConvolver::addInputFrame(const float* data) {
  // data is now from portaudio/audio stream, deinterleave directly to
  // input buffers and then fft to FDL
  for(int i = 0; i < this->num_inputs_; i++) {
    for(int j = 0; j < this->transform_len_; j++) {
      int dl_pos = (this->in_dl_positions_.at(i)+j)%this->transform_len_;
      int buf_pos = MOD(dl_pos-(int)this->buffer_len_+1, (int)this->transform_len_);

      if(j < this->buffer_len_)
        this->in_dl_.at(i).at(dl_pos) = data[i+j*this->num_inputs_];

      this->getInBuffers(i)[j] = this->in_dl_.at(i).at(buf_pos);
    }
  }

  for(int i = 0; i < this->num_inputs_; i++) {
    this->in_dl_positions_.at(i) = (this->in_dl_positions_.at(i)+this->buffer_len_)%
                                    this->transform_len_;

    this->fdl_positions_.at(i) = (this->fdl_positions_.at(i)+1)%
                                  this->num_filter_parts_;

    fftwf_complex* cur = this->getFdlAt(i, this->getFdlPosition(i));
    this->fft(this->getInBuffers(i), cur);
  }
}

void CxxConvolver::getOutputFrame(float* data) {
  for(int j = 0; j < this->buffer_len_; j++) {
    for(int i = 0; i < this->num_outputs_; i++) {
      // Read from the outputbuffer location buffer_len_-1
      data[i+j*this->num_outputs_] = this->getOutputAt(i)[j+this->buffer_len_-1]*this->len_scale_;
    }
  }
}

void CxxConvolver::processInputOutput(unsigned int input, unsigned int output) {
  //Grab the output buffer of the argument output
  float* cur_output = getOutputAt(output);
  // Get the freqyency domain output buffer for the multiply and accumulate
  fftwf_complex* cur_foutput = getFOutputAt(output);

  for(int i = 0; i < this->num_filter_parts_; i++) {
    // Get the filter corresponding to input, output and filter part
    fftwf_complex* freq_filt = this->getFreqFilterAt(input, output, i);

    // Get the FDL corresponding to the input and filter part, -1 due to the
    // initial increment in CxxConvolver::addNewBuffer with the first buffer
    fftwf_complex* cur_input = this->getFdlAt(input, this->getFdlPosition(input)-i);

    // Multiply and accumulate each bin of current FDL and filter part to
    // output buffer

    // TODO: parallelize
    for(int j = 0; j < this->complex_transform_len_; j++)
      complexMultiplyAdd(&cur_input[j], &freq_filt[j], &cur_foutput[j]);
  }

  // ifft to current time-domain output
  this->ifft(cur_foutput, cur_output);
  // zero out the accumulated frequency domain output buffer
  memset(cur_foutput, 0, this->complex_transform_len_*sizeof(fftwf_complex));
}

void CxxConvolver::processOutput(unsigned int output) {
  //Grab the output buffer of the argument output
  float* cur_output = getOutputAt(output);
  // Get the freqyency domain output buffer for the multiply and accumulate
  fftwf_complex* cur_foutput = getFOutputAt(output);

  for (int input = 0; input < this->num_inputs_; input++) {
    for(int i = 0; i < this->num_filter_parts_; i++) {
      // Get the filter corresponding to input, output and filter part
      fftwf_complex* freq_filt = this->getFreqFilterAt(input, output, i);

      // Get the FDL corresponding to the input and filter part, -1 due to the
      // initial increment in CxxConvolver::addNewBuffer with the first buffer
      fftwf_complex* cur_input = this->getFdlAt(input, this->getFdlPosition(input)-i);

      // Multiply and accumulate each bin of current FDL and filter part to
      // output buffer
      // TODO: parallelize
      for(int j = 0; j < this->complex_transform_len_; j++)
        complexMultiplyAdd(&cur_input[j], &freq_filt[j], &cur_foutput[j]);
    }
  }
  // ifft to current time-domain output
  this->ifft(cur_foutput, cur_output);
  // zero out the accumulated frequency domain output buffer
  memset(cur_foutput, 0, this->complex_transform_len_*sizeof(fftwf_complex));
}
