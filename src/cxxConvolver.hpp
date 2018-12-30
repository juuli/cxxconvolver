#ifndef CXX_CONVOLVER_HPP
#define CXX_CONVOLVER_HPP

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

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <numeric>
#include <fftw3.h>
#include "json.hpp"

#ifdef _WIN32
  #define DLLEXPORT extern "C" __declspec( dllexport )
#else
  #define DLLEXPORT
#endif

#define MOD(a,b) ((((a)%(b))+(b))%(b))

using json = nlohmann::json;

typedef std::vector< std::vector< std::vector<float> > > FilterArray;
typedef std::vector< std::vector< fftwf_complex* > > FFilterArray;

inline void complexMultiplyAdd(const fftwf_complex* a, const fftwf_complex* b,
                               fftwf_complex* output) {
  output[0][0] += a[0][0]*b[0][0]-a[0][1]*b[0][1];
  output[0][1] += a[0][0]*b[0][1]+a[0][1]*b[0][0];
}

// C interface
extern "C" {
  DLLEXPORT void* initialize(unsigned int num_inputs,
                             unsigned int num_outputs,
                             unsigned int filter_len,
                             unsigned int buffer_len);
                             
  DLLEXPORT void destroy(void* cc_);
  DLLEXPORT void addInputFrame(void* cc_, const float* data);
  DLLEXPORT void processOutput(void* cc_, unsigned int output);
  DLLEXPORT void getOutputFrame(void* cc_, float* data);
  DLLEXPORT void setAllSwap(void* cc_);
  DLLEXPORT int getNumSwapping(void* cc_);
  DLLEXPORT int isSwapping(void* cc_, unsigned int in, unsigned int out);

  DLLEXPORT void updateFilter(void* cc_, unsigned int in, unsigned int out,
                              float* filter, unsigned int filter_len);

  DLLEXPORT void execPartSwap(void* cc_, unsigned int in, unsigned int out,
                              float* filter, unsigned int filter_len);

}

class CxxConvolver {
  public:
    CxxConvolver()
    : num_outputs_(0),
      num_inputs_(0),
      filter_len_(0),
      num_filters_(0),
      num_filter_parts_(0),
      buffer_len_(0),
      transform_len_(0)
    {}

    ~CxxConvolver() {}

  private:

  // Plans and variables for filter partitioning
  FilterArray time_domain_filters_;
  FFilterArray freq_domain_filters_;

  // A structure for filter swap in parts
  FilterArray time_domain_swap_filters_;
  FFilterArray freq_domain_swap_filters_;

  // Swapped filter part position for each output
  std::vector< std::vector< int > > swap_part_positions_;
  std::vector< std::vector< bool > > swapping_;

  fftwf_plan fft_;
  fftwf_plan ifft_;

  unsigned int num_outputs_;
  unsigned int num_inputs_;
  unsigned int filter_len_;
  unsigned int num_filters_;
  unsigned int buffer_len_;
  unsigned int transform_len_;
  unsigned int complex_transform_len_;
  int num_filter_parts_;
  float len_scale_;

  std::vector< std::vector<float> > in_dl_;
  std::vector< std::vector<float> > in_buffers_padded_;
  std::vector< std::vector<float> > out_buffers_padded_;
  std::vector< fftwf_complex* > f_out_buffers_padded_;

  FFilterArray input_fdl_;
  std::vector<int> fdl_positions_;
  std::vector<int> in_dl_positions_;

  public:

  unsigned int getNumOutputs() const {return this->num_outputs_;}
  unsigned int getNumInputs() const {return this->num_inputs_;}
  unsigned int getFilterLen() const {return this->filter_len_;}
  unsigned int getBufferLen() const {return this->buffer_len_;}
  unsigned int getTransformLen() const {return this->transform_len_;}
  unsigned int getComplexTransformLen() const {return this->complex_transform_len_;}
  unsigned int getNumFilters() const {return this->getNumInputs()*this->getNumOutputs();}
  unsigned int getNumFilterParts() const {return this->num_filter_parts_;}

  bool isSwapping(unsigned int in, unsigned int out) const {
    return this->swapping_[in][out];
  }

  int getNumSwapping() const {
    int ret = 0;
    for(int i = 0; i < this->num_inputs_; i++)
      ret+=getNumSwappingAt(i);
    return ret;
  }

  int getNumSwappingAt(unsigned int in) const {
    return std::accumulate(this->swapping_[in].begin(),
                           this->swapping_[in].end(), 0);
  }

  int getSwapPartPosition(unsigned int in, unsigned int out) const {
    return this->swap_part_positions_[in][out];
  }

  void setSwap(unsigned int in, unsigned int out, bool val = true) {
    this->swapping_[in][out] = val;
  }

  void setAllSwap(bool val=true) {
    for(int i = 0; i < this->num_inputs_; i++)
      this->swapping_[i].assign(this->num_outputs_, true);
  }

  // Return int, modulo between int and unsigned int goes apeshit
  int getFdlPosition(unsigned int input) const
    { return (int)this->fdl_positions_.at(input);}

  float* getInBuffers(unsigned int input)
    { return &(in_buffers_padded_.at(input)[0]); }

  // Get size of the partitioned and padded filter in time domain
  unsigned int getFilterContainerSizeT() const {
    return this->num_filter_parts_*this->transform_len_;
  }

  // Get size for the partitioned filter in frequency domain (half spectrum)
  unsigned int getFilterContainerSizeF() const {
    return this->num_filter_parts_*(this->transform_len_/2+1);
  }

  float* getTimeFilterAt(unsigned int in, unsigned int out) {
    return &(this->time_domain_filters_.at(in).at(out)[0]);
  }

  float* getTimeSwapFilterAt(unsigned int in, unsigned int out) {
    return &(this->time_domain_swap_filters_.at(in).at(out)[0]);
  }

  fftwf_complex* getFreqFilterAt(unsigned int in, unsigned int out,
                                 unsigned int part = 0) {
    unsigned int idx = part*this->complex_transform_len_;
    return (this->freq_domain_filters_.at(in).at(out)+idx);
  }

  fftwf_complex* getFreqSwapFilterAt(unsigned int in, unsigned int out,
                                     unsigned int part = 0) {
    unsigned int idx = part*this->complex_transform_len_;
    return (this->freq_domain_swap_filters_.at(in).at(out)+idx);
  }

  fftwf_complex* getFdlAt(unsigned int in, int part) {
    return this->input_fdl_.at(in).at(MOD(part, this->num_filter_parts_));
  }

  float* getOutputAt(unsigned int out) {
    return &(this->out_buffers_padded_.at(out)[0]);
  }

  fftwf_complex* getFOutputAt(unsigned int out) {
    return this->f_out_buffers_padded_.at(out);
  }

  void initializePartitioning(unsigned int num_inputs,
                              unsigned int num_outputs,
                              unsigned int filter_len,
                              unsigned int buffer_len);

  void destroyPartitioning();

  void updateFilter(unsigned int in, unsigned int out,
                    std::vector<float>& filter);

  void updateFilter(unsigned int in, unsigned int out,
                    float* filter, unsigned int filter_len);

  void swapPart(unsigned int in, unsigned int out,
                float* filter, unsigned int filter_len,
                unsigned int part);

  void execPartSwap(unsigned int in, unsigned int out,
                    float* filter, unsigned int filter_len);

  void addNewBuffer(unsigned int input, float* data);

  void addInputFrame(const float* data);

  void processInputOutput(unsigned int input, unsigned int output);

  void processOutput(unsigned int output);

  void getOutputFrame(float* data);

  void fft(float* input, fftwf_complex* output);

  void ifft(fftwf_complex* input, float* output);

};


#endif
