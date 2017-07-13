#ifndef CAFFE_COMPLEX_IMAGE_DATA_LAYER_HPP_
#define CAFFE_COMPLEX_IMAGE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>
//#include "caffe/util/image_cropping_utils.hpp"
#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/image_cropping_utils.hpp"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ComplexImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ComplexImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ComplexImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ComplexImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  vector<std::pair<std::string, int> > lines_;
  //cv::Mat m_labels;
  //vector<std::pair<int, int>> lines_;
  //cv::Mat m_fullImage;

  int lines_id_;
  cv::Mat m_fullImage;
  int numOfPoints;
};


}  // namespace caffe

#endif  // CAFFE_COMPLEX_IMAGE_DATA_LAYER_HPP_
