#ifndef CAFFE_POSE_DATA_LAYER_HPP_
#define CAFFE_POSE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class PoseDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit PoseDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~PoseDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  vector<std::pair<std::string, int> > lines_;
  int lines_id_;

  map<string, cv::VideoCapture> mpv_;
  map<string, MATFile*> mpf_;
  cv::Mat ReadFrameToCVMat(const string& root, const string& video, const int frame_no);
  double Uniform(const double min, const double max);
};


}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
