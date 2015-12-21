#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/pose_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include "mat.h"

namespace caffe {

template <typename Dtype>
PoseDataLayer<Dtype>::~PoseDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
cv::Mat PoseDataLayer<Dtype>::ReadFrameToCVMat(const string& root, const string& video, const int frame_no) {
  if (mpv_.find(video) == mpv_.end()) {
	MATFile *f = matOpen((root + video).c_str());
	mxArray *ps = matGetVariable(f, "Name");
	char buf[255];
	mxGetString(ps, buf, 255);
	mpv_[video] = cv::VideoCapture(root + string(buf));
	mpf_[video] = f;
  }
  VideoCapture &vid = mpv_[video];
  vid.set(CV_CAP_PROP_POS_FRAMES, frame_no);
  cv::Mat frame;
  vid.read(frame);
  return frame;
}

template <typename Dtype>
double PoseDataLayer<Dtype>::Uniform(const double min, const double max) {
    double random = ((double) rand()) / (double) RAND_MAX;
    double diff = max - min;
    double r = random * diff;
    return min + r;
}

template <typename Dtype>
void PoseDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.pose_data_param().new_height();
  const int new_width  = this->layer_param_.pose_data_param().new_width();
  const bool is_color  = this->layer_param_.pose_data_param().is_color();
  string root_folder = this->layer_param_.pose_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.pose_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int frame_no;
  while (infile >> filename >> frame_no) {
    lines_.push_back(std::make_pair(filename, frame_no));
  }

  if (this->layer_param_.pose_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.pose_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.pose_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadFrameToCVMat(root_folder, lines_[lines_id_].first,
                                    lines_[lines_id_].second);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.pose_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(4, batch_size);
  label_shape[1] = 14;
  label_shape[2] = 32;
  label_shape[3] = 32;
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void PoseDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void PoseDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  PoseDataParameter pose_data_param = this->layer_param_.pose_data_param();
  const int batch_size = pose_data_param.batch_size();
  const int new_height = pose_data_param.new_height();
  const int new_width = pose_data_param.new_width();
  const bool is_color = pose_data_param.is_color();
  string root_folder = pose_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadFrameToCVMat(root_folder + lines_[lines_id_].first,
          lines_[lines_id_].second);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadFrameToCVMat(root_folder, lines_[lines_id_].first,
            lines_[lines_id_].second);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first << lines_[lines_id_].second;
    MATFile *f = mpf_[lines_[lines_id_].first];
    mxArray *pb = matGetVariable(f, "rect");
    int M = mxGetM(pb);
    double *a = (double*) mxGetData(pb);
    vector<double> rec[4];
    for (int i = 0; i < 4; i++) {
    	rec[i] = a[i * M + lines_[lines_id_].second];
    }
    if (rec[2] < rec[3]) {
    	rec[0] = rec[0] - (rec[3] - rec[2]) / 2;
    	rec[2] = rec[3];
    	rec[0] += Uniform(-rec[2] / 5, rec[2] / 5);
    	if (rec[0] < 0) {
    		rec[0] = 0;
    	}
    	if (rec[0] + rec[2] > cv_img.get(cv::CV_CAP_PROP_FRAME_WIDTH)) {
    		rec[0] = v_img.get(cv::CV_CAP_PROP_FRAME_WIDTH) - rec[2];
    	}
    }
    else {
    	rec[1] = rec[1] - (rec[2] - rec[3]) / 2;
    	rec[3] = rec[2];
    	rec[1] += Uniform(-rec[2] / 5, rec[2] / 5);
    	if (rec[1] < 0) {
    		rec[1] = 0;
    	}
    	if (rec[1] + rec[3] > cv_img.get(cv::CV_CAP_PROP_FRAME_HEIGHT)) {
    		rec[1] = v_img.get(cv::CV_CAP_PROP_FRAME_HEIGHT) - rec[3];
    	}
    }
    cv_img = cv_img(cv::Rect(rec[1], rec[0], rec[1] + rec[3], rec[0] + rec[2]));
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    mxArray *pp = matGetVariable(f, "vals");
    offset = batch->label_.offset(item_id);
    prefetch_label[item_id] = lines_[lines_id_].second;

    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.pose_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(PoseDataLayer);
REGISTER_LAYER_CLASS(PoseData);

}  // namespace caffe
#endif  // USE_OPENCV
