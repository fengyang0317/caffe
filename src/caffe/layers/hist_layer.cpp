#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/hist_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void HistLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  HistParameter hist_param = this->layer_param_.hist_param();
  if (hist_param.global_pooling()) {
    CHECK(!(hist_param.has_kernel_size() ||
      hist_param.has_kernel_h() || hist_param.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
  } else {
    CHECK(!hist_param.has_kernel_size() !=
      !(hist_param.has_kernel_h() && hist_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK(hist_param.has_kernel_size() ||
      (hist_param.has_kernel_h() && hist_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  }
  CHECK((!hist_param.has_pad() && hist_param.has_pad_h()
      && hist_param.has_pad_w())
      || (!hist_param.has_pad_h() && !hist_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!hist_param.has_stride() && hist_param.has_stride_h()
      && hist_param.has_stride_w())
      || (!hist_param.has_stride_h() && !hist_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  global_pooling_ = hist_param.global_pooling();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  } else {
    if (hist_param.has_kernel_size()) {
      kernel_h_ = kernel_w_ = hist_param.kernel_size();
    } else {
      kernel_h_ = hist_param.kernel_h();
      kernel_w_ = hist_param.kernel_w();
    }
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!hist_param.has_pad_h()) {
    pad_h_ = pad_w_ = hist_param.pad();
  } else {
    pad_h_ = hist_param.pad_h();
    pad_w_ = hist_param.pad_w();
  }
  if (!hist_param.has_stride_h()) {
    stride_h_ = stride_w_ = hist_param.stride();
  } else {
    stride_h_ = hist_param.stride_h();
    stride_w_ = hist_param.stride_w();
  }
  if (global_pooling_) {
    CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
      << "With Global_pooling: true; only pad = 0 and stride = 1";
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }
  CHECK_GT(hist_param.center_size(), 0);
  for (int c = 0; c < hist_param.center_size(); ++c) {
    center_.push_back(hist_param.center(c));
  }
  sigma_ = hist_param.sigma();
}

template <typename Dtype>
void HistLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels() * center_.size();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  }
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }
  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
}

template <typename Dtype>
void HistLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  Dtype dis = 0;
  for (int i = 0; i < top_count; ++i) {
    top_data[i] = 0;
  }
  // The main loop
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          int hstart = ph * stride_h_ - pad_h_;
          int wstart = pw * stride_w_ - pad_w_;
          int hend = min(hstart + kernel_h_, height_ + pad_h_);
          int wend = min(wstart + kernel_w_, width_ + pad_w_);
          int pool_size = (hend - hstart) * (wend - wstart);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          hend = min(hend, height_);
          wend = min(wend, width_);
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              dis = bottom_data[h * width_ + w] - center_[c % center_.size()];
              top_data[ph * pooled_width_ + pw] +=
                  exp(-dis * dis / sigma_);
            }
          }
          top_data[ph * pooled_width_ + pw] /= pool_size;
        }
      }
      // compute offset
      if ((c + 1) % center_.size() == 0) {
        bottom_data += bottom[0]->offset(0, 1);
      }
      top_data += top[0]->offset(0, 1);
    }
  }
}

template <typename Dtype>
void HistLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          int hstart = ph * stride_h_ - pad_h_;
          int wstart = pw * stride_w_ - pad_w_;
          int hend = min(hstart + kernel_h_, height_ + pad_h_);
          int wend = min(wstart + kernel_w_, width_ + pad_w_);
          int pool_size = (hend - hstart) * (wend - wstart);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          hend = min(hend, height_);
          wend = min(wend, width_);
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              bottom_diff[h * width_ + w] +=
                2 * top_diff[ph * pooled_width_ + pw] *
                (bottom_data[h * width_ + w] - center_[c % center_.size()]) /
                sigma_ / pool_size;
            }
          }
        }
      }
      // offset
      if ((c + 1) % center_.size() == 0) {
        bottom_diff += bottom[0]->offset(0, 1);
      }
      top_diff += top[0]->offset(0, 1);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(HistLayer);
#endif

INSTANTIATE_CLASS(HistLayer);
REGISTER_LAYER_CLASS(Hist);

}  // namespace caffe
