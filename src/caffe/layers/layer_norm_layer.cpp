#include <algorithm>
#include <vector>

#include "caffe/layers/layer_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LayerNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LayerNormParameter param = this->layer_param_.layer_norm_param();
  eps_ = param.eps();
}

template <typename Dtype>
void LayerNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);

  vector<int> sz;
  sz.push_back(bottom[0]->shape(0));
  mean_.Reshape(sz);
  variance_.Reshape(sz);
  temp_.ReshapeLike(*bottom[0]);
  x_norm_.ReshapeLike(*bottom[0]);

  int spatial_dim = bottom[0]->count()/bottom[0]->shape(0);
  if (spatial_sum_multiplier_.num_axes() == 0 ||
      spatial_sum_multiplier_.shape(0) != spatial_dim) {
    sz[0] = spatial_dim;
    spatial_sum_multiplier_.Reshape(sz);
    Dtype* multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
    caffe_set(spatial_sum_multiplier_.count(), Dtype(1), multiplier_data);
  }
}

template <typename Dtype>
void LayerNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/bottom[0]->shape(0);

  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }

  // compute mean
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, spatial_dim,
      1. / spatial_dim, bottom_data,
      spatial_sum_multiplier_.cpu_data(), 0.,
      mean_.mutable_cpu_data());

  // subtract mean
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num,
      spatial_dim, 1, -1, mean_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 1., top_data);

  // compute variance using var(X) = E((X-EX)^2)
  caffe_powx(top[0]->count(), top_data, Dtype(2),
      temp_.mutable_cpu_data());  // (X-EX)^2
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, spatial_dim,
      1. / spatial_dim, temp_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 0.,
      variance_.mutable_cpu_data());  // E((X_EX)^2)

  // normalize variance
  caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
  caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
             variance_.mutable_cpu_data());

  // replicate variance to input size
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num,
      spatial_dim, 1, 1., variance_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
  caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
  // TODO(cdoersch): The caching is only needed because later in-place layers
  //                 might clobber the data.  Can we skip this if they won't?
  caffe_copy(x_norm_.count(), top_data,
      x_norm_.mutable_cpu_data());
}

template <typename Dtype>
void LayerNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff;
  if (bottom[0] != top[0]) {
    top_diff = top[0]->cpu_diff();
  } else {
    caffe_copy(x_norm_.count(), top[0]->cpu_diff(), x_norm_.mutable_cpu_diff());
    top_diff = x_norm_.cpu_diff();
  }
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* top_data = x_norm_.cpu_data();
  int num = bottom[0]->shape()[0];
  int spatial_dim = bottom[0]->count()/bottom[0]->shape(0);
  // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
  //
  // dE(Y)/dX =
  //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
  //     ./ sqrt(var(X) + eps)
  //
  // where \cdot and ./ are hadamard product and elementwise division,
  // respectively, dE/dY is the top diff, and mean/var/sum are all computed
  // along all dimensions except the channels dimension.  In the above
  // equation, the operations allow for expansion (i.e. broadcast) along all
  // dimensions except the channels dimension where required.

  // sum(dE/dY \cdot Y)
  caffe_mul(temp_.count(), top_data, top_diff, bottom_diff);
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, spatial_dim, 1.,
      bottom_diff, spatial_sum_multiplier_.cpu_data(), 0.,
      mean_.mutable_cpu_data());

  // reshape (broadcast) the above
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num,
      spatial_dim, 1, 1., mean_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 0., bottom_diff);

  // sum(dE/dY \cdot Y) \cdot Y
  caffe_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, spatial_dim, 1.,
      top_diff, spatial_sum_multiplier_.cpu_data(), 0.,
      mean_.mutable_cpu_data());
  // reshape (broadcast) the above to make
  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num,
      spatial_dim, 1, 1., mean_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 1., bottom_diff);

  // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  caffe_cpu_axpby(temp_.count(), Dtype(1), top_diff,
      Dtype(-1. / spatial_dim), bottom_diff);

  // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
  // pass.
  caffe_div(temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(LayerNormLayer);
#endif

INSTANTIATE_CLASS(LayerNormLayer);
REGISTER_LAYER_CLASS(LayerNorm);
}  // namespace caffe
