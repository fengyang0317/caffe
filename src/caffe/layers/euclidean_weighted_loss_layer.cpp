#include <vector>

#include "caffe/layers/euclidean_weighted_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

namespace caffe {

template <typename Dtype>
void EuclideanWeightedLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanWeightedLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;

  bool visualise = this->layer_param_.visualise();
  if (visualise)
  {
      cv::namedWindow("pre", CV_WINDOW_AUTOSIZE);
      cv::namedWindow("gt", CV_WINDOW_AUTOSIZE);
      cv::namedWindow("img", CV_WINDOW_AUTOSIZE);
      cv::namedWindow("img_fuse", CV_WINDOW_AUTOSIZE);
      const int num_images = bottom[1]->num();
      const int label_height = bottom[1]->height();
      const int label_width = bottom[1]->width();
      const int label_channels = bottom[0]->channels();
      const int img_height = bottom[2]->height();
      const int img_width = bottom[2]->width();
      const int img_channels = bottom[2]->channels();
      cv::Mat pre_map, gt_map, img, img_f;
      if (img_channels == 3)
    	  img = cv::Mat::zeros(img_height, img_width, CV_32FC3);
      else
    	  img = cv::Mat::zeros(img_height, img_width, CV_32FC1);
      int image_idx;
      const int img_channel_size = img_height * img_width;
      const int img_img_size = img_channel_size * img_channels;
      const int label_channel_size = label_height * label_width;
      const int label_img_size = label_channel_size * label_channels;
      const Dtype *b2 = bottom[2]->cpu_data();
      const Dtype *b1 = bottom[1]->cpu_data();
      const Dtype *b0 = bottom[0]->cpu_data();
      //LOG(INFO) << "image num channels " << img_channels;

      Dtype mean_val[3] = {84.6888, 91.7211, 140.382};

      for (int n = 0; n < num_images; n++)
      {
          pre_map = cv::Mat::ones(label_height, (label_width + 1) * label_channels, CV_32FC1);
          gt_map = cv::Mat::ones(label_height, (label_width + 1) * label_channels, CV_32FC1);
          for (int c = 0; c < img_channels; c++)
          {
              for(int i = 0; i < img_height; i++)
              {
                  for(int j = 0; j < img_width; j++)
                  {
                      image_idx = n * img_img_size + c * img_channel_size + i * img_height + j;
                      if (img_channels == 3)
                    	  img.at<cv::Vec3f>(i, j)[c] = (float) (b2[image_idx] + mean_val[c]) / 255;
                      else
                    	  img.at<float>(i, j) = (float) (b2[image_idx] + mean_val[c]) / 255;
                  }
              }
          }
          if (img_channels == 3)
        	  cv::cvtColor(img, img_f, CV_BGR2GRAY);
          else
        	  img.copyTo(img_f);
          for (int c = 0; c < label_channels; c++)
          {
              for(int i = 0; i < label_height; i++)
              {
                  for(int j = 0; j < label_width; j++)
                  {
                      image_idx = n * label_img_size + c * label_channel_size + i * label_height + j;
                      pre_map.at<float>(i, j + c * (label_width + 1)) = (float) b0[image_idx];
                      gt_map.at<float>(i, j + c * (label_width + 1)) = (float) b1[image_idx];
                  }
              }
              cv::Mat vm = gt_map(cv::Rect(c * (label_width + 1), 0, label_width, label_height));
              cv::resize(vm, vm, cv::Size(img_height, img_width));
              //LOG(INFO) << "fuse channels " << img_f.channels();
              img_f += vm;
          }
          //int vc = this->layer_param_.visualise_channel();
          cv::imshow("img",img);
          cv::imshow("img_fuse", img_f);
		  cv::resize(gt_map, gt_map, cv::Size(0, 0), 4, 4);
          cv::imshow("gt",gt_map);
		  cv::resize(pre_map, pre_map, cv::Size(0, 0), 4, 4);
          cv::imshow("pre",pre_map);
          cv::waitKey(0);
      }
  }
}

template <typename Dtype>
void EuclideanWeightedLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanWeightedLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanWeightedLossLayer);
REGISTER_LAYER_CLASS(EuclideanWeightedLoss);

}  // namespace caffe
