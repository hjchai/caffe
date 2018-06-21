#ifndef CAFFE_YOLO_OUTPUT_LAYER_HPP_
#define CAFFE_YOLO_OUTPUT_LAYER_HPP_

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class YoloOutputLayer : public Layer<Dtype> {
public:
    explicit YoloOutputLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetup(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top);
protected:
private:
};

} // namespace caffe

#endif // CAFFE_YOLO_OUTPUT_LAYER_HPP_
