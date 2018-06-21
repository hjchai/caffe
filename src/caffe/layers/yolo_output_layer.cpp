#include "caffe/layers/yolo_output_layer.hpp"

namespace caffe {

template <typename Dtype>
void YoloOutputLayer<Dtype>::LayerSetup(const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top) {
    const YoloOutputParameter& yolo_output_param = this->layer_param_.yolo_output_param();
    CHECK(yolo_output_param.has_num_classes()) << "Must specify num_classes";
}

}
