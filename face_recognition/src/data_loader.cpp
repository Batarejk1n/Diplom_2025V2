#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class FaceDataset : public torch::data::Dataset<FaceDataset> {
public:
    explicit FaceDataset(const std::vector<std::string>& paths, const std::vector<int>& labels);
    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;

private:
    std::vector<std::string> paths_;
    std::vector<int> labels_;
    cv::CascadeClassifier face_cascade;
};

