#pragma once
#include "common.h"

class HandLandmarks
{
public: HandLandmarks();
	  ~HandLandmarks();
	  int detectHandLandmarks();
private:
	  cv::Mat preprocessImage(const cv::Mat& image);
	  std::vector<float> HandLandmarks::blobFromImage(const cv::Mat& image);
};

