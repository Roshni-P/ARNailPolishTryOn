#pragma once
#include "common.h"

class HandLandmarks;

class IHandDetector
{
public:
	virtual HandLandmarks detectHands(const cv::Mat& frame) = 0;
};