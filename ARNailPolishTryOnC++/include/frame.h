#pragma once
/// @file frame.h
/// @brief Capture video on live camera
/// @namespace cv

#include "common.h"

/**
 * @class Frame
 * @brief Captures live camera feed
 */
class Frame
{
public:
	Frame();
	~Frame();
	int captureFrame();
private:
	cv::Mat frame;
};
