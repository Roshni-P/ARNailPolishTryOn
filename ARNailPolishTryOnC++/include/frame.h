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
	cv::Mat getFrame() const { return frame; }
private:
	cv::Mat frame;
};
