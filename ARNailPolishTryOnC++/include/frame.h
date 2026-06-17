#pragma once
/// @file frame.h
/// @brief Capture video on live camera
/// @namespace cv

#include "common.h"

/**
 * @class FrameQueue
 * @brief Captures live camera feed with the latest 
 * frame, at the top of queue
 */

class FrameQueue
{
public:
	FrameQueue();
	~FrameQueue();
	bool getLatestFrame(cv::Mat& frame);
	int captureFrame();
	void stopCapture();
private:
	cv::VideoCapture cap;
	cv::Mat frame;
	std::thread workerThread;
	std::atomic<bool> isRunning;
	std::queue<cv::Mat> queue;
	std::mutex mtx;
	std::condition_variable cv;
};
