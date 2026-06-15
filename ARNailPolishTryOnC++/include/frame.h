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
	bool getFrame(cv::Mat& outputFrame);
	void stopCapture();
private:
	cv::VideoCapture cap;
	cv::Mat frame;
	std::thread workerThread;
	std::atomic<bool> isRunning;
	std::mutex frameMutex;
};

class FrameQueue
{
public:
	void push(cv::Mat& frame);
	bool pop(cv::Mat& frame);
private:
	std::queue<cv::Mat> queue;
	std::mutex mtx;
	std::condition_variable cv;
};
