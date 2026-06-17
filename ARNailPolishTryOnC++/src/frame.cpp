#include "frame.h"
#include <iostream>

FrameQueue::FrameQueue() :isRunning(false)
{
	cap.open(0, cv::CAP_DSHOW);
	if (!cap.isOpened())
	{
		std::cerr << "Error: Unable to open camera." << std::endl;
		return;
	}
	
	isRunning = true;
	workerThread = std::thread(&FrameQueue::captureFrame, this);
}

FrameQueue::~FrameQueue()
{
	stopCapture();
}

void FrameQueue::stopCapture()
{
	if (isRunning)
	{
		isRunning = false;
		if (workerThread.joinable())
		{
			workerThread.join();
		}
		
		cap.release();
		std::cout << "Stopping video capture..." << std::endl;
	}
}

/**
 * @brief Captures frames from Live camera
 * @param
 * @return error code - 0 if no errors
 */
int FrameQueue::captureFrame()
{
	cv::Mat tempFrame;

	while (isRunning)
	{
		cap >> tempFrame;
		if (tempFrame.empty()) continue;

		{
			std::lock_guard<std::mutex> lock(mtx);
			frame = tempFrame.clone();
		}

		//std::this_thread::sleep_for(std::chrono::milliseconds(5));

		if (cv::waitKey(1) == 27) // ESC key, to break
			break;
	}

	return 0;
}

bool FrameQueue::getLatestFrame(cv::Mat& latestFrame)
{
	std::lock_guard<std::mutex> lock(mtx);
	if (frame.empty())	return false;

	latestFrame = frame.clone();
	
	return true;
}