#include "frame.h"
#include <iostream>

Frame::Frame():isRunning(false)
{
	if (!cap.isOpened())
	{
		std::cerr << "Error: Unable to open camera." << std::endl;
		return;
	}
	cv::namedWindow("Live Camera!", cv::WINDOW_AUTOSIZE);

	isRunning = true;
	//workerThread = std::thread(&Frame::captureFrame, this);
}

Frame::~Frame()
{
	stopCapture();
}

/**
 * @brief Captures frames from Live camera
 * @param 
 * @return error code - 0 if no errors
 */
int Frame::captureFrame()
{
	cv::Mat tempFrame;

	while (isRunning)
	{
		cap >> tempFrame;
		if (tempFrame.empty()) continue;

		{
			std::lock_guard<std::mutex> lock(frameMutex);
			frame = tempFrame.clone();
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(5));
		cv::imshow("Live Camera!", frame);

		if (cv::waitKey(1) == 27) // ESC key, to break
			break;
	}

	return 0;
}

bool Frame::getFrame(cv::Mat& outputFrame)
{
	std::lock_guard<std::mutex> lock(frameMutex);
	if (frame.empty())
		return false;

	outputFrame = frame.clone();

	return true;
}

void Frame::stopCapture()
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

/// <summary>
/// Queue of frames, with the latest frame on top
/// </summary>
/// <param name="frame"></param>
void FrameQueue::push(cv::Mat& frame)
{
	std::lock_guard<std::mutex> lock(mtx);
	if (queue.size() >= MAX_QSIZE)
	{
		queue.pop();
	}
	queue.push(frame);
	cv.notify_one();
}

bool FrameQueue::pop(cv::Mat& frame)
{
	std::unique_lock<std::mutex> lock(mtx);

	cv.wait(lock, [this]() { return !queue.empty(); });

	frame = queue.front();
	queue.pop();

	return true;
}