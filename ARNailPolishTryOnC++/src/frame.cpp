#include "frame.h"
#include <iostream>

Frame::Frame()
{
}

Frame::~Frame()
{
}

/**
 * @brief Captures frames from Live camera
 * @param 
 * @return error code - 0 if no errors
 */
int Frame::captureFrame()
{
	cv::VideoCapture cap(0);

	if (!cap.isOpened())
	{
		std::cerr << "Error: Unable to open camera." << std::endl;
		return -1;
	}

	cv::namedWindow("Live Camera!", cv::WINDOW_AUTOSIZE);

	while (true)
	{
		cv::Mat frame;
		bool bRet = cap.read(frame);
		if (!bRet || frame.empty())
		{
			std::cerr << "Error reading frame." << std::endl;
			return -1;
		}

		cv::imshow("Live Camera!", frame);

		if (cv::waitKey(1) == 27) // ESC key, to break
			break;
	}

	cap.release();
	cv::destroyAllWindows();

	return 0;
}