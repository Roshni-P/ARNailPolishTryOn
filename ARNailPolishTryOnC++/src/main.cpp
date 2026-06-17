#include <iostream>
#include "frame.h"
#include "common.h"
#include "modelFactory.h"
#include "handDetector.h"


int main()
{

/*	Frame obj;
	obj.captureFrame();
	*/

	std::unique_ptr<HandDetector> handDetector = ModelFactory::Create<ModelType::HandDetector>();
	handDetector->LoadModel("D:/ARNailPolishTryOn/ARNailPolishTryOnC++/models/Google/hand_detector.onnx");

	FrameQueue cam;
	std::cout << "Started Live Feed." << std::endl;
	cv::Mat displayFrame;
	while (true)
	{
		cam.getLatestFrame(displayFrame);
		if (displayFrame.empty())
			continue;

		//cv::imshow("Live Camera!", displayFrame);

		handDetector->DetectFrame(displayFrame);

		if (cv::waitKey(1) == 27) break;
	}

	std::cout << "Exiting App!" << std::endl;
	cv::destroyAllWindows();
	return 1;
}