#include <iostream>
#include "frame.h"
#include "common.h"
#include "modelFactory.h"
#include "handDetector.h"
#include "handLandmarksDetector.h"

int main()
{

	std::unique_ptr<HandDetector> handDetector = ModelFactory::Create<ModelType::HandDetector>();
	handDetector->LoadModel("D:/ARNailPolishTryOn/ARNailPolishTryOnC++/models/Google/hand_detector.onnx");

	std::unique_ptr<HandLandmarksDetector> handLandmarkDetector = ModelFactory::Create<ModelType::HandLandmark>();
	handLandmarkDetector->LoadModel("D:/ARNailPolishTryOn/ARNailPolishTryOnC++/models/Google/hand_landmarks_detector.onnx");

	FrameQueue cam;
	std::cout << "Started Live Feed." << std::endl;
	cv::Mat displayFrame;
	cv::Mat handROI, handLandmarks;
	while (true)
	{
		cam.getLatestFrame(displayFrame);
		if (displayFrame.empty())
			continue;

		handDetector->DetectFrame(displayFrame, handROI);
		if(!handROI.empty())
			handLandmarkDetector->DetectLandmarks(handROI, handLandmarks);

		if (cv::waitKey(1) == 27) break;
	}

	std::cout << "Exiting App!" << std::endl;
	cv::destroyAllWindows();
	return 1;
}