#include <iostream>
#include "frame.h"
#include "common.h"
#include "handLandmarks.h"

int main()
{
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
	std::cout << "ONNX Runtime initialized successfully!" << std::endl;

/*	Frame obj;
	obj.captureFrame();
	*/

	HandLandmarks hl;
	hl.detectHandLandmarks();

	std::cout << "Exiting App!" << std::endl;

	return 1;
}