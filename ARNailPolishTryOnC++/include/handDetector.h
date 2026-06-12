#pragma once
#include "common.h"

class IDetector
{
public:
	IDetector();
	virtual ~IDetector();
	inline float sigmoid(float x) {
		return 1.0f / (1.0f + exp(-x));
	}
protected:
	virtual int LoadModel(std::string path) { return -1; }
	virtual int DetectFrame(const Frame& frame) { return -1;  }
	cv::dnn::Net net;
};

class HandDetector : public IDetector
{
public:
	std::vector<Anchor> generateAnchors(int image_width, int image_height);
	int LoadModel(std::string path) override;
	int DetectFrame(const Frame& frame) override;
private:

};