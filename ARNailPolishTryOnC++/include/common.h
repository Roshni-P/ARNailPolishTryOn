#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>

// Forward Declaration
class HandLandmarksDetector;
class IDetector;
class NailDetector;
class HandDetector;
class FrameQueue;

struct Anchor {
    float x_center, y_center, width, height;
};

const int INPUT_WIDTH = 256;
const int INPUT_HEIGHT = 256;
const float SCORE_THRESHOLD = 0.85f; // Raise this to filter out noise/nearby areas (0.75 - 0.85 is sweet spot)
const float NMS_THRESHOLD = 0.40f;
const int MAX_QSIZE = 2;


