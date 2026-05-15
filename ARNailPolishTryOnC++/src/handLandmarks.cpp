#include "handLandmarks.h"

HandLandmarks::HandLandmarks()
{}

HandLandmarks::~HandLandmarks()
{}

int HandLandmarks::detectHandLandmarks()
{
	// Load ONNX model
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::wstring detectorPath = L"D:/ARNailPolishTryOnC++/models/hand_detector.onnx";
    Ort::Session detectorSession(env, detectorPath.c_str(), session_options);

    std::cout << "Model loaded successfully!\n";

    // Load image
    cv::Mat image = cv::imread("D:/ARNailPolishTryOnC++/hand.jpg");
    if (image.empty())
    {
        std::cerr << "Failed to load image\n";
        return -1;
    }
    
    Ort::AllocatorWithDefaultOptions allocator;
    cv::Mat inputImage;
    cv::Mat handROI;
    try
    {
        auto inputNameAllocated = detectorSession.GetInputNameAllocated(0, allocator);

        std::string inputName = inputNameAllocated.get();
        const char* inputNames[] =
        {
            inputName.c_str()
        };

        auto inputShapeInfo =
            detectorSession.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();

        std::vector<int64_t> inputDims =
            inputShapeInfo.GetShape();

        for (auto d : inputDims)
            std::cout << d << std::endl;

        inputImage = preprocessImage(image);

        // Run Detector
        auto outputNameAllocated =
            detectorSession.GetOutputNameAllocated(0, allocator);

        std::string outputName = outputNameAllocated.get();

        const char* outputNames[] =
        {
            outputName.c_str()
        };

        // Create Input Tensor
        std::vector<float> inputTensorValues =
            blobFromImage(inputImage);

        std::array<int64_t, 4> inputShape{ 1, 3, 256, 256 };

        Ort::MemoryInfo memoryInfo =
            Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,
                OrtMemTypeDefault);

        Ort::Value inputTensor =
            Ort::Value::CreateTensor<float>(
                memoryInfo,
                inputTensorValues.data(),
                inputTensorValues.size(),
                inputShape.data(),
                inputShape.size());

        auto outputTensors = detectorSession.Run(
            Ort::RunOptions{ nullptr },
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1);

        // Run Detector Output
        float* output =
            outputTensors[0].GetTensorMutableData<float>();
        float cx = output[0];
        float cy = output[1];
        float w = output[2];
        float h = output[3];
        float conf = output[4];

        // Convert Bounding Box to Image Coordinates
        int x1 = static_cast<int>((cx - w / 2) * inputImage.cols);
        int y1 = static_cast<int>((cy - h / 2) * inputImage.rows);

        int boxW = static_cast<int>(w * inputImage.cols);
        int boxH = static_cast<int>(h * inputImage.rows);

        cv::Rect handRect(x1, y1, boxW, boxH);

        handRect &= cv::Rect(0, 0, inputImage.cols, inputImage.rows);

        // Crop Hand ROI
        handROI = inputImage(handRect).clone();

        // Preprocess Landmark Input
        cv::Mat landmarkInput = preprocessImage(handROI);

        // Landmarks detection -------------
        std::wstring landmarkPath = L"D:/ARNailPolishTryOnC++/models/hand_landmark_detector.onnx";
        Ort::Session landmarkSession(env, landmarkPath.c_str(), session_options);

        inputNameAllocated = landmarkSession.GetInputNameAllocated(0, allocator);

        inputName = inputNameAllocated.get();
        const char* inputNamesLandmark[] =
        {
            inputName.c_str()
        };

        inputShapeInfo =
            landmarkSession.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();

        std::vector<int64_t> inputDims1 =
            inputShapeInfo.GetShape();

        for (auto d : inputDims1)
            std::cout << d << std::endl;

        // Run Detector
        outputNameAllocated =
            landmarkSession.GetOutputNameAllocated(0, allocator);

        outputName = outputNameAllocated.get();

        const char* outputNamesLandmark[] =
        {
            outputName.c_str()
        };

       // Run Landmark Model
        std::vector<float> landmarkBlob =
            blobFromImage(landmarkInput);

        std::array<int64_t, 4> landmarkShape{ 1,3,256,256 };

        Ort::Value landmarkTensor =
            Ort::Value::CreateTensor<float>(
                memoryInfo,
                landmarkBlob.data(),
                landmarkBlob.size(),
                landmarkShape.data(),
                landmarkShape.size());

        auto landmarkOutputs = landmarkSession.Run(
            Ort::RunOptions{ nullptr },
            inputNamesLandmark,
            &landmarkTensor,
            1,
            outputNamesLandmark,
            1);

        // Read Landmark Output
        float* lm =
            landmarkOutputs[0].GetTensorMutableData<float>();

        for (int i = 0; i < 21; ++i)
        {
            float x = lm[i * 3 + 0];
            float y = lm[i * 3 + 1];
            float z = lm[i * 3 + 2];

            int px = static_cast<int>(x * handRect.width) + handRect.x;
            int py = static_cast<int>(y * handRect.height) + handRect.y;

            cv::circle(inputImage, cv::Point(px, py), 3,
                cv::Scalar(0, 0, 255), -1);
        }
    }
    catch (const Ort::Exception& e)
    {
        std::cout << e.what() << std::endl;
    }

    //cv::imshow("Landmarks", inputImage);
    cv::imshow("Hand ROI", handROI);
    cv::waitKey(0);


    return 0;
}

cv::Mat HandLandmarks::preprocessImage(const cv::Mat& image)
{
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(256, 256));

    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    return resized;
}

std::vector<float> HandLandmarks::blobFromImage(const cv::Mat& image)
{
    std::vector<float> blob;

    int channels = 3;
    int height = image.rows;
    int width = image.cols;

    blob.resize(channels * height * width);

    for (int c = 0; c < channels; ++c)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                blob[c * width * height + y * width + x] =
                    image.at<cv::Vec3f>(y, x)[c];
            }
        }
    }

    return blob;
}