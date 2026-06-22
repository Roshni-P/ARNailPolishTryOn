#include "handLandmarks.h"

HandLandmarks::HandLandmarks()
{}

HandLandmarks::~HandLandmarks()
{}

int HandLandmarks::detectHandLandmarks()
{
    
    Ort::AllocatorWithDefaultOptions allocator;
    cv::Mat inputImage;
    cv::Mat handROI;
    try
    {



        // Preprocess Landmark Input
        cv::Mat landmarkInput = preprocessImage(handROI);

        // Landmarks detection -------------
        std::wstring landmarkPath = L"D:/ARNailPolishTryOn/ARNailPolishTryOnC++/models/hand_landmark_detector.onnx";
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


