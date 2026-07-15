#include "handLandmarksDetector.h"

int HandLandmarksDetector::LoadModel(std::string path)
{
    net = cv::dnn::readNetFromONNX(path);
    if (net.empty()) {
        std::cerr << "Error: Could not load Hand Detector model. Check the path!" << std::endl;
        return -1;
    }

    return 1;
}

int HandLandmarksDetector::DetectLandmarks(cv::Mat& handROI, std::vector<cv::Point2f>& handLandmarks)
{
    try
    {
        cv::Mat rgb_frame, resized, floated;
        // Pre-allocate the static 256x256 canvas to save processing time
        cv::Mat letterbox = cv::Mat::zeros(256, 256, CV_8UC3);

        std::vector<std::string> out_names = net.getUnconnectedOutLayersNames();
        std::vector<cv::Mat> outputs;

        // Convert to RGB
        cv::cvtColor(handROI, rgb_frame, cv::COLOR_BGR2RGB);
        // 2. Calculate Dynamic Letterbox Scalers for current frame size
        int w = rgb_frame.cols;
        int h = rgb_frame.rows;
        float scale = std::min(256.0f / w, 256.0f / h);
        int new_w = w * scale;
        int new_h = h * scale;
        int dx = (256 - new_w) / 2;
        int dy = (256 - new_h) / 2;

        // 3. Fast Letterboxing
        cv::resize(rgb_frame, resized, cv::Size(new_w, new_h));

        // Clear previous frame data out of our pre-allocated canvas safely
        letterbox.setTo(cv::Scalar(0, 0, 0));

        // Direct memory copy to the calculated center ROI region
        resized.copyTo(letterbox(cv::Rect(dx, dy, new_w, new_h)));

        // 4. Run Inference
        letterbox.convertTo(floated, CV_32FC3, 1.0 / 255.0);
        int dims[] = { 1, floated.rows, floated.cols, floated.channels() };
        cv::Mat custom_blob(4, dims, CV_32F, floated.data);

        net.setInput(custom_blob);
        net.forward(outputs, out_names);

        const float shift_right_pixels = 25.5f;
        const float shift_below_pixels = 15.5f;
        // 5. Recover Original Coordinates & Draw
        cv::Mat flat_landmarks = outputs[0].reshape(1, 1);
        for (int i = 0; i < 21; ++i) {
            float raw_x = flat_landmarks.at<float>(0, i * 3 + 0);
            float raw_y = flat_landmarks.at<float>(0, i * 3 + 1);

            // Undo padding offset
            float unpadded_x = raw_x - dx;
            float unpadded_y = raw_y - dy;

            // Rescale based on the runtime aspect ratio multipliers
            float pixel_x = (unpadded_x / new_w) * handROI.cols;
            float pixel_y = (unpadded_y / new_h) * handROI.rows;

            pixel_x += shift_right_pixels;
            pixel_y += shift_below_pixels;
            handLandmarks.push_back({ pixel_x, pixel_y });
            
            if (pixel_x >= 0 && pixel_x < handROI.cols && pixel_y >= 0 && pixel_y < handROI.rows) {
                cv::circle(handROI, cv::Point(pixel_x, pixel_y), 4, cv::Scalar(0, 255, 0), -1);
            }
        }
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime exception caught!" << std::endl;
        std::cerr << "Error Message: " << e.what() << std::endl;
        std::cerr << "ORT Error Code: " << e.GetOrtErrorCode() << std::endl;
    }


    return 1;
}

cv::Mat HandLandmarksDetector::getNailMask(const cv::Mat& frame, 
            const std::vector<cv::Point2f>& landmarks, cv::Mat& totalNailMask)
{
    totalNailMask = cv::Mat::zeros(frame.size(), CV_8UC1);
    if (landmarks.empty())
        return totalNailMask;

    // MediaPipe indices for fingertips and their corresponding DIP joints
    // Thumb (4,3), Index (8,7), Middle (12,11), Ring (16,15), Pinky (20,19)
    std::vector<std::pair<int, int>> fingerPairs = { {4,3}, {8,7}, {12,11}, {16,15}, {20,19} };

    // 1. First, find the minimum and maximum X/Y values in your raw landmarks
    float minX = 9999, maxX = -9999;
    float minY = 9999, maxY = -9999;

    for (const auto& lm : landmarks) {
        if (lm.x < minX) minX = lm.x;
        if (lm.x > maxX) maxX = lm.x;
        if (lm.y < minY) minY = lm.y;
        if (lm.y > maxY) maxY = lm.y;
    }

    float handWidthRaw = maxX - minX;
    float handHeightRaw = maxY - minY;

    // 2. Prevent division by zero if landmarks are corrupt
    if (handWidthRaw == 0) handWidthRaw = 1;
    if (handHeightRaw == 0) handHeightRaw = 1;

    // 3. Process the fingertips with dynamic normalization
    for (const auto& pair : fingerPairs) {
        // 1. Map landmarks directly since they are already in absolute pixel space
        cv::Point2f tip(landmarks[pair.first].x, landmarks[pair.first].y);
        cv::Point2f dip(landmarks[pair.second].x, landmarks[pair.second].y);

        // 2. Compute the distance (joint length)
        double dist = cv::norm(tip - dip);

        // Determine ROI size (tweak 0.65 to scale the box size up or down)
        int roiSize = static_cast<int>(dist * 0.65);
        if (roiSize < 15) roiSize = 30;

        // 3. Compute direction vector from DIP to TIP
        // This points exactly towards the direction the finger is facing
        cv::Point2f direction = tip - dip;
        if (dist > 0) {
            direction.x /= dist;
            direction.y /= dist;
        }

        // 4. Push the center of the box slightly past the tip landmark along the finger's vector
        // This guarantees the box shifts toward the nail bed, even if the hand is sideways or upside down
        float offsetMagnitude = dist * 0.15f;
        cv::Point center(
            static_cast<int>(tip.x + direction.x * offsetMagnitude),
            static_cast<int>(tip.y + direction.y * offsetMagnitude)
        );

        // 5. Create the ROI centered on our adjusted coordinates
        cv::Rect roi(center.x - roiSize / 2, center.y - roiSize / 2, roiSize, roiSize);

        // Clamp to frame boundaries
        roi &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (roi.width <= 0 || roi.height <= 0) continue;

        // Draw the box on the frame
        cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2);

        // ... continue with your HSV crop and mask logic ...
        cv::Mat fingerCrop = frame(roi);
        cv::Mat hsv, colorMask;
        cv::cvtColor(fingerCrop, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(0, 20, 100), cv::Scalar(20, 150, 255), colorMask);
        colorMask.copyTo(totalNailMask(roi));
    }
    // Clean up noise
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(totalNailMask, totalNailMask, cv::MORPH_CLOSE, kernel);

    return totalNailMask;
}