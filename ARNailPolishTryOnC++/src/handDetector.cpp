#include "handDetector.h"
#include "frame.h"

IDetector::IDetector()
{
}

IDetector::~IDetector()
{}


std::vector<Anchor> HandDetector::generateAnchors(int image_width, int image_height) {
    std::vector<Anchor> anchors;

    // MediaPipe Production SSD Anchor Configuration
    std::vector<int> strides = { 8, 16, 32, 32, 32, 32 };
    std::vector<int> num_layers = { 2, 2, 6, 6, 6, 6 }; // Anchor counts per feature pixel

    for (size_t i = 0; i < strides.size(); ++i) {
        int stride = strides[i];
        int feature_map_width = std::ceil((float)image_width / stride);
        int feature_map_height = std::ceil((float)image_height / stride);

        for (int y = 0; y < feature_map_height; ++y) {
            for (int x = 0; x < feature_map_width; ++x) {
                for (int anchor_idx = 0; anchor_idx < num_layers[i]; ++anchor_idx) {
                    Anchor anchor;
                    anchor.x_center = (x + 0.5f) / feature_map_width;
                    anchor.y_center = (y + 0.5f) / feature_map_height;
                    anchor.width = 1.0f;
                    anchor.height = 1.0f;
                    anchors.push_back(anchor);
                }
            }
        }
    }
    return anchors;
}

int HandDetector::LoadModel(std::string path)
{
    net = cv::dnn::readNetFromONNX(path);
    if (net.empty()) {
        std::cerr << "Error: Could not load Hand Detector model. Check the path!" << std::endl;
        return -1;
    }

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    return 0;
}

int HandDetector::DetectFrame(cv::Mat& latestFrame, cv::Mat& handROI)
{
    std::vector<Anchor> anchors = generateAnchors(INPUT_WIDTH, INPUT_HEIGHT);
    int frameWidth = latestFrame.cols;
    int frameHeight = latestFrame.rows;

    // 1. Set up the allocation parameters explicitly
    cv::dnn::Image2BlobParams blobParams;
    blobParams.scalefactor = cv::Scalar::all(1.0 / 255.0); // Normalize pixels to [0,1]
    blobParams.size = cv::Size(256, 256);
    blobParams.mean = cv::Scalar(0, 0, 0);
    blobParams.swapRB = true;
    blobParams.ddepth = CV_32F;
    blobParams.datalayout = cv::dnn::DNN_LAYOUT_NHWC;

    // 2. Create the blob using the specialized parameter method
    cv::Mat blob = cv::dnn::blobFromImageWithParams(latestFrame, blobParams);

    net.setInput(blob);

    // 4. Run forward pass
    std::vector<cv::Mat> outputs;
    std::vector<std::string> outNames = net.getUnconnectedOutLayersNames();
    net.forward(outputs, outNames);

    std::cout << "-----Output 0 shape: " << outputs[0].size << " | Output 1 shape: " << outputs[1].size << std::endl;
    cv::Mat rawBoxes = outputs[0].reshape(1, 3584);  // Force to 3584 rows x 18 cols
    cv::Mat rawScores = outputs[1].reshape(1, 3584); // Force to 3584 rows x 1 col

    std::vector<cv::Rect> bboxes;
    std::vector<float> confidences;

    for (int i = 0; i < 3584; ++i) {
        float raw_score = rawScores.at<float>(i, 0);

        // Apply Sigmoid to get actual 0.0 - 1.0 confidence
        float score = 1.0f / (1.0f + std::exp(-raw_score));

        if (score > SCORE_THRESHOLD) {
            Anchor anchor = anchors[i];

            // Parse box regression parameters (First 4 elements out of 18)
            float dx = rawBoxes.at<float>(i, 0);
            float dy = rawBoxes.at<float>(i, 1);
            float dw = rawBoxes.at<float>(i, 2);
            float dh = rawBoxes.at<float>(i, 3);

            if (std::isnan(dx) || std::isnan(dy) || std::isnan(dw) || std::isnan(dh) ||
                std::isinf(dx) || std::isinf(dy)) {
                continue;
            }

            float box_scale_x = 256.0f;
            float box_scale_y = 256.0f;

            // Center coordinates (linear)
            float cx = (dx / box_scale_x) * anchor.width + anchor.x_center;
            float cy = (dy / box_scale_y) * anchor.height + anchor.y_center;

            // LINEAR decoding
            float w = (dw / box_scale_x) * anchor.width;
            float h = (dh / box_scale_y) * anchor.height;

            float box_expand_factor = 3.5f;
            w *= box_expand_factor;
            h *= box_expand_factor;

            // Convert to pixel space
            int w_rect = static_cast<int>(w * frameWidth);
            int h_rect = static_cast<int>(h * frameHeight);
            int x_rect = static_cast<int>((cx - w / 2.0f) * frameWidth);
            int y_rect = static_cast<int>((cy - h / 2.0f) * frameHeight);

            cv::Rect box(x_rect, y_rect, w_rect, h_rect);
            // Clip box to ensure it doesn't fall off image borders
            box &= cv::Rect(0, 0, frameWidth, frameHeight);

            if (box.width > 0 && box.height > 0) {
                bboxes.push_back(box);
                confidences.push_back(score);
            }
        }
    }

    // Apply Non-Maximum Suppression (NMS) to eliminate overlapping boxes
    std::vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

    // Extract and Draw Hand ROI
    for (int idx : indices) {
        cv::Rect rectROI = bboxes[idx];

        // Draw bounding box on original frame
        cv::rectangle(latestFrame, rectROI, cv::Scalar(0, 255, 0), 2);
        cv::putText(latestFrame, "Hand: " + std::to_string(confidences[idx]).substr(0, 4),
            cv::Point(rectROI.x, rectROI.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

        // Crop the Hand ROI
        handROI = latestFrame(rectROI);
    }

    return 0;
}