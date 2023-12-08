#include "algorithm.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat inputImage = cv::imread("noise4-3-2.png", cv::IMREAD_COLOR);
    //inputImage = cv::imread("angel-5-5.png", cv::IMREAD_COLOR);
    //inputImage = cv::imread("circle-5-7.png", cv::IMREAD_COLOR);
    inputImage = cv::imread("compression4-5-15.png", cv::IMREAD_COLOR);

    if(inputImage.empty()) {
        std::cerr << "Error: Image not found." << std::endl;
        return -1;
    }

    inputImage.convertTo(inputImage, CV_64FC3, 1.0 / 255); 
    cv::imshow("inputImage", inputImage);

    try {
        int k = 5;
        int iter = 15;
        cv::Mat result;
        for (int run = 0; run < 10; ++run) {
            int64 start = cv::getTickCount();

            result = bilateralTextureFilter(inputImage, k, iter);

            int64 end = cv::getTickCount();
            double timeSec = (end - start) / cv::getTickFrequency();

            std::cout << "Run " << run + 1 << ": Processing time = " << timeSec << " seconds" << std::endl;

            cv::imshow("result", result);
            cv::waitKey(1);
        }

        cv::waitKey(0);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
