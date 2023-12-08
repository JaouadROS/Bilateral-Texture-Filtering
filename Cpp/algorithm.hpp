#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;

// Assuming computeGuidance and computeBlurAndMRTV are defined elsewhere
cv::Mat computeGuidance(const cv::Mat& B, const cv::Mat& mRTV, int k);
void computeBlurAndMRTV(const cv::Mat& I, int k, cv::Mat& B, cv::Mat& mRTVs);

cv::Mat bilateralTextureFilter(const cv::Mat& I, int k, int iter) {
    if (I.type() != CV_64FC1 && I.type() != CV_64FC3) {
        throw std::invalid_argument("Input image must be double precision with 1 or 3 channels.");
    }

    if (k < 3 || k % 2 == 0) {
        throw std::invalid_argument("Patch size k must be odd and at least 3.");
    }
    if (iter < 1) {
        throw std::invalid_argument("There must be at least one iteration.");
    }

    const int dimX = I.rows;
    const int dimY = I.cols;
    const int channels = I.channels();
    const int s = 2 * k - 1;
    const int half_s = s / 2;
    const double sigma_s = k - 1;
    const double sigma_r = 0.025 * sqrt(channels);

    cv::Mat gaussianKernel = cv::getGaussianKernel(s, sigma_s, CV_64F);
    const cv::Mat f = gaussianKernel * gaussianKernel.t();

    cv::Mat J = I.clone();
    std::vector<cv::Mat> Ic, Bc(channels), mRTVsc(channels);

    for (int m = 0; m < iter; ++m) {
        cv::split(J, Ic);

        for (int i = 0; i < channels; ++i) {
            computeBlurAndMRTV(Ic[i], k, Bc[i], mRTVsc[i]);
        }

        // Sum mRTVs and divide by number of channels
        cv::Mat mRTV = cv::Mat::zeros(mRTVsc[0].size(), mRTVsc[0].type());
        for (const auto& mRTV_channel : mRTVsc) {
            mRTV += mRTV_channel;
        }
        mRTV /= channels;

        cv::Mat B;
        cv::merge(Bc, B);
        cv::Mat G_prime = computeGuidance(B, mRTV, k);

        cv::parallel_for_(cv::Range(0, J.rows), [&](const cv::Range& range) {
            for (int i = range.start; i < range.end; ++i) {
                for (int j = 0; j < I.cols; ++j) {
                    for (int c = 0; c < channels; ++c) {
                        int minX = std::max(i - half_s, 0);
                        int minY = std::max(j - half_s, 0);
                        int maxX = std::min(i + half_s + 1, I.rows);
                        int maxY = std::min(j + half_s + 1, I.cols);

                        cv::Rect roi(minY, minX, maxY - minY, maxX - minX);
                        cv::Mat G_prime_patch = G_prime(roi).clone();

                        cv::Mat g = cv::Mat::zeros(G_prime_patch.size(), CV_64F);
                        for (int x = 0; x < g.rows; ++x) {
                            for (int y = 0; y < g.cols; ++y) {
                                g.at<double>(x, y) = std::exp(-std::pow(G_prime_patch.at<cv::Vec3d>(x, y)[c] - G_prime.at<cv::Vec3d>(i, j)[c], 2) / (2 * std::pow(sigma_r, 2)));
                            }
                        }

                        cv::Mat fg = g.mul(f(cv::Range(minX - i + half_s, maxX - i + half_s), cv::Range(minY - j + half_s, maxY - j + half_s)));
                        
                        cv::Mat I_patch = Ic[c](roi);
                        double sumFg = cv::sum(fg)[0];
                        double response = sumFg != 0 ? cv::sum(I_patch.mul(fg))[0] / sumFg : 0;
                        J.at<cv::Vec3d>(i, j)[c] = response;
                    }
                }
            }
        });

    }

    return J;
}

cv::Mat computeGuidance(const cv::Mat& B, const cv::Mat& mRTV, int k) {
    int dimX = B.rows;
    int dimY = B.cols;
    int halfKernel = k / 2;
    double sigma_alpha = 5 * k;

    cv::Mat G = cv::Mat::zeros(B.size(), B.type());
    cv::Mat mRTV_min = cv::Mat::zeros(mRTV.size(), mRTV.type());

    cv::parallel_for_(cv::Range(0, dimX), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            for (int j = 0; j < dimY; ++j) {
                int minY = std::max(0, i - halfKernel);
                int minX = std::max(0, j - halfKernel);
                int maxY = std::min(i + halfKernel, (dimX-1));
                int maxX = std::min(j + halfKernel, (dimY-1));

                int width = maxX - minX + 1;
                int height = maxY - minY + 1;
                cv::Rect patchRect(minX, minY, width, height);
                cv::Mat mRTV_patch = mRTV(patchRect);

                double minVal;
                cv::Point minLoc;
                cv::minMaxLoc(mRTV_patch, &minVal, nullptr, &minLoc, nullptr);
                mRTV_min.at<double>(i, j) = minVal;

                for (int c = 0; c < B.channels(); ++c) {
                    G.at<cv::Vec3d>(i, j)[c] = B.at<cv::Vec3d>(minY + minLoc.y, minX + minLoc.x)[c];
                }
            }
        }
    });

    cv::Mat alphaExpr = -sigma_alpha * (mRTV - mRTV_min);
    cv::Mat alpha;
    cv::exp(alphaExpr, alpha);
    alpha = 2 * (1 / (1 + alpha) - 0.5);

    cv::Mat G_prime = B.clone();
    for (int c = 0; c < B.channels(); ++c) {
        G_prime.forEach<cv::Vec3d>([&](cv::Vec3d& pixel, const int* position) -> void {
            int x = position[0];
            int y = position[1];
            pixel[c] = alpha.at<double>(x, y) * G.at<cv::Vec3d>(x, y)[c] +
                       (1 - alpha.at<double>(x, y)) * B.at<cv::Vec3d>(x, y)[c];
        });
    }

    return G_prime;
}

void computeBlurAndMRTV(const cv::Mat& inputImage, int k, cv::Mat& B, cv::Mat& mRTV) {
    const int dimX = inputImage.rows;
    const int dimY = inputImage.cols;
    const int halfKernel = k / 2;
    const double eps = std::numeric_limits<double>::epsilon();

    B = cv::Mat(inputImage.rows, inputImage.cols, CV_64F);
    mRTV = cv::Mat(inputImage.rows, inputImage.cols, CV_64F);

    // Define the Sobel kernel (h)
    cv::Mat h = (cv::Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);

    // Compute gradients in X and Y directions
    cv::Mat Gx, Gy;
    cv::filter2D(inputImage, Gx, CV_64F, h.t(), cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(inputImage, Gy, CV_64F, h, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

    // Compute gradient magnitude
    cv::Mat Ixy;
    cv::magnitude(Gx, Gy, Ixy);

    cv::parallel_for_(cv::Range(0, dimX), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            for (int j = 0; j < dimY; ++j) {
                // Extract the local patch
                int minX = std::max(i - halfKernel, 0);
                int minY = std::max(j - halfKernel, 0);
                int maxX = std::min(i + halfKernel + 1, dimX);
                int maxY = std::min(j + halfKernel + 1, dimY);

                cv::Rect roi(minY, minX, maxY - minY, maxX - minX);

                cv::Mat I_patch = inputImage(roi);
                cv::Mat Ixy_patch = Ixy(roi);
                
                // Compute the average intensity
                B.at<double>(i, j) = cv::mean(I_patch)[0];

                // Compute the mRTV based on Eq. (4)
                double minVal, maxVal, maxIxy;
                cv::minMaxLoc(I_patch, &minVal, &maxVal);
                cv::minMaxLoc(Ixy_patch, nullptr, &maxIxy);
                double sumIxy = cv::sum(Ixy_patch)[0];
                mRTV.at<double>(i, j) = (maxVal - minVal) * maxIxy / (sumIxy + eps);
            }
        }
    });
}
