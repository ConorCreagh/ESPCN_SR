// ESPCN_SR.cpp 

#include <iostream>
#include <opencv2/opencv.hpp>

// Function to perform ESPCN super-resolution
cv::Mat superResolveImage(const cv::Mat& lowResImage, cv::dnn::Net& espcnModel) {
    // Create a blob from the low-resolution image
    cv::Mat inputBlob = cv::dnn::blobFromImage(lowResImage, 1.0, cv::Size(), cv::Scalar(), true, false);

    // Set the blob as the input to the ESPCN model
    espcnModel.setInput(inputBlob);

    // Forward pass through the network to obtain the super-resolved image
    cv::Mat outputBlob = espcnModel.forward();

    // Reshape the output blob to obtain the high-resolution image
    cv::Mat highResImage = outputBlob.reshape(1, outputBlob.size[2]);

    // Convert the high-resolution image to the correct format
    highResImage.convertTo(highResImage, CV_8UC3, 255.0);

    // Return the super-resolved image
    return highResImage;
}

int main() {
    // Load the low-resolution image
    cv::Mat lowResImage = cv::imread("C:/Users/drume/OneDrive/Pictures/konijn_test.jpg");

    if (lowResImage.empty()) {
        std::cout << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Load the ESPCN model
    cv::dnn::Net espcnModel = cv::dnn::readNetFromCaffe("espcn_model.prototxt", "espcn_model.caffemodel");

    if (espcnModel.empty()) {
        std::cout << "Could not load the ESPCN model!" << std::endl;
        return -1;
    }

    // Perform image super-resolution
    cv::Mat highResImage = superResolveImage(lowResImage, espcnModel);

    // Display the low-resolution and high-resolution images
    cv::namedWindow("Low Resolution Image", cv::WINDOW_NORMAL);
    cv::imshow("Low Resolution Image", lowResImage);

    cv::namedWindow("High Resolution Image", cv::WINDOW_NORMAL);
    cv::imshow("High Resolution Image", highResImage);

    cv::waitKey(0);

    return 0;
}

