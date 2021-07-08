#include "detection.h"

using namespace cv;
using namespace std;

// PanoramicImage Class
// constructor
Detection::Detection(String pattern) {

    pattern = pattern + "/*.png";
    vector<String> fn;
    try {
        glob(pattern, fn, false);
    }
    catch (exception& ex) {
        cout<<"Invalid pattern name"<<endl;
    }

    if (!fn.size()) {
        
        cout<<"Error loading the images"<<endl;
    }
    else {
        
        int count = fn.size(); // Number of images
        for (int i = 0; i < count; i++) {
            
            test_images.push_back(imread(fn[i], IMREAD_COLOR));
            namedWindow("Image");
            imshow("Image", test_images.at(i));
            waitKey(0);
        }
    }
}

void Detection::pyramid(double scale, Mat image, int stepSize, int windowSize) {
    
    int debug = 0;
    while (image.rows >= windowSize && image.cols >= windowSize) {
        
        // Sliding window
        int i = 0; // row
        while (i <= image.rows - windowSize) {
            
            for (int j = 0; j <= image.cols - windowSize; j += stepSize) {
                
                Rect rect(j, i, windowSize, windowSize);
                Mat draw_window = image.clone();
                rectangle(draw_window, rect, cv::Scalar(0, 255, 0), 4);
                imshow("Window", draw_window);
                waitKey(0);
            
            }
            i += stepSize;
        }
    
        Size size(image.cols/scale, image.rows/scale);
        resize(image, image, size);
    }
}

void Detection::HOG(Mat image) {
    
    // Convert rgb image into gray image
    Mat img_gray;
    cvtColor(image, img_gray, COLOR_BGR2GRAY);
    
    // Normalize gray image
    img_gray.convertTo(img_gray, CV_32F, 1.0 / 255, 0);
    imshow("Norma", img_gray);
    waitKey(0);
    
    // Gamma correction (each pixel x replaced by sqrt(x))
    Mat img_gamma;
    pow(img_gray, 0.5, img_gamma);
    imshow("Gamma", img_gamma);
    waitKey(0);
    
    
}

void Detection::preprocessing(String pattern) {
    
    
}

