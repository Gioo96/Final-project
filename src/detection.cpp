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

void Detection::preprocessing(String pattern) {
    
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(4);

    for (int i = 0; i < test_images.size(); i ++){
        
        // CLAHE GRAY_IMG
        Mat image_gray;
        cvtColor(test_images.at(i), image_gray, COLOR_BGR2GRAY);
        Mat image_clahe;
        clahe->apply(image_gray, image_clahe);
        imshow("Img" + to_string(i), test_images.at(i));
        waitKey(0);
        imshow("Img" + to_string(i) + ": Clahe applied", image_clahe);
        waitKey(0);
        
        
        // CLAHE H
        //
        //
        Mat hsv_image_h;
        cvtColor(test_images.at(i), hsv_image_h, COLOR_BGR2HSV);

        // Extract the HSV channel
        vector<Mat> hsv_planes_h(3);
        split(hsv_image_h, hsv_planes_h);  // now we have the L image in lab_planes[0]
        
        // Apply the CLAHE algorithm to the H channel
        Mat dst_h;
        clahe->apply(hsv_planes_h[0], dst_h);

        // Merge the the color planes back into an HSV image
        dst_h.copyTo(hsv_planes_h[0]);
        merge(hsv_planes_h, hsv_image_h);

        // convert back to RGB
        Mat image_clahe_h;
        cvtColor(hsv_image_h, image_clahe_h, COLOR_HSV2BGR);
        
        imshow("Img" + to_string(i) + ": Clahe applied H", image_clahe_h);
        waitKey(0);
        
        
        // CLAHE S
        //
        //
        Mat hsv_image_s;
        cvtColor(test_images.at(i), hsv_image_s, COLOR_BGR2HSV);
        
        // Extract the HSV channel
        vector<Mat> hsv_planes_s(3);
        split(hsv_image_s, hsv_planes_s);  // now we have the L image in lab_planes[0]
        
        // Apply the CLAHE algorithm to the H channel
        Mat dst_s;
        clahe->apply(hsv_planes_s[1], dst_s);

        // Merge the the color planes back into an HSV image
        dst_s.copyTo(hsv_planes_s[1]);
        merge(hsv_planes_s, hsv_image_s);

        // convert back to RGB
        Mat image_clahe_s;
        cvtColor(hsv_image_s, image_clahe_s, COLOR_HSV2BGR);
        
        imshow("Img" + to_string(i) + ": Clahe applied S", image_clahe_s);
        waitKey(0);
        
        
        // CLAHE V
        //
        //
        Mat hsv_image_v;
        cvtColor(test_images.at(i), hsv_image_v, COLOR_BGR2HSV);
        
        // Extract the HSV channel
        vector<Mat> hsv_planes_v(3);
        split(hsv_image_v, hsv_planes_v);  // now we have the L image in lab_planes[0]
        
        // Apply the CLAHE algorithm to the H channel
        Mat dst_v;
        clahe->apply(hsv_planes_v[2], dst_v);

        // Merge the the color planes back into an HSV image
        dst_v.copyTo(hsv_planes_v[2]);
        merge(hsv_planes_v, hsv_image_v);

        // convert back to RGB
        Mat image_clahe_v;
        cvtColor(hsv_image_v, image_clahe_v, COLOR_HSV2BGR);
        
        imshow("Img" + to_string(i) + ": Clahe applied V", image_clahe_v);
        waitKey(0);
        
        // CLAHE LAB (L)
        //
        //
        Mat lab_image_l;
        cvtColor(test_images.at(i), lab_image_l, COLOR_BGR2Lab);
        
        // Extract the Lab channel
        vector<Mat> lab_planes(3);
        split(lab_image_l, lab_planes);  // now we have the L image in lab_planes[0]
        
        // Apply the CLAHE algorithm to the L channel
        Mat dst_l;
        clahe->apply(lab_planes[0], dst_l);

        // Merge the the color planes back into an HSV image
        dst_l.copyTo(lab_planes[0]);
        merge(lab_planes, lab_image_l);

        // convert back to RGB
        Mat image_clahe_l;
        cvtColor(lab_image_l, image_clahe_l, COLOR_Lab2BGR);
        
        imshow("Img" + to_string(i) + ": Clahe applied L", image_clahe_l);
        waitKey(0);
        
        // DETAIL ENHANCE
        Mat dst;
        detailEnhance(test_images.at(i), dst);
        imshow("Img" + to_string(i) + ": Detail enhance", dst);
        waitKey(0);
        
        
        
        // HISTOGRAM EQUA V CHANNEL
        //
        //
        Mat hsv_im_v;
        cvtColor(test_images.at(i), hsv_im_v, COLOR_BGR2HSV);

        // Extract the HSV channel
        vector<Mat> hsv_planes_histeqq(3);
        split(hsv_im_v, hsv_planes_histeqq);

        // Apply the Hist eq algorithm to the H channel
        Mat dst_v_histeqq;
        equalizeHist(hsv_planes_histeqq[2], dst_v_histeqq);

        // Merge the the color planes back into an HSV image
        dst_v_histeqq.copyTo(hsv_planes_histeqq[2]);
        merge(hsv_planes_histeqq, hsv_im_v);

        // convert back to RGB
        Mat image_clahe_v_histeqq;
        cvtColor(hsv_im_v, image_clahe_v_histeqq, COLOR_HSV2BGR);

        imshow("Img" + to_string(i) + ": HISTEQ applied V", image_clahe_v_histeqq);
        waitKey(0);
    }
    
}

void Detection::Kmeans_segmentation(int k) {
    
   
    Mat data;
    Mat labels;
    vector<Point3f> centers;
    
    //kmeans(data, k, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_RANDOM_CENTERS, centers);
    double compactness = kmeans(data, k, labels, TermCriteria(TermCriteria::EPS, 10, 1.0), 3, KMEANS_RANDOM_CENTERS, centers);
    
}

