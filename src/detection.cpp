#include "detection.h"

using namespace cv;
using namespace std;

// PanoramicImage Class
// constructor
Detection::Detection(String pattern) {

    pattern = pattern + "/*.jpg";
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

void Detection::pyramid(double scale, Mat image, int stepSize, int windowSize_rows, int windowSize_cols, String model_path_pb) {
    
    int tot_num_boats = 0; // Total number of detected boats

    Mat clustered_image = image.clone();
    Mat total_rects = image.clone();
    
    vector<int> probabilities;
    vector<Rect> boat_rects;

    vector<Rect> rects;
    while (image.rows >= windowSize_rows && image.cols >= windowSize_cols) {
        
        // Sliding window
        int i = 0; // row
        while (i <= image.rows - windowSize_rows) {
            
            for (int j = 0; j <= image.cols - windowSize_cols; j += stepSize) { // col
    
                Rect rect(j, i, windowSize_cols, windowSize_rows);
                Mat draw_window = image.clone();
                Mat window = draw_window(rect);
            
                // Resize window (it has to be 300x300 since CNN was trained with images with such sizes)
                resize(window, window, Size(300, 300));
                // Load CNN model
                dnn::Net net = dnn::readNetFromTensorflow(model_path_pb);
                Mat img_toNet_blob = cv::dnn::blobFromImage(window);
                net.setInput(img_toNet_blob);
                Mat prob;
                net.forward(prob);
                cout<<"Probability: "<<prob<<endl;
                
                if (prob.at<float>(0) >= 0.8) {

                    imshow("Boat", window);
                    waitKey(1);
                    rectangle(draw_window, rect, cv::Scalar(0, 255, 0), 3);
                    probabilities.push_back(round(prob.at<float>(0)));
                    // Save sub-images detected as boats into rects, rect_cluster
                    //rect_cluster_08.push_back(rect);
                    rects.push_back(rect);
                    //
                    tot_num_boats ++;
                    imshow("RECT", window);
                    waitKey(0);
                }
                
                else {

                    imshow("No Boat", window);
                    waitKey(1);
                    rectangle(draw_window, rect, cv::Scalar(0, 255, 0), 3);
                    probabilities.push_back(round(prob.at<float>(0)));
                    if (round(prob.at<float>(0)) == 1) {
                        
                        boat_rects.push_back(rect);
                    }
                    
                }
                
                imshow("Window", draw_window);
                waitKey(1);
            
            }
            i += stepSize;
        }
    
        Size size(image.cols/scale, image.rows/scale);
        resize(image, image, size);
    }
    if (tot_num_boats <= 1) {
        
        for (int i=0; i<boat_rects.size(); i ++) {
            
            rectangle(total_rects, boat_rects.at(i), cv::Scalar(rand() & 255, rand() & 255, rand() & 255), 3);
        }
        groupRectangles(boat_rects, 2, 0.3);
        
        for (int i=0; i<boat_rects.size(); i++) {
    
            rectangle(clustered_image, boat_rects.at(i), cv::Scalar(0, 255, 0), 3);
        }
        
    }
    else {
        
        for (int i=0; i< rects.size(); i++) {
            
            rectangle(total_rects, rects.at(i), cv::Scalar(rand() & 255, rand() & 255, rand() & 255), 3);
        }
        
        groupRectangles(rects, 3, 0.3);
        for (int i=0; i<rects.size(); i++) {
    
            rectangle(clustered_image, rects.at(i), cv::Scalar(0, 255, 0), 3);
        }
        //groupRectangles(rects, 1, 0.3);
    }
    cout<<tot_num_boats<<endl;
    
    imshow("Result merging", clustered_image);
    imshow("Result total", total_rects);
    waitKey(0);
    
    //groupRectangles(rect_cluster, 1, 0.3);

//    for (int i=0; i< rect_cluster.size(); i++) {
//
//        rectangle(clustered_image, rect_cluster.at(i), cv::Scalar(0, 255, 0), 3);
//    }
}




//
//
//
void Detection::dataset(String path) {
    
    int stepSize = 50;
    //Kaggle: windowSize = 200
    //venice: windowSize = 240
    int windowSize = 240;
    double scale = 1.5;

    int num_images = 50;
    vector<String> fn;
    glob(path, fn);
    int tot_numb_images = fn.size();
    cout<<tot_numb_images<<endl;
    
    for (int num_im = 0; num_im < tot_numb_images; num_im ++) {
        
        Mat im = imread(fn.at(num_im), IMREAD_COLOR);
        
        int num_noboat = 0;
        int num_boat = 0;
        
        while((im.rows >= windowSize) && (im.cols >= windowSize)) {

            // Sliding window
            int i = 0; // row

            int count_slidWindow = 0;
            while (i <= im.rows - windowSize) {

                for (int j = 0; j <= im.cols - windowSize; j += stepSize) {

                    String model_path_pb = "/Users/gioel/Documents/Control\ System\ Engineering/Computer\ Vision/Final\ Project/Model/boat_class.pb";
                    //Kaggle: Rect rect(j, i, 200, windowSize);
                    Rect rect(j, i, 200, windowSize);
                    Mat sub_im = im.clone();
                    sub_im = sub_im(rect);

//                        imshow("Window", sub_im);
//                        waitKey(1);

                    Mat draw_window = im.clone();
                    rectangle(draw_window, rect, cv::Scalar(0, 255, 0), 1);
                    imshow("Window", draw_window);
                    waitKey(1);

                    if (count_slidWindow % 2 == 0) {

                        resize(sub_im, sub_im, Size(300, 300));
                        dnn::Net net = dnn::readNetFromTensorflow(model_path_pb);
                        Mat img_toNet_blob = cv::dnn::blobFromImage(sub_im);
                        net.setInput(img_toNet_blob);
                        Mat res;
                        net.forward(res);

                        if (round(res.at<float>(0)) == 1) {

                            imshow("Boat", sub_im);
                            waitKey(1);
                            imwrite("/Users/gioel/Desktop/Dataset_tot/Boat/img" + to_string(num_im) + "_" + to_string(num_boat) + ".jpg", sub_im);
                            num_boat ++;
                            //rectangle(draw_window, rect, cv::Scalar(0, 255, 0), 1);
                        }

                        if (round(res.at<float>(0)) == 0) {

                            imshow("No Boat", sub_im);
                            waitKey(1);
                            imwrite("/Users/gioel/Desktop/Dataset_tot/No Boat/img" + to_string(num_im) + "_" + to_string(num_noboat) + ".jpg", sub_im);
                            num_noboat ++;
                            //rectangle(draw_window, rect, cv::Scalar(0, 255, 0), 1);
                        }
                    }
                    count_slidWindow ++;
                }
                // Kaggle: i+= 50
                i += 80;
            }
            
            Size size(im.cols/scale, im.rows/scale);
            resize(im, im, size);
            cout<<num_im<<endl;
        }
    }
}

//void Detection::pyramid(double scale, Mat image, int stepSize, int windowSize, String model_path_pb) {
//
//    Mat immm = image.clone();
//    Mat tot = image.clone();
//
//    int c = 0;
//    int im_c = 0;
//
//    vector<Rect> rect_cluster;
//    vector<Rect> rects;
//    while (image.rows >= windowSize && image.cols >= windowSize) {
//
//        // Sliding window
//        int i = 0; // row
//        while (i <= image.rows - windowSize) {
//
//            for (int j = 0; j <= image.cols - windowSize; j += stepSize) {
//
//                Rect rect(j, i, windowSize, windowSize);
//                Mat draw_window = image.clone();
//                Mat window = draw_window(rect);
////                if (j == 0) {
////
////                    imshow("okk", aa);
////                    waitKey(0);
////                }
//
//                //Mat img_toNet = Mat(draw_window,rect); // Img to give to Net
//                resize(window, window, Size(300, 300));
//                dnn::Net net = dnn::readNetFromTensorflow(model_path_pb);
//                Mat img_toNet_blob = cv::dnn::blobFromImage(window);
//                net.setInput(img_toNet_blob);
//                Mat res;
//                net.forward(res);
//                cout<<res<<endl;
//
//                if (res.at<float>(0) >= 0.8) {
//
//                    imshow("Boat", window);
//                    waitKey(1);
//                    rectangle(draw_window, rect, cv::Scalar(0, 255, 0), 1);
//                    rect_cluster.push_back(rect);
//                    rects.push_back(rect);
//                }
//
//                if (res.at<float>(0) < 0.8) {
//
//                    imshow("No Boat", window);
//                    waitKey(1);
//                    rectangle(draw_window, rect, cv::Scalar(0, 255, 0), 1);
//                }
//
//                if (c < 250 && c % 5 == 0) {
//
//                    imwrite("/Users/gioel/Desktop/Test_noboat/img" + to_string(9) + "_" + to_string(im_c) + ".jpg", window);
//                    im_c ++;
//                }
//                imshow("Window", draw_window);
//                waitKey(0);
//
//                c ++;
//            }
//            i += stepSize;
//        }
//
//        Size size(image.cols/scale, image.rows/scale);
//        resize(image, image, size);
//    }
//    groupRectangles(rect_cluster, 2, 0.3);
//
//    for (int i=0; i< rect_cluster.size(); i++) {
//
//        rectangle(immm, rect_cluster.at(i), cv::Scalar(0, 255, 0), 1);
//    }
//
//    for (int i=0; i< rects.size(); i++) {
//
//        rectangle(tot, rects.at(i), cv::Scalar(rand() & 255, rand() & 255, rand() & 255), 1);
//    }
//    imshow("Result merging", immm);
//
//
//    imshow("Result total", tot);
//    waitKey(0);
//}

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
    
    for (int i=0; i<test_images.size(); i++) {
        
        Mat data;
        test_images.at(i).convertTo(data, CV_32F);
        
        if (i == 6) {
            
            // COLOR
            cout<<"Size data before: "<<data.size()<<endl;
            data = data.reshape(1, data.total());
            cout<<"Size data after: "<<data.size()<<endl;
            
            Mat labels;
            Mat centers;
            
            kmeans(data, k, labels, TermCriteria(TermCriteria::EPS, 10, 1.0), 3, KMEANS_RANDOM_CENTERS, centers);
            
            cout<<"size labels: "<<labels.size()<<endl;
            cout<<"First center: "<<centers.at<Vec3f>(0)<<" ,  Second center: "<<centers.at<Vec3f>(1)<<endl;
            
            //Mat p(Size(test_images.at(i).cols, test_images.at(i).rows), CV_8UC3);

            Mat out = data.reshape(3,data.rows);
            
            for (int j = 0; j < labels.rows; j ++) {
             
                int center_id = labels.at<int>(j);
                out.at<Vec3f>(j) = centers.at<Vec3f>(center_id);
            }
            out = out.reshape(3, test_images.at(i).rows);
            out.convertTo(out, CV_8UC3);
            
            imshow("Color segmentation", out);
            waitKey(0);
            
            imshow("Image 0", test_images.at(i));
            waitKey(0);
            
            // COLOR + POSITION
            
            Mat data_colpos(Size(5, test_images.at(i).cols * test_images.at(i).rows), CV_32F);
            cout<<data_colpos.size()<<endl;
            cout<<"Rows: "<<data_colpos.rows<<endl;
            cout<<"Cols: "<<data_colpos.cols<<endl;
            
            cout<<data.at<Vec3f>(0)[2]<<endl;
            for (int j = 0; j < data_colpos.rows; j ++) {
            
                
                
                data_colpos.at<float>(j,0) = data.at<Vec3f>(j)[0];
                data_colpos.at<float>(j,1) = data.at<Vec3f>(j)[1];
                data_colpos.at<float>(j,2) = data.at<Vec3f>(j)[2];
                //cout<<data.at<uchar>(j,k)<<endl;
            }
            
            for (int j = 0; j < test_images.at(i).cols; j ++) {
                
                for (int k = 0; k < test_images.at(i).rows; k ++) {
                    
                    data_colpos.at<float>(j+k,3) = k;
                    data_colpos.at<float>(j+k,4) = j;
                }
            }
        
        }
        
        
    }

}

void Detection::otsu_segmentation() {
    
    for (int i = 0; i < test_images.size(); i ++) {
        
        // Convert image in gray scale
        Mat gray_img;
        cvtColor(test_images.at(i), gray_img, COLOR_BGR2GRAY);
        
        // Smooth the image with a gaussian filter
        Mat gray_img_smoothed;
        GaussianBlur(gray_img, gray_img_smoothed, Size(3,3), 0);
        if (i == 6) {
            
            Mat otsu_img;
            double th = threshold(gray_img_smoothed, otsu_img, 0, 255, THRESH_OTSU);
            cout<<"threshold "<<th<<endl;
            
            imshow("color", test_images.at(i));
            waitKey(0);
            imshow("gray", gray_img);
            waitKey(0);
            imshow("gray smoothed", gray_img_smoothed);
            waitKey(0);
            imshow("otsu", otsu_img);
            waitKey(0);
        }
    }
    
    
}

void Detection::showAndCompute_sift() {
    
    for (int i = 0; i < test_images.size(); i ++) {
        
        if (i == 6) {
            
            // COMPUTE KEYPOINTS & DESCRIPTORS
            Ptr<SIFT> sift = SIFT::create();
            
            vector<Point2f> coords_keypoints;
            vector<KeyPoint> list_keypoints;
            Mat list_descriptors;
                
            sift->detectAndCompute(test_images.at(i), Mat(), list_keypoints, list_descriptors);
            
            // Visualize keypoints
            //vector<Scalar> color = {Scalar(0,0,255), Scalar(255,0,0), Scalar(0,255,0), Scalar(50,50,50)};
            
            Mat img_keypoints = test_images.at(i).clone();
            drawKeypoints(img_keypoints, list_keypoints, img_keypoints);
            
            
            imshow("Visualizekeypoints", img_keypoints);
            waitKey(0);

//            // GET GOOD KEYPOINTS
//            vector<vector<KeyPoint>> allgood_keypoints;
//            Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2, true);
//            for (int i = 0; i < images_dataset.size(); i ++) {
//
//                // Get all matches of first video frame and image 'i' of the dataset
//                vector<DMatch> matches;
//                matcher->match(descriptors_frame, list_descriptors_dataset.at(i), matches);
//
//                // Get minimum distance between descriptors
//                double min_distance = 200;
//                for (int j = 0; j < matches.size(); j++) {
//
//                    //if (matches.at(j).distance < min_distance && matches.at(j).distance > 0) {
//                    if (matches.at(j).distance < min_distance) {
//
//                        min_distance = matches.at(j).distance;
//                    }
//                }
//
//                // Refine matches
//                vector<DMatch> refined_matches;
//                for (int j = 0; j < matches.size(); j++) {
//
//                    if (matches.at(j).distance < 3 * min_distance) {
//
//                        refined_matches.push_back(matches.at(j));
//                    }
//                }
//
//                // Good matches (RANSAC)
//                vector<Point2f> scene, object;
//                for (int j = 0; j < refined_matches.size(); j++) {
//
//                    // Refined matches between the dataset image and first framed img (Pixel cords)
//                    scene.push_back(keypoints_frame.at(refined_matches.at(j).queryIdx).pt);
//                    object.push_back(list_keypoints_dataset.at(i).at(refined_matches.at(j).trainIdx).pt);
//                }
//
//                Mat H_single;
//                vector<int> mask; // mask will contain 0 if the match is wrong
//                H_single = findHomography(object, scene, mask, RANSAC);
//                H.push_back(H_single);
//
//                // Good matches
//                vector<DMatch> good_matches;
//                for (int j=0; j<mask.size(); j++) {
//
//                    if (mask.at(j) != 0) {
//
//                        good_matches.push_back(refined_matches.at(j));
//                    }
//                }
//
//                // Good keypoints
//                vector<KeyPoint> good_keypoints;
//                vector<Point2f> coords_keypoints;
//
//
//                for (int j=0; j<good_matches.size(); j++) {
//
//                    good_keypoints.push_back(keypoints_frame.at(good_matches.at(j).queryIdx));
//                    coords_keypoints.push_back(good_keypoints.at(j).pt);
//
//                    //cout <<" Punto"+to_string(j)<< good_keypoints.at(j).pt << endl;
//                }
//
//                allcoords_keypoints.push_back(coords_keypoints);
//                allgood_keypoints.push_back(good_keypoints);
//            }
            
//            // Visualize good keypoints
//            vector<Scalar> color = {Scalar(0,0,255), Scalar(255,0,0), Scalar(0,255,0), Scalar(50,50,50)};
//
//            img_keypoints = images_frame.at(0).clone();
//
//            for (int i=0; i<images_dataset.size(); i++) {
//
//                drawKeypoints(img_keypoints, allgood_keypoints.at(i), img_keypoints, color.at(i));
//            }
//
//            imshow("Visualize good keypoints", img_keypoints);
//            waitKey(0);
        }
    }
}

vector<Rect> Detection::selective_search(Mat image, String method) {
    
    Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> sel_search = cv::ximgproc::segmentation::createSelectiveSearchSegmentation();
    sel_search->setBaseImage(image);
    
    if (method == "fast") {
        
        sel_search->switchToSelectiveSearchFast(); // Fast but less accurate version of selective search
    }
    else {
        
        sel_search->switchToSelectiveSearchQuality(); // Slower but more accurate version
    }
    
    vector<Rect> rects;
    sel_search->process(rects);
    return rects;
}
