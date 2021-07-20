#include "detection.h"

using namespace cv;
using namespace std;

Dataset::Dataset(String images_path, String model_CNN_path) {
    
    // Create Boat
    std::__fs::filesystem::create_directories("../data/Boat");
    std::__fs::filesystem::create_directories("../data/No Boat");
    
    int stepSize = 50;
    int windowSize = 300;
    double scale = 1.5;

    // Read images
    vector<String> fn;
    glob(images_path, fn);
    
    // Total number of images
    int tot_numb_images = fn.size();
    
    for (int num_im = 0; num_im < tot_numb_images; num_im ++) {
        
        // Read the image
        Mat im = imread(fn.at(num_im), IMREAD_COLOR);
        
        // Number of No_boats
        int num_noboat = 0;
        
        // Number of Boats
        int num_boat = 0;
        
        while((im.rows >= windowSize) && (im.cols >= windowSize)) {

            // Sliding window
            int i = 0; // row

            int count_slidWindow = 0;
            while (i <= im.rows - windowSize) {

                for (int j = 0; j <= im.cols - windowSize; j += stepSize) {

                    Rect rect(j, i, windowSize, windowSize);
                    Mat sub_im = im.clone();
                    sub_im = sub_im(rect); // Sub-image

                    Mat draw_window = im.clone();
                    rectangle(draw_window, rect, cv::Scalar(0, 255, 0), 1);
                    imshow("Sliding window", draw_window);
                    waitKey(1);

                    // Save Boat-No_Boat images in the corresponding local directories
                    if (count_slidWindow % 2 == 0) {

                        // CNN prediction
                        dnn::Net net = dnn::readNetFromTensorflow(model_CNN_path);
                        Mat img_toNet_blob = cv::dnn::blobFromImage(sub_im);
                        net.setInput(img_toNet_blob);
                        Mat res;
                        net.forward(res);

                        // sub_im is detected as Boat
                        if (round(res.at<float>(0)) == 1) {

                            
                            imwrite("../data/Boat/img" + to_string(num_im) + "_" + to_string(num_boat) + ".jpg", sub_im);
                            num_boat ++;
                        }

                        else {

                            imwrite("../data/No Boat/img" + to_string(num_im) + "_" + to_string(num_noboat) + ".jpg", sub_im);
                            num_noboat ++;
                        }
                    }
                    
                    count_slidWindow ++;
                }
    
                i += 80;
            }
            
            Size size(im.cols/scale, im.rows/scale);
            resize(im, im, size);
        }
    }
}

// Detection Class
// constructor
Detection::Detection(String pattern, int flag) {

    if (flag == 1) {
    
        pattern = pattern + "/*.jpg";
    }
    else {
        
        pattern = pattern + "/*.png";
    }
    
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
    
void Detection::sliding_window(Mat image, int stepSize, int windowSize_rows, int windowSize_cols, String model_CNN_pb, bool use_cnn, vector<pair<Point, Point>> lines) {
    
    if (use_cnn == false) {
        
        // DILATE ABOVE
        // Find min .y
        Mat dst_dilate;
        int current_y = max(lines.at(0).first.y, lines.at(0).second.y);
        for (int k = 1; k < lines.size(); k ++) {
            
            current_y = max(current_y, lines.at(k).first.y);
            current_y = max(current_y, lines.at(k).second.y);
        }
        Mat dilate_subimg = image(Rect(0, 0, image.cols, current_y));
        Mat str_el_dilate = getStructuringElement(MORPH_RECT, Size(11,11));
        dilate(dilate_subimg, dst_dilate, str_el_dilate, Point(-1,-1), 5);
        imshow("DILATE ABOVE", dst_dilate);
        waitKey(0);
        
        // ERODE BELOW
        Mat dst_erode;
        Mat erode_subimg = image(Rect(0, current_y, image.cols, image.rows - current_y));
        Mat str_el_erode = getStructuringElement(MORPH_RECT, Size(7,7));
        erode(erode_subimg, dst_erode, str_el_erode, Point(-1,-1), 2);
        dilate(dst_erode, dst_erode, str_el_erode, Point(-1,-1), 2);
        imshow("EROSION BELOW", dst_erode);
        waitKey(0);
        
        Mat dst_erode_dilate = image.clone();

        vconcat(dst_dilate, dst_erode, dst_erode_dilate);
    //        Mat dst_erode_dilate(image, Rect(0, 0, image.cols, current_y));
    //        dst_dilate.copyTo(dst_erode_dilate);
    //        //Mat dst_erode_dilate_1 = dst_erode_dilate.clone();
    //        Mat dst_erode_dilate_1(dst_erode_dilate, Rect(0, current_y, image.cols, image.rows - current_y));
    //        dst_erode.copyTo(dst_erode_dilate_1);
        
        
        imshow("RESULTTTTTTT", dst_erode_dilate);
        waitKey(0);
    }
    
    else {
        
        int tot_num_boats_08 = 0; // Total number of detected boats (counted if prob >= 0.8), where prob is the probability for the sub-image to belong to the class "Boat" given by the CNN

        int tot_num_boats_05 = 0; // Total number of detected boats (counted if prob >= 0.5), where prob is the probability for the sub-image to belong to the class "Boat" given by the CNN
        
        Mat clustered_image = image.clone();
        Mat total_rects = image.clone();
        Mat image_allRects = image.clone();
        
        vector<Rect> a;
        Mat aaa = image.clone();
        Mat image_final_rect = image.clone();
        Mat image_final = image.clone();
        Mat final_img = image.clone();
        
        vector<double> probabilities_08;
        vector<double> probabilities_05;
        
        vector<Rect> temp_rects;

        vector<Rect> rects_08;
        vector<Rect> rects_05;
        int merged = 0;
        
        // Modify windowSize_rows, windowSize_cols if one of them is greater than the size of the image
//        if (image.rows > windowSize_rows || image.cols > windowSize_cols) {
//
//            windowSize_rows = static_cast<int>(image.rows / 8);
//            windowSize_cols = static_cast<int>(image.cols / 6);
//        }
        
            
        // Sliding window : only the first level of the pyramid is considered
        int i = 0; // row
        
        while (i <= image.rows - windowSize_rows) {
            
            for (int j = 0; j <= image.cols - windowSize_cols; j += stepSize) { // col
                
                Rect rect(j, i, windowSize_cols, windowSize_rows);
                Mat draw_window = image.clone(); // Image in which the sliding windows are drawn
                Mat window = draw_window(rect); // Sub-image to be given to the CNN for classification
            
                // Resize window (it has to be 300x300 since CNN was trained with images with such sizes)
                resize(window, window, Size(300, 300));
                
                // Load CNN model
                dnn::Net net = dnn::readNetFromTensorflow(model_CNN_pb);
                Mat img_toNet_blob = cv::dnn::blobFromImage(window);
                net.setInput(img_toNet_blob);
                Mat prob;
                net.forward(prob);
                
                // Probability >= 0.8
                if (prob.at<float>(0) >= 0.8) {

                    imshow("Boat", window);
                    waitKey(1);
                    probabilities_08.push_back(static_cast<double>(prob.at<float>(0)));
                    
                    
                    rects_08.push_back(rect);
                    tot_num_boats_08 ++;
                }
                
                else {
                    
                    imshow("No Boat", window);
                    waitKey(1);
                    
                    if (round(prob.at<float>(0)) == 1) {
                        
                        
                        rects_05.push_back(rect);
                        probabilities_05.push_back(round(prob.at<float>(0)));
                    }
                }
                
                rectangle(draw_window, rect, cv::Scalar(0, 255, 0), 3);
                imshow("Window", draw_window);
                waitKey(1);
            }
            
            i += stepSize;
        }

        if (tot_num_boats_08 == 0 && tot_num_boats_05 > 0) {
            
            for (int i = 0; i < rects_05.size(); i ++) {
                
                Rect rect_temporal = rects_05.at(i);
                String boat = "BOAT N.";
                putText(image_final, boat + to_string(i+1) , Point(rect_temporal.x,rect_temporal.y-5),FONT_HERSHEY_COMPLEX, 0.8, Scalar(0,0,0), 2, FILLED);
                rectangle(image_final, rects_05.at(i), cv::Scalar(rand() & 255, rand() & 255, rand() & 255), 3);
            }
        }
        
        else if (tot_num_boats_08 > 0) {
            
            for (int i = 0; i < rects_08.size(); i ++) {
                
                rectangle(image_allRects, rects_08.at(i), cv::Scalar(rand() & 255, rand() & 255, rand() & 255), 3);
            }
            
            vector<double> probabilties_output(rects_08.size());
            
            if (rects_08.size() > 1) {

                temp_rects = rects_08;
                
                // rect_08 now contains the seed rectangles
                groupRectangles(rects_08, 4, 0.53);

                
                // Draw seed rectangles
                for (int i = 0; i < rects_08.size(); i ++) {
                    
                
                    rectangle(clustered_image, rects_08.at(i), cv::Scalar(0, 255, 0), 3);
                }
            
                for (int index = 0; index < rects_08.size(); index ++) {
                    
                    probabilties_output.at(index) = static_cast<double>(0);
                    
                }
                
            }
    
            vector<Rect> output_rect = rect_return(temp_rects, rects_08, probabilities_08, probabilties_output);
                
            vector<Rect> output_first_rect = output_rect;
        
            groupRectangles(output_rect, 1, 0.7);
            
            if (output_rect.size() > 0) {
                
                vector<Rect> output_second_rect = rect_return(output_first_rect, output_rect, probabilities_08, probabilties_output);
                
                for (int i = 0; i < output_second_rect.size();i ++) {
                    
                    Rect rect_temporal = output_second_rect.at(i);
                    String boat = "BOAT N.";
                    putText(image_final, boat+to_string(i+1) , Point(rect_temporal.x,rect_temporal.y-5),FONT_HERSHEY_COMPLEX, 0.8, Scalar(0,0,0), 2, FILLED);
                    rectangle(image_final, output_second_rect.at(i), cv::Scalar(rand() & 255, rand() & 255, rand() & 255), 3);
                }
            }
            
            else {
                
                for (int i = 0; i < output_first_rect.size(); i ++) {
                    
                    Rect rect_temporal = output_first_rect.at(i);
                    String boat = "BOAT N.";
                    putText(image_final, boat+to_string(i+1) , Point(rect_temporal.x,rect_temporal.y-5),FONT_HERSHEY_COMPLEX, 0.8, Scalar(0,0,0), 2, FILLED);
                    rectangle(image_final, output_first_rect.at(i), cv::Scalar(rand() & 255, rand() & 255, rand() & 255), 3);
                }
            }
        }
        
        imshow("Detected boats", image_final);
        waitKey(0);
    }
}


vector<Rect> Detection::rect_return(vector<Rect> all_rects, vector<Rect> seed_rects, vector<double> input_probabilities, vector<double> &output_probabilities) {
    
    int distance;
    int change;
    
   
    // standard 1.4 (it identify how interesection is needed)
    int scaling_venice = 2;
    int scaling_kaggle = 1;
    
    vector<Rect> output_rect;
    vector<int> tot_x;
    vector<int> tot_y;
    vector<int> tot_height;
    vector<int> tot_width;
    vector<double> vec_probability;
    
    
    

    
    for (int k = 0; k < seed_rects.size(); k ++) {
        
        
        change = 0;
        
        Rect current_rect = seed_rects.at(k);
        int x_current = current_rect.x;
        int y_current = current_rect.y;
        int height_current = current_rect.height;
        int width_current = current_rect.width;
    
        cout << "X_current number: " << to_string(k) << endl;

        
        for (int h = 0; h < all_rects.size(); h ++) {
            
            //cout<< "PROBABILIT° DI STO CAZZO 000 "<<output_probabilities.at(0)<<endl;
            //cout << "X_current_all number: " << to_string(h) << endl;
            
            Rect current_all_rect = all_rects.at(h);
            int x_all_current = current_all_rect.x;
            int y_all_current = current_all_rect.y;
            int height_all_current = current_all_rect.height;
            int width_all_current = current_all_rect.width;
            
            
            cout << "COORDS CURRENT"<<endl;
            cout << "x_current " << to_string(k) <<" "<< to_string(x_current)<<endl;
            cout << "y_current " << to_string(k)<< " "<<to_string(y_current)<<endl;
            cout << "width_current " << to_string(k)<<" "<<to_string(width_current)<<endl;
            cout << "height_current " << to_string(k)<<" "<<to_string(height_current)<<endl;
            
            
            cout << "COORDS ALL CURRENT"<<endl;
            cout << "x_all_current " << to_string(h)<<" "<< to_string(x_all_current)<<endl;
            cout << "y_all_current " << to_string(h)<<" "<<to_string(y_all_current)<<endl;
            cout << "width_all_current " << to_string(h)<<" "<<to_string(width_all_current)<<endl;
            cout << "height_all_current " << to_string(h)<<" "<<to_string(height_all_current)<<endl;
            
            //if more right
            if (x_all_current <= x_current+width_current/scaling_kaggle && x_all_current >= x_current && x_all_current+width_all_current>= x_current+width_current){
                if (y_all_current <= y_current+height_current/scaling_kaggle && y_all_current >= y_current ){
                    
                    //y_all_current+height_all_current>y_current+height_current
                    
                    cout << "WORKSSSSS 1" << endl;
                    height_current = height_current + y_current;
                    width_current = width_current + (x_all_current+width_all_current-x_current-width_current);
                    height_current = height_current + (y_all_current+height_all_current-y_current-height_current);
                    output_probabilities.at(k) = output_probabilities.at(k) + input_probabilities.at(h);
                    //cout<< "PROBABILIT° DI STO CAZZO 1 "<<output_probabilities.at(k)<<endl;
                    //cout<< "PROBABILIT° DI STO CAZZO 2 "<<input_probabilities.at(h)<<endl;
                    change++;
                }
                if (y_all_current+height_all_current/scaling_kaggle>=y_current && y_all_current<=y_current){
                    cout<<"WORKSSSSS 2"<<endl;
                    
                    width_current = width_current + (x_all_current+width_all_current-x_current-width_current);
                    y_current = y_all_current;
                    output_probabilities.at(k) = output_probabilities.at(k) + input_probabilities.at(h);
                    change++;
                }
                    
            }
            //MORE LEFT
            else if (x_all_current+width_all_current/scaling_kaggle>=x_current && x_all_current<=x_current){
                //More down
                if (y_all_current <= y_current+height_current/scaling_kaggle && y_all_current >= y_current){
                    
                    //y_all_current+height_all_current>y_current+height_current
                    
                    cout<<"WORKSSSSS 3"<<endl;
                    width_current = width_current + x_current-x_all_current;
                    x_current = x_all_current;
                    height_current = y_all_current+height_all_current-y_current;
                    output_probabilities.at(k) = output_probabilities.at(k) + input_probabilities.at(h);
                    change++;
                    }
                //More up
                if (y_all_current+height_all_current/scaling_kaggle>=y_current && y_all_current<=y_current){
                    height_current = y_current-y_all_current+height_current;
                    y_current = y_all_current;
                    width_current = x_current-x_all_current + width_current;
                    x_current = x_all_current;
                    output_probabilities.at(k) = output_probabilities.at(k) + input_probabilities.at(h);
                    change++;
                    cout<<"WORKSSSSS 4"<<endl;
                }
            }
            cout<<"===================================="<<endl;
            output_probabilities.at(k) = output_probabilities.at(k)/change;
            //cout << "PROBABILITA' FINALE " << output_probabilities.at(k) <<endl;
        }
        //cout << "DIMENSIONE PROBABILITY " <<output_probabilities.size() << endl;
        Rect alone_output_rect = Rect(x_current, y_current, width_current, height_current);
        output_rect.push_back(alone_output_rect);
    }
    return output_rect;
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
        imshow("Img" + to_string(i) + ": Clahe applied on Gray image", image_clahe);
        waitKey(0);
        
        
        // CLAHE H
        //
        //
//        Mat hsv_image_h;
//        cvtColor(test_images.at(i), hsv_image_h, COLOR_BGR2HSV);
//
//        // Extract the HSV channel
//        vector<Mat> hsv_planes_h(3);
//        split(hsv_image_h, hsv_planes_h);  // now we have the L image in lab_planes[0]
//
//        // Apply the CLAHE algorithm to the H channel
//        Mat dst_h;
//        clahe->apply(hsv_planes_h[0], dst_h);
//
//        // Merge the the color planes back into an HSV image
//        dst_h.copyTo(hsv_planes_h[0]);
//        merge(hsv_planes_h, hsv_image_h);
//
//        // convert back to RGB
//        Mat image_clahe_h;
//        cvtColor(hsv_image_h, image_clahe_h, COLOR_HSV2BGR);
//
//        imshow("Img" + to_string(i) + ": Clahe applied H", image_clahe_h);
//        waitKey(0);
        
        
        // CLAHE S
        //
        //
//        Mat hsv_image_s;
//        cvtColor(test_images.at(i), hsv_image_s, COLOR_BGR2HSV);
//
//        // Extract the HSV channel
//        vector<Mat> hsv_planes_s(3);
//        split(hsv_image_s, hsv_planes_s);  // now we have the L image in lab_planes[0]
//
//        // Apply the CLAHE algorithm to the H channel
//        Mat dst_s;
//        clahe->apply(hsv_planes_s[1], dst_s);
//
//        // Merge the the color planes back into an HSV image
//        dst_s.copyTo(hsv_planes_s[1]);
//        merge(hsv_planes_s, hsv_image_s);
//
//        // convert back to RGB
//        Mat image_clahe_s;
//        cvtColor(hsv_image_s, image_clahe_s, COLOR_HSV2BGR);
//
//        imshow("Img" + to_string(i) + ": Clahe applied S", image_clahe_s);
//        waitKey(0);
        
        
        // CLAHE V
        //
        //
//        Mat hsv_image_v;
//        cvtColor(test_images.at(i), hsv_image_v, COLOR_BGR2HSV);
//
//        // Extract the HSV channel
//        vector<Mat> hsv_planes_v(3);
//        split(hsv_image_v, hsv_planes_v);  // now we have the L image in lab_planes[0]
//
//        // Apply the CLAHE algorithm to the H channel
//        Mat dst_v;
//        clahe->apply(hsv_planes_v[2], dst_v);
//
//        // Merge the the color planes back into an HSV image
//        dst_v.copyTo(hsv_planes_v[2]);
//        merge(hsv_planes_v, hsv_image_v);
//
//        // convert back to RGB
//        Mat image_clahe_v;
//        cvtColor(hsv_image_v, image_clahe_v, COLOR_HSV2BGR);
//
//        imshow("Img" + to_string(i) + ": Clahe applied V", image_clahe_v);
//        waitKey(0);
        
        // CLAHE LAB (L)
        //
        //
        Mat lab_image;
        cvtColor(test_images.at(i), lab_image, COLOR_BGR2Lab);
        
        // Extract the Lab channel
        vector<Mat> lab_planes(3);
        split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]
        
        // Apply the CLAHE algorithm to the L channel
        Mat dst_l;
        clahe->apply(lab_planes[0], dst_l);

        // Merge the the color planes back into an HSV image
        dst_l.copyTo(lab_planes[0]);
        merge(lab_planes, lab_image);

        // convert back to RGB
        Mat image_clahe_l;
        cvtColor(lab_image, image_clahe_l, COLOR_Lab2BGR);
        
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

void Detection::Kmeans_color_segmentation(int k) {
    
    for (int i=0; i<test_images.size(); i++) {
        
        if (i == 3) {
            
            Mat image_test = test_images.at(i).clone();
            cout<<"R "<<image_test.rows<<endl;
            cout<<"C "<<image_test.cols<<endl;
            GaussianBlur(image_test, image_test, Size (27,27), 0, 0);
            imshow("IMAGE", image_test);
            waitKey(0);
            
            Mat data;
            image_test.convertTo(data, CV_32F);
            
            // COLOR
            
            data = data.reshape(1, data.total());
          
            
            Mat labels;
            Mat centers;
            
            kmeans(data, k, labels, TermCriteria(TermCriteria::EPS, 10, 1.0), 3, KMEANS_RANDOM_CENTERS, centers);
            
            cout<<"size labels: "<<labels.size()<<endl;
            cout<<"First center: "<<centers.at<Vec3f>(0)<<" ,  Second center: "<<centers.at<Vec3f>(1)<<endl;
            
            //Mat p(Size(test_images.at(i).cols, test_images.at(i).rows), CV_8UC3);

            Mat color_segmented_img = data.reshape(3,data.rows);
            
            for (int j = 0; j < labels.rows; j ++) {
             
                int center_id = labels.at<int>(j);
                color_segmented_img.at<Vec3f>(j) = centers.at<Vec3f>(center_id);
            }
            color_segmented_img = color_segmented_img.reshape(3, test_images.at(i).rows);
            color_segmented_img.convertTo(color_segmented_img, CV_8UC3);
            
            // COLOR SEGMENTATION
            imshow("Color segmentation", color_segmented_img);
            waitKey(0);
            
            
            // CANNY EDGE
            // Find edges (input image of HoughLine)
            int threshold1_Canny = 90;
            int count_th_Canny = 1000;
            int threshold2_Canny = 155;
            int aperture_size_Canny = 3;
            Canny_edge canny(color_segmented_img, threshold1_Canny, threshold2_Canny, aperture_size_Canny);
            canny.doAlgorithm();
            
            cout<<"Threshold 1: " + to_string(canny.getThreshold1())<<endl;
            cout<<"Threshold 2: " + to_string(canny.getThreshold2())<<endl;
            namedWindow("Canny output (input image of HoughLine)", WINDOW_AUTOSIZE);
            createTrackbar("Threshold 2", "Canny output (input image of HoughLine)", &threshold2_Canny, count_th_Canny);
            createTrackbar("Threshold 1", "Canny output (input image of HoughLine)", &threshold1_Canny, count_th_Canny);
            imshow("Canny output (input image of HoughLine)", canny.getResult());
            createTrackbar("Threshold 1", "Canny output (input image of HoughLine)", &threshold1_Canny, count_th_Canny, canny.onCannyThreshold_1,static_cast<void*>(&canny));
            createTrackbar("Threshold 2", "Canny output (input image of HoughLine)", &threshold2_Canny, count_th_Canny, canny.onCannyThreshold_2,static_cast<void*>(&canny));
            waitKey(0);
            
            // HOUGH LINE
            // Find line which delimits the sea
            int rho_HoughLine = 1;
            int thetaden_HoughLine = 44; // denominator of theta parameter
            int votes_threshold = 100;
            HoughLine hough_l(canny.getResult(), rho_HoughLine, CV_PI/thetaden_HoughLine, votes_threshold);
            hough_l.doAlgorithm();
            
            cout<<"Theta: " + to_string(hough_l.getTheta())<<endl;
            namedWindow("HoughLine output", WINDOW_AUTOSIZE);
            createTrackbar("Theta denominator", "HoughLine output", &thetaden_HoughLine, 360);
            imshow("HoughLine output",hough_l.getResult());
            createTrackbar("Theta denominator", "HoughLine output", &thetaden_HoughLine, 360, hough_l.onHoughLineThetaden,static_cast<void*>(&hough_l));
            imshow("HoughLine output", hough_l.getResult());
            waitKey(0);
            
            // HOUGH LINE OUTPUT
            imshow("HOUGH", hough_l.getResult());
            waitKey(0);
            
            // Corrdinate pixels of the pair of points foreach line
            std::vector<pair<Point, Point>> lines = hough_l.getLines();
            cout<<"aaa"<<lines.size()<<endl;
            Mat color_segmened_lines_img = color_segmented_img.clone();
            for (int i = 0; i < lines.size(); i ++) {

                hough_l.drawStraightLine(color_segmened_lines_img, Point(lines.at(i).first) , Point(lines.at(i).second), Scalar(0, 255, 0));
                //line(color_segmented_img, Point(lines.at(i).first) , Point(lines.at(i).second), Scalar(0, 255, 0), 2, LINE_AA);
            }
            
            imshow("Lines drawn in Segmented image", color_segmened_lines_img);
            waitKey(0);
            
            // Sliding window for better segmentation output
            sliding_window(color_segmented_img, 50, 150, 150, "Not used", false, lines);
            
            
            
            
//            // COLOR + POSITION
//
//            Mat data_colpos(Size(5, test_images.at(i).cols * test_images.at(i).rows), CV_32F);
//            cout<<data_colpos.size()<<endl;
//            cout<<"Rows: "<<data_colpos.rows<<endl;
//            cout<<"Cols: "<<data_colpos.cols<<endl;
//
//            for (int j = 0; j < data_colpos.rows; j ++) {
//
//                data_colpos.at<float>(j,0) = data.at<Vec3f>(j)[0];
//                data_colpos.at<float>(j,1) = data.at<Vec3f>(j)[1];
//                data_colpos.at<float>(j,2) = data.at<Vec3f>(j)[2];
//
//                data_colpos.at<float>(j,3) = 0;
//                data_colpos.at<float>(j,4) = 0;
//            }
//
//            float r = 0;
//            float c = 0;
//            for (int j = 0; j < data_colpos.rows; j ++) {
//
//                r = floor(j/image_test.cols);
//                if (j % image_test.cols == 0) {
//
//                    c = 0;
//                }
//                data_colpos.at<float>(j,3) = r;
//                data_colpos.at<float>(j,4) = c;
//                c ++;
//            }
//
//            Mat labels_colpos;
//            Mat centers_colpos;
//
//            kmeans(data_colpos, k, labels_colpos, TermCriteria(TermCriteria::EPS, 10, 1.0), 3, KMEANS_RANDOM_CENTERS, centers_colpos);
//
//            cout<<"size labels POS: "<<labels_colpos.size()<<endl;
//            cout<<"Centers POS: "<<centers_colpos<<endl;
//
//            //Mat p(Size(test_images.at(i).cols, test_images.at(i).rows), CV_8UC3);
//
//            Mat out_colpos = data_colpos.reshape(5,data_colpos.rows);
//
//            for (int j = 0; j < labels_colpos.rows; j ++) {
//
//                int center_id = labels_colpos.at<int>(j);
//                out_colpos.at<Vec3f>(j) = centers_colpos.at<Vec3f>(center_id);
//            }
//            out_colpos = out_colpos.reshape(5, test_images.at(i).rows);
//            cout<<"OUT "<<out_colpos.channels()<<endl;
//
//            Mat out_colpos_c3;
//            Mat channels_5[5];
//            // Split the 5 channels;
//            split(out_colpos, channels_5);
//            //then merge them back
//            Mat channels_3[3] = {channels_5[0] , channels_5[1], channels_5[2]};
//
//            merge(channels_3, 3, out_colpos_c3);
//            out_colpos_c3.convertTo(out_colpos_c3, CV_8UC3);
//
//            imshow("COLOR POSITION", out_colpos_c3);
//            waitKey(0);
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



//
//
//
//
// CANNY + HOUGH
// CANNY Class
// Constructor
Canny_edge::Canny_edge(cv::Mat input_img, int th1, int th2, int apertureSize) {
    
    if (apertureSize % 2 == 0)
        apertureSize++;
    input_image_canny = input_img;
    aperture_size_Canny = apertureSize;
    threshold1_Canny = th1;
    threshold2_Canny = th2;
}

// doAlgorithm
void Canny_edge::doAlgorithm() {
    
    cv::Canny(input_image_canny, result_image_canny, threshold1_Canny, threshold2_Canny, aperture_size_Canny);
}

// setThreshold1
void Canny_edge::setThreshold1(int th1) {
    
    threshold1_Canny = th1;
}

// setThreshold2
void Canny_edge::setThreshold2(int th2) {
    
    threshold2_Canny = th2;
}

// getThreshold1
int Canny_edge::getThreshold1() {
    
    return threshold1_Canny;
}

// getThreshold2
int Canny_edge::getThreshold2() {
    
    return threshold2_Canny;
}

cv::Mat Canny_edge::getResult() {
    
    return result_image_canny;
}


// HOUGH LINE Class
// Constructor
HoughLine::HoughLine(cv::Mat input_img, int rho, double theta, int threshold){
    
    input_image_hough = input_img;
    rho_HoughLine = rho;
    theta_HoughLine = theta;
    threshold_HoughLine = threshold;
}

// doAlgorithm
void HoughLine::doAlgorithm() {
    
    vector<Vec2f> lines_theta_rho;
    cv::HoughLines(input_image_hough, lines_theta_rho, rho_HoughLine, theta_HoughLine, threshold_HoughLine);
    Mat cdst;
    cv::cvtColor(input_image_hough, cdst, COLOR_GRAY2BGR);
    vector<pair<Point, Point>> all_point_couple;
    for( size_t i = 0; i < lines_theta_rho.size(); i++ ) {
        
            float rho = lines_theta_rho[i][0], theta = lines_theta_rho[i][1];
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + 1000*(-b));
            pt1.y = cvRound(y0 + 1000*(a));
            pt2.x = cvRound(x0 - 1000*(-b));
            pt2.y = cvRound(y0 - 1000*(a));
//            pt1.x = cvRound(x0 + 100*(+b));
//            pt1.y = cvRound(y0 + 1000*(a));
//            pt2.x = cvRound(x0 - 1000*(-b));
//            pt2.y = cvRound(y0 - 1000*(a));
            //cv::line(cdst, pt1, pt2, Scalar(0,0,255), 2, LINE_AA);
            //cv::circle(cdst, pt1, 1, cv::Scalar(255,0,255), -1);
            //cv::circle(cdst, pt2, 1, cv::Scalar(255,0,255), -1);
            //cout<<"p1 "<<pt1<<endl;
            //cout<<"p2 "<<pt2<<endl;
            Point point1_x_y(pt1.x, pt1.y);
            Point point2_x_y(pt2.x, pt2.y);
    
            pair<Point, Point> point_couple;
            drawStraightLine(cdst, point1_x_y, point2_x_y, Scalar(0, 255, 0));
            //drawFullImageLine(cdst, point_couple, Scalar(0, 255, 0));
            
            point_couple.first = point1_x_y;
            point_couple.second = point2_x_y;
            all_point_couple.push_back(point_couple);
        }
    lines = all_point_couple;
    result_image_hough = cdst;
}

// Set rho for HoughLine
void HoughLine::setRho(int rho) {
    
    rho_HoughLine = rho;
}

// Set theta for HoughLine
void HoughLine::setTheta(double theta) {
    
    theta_HoughLine = theta;
}

// Set threshold for HoughLine
void HoughLine::setThreshold(int threshold) {
    
    threshold_HoughLine = threshold;
}

// Get rho
int HoughLine::getRho() {
    
    return rho_HoughLine;
}


// Get theta
double HoughLine::getTheta() {
    
    return theta_HoughLine;
}

// Get threshold
int HoughLine::getThreshold() {
    
    return threshold_HoughLine;
}

std::vector<pair<Point, Point>> HoughLine::getLines() {
    
    return lines;
}

cv::Mat HoughLine::getResult() {
    
    return result_image_hough;
}

void HoughLine::drawStraightLine(cv::Mat& img, cv::Point p1, cv::Point p2, cv::Scalar color) {
    
        Point p, q;
        // Check if the line is a vertical line because vertical lines don't have slope
        if (p1.x != p2.x)
        {
                p.x = 0;
                q.x = img.cols;
                // Slope equation (y1 - y2) / (x1 - x2)
                float m = (p1.y - p2.y) / (p1.x - p2.x);
                // Line equation:  y = mx + b
                float b = p1.y - (m * p1.x);
                p.y = m * p.x + b;
                q.y = m * q.x + b;
        }
        else
        {
                p.x = q.x = p2.x;
                p.y = 0;
                q.y = img.rows;
        }

        cv::line(img, p, q, color, 1);
}

// CANNY
void Canny_edge::onCannyThreshold_1(int pos, void *userdata) {

    Canny_edge* canny = static_cast<Canny_edge*>(userdata);
    canny->setThreshold1(pos);
    cout<<"Threshold 1: " + to_string(canny->getThreshold1())<<endl;
    cout<<"Threshold 2: " + to_string(canny->getThreshold2())<<endl;
    canny->doAlgorithm();
    imshow("Canny output (input image of HoughLine)", canny->getResult());
}

void Canny_edge::onCannyThreshold_2(int pos, void *userdata) {

    Canny_edge* canny = static_cast<Canny_edge*>(userdata);
    canny->setThreshold2(pos);
    cout<<"Threshold 1: " + to_string(canny->getThreshold1())<<endl;
    cout<<"Threshold 2: " + to_string(canny->getThreshold2())<<endl;
    canny->doAlgorithm();
    imshow("Canny output (input image of HoughLine)", canny->getResult());
}

// HOUGH LINE
void HoughLine::onHoughLineThetaden(int pos, void *userdata) {

    HoughLine* hough_l = static_cast<HoughLine*>(userdata);
    hough_l->setTheta(CV_PI/pos);
    cout<<"Theta: " + to_string(hough_l->getTheta())<<endl;
    cout<<"Theta_den: " + to_string(CV_PI/hough_l->getTheta())<<endl;
    hough_l->doAlgorithm();
    imshow("HoughLine output", hough_l->getResult());
}
