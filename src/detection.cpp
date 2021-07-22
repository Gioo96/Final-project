#include "detection.h"

using namespace cv;
using namespace std;


//
// DATASET CLASS
//
// CONSTRUCTOR
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


//
// BASE CLASS
//
// CONSTRUCTOR
Base::Base(String pattern, int fl) {

    flag = fl;
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
            waitKey(1);
        }
    }
}


//
// DETECTION CLASS
//
// CONSTRUCTOR
Detection::Detection(String pattern, int fl) : Base(pattern, fl) {

}
    
// METHODS
// sliding_window
void Detection::sliding_window(Mat image, int stepSize, int windowSize_rows, int windowSize_cols, String model_CNN_pb) {
    
        
    int tot_num_boats_08 = 0; // Total number of detected boats (counted if prob >= 0.8), where prob is the probability for the sub-image to belong to the class "Boat" given by the CNN

    int tot_num_boats_05 = 0; // Total number of detected boats (counted if prob >= 0.5), where prob is the probability for the sub-image to belong to the class "Boat" given by the CNN
    
    Mat clustered_image = image.clone();
    Mat total_rects = image.clone();
    Mat image_allRects = image.clone();
    
    vector<Rect> a;
    Mat aaa = image.clone();
    Mat image_final_rect = image.clone();
    Mat image_seeds = image.clone();
    Mat image_final = image.clone();
    Mat final_img = image.clone();
    
    vector<double> probabilities_08;
    vector<double> probabilities_05;
    
    vector<Rect> temp_rects;

    vector<Rect> rects_08;
    vector<Rect> rects_05;
    int merged = 0;
        
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
        
        cout<<"IMAGE 05"<<endl;
        for (int i = 0; i < rects_05.size(); i ++) {
            
            Rect rect_temporal = rects_05.at(i);
            String boat = "BOAT N.";
            putText(image_final, boat + to_string(i+1) , Point(rect_temporal.x,rect_temporal.y-5),FONT_HERSHEY_COMPLEX, 0.8, Scalar(0,0,0), 2, FILLED);
            rectangle(image_final, rects_05.at(i), cv::Scalar(rand() & 255, rand() & 255, rand() & 255), 3);
        }
    }
    
    else if (tot_num_boats_08 > 0) {
        
        cout<<"IMAGE 08"<<endl;
        for (int i = 0; i < rects_08.size(); i ++) {
            
            rectangle(image_allRects, rects_08.at(i), cv::Scalar(rand() & 255, rand() & 255, rand() & 255), 3);
        }
        
        vector<double> probabilties_output(rects_08.size());
        
        if (rects_08.size() > 1) {

            temp_rects = rects_08;
            
            // rect_08 now contains the seed rectangles
            groupRectangles(rects_08, 1, 0.8);
            
            // Draw seed rectangles
            for (int i = 0; i < rects_08.size(); i ++) {
                
            
                rectangle(image_seeds, rects_08.at(i), cv::Scalar(0, 255, 0), 3);
            }
        
            for (int index = 0; index < rects_08.size(); index ++) {
                
                probabilties_output.at(index) = static_cast<double>(0);
                
            }
            
        }

        vector<Rect> output_rect = rect_return(temp_rects, rects_08, probabilities_08, probabilties_output);
            
        vector<Rect> output_first_rect = output_rect;
    
        groupRectangles(output_rect, 2, 0.53);
        
        if (output_rect.size() > 0) {
    
            cout<<"debug"<<endl;
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
    
    imshow("All rects", image_allRects);
    waitKey(0);
    
    imshow("Seeds", image_seeds);
    waitKey(0);
    
    imshow("Detected boats", image_final);
    waitKey(0);
}

// rect_return
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
    
    
    

    cout<<"AAAAA"<<seed_rects.size()<<endl;
    
    for (int k = 0; k < seed_rects.size(); k ++) {
        
        
        change = 0;
        
        Rect current_rect = seed_rects.at(k);
        int x_current = current_rect.x;
        int y_current = current_rect.y;
        int height_current = current_rect.height;
        int width_current = current_rect.width;
        
        cout<<"BBBBB"<<all_rects.size()<<endl;
        
        for (int h = 0; h < all_rects.size(); h ++) {
            
            //cout<< "PROBABILIT° DI STO CAZZO 000 "<<output_probabilities.at(0)<<endl;
            //cout << "X_current_all number: " << to_string(h) << endl;
            
            Rect current_all_rect = all_rects.at(h);
            int x_all_current = current_all_rect.x;
            int y_all_current = current_all_rect.y;
            int height_all_current = current_all_rect.height;
            int width_all_current = current_all_rect.width;
            
            
            
            //if more right
            if (x_all_current <= x_current+width_current/scaling_kaggle && x_all_current >= x_current && x_all_current+width_all_current>= x_current+width_current){
                if (y_all_current <= y_current+height_current/scaling_kaggle && y_all_current >= y_current ){
                    
                    //y_all_current+height_all_current>y_current+height_current
                    
                    height_current = height_current + y_current;
                    width_current = width_current + (x_all_current+width_all_current-x_current-width_current);
                    height_current = height_current + (y_all_current+height_all_current-y_current-height_current);
                    //output_probabilities.at(k) = output_probabilities.at(k) + input_probabilities.at(h);
                    //cout<< "PROBABILIT° DI STO CAZZO 1 "<<output_probabilities.at(k)<<endl;
                    //cout<< "PROBABILIT° DI STO CAZZO 2 "<<input_probabilities.at(h)<<endl;
                    change++;
                }
                if (y_all_current+height_all_current/scaling_kaggle>=y_current && y_all_current<=y_current){
                    
                    width_current = width_current + (x_all_current+width_all_current-x_current-width_current);
                    y_current = y_all_current;
                    //output_probabilities.at(k) = output_probabilities.at(k) + input_probabilities.at(h);
                    change++;
                }
                    
            }
            //MORE LEFT
            else if (x_all_current+width_all_current/scaling_kaggle>=x_current && x_all_current<=x_current){
                //More down
                if (y_all_current <= y_current+height_current/scaling_kaggle && y_all_current >= y_current){
                    
                    //y_all_current+height_all_current>y_current+height_current
                    
                    width_current = width_current + x_current-x_all_current;
                    x_current = x_all_current;
                    height_current = y_all_current+height_all_current-y_current;
                    //output_probabilities.at(k) = output_probabilities.at(k) + input_probabilities.at(h);
                    change++;
                    }
                //More up
                if (y_all_current+height_all_current/scaling_kaggle>=y_current && y_all_current<=y_current){
                    height_current = y_current-y_all_current+height_current;
                    y_current = y_all_current;
                    width_current = x_current-x_all_current + width_current;
                    x_current = x_all_current;
                    //output_probabilities.at(k) = output_probabilities.at(k) + input_probabilities.at(h);
                    change++;
                }
            }
            //output_probabilities.at(k) = output_probabilities.at(k)/change;
            //cout << "PROBABILITA' FINALE " << output_probabilities.at(k) <<endl;
        }
        //cout << "DIMENSIONE PROBABILITY " <<output_probabilities.size() << endl;
        Rect alone_output_rect = Rect(x_current, y_current, width_current, height_current);
        output_rect.push_back(alone_output_rect);
    }
    return output_rect;
}


//
// SEGMENTATION CLASS
//
// CONSTRUCTOR
Segmentation::Segmentation(String pattern, int fl) : Base(pattern, fl) {

}

// METHODS
// segmentation
vector<Mat> Segmentation::segmentation() {
        
    vector<Mat> segmented_images;
    
    // KAGGLE DATASET
    if (flag == 1) {
        
        for (int i = 0; i < test_images.size(); i ++) {
            
            Mat image_test = test_images.at(i).clone();
            if (i != 9) {
                
                // Smooth the image
                GaussianBlur(image_test, image_test, Size (27,27), 0, 0);
                
                // KMEANS SEGMENTATION BASED ON COLOR
                // Covert the image into CV_32F
                Mat data;
                image_test.convertTo(data, CV_32F);
                
                // Reshape data
                data = data.reshape(1, data.total());
              
                // Outputs of Kmeans
                Mat labels;
                Mat centers;
                kmeans(data, 2, labels, TermCriteria(TermCriteria::EPS, 10, 1.0), 3, KMEANS_RANDOM_CENTERS, centers);

                // color_segmented_img has size (rows*cols, 1), channels = 3
                Mat color_segmented_img = data.reshape(3,data.rows);
                
                for (int j = 0; j < labels.rows; j ++) {
                 
                    int center_id = labels.at<int>(j);
                    color_segmented_img.at<Vec3f>(j) = centers.at<Vec3f>(center_id);
                }
                
                // color_segmented_img has size (rows, cols), channels = 3
                color_segmented_img = color_segmented_img.reshape(3, test_images.at(i).rows);
                
                // Convert color_segmented_img to CV_8UC3
                color_segmented_img.convertTo(color_segmented_img, CV_8UC3);
                
                // CANNY EDGE
                // Find edges (input image of HoughLine)
                int threshold1_Canny = 90;
                int count_th_Canny = 1000;
                int threshold2_Canny = 155;
                int aperture_size_Canny = 3;
                
                Mat edges;
                Canny(color_segmented_img, edges, threshold1_Canny, threshold2_Canny, aperture_size_Canny);
                
                // HOUGH LINE
                // Find line which delimits the sea
                int rho_HoughLine = 1;
                int thetaden_HoughLine = 50; // denominator of theta parameter
                int votes_threshold = 100;
                
                HoughLine hough_l(edges, rho_HoughLine, CV_PI/thetaden_HoughLine, votes_threshold);
                hough_l.doAlgorithm();
                
                // Corrdinate pixels of the pair of points foreach line
                vector<pair<Point, Point>> lines = hough_l.getLines();
                
                // Result image after dilation
                Mat dst_dilate_above;
                
                // Result image after erosion+dilation
                Mat dst_erode_dilate_below;
                
                // Find max .y
                int current_y = max(lines.at(0).first.y, lines.at(0).second.y);
                for (int k = 1; k < lines.size(); k ++) {
                    
                    current_y = max(current_y, lines.at(k).first.y);
                    current_y = max(current_y, lines.at(k).second.y);
                }
                
                // DILATE ABOVE
                // Sub-image to dilate
                Mat dilate_subimg = color_segmented_img(Rect(0, 0, color_segmented_img.cols, current_y));
                
                // Structuring element
                Mat str_el_dilate = getStructuringElement(MORPH_RECT, Size(11,11));
                dilate(dilate_subimg, dst_dilate_above, str_el_dilate, Point(-1,-1), 15);
                
                // ERODE+DILATE BELOW
                // Sub-image to erode
                Mat erode_subimg = color_segmented_img(Rect(0, current_y, color_segmented_img.cols, color_segmented_img.rows - current_y));
                
                // Structuring element
                Mat str_el_erode_dilate = getStructuringElement(MORPH_RECT, Size(7,7));
                erode(erode_subimg, dst_erode_dilate_below, str_el_erode_dilate, Point(-1,-1), 2);
                dilate(dst_erode_dilate_below, dst_erode_dilate_below, str_el_erode_dilate, Point(-1,-1), 2);
                
                // Resulting image
                Mat dst_erode_dilate = color_segmented_img.clone();
                vconcat(dst_dilate_above, dst_erode_dilate_below, dst_erode_dilate);
                
                Mat original_segmentated;
                hconcat(image_test, dst_erode_dilate, original_segmentated);
                
                imshow("Segmentation img n. " + to_string(i), original_segmentated);
                waitKey(0);
                
                cvtColor(dst_erode_dilate, dst_erode_dilate, COLOR_BGR2GRAY);
                
                // White: sea, Black: not sea
                swap_colors(dst_erode_dilate);
                
                segmented_images.push_back(dst_erode_dilate);
            }
            
            // image.at(9) KAGGLE: OTSU
            else {
                
                // Convert to gray scale image
                Mat gray;
                cvtColor(image_test, gray, COLOR_BGR2GRAY);
                
                // Smoothing
                Mat gray_smoothed;
                GaussianBlur(gray, gray_smoothed, Size(21,21), 0);
                
                // Threshold : Otsu
                Mat gray_smoothed_otsu;
                double thresh = 0;
                double maxval = 255;
                threshold(gray_smoothed, gray_smoothed_otsu, thresh, maxval, THRESH_OTSU);
                
                // White: sea, Black: not sea
                swap_colors(gray_smoothed_otsu);
                
                // Show segmentation result
                Mat show_img;
                cvtColor(gray_smoothed_otsu, show_img, COLOR_GRAY2BGR);
                Mat original_segmentated;
                hconcat(image_test, show_img, original_segmentated);
                imshow("Segmentation img n. " + to_string(i), original_segmentated);
                waitKey(0);
                
                segmented_images.push_back(gray_smoothed_otsu);
            }
        }
    }
    
    // VENICE DATASET
    else {
        
        for (int i = 0; i < test_images.size(); i ++) {
            
            Mat image_test = test_images.at(i).clone();
            // Convert image to gray scale
            //cvtColor(image_test, image_test, COLOR_BGR2GRAY);
            
            // Histogram equalization
            //equalizeHist(image_test, image_test);
            
            // Smooth image
            GaussianBlur(image_test, image_test, Size(11,11), 0);
            
            // Show blurred image
            imshow("Gray scale smoothed image", image_test);
            waitKey(0);
            
            // EDGE SEGMENTATION
            // CANNY EDGE
            // Find edges
            int threshold1_Canny = 90;
            int count_th_Canny = 1000;
            int threshold2_Canny = 155;
            int aperture_size_Canny = 3;
            Canny_edge canny(image_test, threshold1_Canny, threshold2_Canny, aperture_size_Canny);
            canny.doAlgorithm();
            
            cout<<"Threshold 1: " + to_string(canny.getThreshold1())<<endl;
            cout<<"Threshold 2: " + to_string(canny.getThreshold2())<<endl;
            namedWindow("Canny output", WINDOW_AUTOSIZE);
            createTrackbar("Threshold 2", "Canny output", &threshold2_Canny, count_th_Canny);
            createTrackbar("Threshold 1", "Canny output", &threshold1_Canny, count_th_Canny);
            imshow("Canny output", canny.getResult());
            createTrackbar("Threshold 1", "Canny output", &threshold1_Canny, count_th_Canny, canny.onCannyThreshold_1,static_cast<void*>(&canny));
            createTrackbar("Threshold 2", "Canny output", &threshold2_Canny, count_th_Canny, canny.onCannyThreshold_2,static_cast<void*>(&canny));
            waitKey(0);
            
            // Draw contours
            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            findContours(canny.getResult(), contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

            vector<vector<Point>> contours_poly(contours.size());
            vector<Rect> boundRect(contours.size());
            vector<Point2f>centers( contours.size() );
            vector<float>radius( contours.size() );
            for( size_t i = 0; i < contours.size(); i++ )
            {
                approxPolyDP( contours[i], contours_poly[i], 3, true );
                boundRect[i] = boundingRect( contours_poly[i] );
                minEnclosingCircle( contours_poly[i], centers[i], radius[i] );
            }
            
            Mat drawing = Mat::zeros( canny.getResult().size(), CV_8UC3 );
            for( size_t i = 0; i< contours.size(); i++ )
            {
                Scalar color = Scalar(255,0,0);
                drawContours( drawing, contours_poly, (int)i, color );
                rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2 );
                //circle( drawing, centers[i], (int)radius[i], color, 2 );
            }
            cout<<"NUMBER OF BOXES: "<<boundRect.size()<<endl;
            imshow("All rects", drawing);
            waitKey(0);
            // Merged
            int perimeter = 200;
            vector<Rect> boundRect_new;
            for (int i=0; i<boundRect.size(); i++) {

                int perimeter_actual = 2*boundRect.at(i).height + 2*boundRect.at(i).width;
                if (perimeter_actual >= perimeter) {
                    
                    boundRect_new.push_back(boundRect.at(i));
                }
            }
            
//               groupRectangles(boundRect, 1, 0.1);
            Mat merged_drawing = Mat::zeros( canny.getResult().size(), CV_8UC3 );
            
            String path_CNN = "/Users/gioel/Documents/Control\ System\ Engineering/Computer\ Vision/Final\ Project/Model/model_cnn.pb";
            dnn::Net net = dnn::readNetFromTensorflow(path_CNN);
            vector<Rect> aa;
            for ( size_t i = 0; i< boundRect_new.size(); i++ )
            {
                Scalar color = Scalar(255,0,0);
                rectangle( merged_drawing, boundRect_new[i], color, 2 );
                //circle( drawing, centers[i], (int)radius[i], color, 2 );
             
                Mat window = image_test(boundRect_new.at(i));
                //cvtColor(window, window, COLOR_GRAY2BGR);
                resize(window, window, Size(300,300));
                Mat img_toNet_blob = cv::dnn::blobFromImage(window);
                net.setInput(img_toNet_blob);
                Mat prob;
                net.forward(prob);
                
                if (prob.at<float>(0) >= 0.5) {
                    
                    aa.push_back(boundRect_new.at(i));
                }
            }
            cout<<"SIZE CNN "<<aa.size()<<endl;
            imshow("Important rects", merged_drawing);
            waitKey(0);
            
            Mat imm = image_test.clone();
            for( size_t i = 0; i< aa.size(); i++ )
            {
                Scalar color = Scalar(255,0,0);
                rectangle( imm, aa[i], color, 2 );
            }
            imshow("After CNN rects", imm);
            waitKey(0);
        }
    }
    return segmented_images;
}

// pixel_accuracy
vector<double> Segmentation::pixel_accuracy(String ground_truth_segmentation_path, vector<Mat> segmentated_images) {
    
    // Kaggle
    if (flag == 1) {
        
        ground_truth_segmentation_path += "Kaggle/*.jpg";
    }
    
    else {
        
        ground_truth_segmentation_path += "Venice/*jpg";
    }
    
    vector<String> fn;
    glob(ground_truth_segmentation_path, fn);
    vector<Mat> Kaggle_gt; // Ground truth Kaggle images
    for (int i = 0; i < fn.size(); i ++) {
        
        Kaggle_gt.push_back(imread(fn.at(i)));
    }
    
    vector<double> accuracy_images;
    
    cout<<"GT: "<<Kaggle_gt.size()<<endl;
    cout<<"SEGMENT: "<<segmentated_images.size()<<endl;
    // Evaluate pixel acccuracy for any image in segmented_images
    for (int i = 0; i < segmentated_images.size(); i ++) {
        
        cvtColor(segmentated_images.at(i), segmentated_images.at(i), COLOR_GRAY2BGR);
        double intersection_pixels = 0;
        double union_pixels = 0;
        
        cout<<"ROWS: "<<segmentated_images.at(i).rows<<"   "<<Kaggle_gt.at(i).rows<<endl;
        cout<<"COLS: "<<segmentated_images.at(i).cols<<"   "<<Kaggle_gt.at(i).cols<<endl;
        
        for (int r = 0; r < Kaggle_gt.at(i).rows; r ++) {
            
            for (int c = 0; c < Kaggle_gt.at(i).cols; c ++) {
                
                if (segmentated_images.at(i).at<Vec3b>(r,c) == Kaggle_gt.at(i).at<Vec3b>(r,c) && segmentated_images.at(i).at<Vec3b>(r,c) == Vec3b(255, 255, 255)) {
                    
                    intersection_pixels ++;
                    union_pixels ++;
                }
                
                else if (segmentated_images.at(i).at<Vec3b>(r,c) == Vec3b(255, 255, 255) || Kaggle_gt.at(i).at<Vec3b>(r,c) == Vec3b(255, 255, 255)) {
                    
                    union_pixels ++;
                }
            }
        }
        
        // Evaluate metric
        double acc = intersection_pixels / union_pixels;
        accuracy_images.push_back(acc);
    }
    
    return accuracy_images;
}

// swap_colors
void Segmentation::swap_colors(Mat& image) {
    
    // Sea: white, Not sea: black
    int count = 1; // count number of classes
    int c1 = static_cast<int>(image.at<uchar>(0,0));
    int c2;
    while (count < 2) {
        for (int r = 0; r < image.rows; r ++) {
            
            for (int c = 0; c < image.cols; c ++) {
                
                if (image.at<uchar>(r,c) != c1) {
                    
                    c2 = static_cast<int>(image.at<uchar>(r,c));
                    count ++;
                }
            }
        }
    }
    
    uchar color_light, color_dark;
    if (c1 > c2) {
        
        color_light = static_cast<uchar>(c2);
        color_dark = static_cast<uchar>(c1);
    }
    else {
        
        color_light = static_cast<uchar>(c1);
        color_dark = static_cast<uchar>(c2);
    }
    
    for (int r = 0; r < image.rows; r ++) {
        
        for (int c = 0; c < image.cols; c ++) {
            
            if (image.at<uchar>(r,c) == color_light) {
                
                image.at<uchar>(r,c) = 255;
            }
            else {
                
                image.at<uchar>(r,c) = 0;
            }
        }
    }
}

// click
void Segmentation::click(Mat image) {

    Mat mask = image.clone();
    
    int count = 0;
    for (int i=0; i<image.rows; i++) {
        
        for (int j=0; j<image.cols; j++) {
            
            if (image.at<Vec3b>(i,j) == Vec3b(127,255,212)) {

                count ++;
            }
        }
    }
    cout<<"COUNT "<<count<<endl;
    
    //mask = Mat::zeros(image.rows, image.cols, CV_8UC1);
    //Mat image_ex = image;
    imshow("IMAGE", image);
    setMouseCallback("IMAGE", onMouse, static_cast<void*>(&mask));
    waitKey(0);
   
    for (int i=0; i<mask.cols; i++) {
        
        for (int j=0; j<mask.rows; j++) {
            
            if (mask.at<Vec3b>(j,i) == Vec3b(127,255,212)) {
                
                // Sea pixels: white
                mask.at<Vec3b>(j,i) = Vec3b(255,255,255);
            }
            else {
                
                // Not sea pixels: black
                mask.at<Vec3b>(j,i) = Vec3b(0,0,0);
            }
        }
    }
    imshow("Ground truth", mask);
    waitKey(0);
    imwrite("/Users/gioel/Documents/Control\ System\ Engineering/Computer\ Vision/Final\ Project/data/Ground_truth/Segmentation/Venice/img_11.jpg", mask);
}

// onMouse
void Segmentation::onMouse(int event, int x, int y, int f, void* userdata) {
    
    Mat image = *static_cast<Mat*>(userdata);
    Point point;
    
    if (event == EVENT_LBUTTONDOWN && click_pair < 2) {

        point.x = x;
        point.y = y;
        click_pair ++;
        cout<<"Point: "<<point<<endl;
        //vec_points.push_back(point);
        if (click_pair == 1) {
            
            p1 = point;
        }
        else if (click_pair == 2) {
            
            p2 = point;
        }
    }
    
    if (click_pair == 2) {
        
        //Rect rect(vec_points.at(0).x, vec_points.at(0).y, vec_points.at(1).x-vec_points.at(0).x, vec_points.at(0).x);
        //vec_pairs.push_back(pair);
        p1_p2.first = p1;
        p1_p2.second = p2;
        vec_pairs.push_back(p1_p2);
        cout<<"First: "<<p1_p2.first<<"Second: "<<p1_p2.second<<endl;
//        for (int i=0; i<vec_points.size(); i++) {
//
//            rectangle(image, vec_points.at(0), vec_points.at(1), Scalar(255,255,255), FILLED);
//        }
//        for (int i = 0; i<vec_points.size()+1; i++) {
//
//            vec_points.pop_back();
//        }
        rectangle(image, p1_p2.first, p1_p2.second, Scalar(127,255,212), FILLED);
        click_pair = 0;
    }
    
    imshow("IMAGE", image);
    waitKey(1);
}


//
// CANNY_EDGE CLASS
//
// CONSTRUCTOR
Canny_edge::Canny_edge(cv::Mat input_img, int th1, int th2, int apertureSize) {
    
    if (apertureSize % 2 == 0)
        apertureSize++;
    input_image_canny = input_img;
    aperture_size_Canny = apertureSize;
    threshold1_Canny = th1;
    threshold2_Canny = th2;
}

// METHODS
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

// getResult
cv::Mat Canny_edge::getResult() {
    
    return result_image_canny;
}

// onCannyThreshold_1
void Canny_edge::onCannyThreshold_1(int pos, void *userdata) {

    Canny_edge* canny = static_cast<Canny_edge*>(userdata);
    canny->setThreshold1(pos);
    cout<<"Threshold 1: " + to_string(canny->getThreshold1())<<endl;
    cout<<"Threshold 2: " + to_string(canny->getThreshold2())<<endl;
    canny->doAlgorithm();
    imshow("Canny output", canny->getResult());
}

// onCannyThreshold_2
void Canny_edge::onCannyThreshold_2(int pos, void *userdata) {

    Canny_edge* canny = static_cast<Canny_edge*>(userdata);
    canny->setThreshold2(pos);
    cout<<"Threshold 1: " + to_string(canny->getThreshold1())<<endl;
    cout<<"Threshold 2: " + to_string(canny->getThreshold2())<<endl;
    canny->doAlgorithm();
    imshow("Canny output", canny->getResult());
}


//
// HOUGHLINE CLASS
//
// CONSTRUCTOR
HoughLine::HoughLine(cv::Mat input_img, int rho, double theta, int threshold){
    
    input_image_hough = input_img;
    rho_HoughLine = rho;
    theta_HoughLine = theta;
    threshold_HoughLine = threshold;
}

// METHODS
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

// setRho
void HoughLine::setRho(int rho) {
    
    rho_HoughLine = rho;
}

// setTheta
void HoughLine::setTheta(double theta) {
    
    theta_HoughLine = theta;
}

// setThreshold
void HoughLine::setThreshold(int threshold) {
    
    threshold_HoughLine = threshold;
}

// getRho
int HoughLine::getRho() {
    
    return rho_HoughLine;
}


// getTheta
double HoughLine::getTheta() {
    
    return theta_HoughLine;
}

// getThreshold
int HoughLine::getThreshold() {
    
    return threshold_HoughLine;
}

// getLines
std::vector<pair<Point, Point>> HoughLine::getLines() {
    
    return lines;
}

// getResult
cv::Mat HoughLine::getResult() {
    
    return result_image_hough;
}

// drawStraightLine
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

// onHoughLineThetaden
void HoughLine::onHoughLineThetaden(int pos, void *userdata) {

    HoughLine* hough_l = static_cast<HoughLine*>(userdata);
    hough_l->setTheta(CV_PI/pos);
    cout<<"Theta: " + to_string(hough_l->getTheta())<<endl;
    cout<<"Theta_den: " + to_string(CV_PI/hough_l->getTheta())<<endl;
    hough_l->doAlgorithm();
    imshow("HoughLine output", hough_l->getResult());
}
