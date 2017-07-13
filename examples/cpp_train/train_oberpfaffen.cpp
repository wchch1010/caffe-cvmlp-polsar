#include "config.h"
#ifdef TRAIN_OBERPFAFFEN

#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "caffe/PolsarDataTools.hpp"

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;


int main(int argc, char** argv) {

	std::vector<cv::Mat> inputVector;
	//amplitudes of the complex components of the sample covariance matrix C :
	loadRATOberpfaffenhofen("C:\\Users\\Galya\\Desktop\\Master\\Data\\Oberpfaffenhofen\\oph_lexi.rat", inputVector);

	cv::Mat rgbImage(inputVector.at(0).rows, inputVector.at(0).cols, CV_32FC3, cv::Scalar(0, 0, 0));

	//cv::Vec2f(realVal, imagVal)
	for (int row = 0; row < inputVector.at(0).rows; row++) {
		for (int col = 0; col < inputVector.at(0).cols; col++) {

			cv::Vec2f pxValue  = inputVector.at(0).at<cv::Vec2f>(row, col);
			float magnitudeB   = std::sqrt((pxValue[0] * pxValue[0]) + (pxValue[1] * pxValue[1]));

			pxValue			   = inputVector.at(1).at<cv::Vec2f>(row, col);
			float magnitudeG   = std::sqrt((pxValue[0] * pxValue[0]) + (pxValue[1] * pxValue[1]));

			pxValue			   = inputVector.at(2).at<cv::Vec2f>(row, col);
			float magnitudeR   = std::sqrt((pxValue[0] * pxValue[0]) + (pxValue[1] * pxValue[1]));

			rgbImage.at<cv::Vec3f>(row, col)[0] = magnitudeB;
			rgbImage.at<cv::Vec3f>(row, col)[1] = magnitudeG;
			rgbImage.at<cv::Vec3f>(row, col)[2] = magnitudeR;
		}
	}

	cv::namedWindow("RGB by magnitude", cv::WINDOW_NORMAL);
	cv::imshow("RGB by magnitude", rgbImage);

	cv::Mat rgbImage_CV_8U;
	rgbImage.convertTo(rgbImage_CV_8U, CV_8U, 255, 0);

	cv::FileStorage fsWrite("C:/Data/magnitude_7_7/full_images/magnitude_matrix.xml", cv::FileStorage::WRITE);
	fsWrite << "Image" << rgbImage_CV_8U;
	fsWrite.release();


	cv::imwrite("C:/Data/magnitude_7_7/full_images/magnitude_image.png",rgbImage_CV_8U);

	cv::Mat rgbImageLoaded(inputVector.at(0).rows, inputVector.at(0).cols, CV_32FC3, cv::Scalar(0, 0, 0));

	cv::FileStorage fsRead("C:/Data/magnitude_7_7/full_images/magnitude_matrix.xml", cv::FileStorage::READ);
	fsRead["Image"] >> rgbImageLoaded;
	fsRead.release();

	cv::namedWindow("RGB by magnitude loaded", cv::WINDOW_NORMAL);
	cv::imshow("RGB by magnitude loaded", rgbImage);
	cv::waitKey();

	std::cout << "Output :  channels  " << rgbImage.channels() << " ,size" << rgbImage.size() << " ,type  " << rgbImage.type() << std::endl;
	std::cout << "Input  :  channels  " << inputVector.at(0).channels() << " ,size" << inputVector.at(0).size() << " ,type  " << inputVector.at(0).type() << std::endl;
}
#else
int main(int argc, char** argv) {
	LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
#endif 