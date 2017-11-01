#include "config.h"

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
#include "caffe/util/image_cropping_utils.hpp"

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

void create6ChennelsImage() {
	std::vector<cv::Mat> inputVector;
	loadRATOberpfaffenhofen("C:\\Users\\Galya\\Desktop\\Master\\Data\\Oberpfaffenhofen\\oph_lexi.rat", inputVector);

	cv::Mat rgbImage(inputVector.at(0).rows, inputVector.at(0).cols, CV_32FC3, cv::Scalar(0, 0, 0));
	cv::Mat source = cv::Mat::zeros(inputVector.at(0).rows, inputVector.at(0).cols, CV_32FC(6));

	//cv::Vec2f(realVal, imagVal)
	for (int row = 0; row < inputVector.at(0).rows; row++) {
		for (int col = 0; col < inputVector.at(0).cols; col++) {

			cv::Vec2f pxValue = inputVector.at(0).at<cv::Vec2f>(row, col);
			float magnitudeB = std::sqrt((pxValue[0] * pxValue[0]) + (pxValue[1] * pxValue[1]));

			pxValue = inputVector.at(1).at<cv::Vec2f>(row, col);
			float magnitudeG = std::sqrt((pxValue[0] * pxValue[0]) + (pxValue[1] * pxValue[1]));

			pxValue = inputVector.at(2).at<cv::Vec2f>(row, col);
			float magnitudeR = std::sqrt((pxValue[0] * pxValue[0]) + (pxValue[1] * pxValue[1]));

			rgbImage.at<cv::Vec3f>(row, col)[0] = magnitudeB;
			rgbImage.at<cv::Vec3f>(row, col)[1] = magnitudeG;
			rgbImage.at<cv::Vec3f>(row, col)[2] = magnitudeR;
		}
	}

	//cv::Vec2f(realVal, imagVal)
	for (int row = 0; row < inputVector.at(0).rows; row++) {
		for (int col = 0; col < inputVector.at(0).cols; col++) {

			cv::Vec2f pxValueB = inputVector.at(0).at<cv::Vec2f>(row, col);
			cv::Vec2f pxValueG = inputVector.at(1).at<cv::Vec2f>(row, col);
			cv::Vec2f pxValueR = inputVector.at(2).at<cv::Vec2f>(row, col);

			source.at<cv::Vec6f>(row, col)[0] = pxValueB[0];
			source.at<cv::Vec6f>(row, col)[1] = pxValueB[1];
			source.at<cv::Vec6f>(row, col)[2] = pxValueG[0];
			source.at<cv::Vec6f>(row, col)[3] = pxValueG[1];
			source.at<cv::Vec6f>(row, col)[4] = pxValueR[0];
			source.at<cv::Vec6f>(row, col)[5] = pxValueR[1];
		}
	}


	cv::Mat test_source = cv::Mat::zeros(inputVector.at(0).rows, inputVector.at(0).cols, CV_32FC3);

	for (int row = 0; row < inputVector.at(0).rows; row++) {
		for (int col = 0; col < inputVector.at(0).cols; col++) {
			float magnitudeB = std::sqrt((source.at<cv::Vec6f>(row, col)[0] * source.at<cv::Vec6f>(row, col)[0]) + (source.at<cv::Vec6f>(row, col)[1] * source.at<cv::Vec6f>(row, col)[1]));
			float magnitudeG = std::sqrt((source.at<cv::Vec6f>(row, col)[2] * source.at<cv::Vec6f>(row, col)[2]) + (source.at<cv::Vec6f>(row, col)[3] * source.at<cv::Vec6f>(row, col)[3]));
			float magnitudeR = std::sqrt((source.at<cv::Vec6f>(row, col)[4] * source.at<cv::Vec6f>(row, col)[4]) + (source.at<cv::Vec6f>(row, col)[5] * source.at<cv::Vec6f>(row, col)[5]));

			test_source.at<cv::Vec3f>(row, col)[0] = magnitudeB;
			test_source.at<cv::Vec3f>(row, col)[1] = magnitudeG;
			test_source.at<cv::Vec3f>(row, col)[2] = magnitudeR;
		}
	}


	cv::namedWindow("RGB by magnitude", cv::WINDOW_NORMAL);
	cv::imshow("RGB by magnitude", rgbImage);

	cv::namedWindow("test_source", cv::WINDOW_NORMAL);
	cv::imshow("test_source", rgbImage);

	cv::waitKey();

	/*cv::Mat rgbImage_CV_8U;
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
	cv::waitKey();*/

	std::cout << "Output complex   :  channels  " << source.channels() << " ,size" << source.size() << " ,type  " << source.type() << std::endl;
	std::cout << "Output magnitude :  channels  " << rgbImage.channels() << " ,size" << rgbImage.size() << " ,type  " << rgbImage.type() << std::endl;
	std::cout << "Input            :  channels  " << inputVector.at(0).channels() << " ,size" << inputVector.at(0).size() << " ,type  " << inputVector.at(0).type() << std::endl;
}

void createImageForTest() {

	cv::Scalar blue = cv::Scalar(255, 0, 0);
	cv::Scalar green = cv::Scalar(0, 255, 0);
	cv::Scalar red = cv::Scalar(0, 0, 255);
	cv::Scalar yellow = cv::Scalar(0, 255, 255);
	cv::Scalar purple = cv::Scalar(255, 0, 255);

	/*cv::Mat blueImage(3, 3, CV_32FC3, blue);
	cv::Mat greenImage(3, 3, CV_32FC3, green);
	cv::Mat redImage(3, 3, CV_32FC3, red);
	cv::Mat yellowImage(3, 3, CV_32FC3, yellow);
	cv::Mat purpleImage(3, 3, CV_32FC3, purple);*/


	cv::Mat testImage(9, 9, CV_32FC3, cv::Scalar(0, 0, 255));

	for (int i = 0; i < 9; i++) {
		testImage.at<cv::Vec3f>(0, i)[0] = 255;
		testImage.at<cv::Vec3f>(0, i)[1] = 0;
		testImage.at<cv::Vec3f>(0, i)[2] = 0;
	}

	for (int i = 0; i < 9; i++) {
		testImage.at<cv::Vec3f>(1, i)[0] = 0;
		testImage.at<cv::Vec3f>(1, i)[1] = 255;
		testImage.at<cv::Vec3f>(1, i)[2] = 0;
	}

	for (int i = 0; i < 9; i++) {
		testImage.at<cv::Vec3f>(8, i)[0] = 255;
		testImage.at<cv::Vec3f>(8, i)[1] = 255;
		testImage.at<cv::Vec3f>(8, i)[2] = 0;
	}


	for (int i = 2; i < 7; i++) {
		testImage.at<cv::Vec3f>(4, i)[0] = 0;
		testImage.at<cv::Vec3f>(4, i)[1] = 255;
		testImage.at<cv::Vec3f>(4, i)[2] = 255;
	}

	cv::namedWindow("BGR", cv::WINDOW_NORMAL);
	cv::imshow("BGR", testImage);
	cv::waitKey();

	cv::Mat source = cv::Mat::zeros(testImage.rows, testImage.cols, CV_32FC(6));
	//cv::Vec2f(realVal, imagVal)
	for (int row = 0; row < testImage.rows; row++) {
		for (int col = 0; col < testImage.cols; col++) {
			source.at<cv::Vec6f>(row, col)[0] = testImage.at<cv::Vec3f>(row, col)[0];
			source.at<cv::Vec6f>(row, col)[1] = 0;
			source.at<cv::Vec6f>(row, col)[2] = testImage.at<cv::Vec3f>(row, col)[1];
			source.at<cv::Vec6f>(row, col)[3] = 0;
			source.at<cv::Vec6f>(row, col)[4] = testImage.at<cv::Vec3f>(row, col)[2];
			source.at<cv::Vec6f>(row, col)[5] = 0;
		}
	}

	cv::FileStorage fs("C:/Data/color_test/complex.xml", cv::FileStorage::WRITE);
	fs << "matrix" << source;
	fs.release();


	cv::Mat test_source = cv::Mat::zeros(testImage.rows, testImage.cols, CV_32FC3);

	for (int row = 0; row < testImage.rows; row++) {
		for (int col = 0; col < testImage.cols; col++) {
			float magnitudeB = std::sqrt((source.at<cv::Vec6f>(row, col)[0] * source.at<cv::Vec6f>(row, col)[0]) + (source.at<cv::Vec6f>(row, col)[1] * source.at<cv::Vec6f>(row, col)[1]));
			float magnitudeG = std::sqrt((source.at<cv::Vec6f>(row, col)[2] * source.at<cv::Vec6f>(row, col)[2]) + (source.at<cv::Vec6f>(row, col)[3] * source.at<cv::Vec6f>(row, col)[3]));
			float magnitudeR = std::sqrt((source.at<cv::Vec6f>(row, col)[4] * source.at<cv::Vec6f>(row, col)[4]) + (source.at<cv::Vec6f>(row, col)[5] * source.at<cv::Vec6f>(row, col)[5]));

			test_source.at<cv::Vec3f>(row, col)[0] = magnitudeB;
			test_source.at<cv::Vec3f>(row, col)[1] = magnitudeG;
			test_source.at<cv::Vec3f>(row, col)[2] = magnitudeR;
		}
	}

	cv::namedWindow("COMPLEX", cv::WINDOW_NORMAL);
	cv::imshow("COMPLEX", test_source);
	cv::waitKey();

	//cv::imwrite("C:/Data/color_test/test.png", testImage);

	//cv::imwrite("C:/Data/color_test/blue2.png", blueImage);
	//cv::imwrite("C:/Data/color_test/green2.png", greenImage);
	//cv::imwrite("C:/Data/color_test/red2.png", redImage);
	//cv::imwrite("C:/Data/color_test/yellow2.png", yellowImage);
	//cv::imwrite("C:/Data/color_test/purple2.png", purpleImage);
}


int main(int argc, char** argv) {
	createImageForTest();
}
#else
int main(int argc, char** argv) {
	LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV