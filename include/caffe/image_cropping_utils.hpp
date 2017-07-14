#ifndef CAFFE_UTIL_IMAGE_CROPPING_H_
#define CAFFE_UTIL_IMAGE_CROPPING_H_


#include "caffe/common.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "caffe/PolsarDataTools.hpp"

namespace caffe{
	template<typename T>std::vector<T> splitCustomString(const T & str, const T & delimiters);
	cv::Mat getImageByIndex(cv::Mat image, int row, int column, int radius);
	std::vector<std::string> getLabelsPathes(std::string path);
	cv::Mat showInPseudoColorFirst5Classes(cv::Mat labeledImage);
	cv::Mat getMainImage(std::string pathToRatFile);
	cv::Mat getLableMat(std::string path);
}

#endif