#ifndef CAFFE_UTIL_IMAGE_CROPPING_H_
#define CAFFE_UTIL_IMAGE_CROPPING_H_


#include "caffe/common.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "PolsarDataTools.hpp"

namespace caffe{

	template<typename T>
	inline std::vector<T> splitCustomString(const T & str, const T & delimiters) {
		std::vector<T> v;
		typename T::size_type start = 0;
		auto pos = str.find_first_of(delimiters, start);
		while (pos != T::npos) {
			if (pos != start) // ignore empty tokens
				v.emplace_back(str, start, pos - start);
			start = pos + 1;
			pos = str.find_first_of(delimiters, start);
		}
		if (start < str.length()) // ignore trailing delimiter
			v.emplace_back(str, start, str.length() - start); // add what's left of the string
		return v;
	}


	inline cv::Mat getImageByIndex(cv::Mat image, int row, int column, int radius) {

		int indexColumnCopyFrom;
		int indexRowCopyFrom;
		int indexColumn;
		int indexRow;
		int countWl;
		int countHl;
		int countWr;
		int countHr;

		cv::Mat entry = cv::Mat::zeros(2 * radius + 1, 2 * radius + 1, image.type());

		if ((column - radius) < 0) {
			indexColumnCopyFrom = 0;
			indexColumn = radius - column;
			countWl = column;
		}
		else {
			indexColumnCopyFrom = column - radius;
			indexColumn = 0;
			countWl = radius;
		}

		if ((row - radius) < 0) {
			indexRowCopyFrom = 0;
			indexRow = radius - row;
			countHl = row;
		}
		else {
			indexRowCopyFrom = row - radius;
			indexRow = 0;
			countHl = radius;
		}

		if ((column + radius) >= image.cols)
		{
			countWr = (image.cols - 1) - column;
		}
		else {
			countWr = radius;
		}

		if ((row + radius) >= image.rows)
		{
			countHr = image.rows - 1 - row;
		}
		else {
			countHr = radius;
		}

		cv::Mat tmp = image(cv::Rect(indexColumnCopyFrom, indexRowCopyFrom, countWl + countWr + 1, countHr + countHl + 1));

		//std::cout << "tmp.type()" << tmp.type() <<"  image.type() "<< image.type()<< std::endl;

		for (int i = indexColumn, i2 = 0; i2 < tmp.cols; i++, i2++) {
			for (int j = indexRow, j2 = 0; j2 < tmp.rows; j++, j2++) {

				if (image.type() == CV_8UC3)
				{
					entry.at<cv::Vec3b>(j, i)[0] = tmp.at<cv::Vec3b>(j2, i2)[0];
					entry.at<cv::Vec3b>(j, i)[1] = tmp.at<cv::Vec3b>(j2, i2)[1];
					entry.at<cv::Vec3b>(j, i)[2] = tmp.at<cv::Vec3b>(j2, i2)[2];
				}

				if (image.type() == CV_32FC3)
				{
					entry.at<cv::Vec3f>(j, i)[0] = tmp.at<cv::Vec3f>(j2, i2)[0];
					entry.at<cv::Vec3f>(j, i)[1] = tmp.at<cv::Vec3f>(j2, i2)[1];
					entry.at<cv::Vec3f>(j, i)[2] = tmp.at<cv::Vec3f>(j2, i2)[2];
				}

				if (image.type() == CV_32FC(6))
				{
					entry.at<cv::Vec6f>(j, i)[0] = tmp.at<cv::Vec6f>(j2, i2)[0];
					entry.at<cv::Vec6f>(j, i)[1] = tmp.at<cv::Vec6f>(j2, i2)[1];
					entry.at<cv::Vec6f>(j, i)[2] = tmp.at<cv::Vec6f>(j2, i2)[2];
					entry.at<cv::Vec6f>(j, i)[3] = tmp.at<cv::Vec6f>(j2, i2)[3];
					entry.at<cv::Vec6f>(j, i)[4] = tmp.at<cv::Vec6f>(j2, i2)[4];
					entry.at<cv::Vec6f>(j, i)[5] = tmp.at<cv::Vec6f>(j2, i2)[5];
				}
			}
		}



		//std::cout << "Image to be trained with: " << std::endl;
		//std::cout << entry << std::endl;

		//cv::namedWindow("Cropped_for_px", cv::WINDOW_NORMAL);
		//cv::imshow("Cropped_for_px", entry);
		//cv::waitKey(20);

		return entry;
	}



	inline cv::Mat getComplexImageByIndex(std::vector<cv::Mat> inputVector, int row, int column, int radius) {

		int indexColumnCopyFrom;
		int indexRowCopyFrom;
		int indexColumn;
		int indexRow;
		int countWl;
		int countHl;
		int countWr;
		int countHr;

		//cv::Mat entry = cv::Mat::zeros(2 * radius + 1, 2 * radius + 1, image.type());

		if ((column - radius) < 0) {
			indexColumnCopyFrom = 0;
			indexColumn = radius - column;
			countWl = column;
		}
		else {
			indexColumnCopyFrom = column - radius;
			indexColumn = 0;
			countWl = radius;
		}

		if ((row - radius) < 0) {
			indexRowCopyFrom = 0;
			indexRow = radius - row;
			countHl = row;
		}
		else {
			indexRowCopyFrom = row - radius;
			indexRow = 0;
			countHl = radius;
		}

		if ((column + radius) >= inputVector.at(0).cols)
		{
			countWr = (inputVector.at(0).cols - 1) - column;
		}
		else {
			countWr = radius;
		}

		if ((row + radius) >= inputVector.at(0).rows)
		{
			countHr = inputVector.at(0).rows - 1 - row;
		}
		else {
			countHr = radius;
		}

		//cv::Mat tmp = image(cv::Rect(indexColumnCopyFrom, indexRowCopyFrom, countWl + countWr + 1, countHr + countHl + 1));

		cv::Mat tmp1 = inputVector.at(0)(cv::Rect(indexColumnCopyFrom, indexRowCopyFrom, countWl + countWr + 1, countHr + countHl + 1));
		cv::Mat tmp2 = inputVector.at(1)(cv::Rect(indexColumnCopyFrom, indexRowCopyFrom, countWl + countWr + 1, countHr + countHl + 1));
		cv::Mat tmp3 = inputVector.at(2)(cv::Rect(indexColumnCopyFrom, indexRowCopyFrom, countWl + countWr + 1, countHr + countHl + 1));


		cv::Mat rgbImage(tmp3.rows, tmp3.cols, CV_32FC3, cv::Scalar(0, 0, 0));

		for (int row = 0; row < inputVector.at(0).rows; row++) {
			for (int col = 0; col < inputVector.at(0).cols; col++) {

				cv::Vec2f pxValue = tmp1.at<cv::Vec2f>(row, col);
				float magnitudeB = std::sqrt((pxValue[0] * pxValue[0]) + (pxValue[1] * pxValue[1]));

				pxValue = tmp2.at<cv::Vec2f>(row, col);
				float magnitudeG = std::sqrt((pxValue[0] * pxValue[0]) + (pxValue[1] * pxValue[1]));

				pxValue = tmp3.at<cv::Vec2f>(row, col);
				float magnitudeR = std::sqrt((pxValue[0] * pxValue[0]) + (pxValue[1] * pxValue[1]));

				rgbImage.at<cv::Vec3f>(row, col)[0] = magnitudeB;
				rgbImage.at<cv::Vec3f>(row, col)[1] = magnitudeG;
				rgbImage.at<cv::Vec3f>(row, col)[2] = magnitudeR;
			}
		}

		return rgbImage;
	}



	inline std::vector<std::string> getLabelsPathes(std::string path) {
		std::vector<std::string> labelsPath;
		std::ifstream infile(path);
		std::string line;
		while (std::getline(infile, line)) {
			labelsPath.push_back(line);
		}

		return labelsPath;
	}

	inline cv::Mat showInPseudoColorFirst5Classes(cv::Mat labeledImage) {

		cv::Mat input_3channels;
		cv::cvtColor(labeledImage, input_3channels, CV_GRAY2BGR);

		//std::cout << "Mat labeledImage type " << labeledImage.type() << std::endl;
		//std::cout << "Mat input_3channels type " << input_3channels.type() << std::endl;

		cv::Scalar color;
		for (int i = 0; i < labeledImage.rows; ++i)
		{
			for (int j = 0; j < labeledImage.cols; ++j)
			{
				if (labeledImage.at<uchar>(i, j) == 0) {
					color = cv::Scalar(0, 0, 0);
				}
				if (labeledImage.at<uchar>(i, j) == 1) {
					color = cv::Scalar(255, 0, 0);
				}
				if (labeledImage.at<uchar>(i, j) == 2) {
					color = cv::Scalar(0, 255, 0);
				}
				if (labeledImage.at<uchar>(i, j) == 3) {
					color = cv::Scalar(0, 0, 255);
				}
				if (labeledImage.at<uchar>(i, j) == 4) {
					color = cv::Scalar(255, 0, 255);
				}
				if (labeledImage.at<uchar>(i, j) == 5) {
					color = cv::Scalar(0, 255, 255);
				}

				input_3channels.at<cv::Vec3b>(i, j)[0] = color.val[0];
				input_3channels.at<cv::Vec3b>(i, j)[1] = color.val[1];
				input_3channels.at<cv::Vec3b>(i, j)[2] = color.val[2];
			}
		}

		cv::Mat show = (input_3channels.clone());
		return show;
	}
	inline cv::Mat getMainImage(std::string pathToRatFile) {
		std::vector<cv::Mat> inputVector;
	
		//amplitudes of the complex components of the sample covariance matrix C :

		loadRATOberpfaffenhofen(pathToRatFile, inputVector);

		cv::Mat rgbImage(inputVector.at(0).rows, inputVector.at(0).cols, CV_32FC3, cv::Scalar(0, 0, 0));
	

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

		return rgbImage;
	}


	inline cv::Mat getLableMat(std::string path) {

		std::vector<std::string> labelsPath = getLabelsPathes(path);
		cv::Mat lableInt;

		cv::Mat labelBinary = cv::imread(labelsPath.at(0));
		//TODO delet
		labelBinary = labelBinary(cv::Rect(1000, 500, 5, 5));

		cv::Mat outPut = cv::Mat::zeros(labelBinary.size(), CV_8UC1);

		//std::cout <<"-------------------"<<std::endl<< " Creat label MAT: " << path<< std::endl << "-------------------" << std::endl;

		for (int i = 0; i < labelsPath.size(); i++) {

			labelBinary = cv::imread(labelsPath.at(i));

			//TODO delet
			labelBinary = labelBinary(cv::Rect(1000, 500, 5, 5));


			cv::cvtColor(labelBinary, labelBinary, CV_BGR2GRAY, 1);
			//std::cout << "lableInt " << labelBinary.size() << std::endl << labelBinary << std::endl;

			cv::threshold(labelBinary, lableInt, 0, i + 1, CV_THRESH_BINARY);
			//std::cout << labelsPath.at(i) <<"  :  " << i<< std::endl << lableInt << std::endl;
			outPut = outPut + lableInt;
			//std::cout << "outPut after " << "  :  " << i << std::endl << outPut << std::endl;
			//std::cout << " ------- " << i << " ------- " <<std::endl;

		}
		//std::cout << "Final label mat:" << std::endl<< outPut << std::endl<<std::endl;

		/*cv::Mat show = outPut.clone();

		cv::normalize(show, show, 255, 0, cv::NORM_MINMAX);
		cv::cvtColor(show, show, CV_GRAY2BGR, 3);
		cv::namedWindow("Labels", cv::WINDOW_NORMAL);
		cv::imshow("Labels", show);
		cv::waitKey();*/

		std::cout << " -------- Labels mat created ---------- " << std::endl;
		/*	std::cout << (int)outPut.at<uchar>(44 , 101) << std::endl;
		std::cout << (int)outPut.at<uchar>(108, 226) << std::endl;
		std::cout << (int)outPut.at<uchar>(102, 390) << std::endl;
		std::cout << (int)outPut.at<uchar>(6080, 30) << std::endl;
		std::cout << (int)outPut.at<uchar>(96, 84) << std::endl;
		std::cout << " ------------------------ " << std::endl;*/


		cv::Mat colored = showInPseudoColorFirst5Classes(outPut);

		cv::namedWindow("LabelsColored", cv::WINDOW_NORMAL);
		cv::imshow("LabelsColored", colored);
		cv::waitKey(10);

		return outPut;

	}
}

#endif