#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <fstream>
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"


static void loadRATOberpfaffenhofen(std::string fname, std::vector<cv::Mat>& data) {

	bool verbose = true;

	// header info
	unsigned int dim;
	unsigned int* size;
	unsigned int var;
	unsigned int type;
	unsigned int dummy;
	char info[80];

	// open file
	std::fstream file(fname.c_str(), std::ios::in | std::ios::binary);
	if (!file)
		std::cerr << "ERROR: Cannot open file: " << fname << std::endl;

	// read header
	file.read((char*)(&dim), sizeof(dim));
	dim = (dim >> 24) | ((dim << 8) & 0x00FF0000) | ((dim >> 8) & 0x0000FF00) | (dim << 24);
	size = new unsigned int[dim];
	for (int i = 0; i<dim; i++) {
		file.read((char*)(size + i), sizeof(size[i]));
		size[i] = (size[i] >> 24) | ((size[i] << 8) & 0x00FF0000) | ((size[i] >> 8) & 0x0000FF00) | (size[i] << 24);
	}
	file.read((char*)(&var), sizeof(var));
	var = (var >> 24) | ((var << 8) & 0x00FF0000) | ((var >> 8) & 0x0000FF00) | (var << 24);
	file.read((char*)(&type), sizeof(type));
	type = (type >> 24) | ((type << 8) & 0x00FF0000) | ((type >> 8) & 0x0000FF00) | (type << 24);
	file.read((char*)(&dummy), sizeof(dummy));
	file.read((char*)(&dummy), sizeof(dummy));
	file.read((char*)(&dummy), sizeof(dummy));
	file.read((char*)(&dummy), sizeof(dummy));
	file.read(info, sizeof(info));

	if (verbose) {
		std::cout << "Number of image dimensions:\t" << dim << std::endl;
		std::cout << "Image dimensions:\t";
		for (int i = 0; i<dim - 1; i++)
			std::cout << size[i] << " x ";
		std::cout << size[dim - 1] << std::endl;
		std::cout << "Data type:\t" << var << std::endl;
		std::cout << "Type:\t" << type << std::endl;
		std::cout << "Info:\t" << info << std::endl;
	}

	int nChannels = 0, dsize = 0;
	switch (var) {
	case 6: nChannels = 2; dsize = 4; break;
	default: std::cerr << "ERROR: arraytyp not recognized (wrong format?)" << std::endl;
	}
	char* buf = new char[dsize];
	char* swap = new char[dsize];

	int i, j, x, y;
	cv::Mat img, real, imag;
	switch (dim) {
	case 3:

		std::cout << "Data push_back amount of operations " << size[0] << std::endl;

		for (i = 0; i<size[0]; i++)
		{
			real = cv::Mat::zeros(size[2], size[1], CV_32FC1);
			if (nChannels == 2) {
				std::vector<cv::Mat> channels;
				imag = cv::Mat::zeros(size[2], size[1], CV_32FC1);
				channels.push_back(real);
				channels.push_back(imag);
				merge(channels, img);
			}
			else {
				img = real.clone();
			}
			data.push_back(img.clone());
		}


		for (y = 0; y<size[2]; y++)
		{

			for (x = 0; x<size[1]; x++)
			{


				for (i = 0; i<size[0]; i++)
				{
					double realVal, imagVal;


					//file.read((char*)(&buf), dsize);
					file.read(buf, dsize);

					for (int d = 0; d<dsize; d++) swap[d] = buf[dsize - d - 1];
					switch (var) {
					case 1: dsize = 1; realVal = *((char*)swap); break;	// byte
					case 2: dsize = 4; realVal = *((int*)swap); break;	// int
					case 3: dsize = 4; realVal = *((long*)swap); break;	// long
					case 4: dsize = 4; realVal = *((float*)swap); break;	// float
					case 5: dsize = 8; realVal = *((double*)swap); break;	// double
					case 6: dsize = 4;					// complex
						realVal = *((float*)swap);
						//file.read((char*)(&buf), dsize);

						file.read(buf, dsize);


						for (int d = 0; d<dsize; d++) swap[d] = buf[dsize - d - 1];
						imagVal = *((float*)swap); break;
					case 9: dsize = 8;					// dcomplex
						realVal = *((double*)swap);
						file.read((char*)(&buf), dsize);
						for (int d = 0; d<dsize; d++) swap[d] = buf[dsize - d - 1];
						imagVal = *((double*)swap); break;
					case 12: dsize = 4; realVal = *((unsigned int*)swap); break;	// uint
					case 13: dsize = 4; realVal = *((unsigned long*)swap); break;	// ulong
					case 14: dsize = 4; realVal = *((double*)swap); break;	// l64
					case 15: dsize = 4; realVal = *((double*)swap); break;	// ul64
					}
					if (nChannels != 2)
						data.at(i).at<float>(size[2] - y - 1, x) = realVal;
					else
						data.at(i).at<cv::Vec2f>(size[2] - y - 1, x) = cv::Vec2f(realVal, imagVal);
					//cvSet2D(data->at(0), size[1]-y-1, x, cur);
				}
			}
		}
		break;
	default: std::cerr << "ERROR: dim not recognized (wrong format?)" << std::endl;
	}
	delete[](size);
	delete[](buf);
	delete[](swap);
}


static void loadRAT(std::string fname, std::vector<cv::Mat>& data) {

	bool verbose = true;

	// header info
	unsigned int dim;
	unsigned int* size;
	unsigned int var;
	unsigned int type;
	unsigned int dummy;
	char info[80];

	// open file
	std::fstream file(fname.c_str(), std::ios::in | std::ios::binary);
	if (!file)
		std::cerr << "ERROR: Cannot open file: " << fname << std::endl;

	// read header
	file.read((char*)(&dim), sizeof(dim));
	dim = (dim >> 24) | ((dim << 8) & 0x00FF0000) | ((dim >> 8) & 0x0000FF00) | (dim << 24);
	size = new unsigned int[dim];
	for (int i = 0; i<dim; i++) {
		file.read((char*)(size + i), sizeof(size[i]));
		size[i] = (size[i] >> 24) | ((size[i] << 8) & 0x00FF0000) | ((size[i] >> 8) & 0x0000FF00) | (size[i] << 24);
	}
	file.read((char*)(&var), sizeof(var));
	var = (var >> 24) | ((var << 8) & 0x00FF0000) | ((var >> 8) & 0x0000FF00) | (var << 24);
	file.read((char*)(&type), sizeof(type));
	type = (type >> 24) | ((type << 8) & 0x00FF0000) | ((type >> 8) & 0x0000FF00) | (type << 24);
	file.read((char*)(&dummy), sizeof(dummy));
	file.read((char*)(&dummy), sizeof(dummy));
	file.read((char*)(&dummy), sizeof(dummy));
	file.read((char*)(&dummy), sizeof(dummy));
	file.read(info, sizeof(info));

	if (verbose) {
		std::cout << "Number of image dimensions:\t" << dim << std::endl;
		std::cout << "Image dimensions:\t";
		for (int i = 0; i<dim - 1; i++)
			std::cout << size[i] << " x ";
		std::cout << size[dim - 1] << std::endl;
		std::cout << "Data type:\t" << var << std::endl;
		std::cout << "Type:\t" << type << std::endl;
		std::cout << "Info:\t" << info << std::endl;
	}

	int nChannels = 0, dsize = 0;
	switch (var) {
	case 1: nChannels = 1; dsize = 1; break;
	case 2: nChannels = 1; dsize = 4; break;
	case 3: nChannels = 1; dsize = 4; break;
	case 4: nChannels = 1; dsize = 4; break;
	case 5: nChannels = 1; dsize = 8; break;
	case 12: nChannels = 1; dsize = 4; break;
	case 13: nChannels = 1; dsize = 4; break;
	case 14: nChannels = 1; dsize = 8; break;
	case 15: nChannels = 1; dsize = 8; break;
	case 6: nChannels = 2; dsize = 4; break;
	case 9: nChannels = 2; dsize = 8; break;
	default: std::cerr << "ERROR: arraytyp not recognized (wrong format?)" << std::endl;
	}
	char* buf = new char[dsize];
	char* swap = new char[dsize];

	int i, j, x, y;
	cv::Mat img, real, imag;
	switch (dim) {
	case 2:
		real = cv::Mat::zeros(size[1], size[0], CV_32FC1);
		imag = cv::Mat::zeros(size[1], size[0], CV_32FC1);
		for (y = 0; y<size[1]; y++) {
			for (x = 0; x<size[0]; x++) {
				double realVal, imagVal;
				file.read((char*)(&buf), dsize);
				for (int d = 0; d<dsize; d++) swap[d] = buf[dsize - d - 1];
				switch (var) {
				case 1: dsize = 1; realVal = *((char*)swap); break;	// byte
				case 2: dsize = 4; realVal = *((int*)swap); break;	// int
				case 3: dsize = 4; realVal = *((long*)swap); break;	// long
				case 4: dsize = 4; realVal = *((float*)swap); break;	// float
				case 5: dsize = 8; realVal = *((double*)swap); break;	// double
				case 6: dsize = 4;					// complex
					realVal = *((float*)swap);
					file.read((char*)(&buf), dsize);
					for (int d = 0; d<dsize; d++) swap[d] = buf[dsize - d - 1];
					imagVal = *((float*)swap); break;
				case 9: dsize = 8;					// dcomplex
					realVal = *((double*)swap);
					file.read((char*)(&buf), dsize);
					for (int d = 0; d<dsize; d++) swap[d] = buf[dsize - d - 1];
					imagVal = *((double*)swap); break;
				case 12: dsize = 4; realVal = *((unsigned int*)swap); break;	// uint
				case 13: dsize = 4; realVal = *((unsigned long*)swap); break;	// ulong
				case 14: dsize = 4; realVal = *((double*)swap); break;	// l64
				case 15: dsize = 4; realVal = *((double*)swap); break;	// ul64
				}
				real.at<float>(size[1] - y - 1, x) = realVal;
				if (nChannels == 2)
					imag.at<float>(size[1] - y - 1, x) = imagVal;
				//cvSet2D(data->at(0), size[1]-y-1, x, cur);
			}
		}
		if (nChannels == 2) {
			std::vector<cv::Mat> channels;
			channels.push_back(real);
			channels.push_back(imag);
			merge(channels, img);
		}
		else
			img = real.clone();
		data.push_back(img);
		break;
	case 3: for (i = 0; i<size[0]; i++) {
		real = cv::Mat::zeros(size[2], size[1], CV_32FC1);
		if (nChannels == 2) {
			std::vector<cv::Mat> channels;
			imag = cv::Mat::zeros(size[2], size[1], CV_32FC1);
			channels.push_back(real);
			channels.push_back(imag);
			merge(channels, img);
		}
		else
			img = real.clone();
		data.push_back(img.clone());
	}
			for (y = 0; y<size[2]; y++) {
				for (x = 0; x<size[1]; x++) {
					for (i = 0; i<size[0]; i++) {
						double realVal, imagVal;


						//file.read((char*)(&buf), dsize);
						file.read(buf, dsize);



						for (int d = 0; d<dsize; d++) swap[d] = buf[dsize - d - 1];
						switch (var) {
						case 1: dsize = 1; realVal = *((char*)swap); break;	// byte
						case 2: dsize = 4; realVal = *((int*)swap); break;	// int
						case 3: dsize = 4; realVal = *((long*)swap); break;	// long
						case 4: dsize = 4; realVal = *((float*)swap); break;	// float
						case 5: dsize = 8; realVal = *((double*)swap); break;	// double
						case 6: dsize = 4;					// complex
							realVal = *((float*)swap);
							//file.read((char*)(&buf), dsize);

							file.read(buf, dsize);


							for (int d = 0; d<dsize; d++) swap[d] = buf[dsize - d - 1];
							imagVal = *((float*)swap); break;
						case 9: dsize = 8;					// dcomplex
							realVal = *((double*)swap);
							file.read((char*)(&buf), dsize);
							for (int d = 0; d<dsize; d++) swap[d] = buf[dsize - d - 1];
							imagVal = *((double*)swap); break;
						case 12: dsize = 4; realVal = *((unsigned int*)swap); break;	// uint
						case 13: dsize = 4; realVal = *((unsigned long*)swap); break;	// ulong
						case 14: dsize = 4; realVal = *((double*)swap); break;	// l64
						case 15: dsize = 4; realVal = *((double*)swap); break;	// ul64
						}
						if (nChannels != 2)
							data.at(i).at<float>(size[2] - y - 1, x) = realVal;
						else
							data.at(i).at<cv::Vec2f>(size[2] - y - 1, x) = cv::Vec2f(realVal, imagVal);
						//cvSet2D(data->at(0), size[1]-y-1, x, cur);
					}
				}
			}
			break;
	case 4: for (i = 0; i<size[0]; i++) {
		for (j = 0; j<size[1]; j++) {
			real = cv::Mat::zeros(size[3], size[2], CV_32FC1);
			if (nChannels == 2) {
				imag = cv::Mat::zeros(size[3], size[2], CV_32FC1);
				std::vector<cv::Mat> channels;
				channels.push_back(real);
				channels.push_back(imag);
				merge(channels, img);
			}
			else
				img = real.clone();
			data.push_back(img.clone());
		}
	}

			for (y = 0; y<size[3]; y++) {
				for (x = 0; x<size[2]; x++) {
					for (j = 0; j<size[0]; j++) {
						for (i = 0; i<size[1]; i++) {
							double realVal, imagVal;
							file.read((char*)(&buf), dsize);
							for (int d = 0; d<dsize; d++) swap[d] = buf[dsize - d - 1];
							switch (var) {
							case 1: dsize = 1; realVal = *((char*)swap); break;	// byte
							case 2: dsize = 4; realVal = *((int*)swap); break;	// int
							case 3: dsize = 4; realVal = *((long*)swap); break;	// long
							case 4: dsize = 4; realVal = *((float*)swap); break;	// float
							case 5: dsize = 8; realVal = *((double*)swap); break;	// double
							case 6: dsize = 4;					// complex
								realVal = *((float*)swap);
								file.read((char*)(&buf), dsize);
								for (int d = 0; d<dsize; d++) swap[d] = buf[dsize - d - 1];
								imagVal = *((float*)swap); break;
							case 9: dsize = 8;					// dcomplex
								realVal = *((double*)swap);
								file.read((char*)(&buf), dsize);
								for (int d = 0; d<dsize; d++) swap[d] = buf[dsize - d - 1];
								imagVal = *((double*)swap); break;
							case 12: dsize = 4; realVal = *((unsigned int*)swap); break;	// uint
							case 13: dsize = 4; realVal = *((unsigned long*)swap); break;	// ulong
							case 14: dsize = 4; realVal = *((double*)swap); break;	// l64
							case 15: dsize = 4; realVal = *((double*)swap); break;	// ul64
							}
							if (nChannels != 2)
								data.at(j*size[1] + i).at<float>(size[3] - y - 1, x) = realVal;
							else
								data.at(j*size[1] + i).at<cv::Vec2f>(size[3] - y - 1, x) = cv::Vec2f(realVal, imagVal);
						}
					}
					//real.at<float>(size[3]-y-1, x) = realVal;
					//if (nChannels == 2)
					//imag.at<float>(size[3]-y-1, x) = imagVal;
					//cvSet2D(data->at(0), size[1]-y-1, x, cur);
				}
			}
			break;
	}
	delete[](size);
	delete[](buf);
	delete[](swap);
}

#endif
