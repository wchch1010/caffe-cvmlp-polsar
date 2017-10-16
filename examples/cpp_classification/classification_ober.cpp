//#include "config.h"
//#ifdef CLASSIFICATION_OBERPFAFFEN
//
//#include <caffe/caffe.hpp>
//#ifdef USE_OPENCV
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#endif  // USE_OPENCV
//#include <algorithm>
//#include <iosfwd>
//#include <memory>
//#include <string>
//#include <utility>
//#include <vector>
//#include "caffe/util/image_cropping_utils.hpp"
//
//#ifdef USE_OPENCV
//using namespace caffe;  // NOLINT(build/namespaces)
//using std::string;
//
///* Pair (label, confidence) representing a prediction. */
//typedef std::pair<string, float> Prediction;
//
//const int cropFactorTest = 2;
//const int cropFactorTrain = 2;
//const int IMAGE_RADIUS = 20;
//
//class Classifier {
//public:
//	Classifier(const string& model_file,
//		const string& trained_file,
//		const string& label_file);
//
//	std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);
//
//	cv::Scalar  getColorByLabelName(std::string);
//private:
//	void SetMean(const string& mean_file);
//
//	std::vector<float> Predict(const cv::Mat& img);
//
//	void WrapInputLayer(std::vector<cv::Mat>* input_channels);
//
//	void Preprocess(const cv::Mat& img,
//		std::vector<cv::Mat>* input_channels);
//
//private:
//	shared_ptr<Net<float> > net_;
//	cv::Size input_geometry_;
//	int num_channels_;
//	cv::Mat mean_;
//	std::vector<string> labels_;
//	std::vector<cv::Scalar> m_colors;
//};
//
//Classifier::Classifier(const string& model_file,
//	const string& trained_file,
//	const string& label_file) {
//#ifdef CPU_ONLY
//	Caffe::set_mode(Caffe::CPU);
//#else
//	Caffe::set_mode(Caffe::GPU);
//#endif
//
//
//	m_colors.push_back(cv::Scalar(0, 0, 0));
//	m_colors.push_back(cv::Scalar(255, 0, 0));
//	m_colors.push_back(cv::Scalar(0, 255, 0));
//	m_colors.push_back(cv::Scalar(0, 0, 255));
//	m_colors.push_back(cv::Scalar(255, 0, 255));
//	m_colors.push_back(cv::Scalar(0, 255, 255));
//	m_colors.push_back(cv::Scalar(255, 255, 255));
//	m_colors.push_back(cv::Scalar(100, 0, 100));
//
//	/* Load the network. */
//	net_.reset(new Net<float>(model_file, TEST));
//	net_->CopyTrainedLayersFrom(trained_file);
//
//	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
//	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
//
//	Blob<float>* input_layer = net_->input_blobs()[0];
//	num_channels_ = input_layer->channels();
//	CHECK(num_channels_ == 3 || num_channels_ == 1)
//		<< "Input layer should have 1 or 3 channels.";
//	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
//
//	/* Load the binaryproto mean file. */
//	//SetMean(mean_file);
//
//	/* Load labels. */
//	std::ifstream labels(label_file.c_str());
//	CHECK(labels) << "Unable to open labels file " << label_file;
//	string line;
//	while (std::getline(labels, line))
//		labels_.push_back(string(line));
//
//	Blob<float>* output_layer = net_->output_blobs()[0];
//	CHECK_EQ(labels_.size(), output_layer->channels())
//		<< "Number of labels is different from the output layer dimension.";
//}
//
//static bool PairCompare(const std::pair<float, int>& lhs,
//	const std::pair<float, int>& rhs) {
//	return lhs.first > rhs.first;
//}
//
//
//
///* Return the indices of the top N values of vector v. */
//static std::vector<int> Argmax(const std::vector<float>& v, int N) {
//	std::vector<std::pair<float, int> > pairs;
//	for (size_t i = 0; i < v.size(); ++i)
//		pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
//	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);
//
//	std::vector<int> result;
//	for (int i = 0; i < N; ++i)
//		result.push_back(pairs[i].second);
//	return result;
//}
//
//cv::Scalar Classifier::getColorByLabelName(std::string label) {
//
//	if (this->labels_.empty()) {
//		std::cout << "Error!! Empty labels array" << std::endl;
//	}
//	else {
//		for (size_t i = 0; i < labels_.size(); i++)
//		{
//			if (labels_.at(i) == label) {
//				return m_colors.at(i);
//			}
//		}
//	}
//
//	return NULL;
//}
//
///* Return the top N predictions. */
//std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
//	std::vector<float> output = Predict(img);
//
//	N = std::min<int>(labels_.size(), N);
//	std::vector<int> maxN = Argmax(output, N);
//	std::vector<Prediction> predictions;
//	for (int i = 0; i < N; ++i) {
//		int idx = maxN[i];
//		predictions.push_back(std::make_pair(labels_[idx], output[idx]));
//	}
//
//	return predictions;
//}
//
///* Load the mean file in binaryproto format. */
//void Classifier::SetMean(const string& mean_file) {
//	BlobProto blob_proto;
//	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
//
//	/* Convert from BlobProto to Blob<float> */
//	Blob<float> mean_blob;
//	mean_blob.FromProto(blob_proto);
//	CHECK_EQ(mean_blob.channels(), num_channels_)
//		<< "Number of channels of mean file doesn't match input layer.";
//
//	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
//	std::vector<cv::Mat> channels;
//	float* data = mean_blob.mutable_cpu_data();
//	for (int i = 0; i < num_channels_; ++i) {
//		/* Extract an individual channel. */
//		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
//		channels.push_back(channel);
//		data += mean_blob.height() * mean_blob.width();
//	}
//
//	/* Merge the separate channels into a single image. */
//	cv::Mat mean;
//	cv::merge(channels, mean);
//
//	/* Compute the global mean pixel value and create a mean image
//	* filled with this value. */
//	cv::Scalar channel_mean = cv::mean(mean);
//	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
//}
//
//std::vector<float> Classifier::Predict(const cv::Mat& img) {
//	Blob<float>* input_layer = net_->input_blobs()[0];
//	input_layer->Reshape(1, num_channels_,
//		input_geometry_.height, input_geometry_.width);
//	/* Forward dimension change to all layers. */
//	net_->Reshape();
//
//	std::vector<cv::Mat> input_channels;
//	WrapInputLayer(&input_channels);
//
//	Preprocess(img, &input_channels);
//
//	net_->Forward();
//
//	/* Copy the output layer to a std::vector */
//	Blob<float>* output_layer = net_->output_blobs()[0];
//	const float* begin = output_layer->cpu_data();
//	const float* end = begin + output_layer->channels();
//	return std::vector<float>(begin, end);
//}
//
///* Wrap the input layer of the network in separate cv::Mat objects
//* (one per channel). This way we save one memcpy operation and we
//* don't need to rely on cudaMemcpy2D. The last preprocessing
//* operation will write the separate channels directly to the input
//* layer. */
//void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
//	Blob<float>* input_layer = net_->input_blobs()[0];
//
//	int width = input_layer->width();
//	int height = input_layer->height();
//	float* input_data = input_layer->mutable_cpu_data();
//	for (int i = 0; i < input_layer->channels(); ++i) {
//		cv::Mat channel(height, width, CV_32FC1, input_data);
//		input_channels->push_back(channel);
//		input_data += width * height;
//	}
//}
//
//void Classifier::Preprocess(const cv::Mat& img,
//	std::vector<cv::Mat>* input_channels) {
//	/* Convert the input image to the input image format of the network. */
//	cv::Mat sample;
//	if (img.channels() == 3 && num_channels_ == 1)
//		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
//	else if (img.channels() == 4 && num_channels_ == 1)
//		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
//	else if (img.channels() == 4 && num_channels_ == 3)
//		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
//	else if (img.channels() == 1 && num_channels_ == 3)
//		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
//	else
//		sample = img;
//
//	cv::Mat sample_resized;
//	if (sample.size() != input_geometry_)
//		cv::resize(sample, sample_resized, input_geometry_);
//	else
//		sample_resized = sample;
//
//	cv::Mat sample_float;
//	if (num_channels_ == 3)
//		sample_resized.convertTo(sample_float, CV_32FC3);
//	else
//		sample_resized.convertTo(sample_float, CV_32FC1);
//
//	//Galya
//	cv::Mat sample_normalized = sample_float;
//	//cv::subtract(sample_float, mean_, sample_normalized);
//
//	/* This operation will write the separate BGR planes directly to the
//	* input layer of the network because it is wrapped by the cv::Mat
//	* objects in input_channels. */
//	cv::split(sample_normalized, *input_channels);
//
//	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
//		== net_->input_blobs()[0]->cpu_data())
//		<< "Input channels are not wrapping the input layer of the network.";
//}
//template<typename T>
//std::vector<T>
//split(const T & str, const T & delimiters) {
//	std::vector<T> v;
//	typename T::size_type start = 0;
//	auto pos = str.find_first_of(delimiters, start);
//	while (pos != T::npos) {
//		if (pos != start) // ignore empty tokens
//			v.emplace_back(str, start, pos - start);
//		start = pos + 1;
//		pos = str.find_first_of(delimiters, start);
//	}
//	if (start < str.length()) // ignore trailing delimiter
//		v.emplace_back(str, start, str.length() - start); // add what's left of the string
//	return v;
//}
//
////"C:/Users/Galya/Desktop/Master/Code/caffe-windows/examples/Oberpfaffenhofen/data/labels_list.txt"
//
////std::vector<std::string> getLabelsPathes(std::string path) {
////	std::vector<std::string> labelsPath;
////	std::ifstream infile(path);
////	std::string line;
////	while (std::getline(infile, line)) {
////		labelsPath.push_back(line);
////	}
////
////	return labelsPath;
////}
//
//
////cv::Mat getLableMat(std::string path) {
////
////	std::vector<std::string> labelsPath = getLabelsPathes(path);
////
////	cv::Mat labelBinary;
////	cv::Mat lableInt;
////
////	cv::Mat outPut = cv::Mat::zeros(cv::imread(labelsPath.at(0)).size(), CV_8UC1);
////
////	std::cout << "-------------------" << std::endl << " Creat label MAT: " << path << std::endl << "-------------------" << std::endl;
////
////	for (int i = 0; i < labelsPath.size(); i++) {
////
////		labelBinary = cv::imread(labelsPath.at(i));
////		cv::cvtColor(labelBinary, labelBinary, CV_BGR2GRAY, 1);
////		//std::cout << "lableInt " << labelBinary.size() << std::endl << labelBinary << std::endl;
////
////		cv::threshold(labelBinary, lableInt, 0, i + 1, CV_THRESH_BINARY);
////		//std::cout << labelsPath.at(i) << "  :  " << i << std::endl << lableInt << std::endl;
////		outPut = outPut + lableInt;
////		//std::cout << "outPut size " << outPut.size() << std::endl;
////		//std::cout << " ------- " << i << " ------- " <<std::endl;
////		//std::cout << (int)outPut.at<uchar>(100, 803) << std::endl;
////		//std::cout << (int)outPut.at<uchar>(6473, 1361) << std::endl;
////		//std::cout << (int)outPut.at<uchar>(21, 152) << std::endl;
////		//std::cout << (int)outPut.at<uchar>(6257, 737) << std::endl;
////		//std::cout << (int)outPut.at<uchar>(95, 777) << std::endl;
////		//std::cout << " ------------------------ " << std::endl;
////	}
////	
////	return outPut;
////}
//
//void predictOnInput(cv::Mat image, Classifier& classifier) {
//	
//	int radius = 7;
//	cv::Mat entry;
//
//
//	cv::namedWindow("Original", cv::WINDOW_NORMAL);
//	cv::imshow("Original", image);
//	cv::waitKey(20);
//
//	cv::Mat result = cv::Mat::zeros(image.size(), image.type());
//
//	for (int col = 0; col < image.cols; col++) {
//		for (int row = 0; row < image.rows; row++) {
//			entry = getImageByIndex(image, row, col, IMAGE_RADIUS);
//
//
//			/*cv::namedWindow("entry", cv::WINDOW_NORMAL);
//			cv::imshow("entry", entry);
//			cv::waitKey();*/
//
//			/*cv::namedWindow("px: " + std::to_string(row) + std::to_string(col), cv::WINDOW_NORMAL);
//			cv::imshow("px: " + std::to_string(row) + std::to_string(col), entry);
//			cv::waitKey(15);*/
//
//			std::vector<Prediction> predictions = classifier.Classify(entry);
//
//			/* Print the top N predictions. */
//			for (size_t i = 0; i < predictions.size(); ++i) {
//				Prediction p = predictions[i];
//				std::cout << std::fixed << std::setprecision(4) << p.second << " - \"" << p.first << "\"" << std::endl;
//			}
//
//			cv::Scalar color = classifier.getColorByLabelName(predictions[0].first);
//
//			result.at<cv::Vec3b>(row, col)[0] = color.val[0];
//			result.at<cv::Vec3b>(row, col)[1] = color.val[1];
//			result.at<cv::Vec3b>(row, col)[2] = color.val[2];
//
//			std::cout << "   ---------------------   " << std::endl;
//		}
//	}
//
//	cv::namedWindow("Result", cv::WINDOW_NORMAL);
//	cv::imshow("Result", result);
//	cv::waitKey();
//}
//
////
////cv::Mat getLableMat(std::string path) {
////
////	std::vector<std::string> labelsPath = getLabelsPathes(path);
////
////	cv::Mat labelBinary;
////	cv::Mat lableInt;
////
////	cv::Mat outPut = cv::Mat::zeros(cv::imread(labelsPath.at(0)).size(), CV_8UC1);
////
////	//std::cout <<"-------------------"<<std::endl<< " Creat label MAT: " << path<< std::endl << "-------------------" << std::endl;
////
////
////
////	for (int i = 0; i < labelsPath.size(); i++) {
////
////		labelBinary = cv::imread(labelsPath.at(i));
////		cv::cvtColor(labelBinary, labelBinary, CV_BGR2GRAY, 1);
////		//std::cout << "lableInt " << labelBinary.size() << std::endl << labelBinary << std::endl;
////
////		cv::threshold(labelBinary, lableInt, 0, i + 1, CV_THRESH_BINARY);
////		//std::cout << labelsPath.at(i) <<"  :  " << i<< std::endl << lableInt << std::endl;
////		outPut = outPut + lableInt;
////		//std::cout << "outPut size " << outPut.size() << std::endl;
////		//std::cout << " ------- " << i << " ------- " <<std::endl;
////
////	}
////	//std::cout << "Final label mat:" << std::endl<< outPut << std::endl<<std::endl;
////
////	cv::Mat show = outPut.clone();
////
////	cv::normalize(show, show, 255, 0, cv::NORM_MINMAX);
////	cv::cvtColor(show, show, CV_GRAY2RGB, 3);
////	cv::namedWindow("Labels", cv::WINDOW_NORMAL);
////	cv::imshow("Labels", show);
////	cv::waitKey(10);
////
////	return outPut;
////
////}
//
////cv::Mat showInPseudoColorFirst5Classes(cv::Mat labeledImage) {
////
////	cv::Mat input_3channels;
////	cv::cvtColor(labeledImage, input_3channels, CV_GRAY2RGB);
////
////	//std::cout << "Mat labeledImage type " << labeledImage.type() << std::endl;
////	//std::cout << "Mat input_3channels type " << input_3channels.type() << std::endl;
////
////	//std::cout << "Mat labeledImage  :" << std::endl<< labeledImage << std::endl;
////	//std::cout << "Mat input_3channels:  " << std::endl << input_3channels << std::endl;
////
////	cv::Scalar color;
////	for (int i = 0; i < labeledImage.rows; ++i)
////	{
////		for (int j = 0; j < labeledImage.cols; ++j)
////		{
////			if (labeledImage.at<uchar>(i, j) == 0) {
////				//Nothing 
////				color = cv::Scalar(0, 0, 0);
////			}
////			if (labeledImage.at<uchar>(i, j) == 1) {
////				//city == red
////				color = cv::Scalar(0, 0, 255);
////				
////			}
////			if (labeledImage.at<uchar>(i, j) == 2) {
////				//field == yellow
////				color = cv::Scalar(0, 255, 255);
////			
////			}
////			if (labeledImage.at<uchar>(i, j) == 3) {
////				//forest = dark green
////				color = cv::Scalar(15, 78, 5);
////				
////			}
////			if (labeledImage.at<uchar>(i, j) == 4) {
////				//grass =  green
////				color = cv::Scalar(0, 255, 0);
////			}
////			if (labeledImage.at<uchar>(i, j) == 5) {
////				//street =  blue
////				color = cv::Scalar(255, 0, 0);
////			}
////
////			input_3channels.at<cv::Vec3b>(i, j)[0] = color.val[0];
////			input_3channels.at<cv::Vec3b>(i, j)[1] = color.val[1];
////			input_3channels.at<cv::Vec3b>(i, j)[2] = color.val[2];
////		}
////	}
////
////	cv::Mat show = (input_3channels.clone());
////	return show;
////}
//
//
////const int cropFactorTest = 8;
////const int cropFactorTrain = 16;
//
//void cropImages(std::string folder) {
//
//	cv::Mat city_interTest = cv::imread("C:/Data/RGB/full_images/city_inter.png");
//	cv::Mat city_interTrain = cv::imread("C:/Data/RGB/full_images/city_inter.png");
//
//	cv::Mat city_interTestOutput = city_interTest(cv::Rect(0, 0, floor(city_interTest.cols / cropFactorTest), city_interTest.rows - 1));
//	cv::Mat city_interTrainOutput = city_interTrain(cv::Rect(ceil(city_interTrain.cols / cropFactorTrain), 0, floor(city_interTest.cols / cropFactorTrain) - 1, city_interTest.rows - 1));
//
//	cv::namedWindow("test", cv::WINDOW_NORMAL);
//	cv::imshow("test", city_interTestOutput);
//	cv::namedWindow("train", cv::WINDOW_NORMAL);
//	cv::imshow("train", city_interTrainOutput);
//	cv::waitKey();
//
//	imwrite("C:/Data/RGB/train_small/test/city_inter.png", city_interTestOutput);
//	imwrite("C:/Data/RGB/train_small/train/city_inter.png", city_interTrainOutput);
//
//	//-----------------------------------------------------------------------------
//
//	cv::Mat field_interTest = cv::imread("C:/Data/RGB/full_images/field_inter.png");
//	cv::Mat field_interTrain = cv::imread("C:/Data/RGB/full_images/field_inter.png");
//	cv::Mat field_interTestOutput = field_interTest(cv::Rect(0, 0, floor(field_interTest.cols / cropFactorTest), field_interTest.rows - 1));
//	cv::Mat field_interTrainOutput = field_interTrain(cv::Rect(ceil(field_interTrain.cols / cropFactorTrain), 0, floor(field_interTrain.cols / cropFactorTrain) - 1, field_interTrain.rows - 1));
//
//	cv::namedWindow("test", cv::WINDOW_NORMAL);
//	cv::imshow("test", field_interTestOutput);
//	cv::namedWindow("train", cv::WINDOW_NORMAL);
//	cv::imshow("train", field_interTrainOutput);
//	cv::waitKey();
//
//	imwrite("C:/Data/RGB/train_small/test/field_inter.png", field_interTestOutput);
//	imwrite("C:/Data/RGB/train_small/train/field_inter.png", field_interTrainOutput);
//
//
//	//-----------------------------------------------------------------------------
//
//	cv::Mat forest_interTest = cv::imread("C:/Data/RGB/full_images/forest_inter.png");
//	cv::Mat forest_interTrain = cv::imread("C:/Data/RGB/full_images/forest_inter.png");
//	cv::Mat forest_interTestOutput = forest_interTest(cv::Rect(0, 0, floor(forest_interTest.cols / cropFactorTest), forest_interTest.rows - 1));
//	cv::Mat forest_interTrainOutput = forest_interTrain(cv::Rect(ceil(forest_interTrain.cols / cropFactorTrain), 0, floor(forest_interTrain.cols / cropFactorTrain) - 1, forest_interTrain.rows - 1));
//
//	cv::namedWindow("test", cv::WINDOW_NORMAL);
//	cv::imshow("test", forest_interTestOutput);
//	cv::namedWindow("train", cv::WINDOW_NORMAL);
//	cv::imshow("train", forest_interTrainOutput);
//	cv::waitKey();
//
//	imwrite("C:/Data/RGB/train_small/test/forest_inter.png", forest_interTestOutput);
//	imwrite("C:/Data/RGB/train_small/train/forest_inter.png", forest_interTrainOutput);
//
//
//	//-----------------------------------------------------------------------------
//
//	cv::Mat grassland_interTest = cv::imread("C:/Data/RGB/full_images/grassland_inter.png");
//	cv::Mat grassland_interTrain = cv::imread("C:/Data/RGB/full_images/grassland_inter.png");
//	cv::Mat grassland_interTestOutput = grassland_interTest(cv::Rect(0, 0, floor(grassland_interTest.cols / cropFactorTest), grassland_interTest.rows - 1));
//	cv::Mat grassland_interTrainOutput = grassland_interTrain(cv::Rect(ceil(grassland_interTrain.cols / cropFactorTrain), 0,
//		floor(grassland_interTrain.cols / cropFactorTrain) - 1, grassland_interTrain.rows - 1));
//
//	cv::namedWindow("test", cv::WINDOW_NORMAL);
//	cv::imshow("test", grassland_interTestOutput);
//	cv::namedWindow("train", cv::WINDOW_NORMAL);
//	cv::imshow("train", grassland_interTrainOutput);
//	cv::waitKey();
//
//	imwrite("C:/Data/RGB/train_small/test/grassland_inter.png", grassland_interTestOutput);
//	imwrite("C:/Data/RGB/train_small/train/grassland_inter.png", grassland_interTrainOutput);
//
//
//	//-----------------------------------------------------------------------------
//
//	cv::Mat street_interTest = cv::imread("C:/Data/RGB/full_images/street_inter.png");
//	cv::Mat street_interTrain = cv::imread("C:/Data/RGB/full_images/street_inter.png");
//	cv::Mat street_interTestOutput = street_interTest(cv::Rect(0, 0, floor(street_interTest.cols / cropFactorTest), street_interTest.rows - 1));
//	cv::Mat street_interTrainOutput = street_interTrain(cv::Rect(ceil(street_interTrain.cols / cropFactorTrain), 0,
//		floor(street_interTrain.cols / cropFactorTrain) - 1, street_interTrain.rows - 1));
//
//	cv::namedWindow("test", cv::WINDOW_NORMAL);
//	cv::imshow("test", street_interTestOutput);
//	cv::namedWindow("train", cv::WINDOW_NORMAL);
//	cv::imshow("train", street_interTrainOutput);
//	cv::waitKey();
//
//	imwrite("C:/Data/RGB/train_small/test/street_inter.png", street_interTestOutput);
//	imwrite("C:/Data/RGB/train_small/train/street_inter.png", street_interTrainOutput);
//
//
//
//
//
//	//-----------------------------------------------------------------------------
//
//	cv::Mat rgb_imageTest = cv::imread("C:/Data/RGB/full_images/rgb_image.jpg");
//	cv::Mat rgb_imageTrain = cv::imread("C:/Data/RGB/full_images/rgb_image.jpg");
//	cv::Mat rgb_imageTestOutput = rgb_imageTest(cv::Rect(0, 0, floor(rgb_imageTest.cols / cropFactorTest), rgb_imageTest.rows - 1));
//	cv::Mat rgb_imageTrainOutput = rgb_imageTrain(cv::Rect(ceil(rgb_imageTrain.cols / cropFactorTrain), 0,
//		floor(rgb_imageTrain.cols / cropFactorTrain) - 1, rgb_imageTrain.rows - 1));
//
//	cv::namedWindow("test", cv::WINDOW_NORMAL);
//	cv::imshow("test", rgb_imageTestOutput);
//	cv::namedWindow("train", cv::WINDOW_NORMAL);
//	cv::imshow("train", rgb_imageTrainOutput);
//	cv::waitKey();
//
//	imwrite("C:/Data/RGB/train_small/test/rgb_image.png", rgb_imageTestOutput);
//	imwrite("C:/Data/RGB/train_small/train/rgb_image.png", rgb_imageTrainOutput);
//
//}
//
//
//void cropImagesSmall(int x, int y, int width, int height, int x_t, int y_t, int width_t, int height_t, std::string path_to_folder) {
//
//	cv::Mat city_interTest = cv::imread("C:/Data/magnitude_7_7/full_images/city_inter.png");
//	cv::Mat city_interTrain = cv::imread("C:/Data/magnitude_7_7/full_images/city_inter.png");
//
//	cv::Mat city_interTestOutput  = city_interTest(cv::Rect(x_t, y_t, width_t, height_t));
//	cv::Mat city_interTrainOutput = city_interTrain(cv::Rect(x, y, width, height));
//
//	cv::namedWindow("test", cv::WINDOW_NORMAL);
//	cv::imshow("test", city_interTestOutput);
//	cv::namedWindow("train", cv::WINDOW_NORMAL);
//	cv::imshow("train", city_interTrainOutput);
//	//cv::waitKey();
//
//	imwrite(path_to_folder+"test/city_inter.png", city_interTestOutput);
//	imwrite(path_to_folder + "train/city_inter.png", city_interTrainOutput);
//
//	//-----------------------------------------------------------------------------
//
//	cv::Mat field_interTest = cv::imread("C:/Data/magnitude_7_7/full_images/field_inter.png");
//	cv::Mat field_interTrain = cv::imread("C:/Data/magnitude_7_7/full_images/field_inter.png");
//	cv::Mat field_interTestOutput = field_interTest(cv::Rect(x_t, y_t, width_t, height_t));
//	cv::Mat field_interTrainOutput = field_interTrain(cv::Rect(x, y, width, height));
//
//	cv::namedWindow("test", cv::WINDOW_NORMAL);
//	cv::imshow("test", field_interTestOutput);
//	cv::namedWindow("train", cv::WINDOW_NORMAL);
//	cv::imshow("train", field_interTrainOutput);
//	//cv::waitKey();
//
//	imwrite(path_to_folder + "test/field_inter.png", field_interTestOutput);
//	imwrite(path_to_folder + "train/field_inter.png", field_interTrainOutput);
//
//
//	//-----------------------------------------------------------------------------
//
//	cv::Mat forest_interTest = cv::imread("C:/Data/magnitude_7_7/full_images/forest_inter.png");
//	cv::Mat forest_interTrain = cv::imread("C:/Data/magnitude_7_7/full_images/forest_inter.png");
//	cv::Mat forest_interTestOutput = forest_interTest(cv::Rect(x_t, y_t, width_t, height_t));
//	cv::Mat forest_interTrainOutput = forest_interTrain(cv::Rect(x, y, width, height));
//
//	cv::namedWindow("test", cv::WINDOW_NORMAL);
//	cv::imshow("test", forest_interTestOutput);
//	cv::namedWindow("train", cv::WINDOW_NORMAL);
//	cv::imshow("train", forest_interTrainOutput);
//	//cv::waitKey();
//
//	imwrite(path_to_folder + "test/forest_inter.png", forest_interTestOutput);
//	imwrite(path_to_folder + "train/forest_inter.png", forest_interTrainOutput);
//
//
//	//-----------------------------------------------------------------------------
//
//	cv::Mat grassland_interTest = cv::imread("C:/Data/magnitude_7_7/full_images/grassland_inter.png");
//	cv::Mat grassland_interTrain = cv::imread("C:/Data/magnitude_7_7/full_images/grassland_inter.png");
//	cv::Mat grassland_interTrainOutput = grassland_interTrain(cv::Rect(x, y, width, height));
//	cv::Mat grassland_interTestOutput  = grassland_interTest(cv::Rect(x_t, y_t, width_t, height_t));
//	
//
//	cv::namedWindow("test", cv::WINDOW_NORMAL);
//	cv::imshow("test", grassland_interTestOutput);
//	cv::namedWindow("train", cv::WINDOW_NORMAL);
//	cv::imshow("train", grassland_interTrainOutput);
//	//cv::waitKey();
//
//	imwrite(path_to_folder + "test/grassland_inter.png", grassland_interTestOutput);
//	imwrite(path_to_folder + "train/grassland_inter.png", grassland_interTrainOutput);
//
//
//	//-----------------------------------------------------------------------------
//
//	cv::Mat street_interTest = cv::imread("C:/Data/magnitude_7_7/full_images/street_inter.png");
//	cv::Mat street_interTrain = cv::imread("C:/Data/magnitude_7_7/full_images/street_inter.png");
//	cv::Mat street_interTestOutput = street_interTest(cv::Rect(x_t, y_t, width_t, height_t));
//	cv::Mat street_interTrainOutput = street_interTrain(cv::Rect(x, y, width, height));
//
//	cv::namedWindow("test", cv::WINDOW_NORMAL);
//	cv::imshow("test", street_interTestOutput);
//	cv::namedWindow("train", cv::WINDOW_NORMAL);
//	cv::imshow("train", street_interTrainOutput);
//	//cv::waitKey();
//
//	imwrite(path_to_folder + "test/street_inter.png", street_interTestOutput);
//	imwrite(path_to_folder + "train/street_inter.png", street_interTrainOutput);
//
//
//	//-----------------------------------------------------------------------------
//
//	cv::Mat rgb_imageTest = cv::imread("C:/Data/magnitude_7_7/full_images/rgb_image.png");
//	cv::Mat rgb_imageTrain = cv::imread("C:/Data/magnitude_7_7/full_images/rgb_image.png");
//	cv::Mat rgb_imageTestOutput = rgb_imageTest(cv::Rect(x_t, y_t, width_t, height_t));
//	cv::Mat rgb_imageTrainOutput = rgb_imageTrain(cv::Rect(x, y, width, height));
//
//	cv::namedWindow("test", cv::WINDOW_NORMAL);
//	cv::imshow("test", rgb_imageTestOutput);
//	cv::namedWindow("train", cv::WINDOW_NORMAL);
//	cv::imshow("train", rgb_imageTrainOutput);
//	cv::waitKey(20);
//
//	imwrite(path_to_folder + "test/rgb_image.png", rgb_imageTestOutput);
//	imwrite(path_to_folder + "train/rgb_image.png", rgb_imageTrainOutput);
//
//
//
//	//-----------------------------------------------------------------------------
//
//	cv::Mat magnitude_imageTest = cv::imread("C:/Data/magnitude_7_7/full_images/magnitude_image.png");
//	cv::Mat magnitude_imageTrain = cv::imread("C:/Data/magnitude_7_7/full_images/magnitude_image.png");
//	cv::Mat magnitude_imageTestOutput = magnitude_imageTest(cv::Rect(x_t, y_t, width_t, height_t));
//	cv::Mat magnitude_imageTrainOutput = magnitude_imageTrain(cv::Rect(x, y, width, height));
//
//	cv::namedWindow("test", cv::WINDOW_NORMAL);
//	cv::imshow("test", magnitude_imageTestOutput);
//	cv::namedWindow("train", cv::WINDOW_NORMAL);
//	cv::imshow("train", magnitude_imageTrainOutput);
//	cv::waitKey(20);
//
//	imwrite(path_to_folder + "test/magnitude_image.png", magnitude_imageTestOutput);
//	imwrite(path_to_folder + "train/magnitude_image.png", magnitude_imageTrainOutput);
//}
//
//
//
//void createImages() {
//
//	
//	cv::Mat rgb(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
//
//	cv::Mat city(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
//	cv::Mat field(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
//	cv::Mat forest(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
//	cv::Mat grassland(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
//	cv::Mat street(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
//
//
//	cv::rectangle(rgb,   cv::Point(0, 0), cv::Point(10, 100), cv::Scalar(255, 0, 0), -1);
//	cv::rectangle(city, cv::Point(0, 0), cv::Point(10, 100), cv::Scalar(255, 255, 255), -1);
//
//	cv::rectangle(rgb, cv::Point(20, 15), cv::Point(30, 30), cv::Scalar(0, 255, 0), -1);
//	cv::rectangle(field, cv::Point(20, 15), cv::Point(30, 30), cv::Scalar(255, 255, 255), -1);
//
//	cv::rectangle(rgb, cv::Point(40, 40), cv::Point(45, 45), cv::Scalar(0, 0, 255), -1);
//	cv::rectangle(forest, cv::Point(40, 40), cv::Point(45, 45), cv::Scalar(255, 255, 255), -1);
//
//	cv::rectangle(rgb, cv::Point(48, 46), cv::Point(50, 50), cv::Scalar(255, 0, 255), -1);
//	cv::rectangle(grassland, cv::Point(48, 46), cv::Point(50, 50), cv::Scalar(255, 255, 255), -1);
//
//	cv::rectangle(rgb, cv::Point(80, 70), cv::Point(100, 90), cv::Scalar(255, 255, 0), -1);
//	cv::rectangle(street, cv::Point(80, 70), cv::Point(100, 90), cv::Scalar(255, 255, 255), -1);
//
//
//
//
//
//
//	imwrite("C:/Data/RGB/test/rgb_image.png", rgb);
//	imwrite("C:/Data/RGB/train/rgb_image.png", rgb);
//
//	cv::namedWindow("rgb_image", cv::WINDOW_NORMAL);
//	cv::imshow("rgb_image", rgb);
//	cv::waitKey(20);
//
//	imwrite("C:/Data/RGB/test/city_inter.png", city);
//	imwrite("C:/Data/RGB/train/city_inter.png", city);
//
//	cv::namedWindow("city", cv::WINDOW_NORMAL);
//	cv::imshow("city", city);
//	cv::waitKey(20);
//
//	imwrite("C:/Data/RGB/test/field_inter.png", field);
//	imwrite("C:/Data/RGB/train/field_inter.png", field);
//
//	cv::namedWindow("field", cv::WINDOW_NORMAL);
//	cv::imshow("field", field);
//	cv::waitKey(20);
//
//	imwrite("C:/Data/RGB/test/forest_inter.png", forest);
//	imwrite("C:/Data/RGB/train/forest_inter.png", forest);
//
//	cv::namedWindow("forest", cv::WINDOW_NORMAL);
//	cv::imshow("forest", forest);
//	cv::waitKey(20);
//
//	imwrite("C:/Data/RGB/test/grassland_inter.png", grassland);
//	imwrite("C:/Data/RGB/train/grassland_inter.png", grassland);
//
//	cv::namedWindow("grassland", cv::WINDOW_NORMAL);
//	cv::imshow("grassland", grassland);
//	cv::waitKey(20);
//
//	imwrite("C:/Data/RGB/test/street_inter.png", street);
//	imwrite("C:/Data/RGB/train/street_inter.png", street );
//
//	cv::namedWindow("street", cv::WINDOW_NORMAL);
//	cv::imshow("street", street);
//	cv::waitKey();
//
//
//
//
//}
//
//
//void createClassification() {
//
//
//	cv::Mat rgb(10, 10, CV_8UC3, cv::Scalar(0, 0, 0));
//
//	cv::rectangle(rgb, cv::Point(0, 0),   cv::Point(9, 1), cv::Scalar(255, 0, 0), -1);
//	cv::rectangle(rgb, cv::Point(0, 2), cv::Point(10, 5), cv::Scalar(0, 255, 0), -1);	
//	cv::rectangle(rgb, cv::Point(8, 6), cv::Point(9, 6), cv::Scalar(0, 0, 255), -1);	
//	cv::rectangle(rgb, cv::Point(8, 8), cv::Point(9, 9), cv::Scalar(255, 0, 255), -1);	
//	cv::rectangle(rgb, cv::Point(4, 4), cv::Point(5, 5), cv::Scalar(255, 255, 0), -1);
//
//	imwrite("C:/Data/RGB/classification.png", rgb);
//	imwrite("C:/Data/RGB/classification.png", rgb);
//
//	cv::namedWindow("classification.png", cv::WINDOW_NORMAL);
//	cv::imshow("classification.png", rgb);
//	cv::waitKey(20);
//}
//
////"C:/Data/RGB/train/
//void saveCropped(std::string path) {
//
//	cv::Mat labeledImage = getLableMat(path+"labels_list.txt");
//	cv::imwrite(path+"labels.png", labeledImage);
//
//	cv::Mat show = (labeledImage.clone());
//	cv::normalize(show, show, 255, 0, cv::NORM_MINMAX);
//	cv::cvtColor(show, show, CV_GRAY2RGB, 3);
//	//std::cout << show << std::endl;
//	/*cv::namedWindow("TrainLabels", cv::WINDOW_NORMAL);
//	cv::imshow("TrainLabels", show);
//	cv::waitKey();*/
//
//	cv::Mat result = showInPseudoColorFirst5Classes(labeledImage);
//	cv::imwrite(path+"labels_color.png", result);
//
//	cv::namedWindow("TrainColor", cv::WINDOW_NORMAL);
//	cv::imshow("TrainColor", result);
//	cv::waitKey(20);
//}
//
//int main(int argc, char** argv) {
//
//	//createClassification();
// 	//cropImagesSmall(2 ,2, 1141, 6636, 1142, 2, 246, 6636, "C:/Data/magnitude_7_7/");
//	//saveCropped("C:/Data/magnitude_7_7/train/");
//	//saveCropped("C:/Data/magnitude_7_7/test/");
//	//saveCropped("C:/Data/RGB/full_images/");
//	
//	//saveCropped();
//	
//	
//	//-----------------------------------------------------------------------------
//
//	//std::vector<cv::Mat> data;
//
//	////loadRATOberpfaffenhofen("C:\\Users\\Galya\\Desktop\\Master\\Data\\Oberpfaffenhofen\\oph_lexi.rat", data);
//
//	//::google::InitGoogleLogging(argv[0]);
//
//	//string model_file = "C:/Data/FullyConnected/ober_quick.prototxt";// argv[1];
//	//string trained_file = "C:/Data/FullyConnected/trained_model/snapshot_prefix_iter_10000.caffemodel";// argv[1];
//	//string label_file = "C:/Data/FullyConnected/labels.txt";
//
//	//////string label_file = argv[3];
//
//	//Classifier classifier(model_file, trained_file, label_file);
//	//cv::Mat img = cv::imread("C:/Data/RGB/classify.png");
//	//predictOnInput(img, classifier);
//	////CHECK(!img.empty()) << "Unable to decode image " << file;
//
//
//	//std::vector<Prediction> predictions = classifier.Classify(img);
//
//	///* Print the top N predictions. */
//	//for (size_t i = 0; i < predictions.size(); ++i) {
//	//	Prediction p = predictions[i];
//	//	std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
//	//		<< p.first << "\"" << std::endl;
//	//}
//}
//#else
//int main(int argc, char** argv) {
//	LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
//}
//#endif  // USE_OPENCV
//#endif 