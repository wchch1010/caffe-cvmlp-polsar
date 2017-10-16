#include "config.h"
#ifdef CLASSIFICATION_OBERPFAFFEN

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

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

const int cropFactorTest = 2;
const int cropFactorTrain = 2;
const int IMAGE_RADIUS = 20;

class Classifier {
public:
	Classifier(const string& model_file,
		const string& trained_file,
		const string& label_file);

	std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

	cv::Scalar  getColorByLabelName(std::string);
private:
	void SetMean(const string& mean_file);

	std::vector<float> Predict(const cv::Mat& img);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

private:
	shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
	std::vector<string> labels_;
	std::vector<cv::Scalar> m_colors;
};

Classifier::Classifier(const string& model_file,
	const string& trained_file,
	const string& label_file) {
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

	m_colors.push_back(cv::Scalar(255, 0, 0));
	m_colors.push_back(cv::Scalar(0, 255, 0));
	m_colors.push_back(cv::Scalar(0, 0, 255));
	m_colors.push_back(cv::Scalar(0, 255, 255));
	m_colors.push_back(cv::Scalar(255, 255, 0));

	/*m_colors.push_back(cv::Scalar(0, 0, 0));
	m_colors.push_back(cv::Scalar(255, 0, 0));
	m_colors.push_back(cv::Scalar(0, 255, 0));
	m_colors.push_back(cv::Scalar(0, 0, 255));
	m_colors.push_back(cv::Scalar(255, 0, 255));
	m_colors.push_back(cv::Scalar(0, 255, 255));
	m_colors.push_back(cv::Scalar(255, 255, 255));
	m_colors.push_back(cv::Scalar(100, 0, 100));*/

	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	/* Load the binaryproto mean file. */
	//SetMean(mean_file);

	/* Load labels. */
	std::ifstream labels(label_file.c_str());
	CHECK(labels) << "Unable to open labels file " << label_file;
	string line;
	while (std::getline(labels, line))
		labels_.push_back(string(line));

	Blob<float>* output_layer = net_->output_blobs()[0];
	CHECK_EQ(labels_.size(), output_layer->channels())
		<< "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}



/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

cv::Scalar Classifier::getColorByLabelName(std::string label) {

	if (this->labels_.empty()) {
		std::cout << "Error!! Empty labels array" << std::endl;
	}
	else {
		for (size_t i = 0; i < labels_.size(); i++)
		{
			if (labels_.at(i) == label) {
				return m_colors.at(i);
			}
		}
	}

	return NULL;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
	std::vector<float> output = Predict(img);

	N = std::min<int>(labels_.size(), N);
	std::vector<int> maxN = Argmax(output, N);
	std::vector<Prediction> predictions;
	for (int i = 0; i < N; ++i) {
		int idx = maxN[i];
		predictions.push_back(std::make_pair(labels_[idx], output[idx]));
	}

	return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), num_channels_)
		<< "Number of channels of mean file doesn't match input layer.";

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	cv::merge(channels, mean);

	/* Compute the global mean pixel value and create a mean image
	* filled with this value. */
	cv::Scalar channel_mean = cv::mean(mean);
	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->Forward();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Classifier::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	//Galya
	cv::Mat sample_normalized = sample_float;
	//cv::subtract(sample_float, mean_, sample_normalized);

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}
template<typename T>
std::vector<T>
split(const T & str, const T & delimiters) {
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

void predictOnInput(cv::Mat image, Classifier& classifier) {
	
	int radius = 1;
	cv::Mat entry;

	cv::namedWindow("Original", cv::WINDOW_NORMAL);
	cv::imshow("Original", image);
	cv::waitKey(20);

	cv::Mat result = cv::Mat::zeros(image.size(), image.type());

	for (int col = 0; col < image.cols; col++) {
		for (int row = 0; row < image.rows; row++) {
			entry = getImageByIndex(image, row, col, IMAGE_RADIUS);


			/*cv::namedWindow("entry", cv::WINDOW_NORMAL);
			cv::imshow("entry", entry);
			cv::waitKey();*/

			/*cv::namedWindow("px: " + std::to_string(row) + std::to_string(col), cv::WINDOW_NORMAL);
			cv::imshow("px: " + std::to_string(row) + std::to_string(col), entry);
			cv::waitKey(15);*/

			std::vector<Prediction> predictions = classifier.Classify(entry);

			/* Print the top N predictions. */
			for (size_t i = 0; i < predictions.size(); ++i) {
				Prediction p = predictions[i];
				std::cout << std::fixed << std::setprecision(4) << p.second << " - \"" << p.first << "\"" << std::endl;
			}

			cv::Scalar color = classifier.getColorByLabelName(predictions[0].first);

			result.at<cv::Vec3b>(row, col)[0] = color.val[0];
			result.at<cv::Vec3b>(row, col)[1] = color.val[1];
			result.at<cv::Vec3b>(row, col)[2] = color.val[2];

			std::cout << "   ---------------------   " << std::endl;
		}
	}

	cv::namedWindow("Result", cv::WINDOW_NORMAL);
	cv::imshow("Result", result);
	cv::waitKey();
}

int main(int argc, char** argv) {

	string model_file = "C:/Data/color_test/lenet.prototxt";
	string trained_file = "C:/Data/color_test/lenet_color_test_iter_500.caffemodel";
	string label_file = "C:/Data/color_test/labels_test.txt";

	Classifier classifier(model_file, trained_file, label_file);
	cv::Mat img = cv::imread("C:/Data/color_test/test.png");
	predictOnInput(img, classifier);

	std::vector<Prediction> predictions = classifier.Classify(img);

	/* Print the top N predictions. */
	for (size_t i = 0; i < predictions.size(); ++i) {
		Prediction p = predictions[i];
		std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
			<< p.first << "\"" << std::endl;
	}
}
#else
int main(int argc, char** argv) {
	LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
#endif 