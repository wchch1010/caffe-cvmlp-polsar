 #ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <random>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/complex_image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


namespace caffe { 

template <typename Dtype>
ComplexImageDataLayer<Dtype>::~ComplexImageDataLayer<Dtype>() {
  this->StopInternalThread();
}
//33
const int IMAGE_RADIUS  = 13;
const int NUM_OF_TRAIN_IMAGES  = 10000;
const int NUM_OF_TEST_IMAGES   = 2000;



void getImageLinesAndLabels(int numOfPoints, cv::Mat image, int width, int height, string root_folder, vector<std::pair<std::string, int>>& lines) {


	std::random_device rdRows;     // only used once to initialise (seed) engine
	std::mt19937 rngRows(rdRows());    // random-number engine used (Mersenne-Twister in this case)
	std::uniform_int_distribution<int> uniRows(0, image.rows-1); // guaranteed unbiased

	
	std::random_device rdCols;     // only used once to initialise (seed) engine
	std::mt19937 rngCols(rdCols());    // random-number engine used (Mersenne-Twister in this case)
	std::uniform_int_distribution<int> uniCols(0, image.cols - 1); // guaranteed unbiased

	std::map<std::string, int> city;
	std::map<std::string, int> field;
	std::map<std::string, int> forest;
	std::map<std::string, int> grass;
	std::map<std::string, int> street;
	std::map<std::string, int> none;


	int total_city;
	int total_field;
	int total_forest;
	int total_grass;
	int total_street;
	int total_none;

	std::map<std::string, int>::iterator it;
	bool bInsert = false;
	cv::Mat labels = getLableMat(root_folder + "/labels_list.txt");

	std::vector<std::string> labelsPath = getLabelsPathes(root_folder + "/labels_list.txt");
	cv::Mat labelBinary;
	bool full_city = false;
	bool full_field = false;
	bool full_foresty = false;
	bool full_grass = false;
	bool full_street = false;
	bool full_none = false;
	
	labelBinary = cv::imread(labelsPath.at(0));
	cv::cvtColor(labelBinary, labelBinary, CV_BGR2GRAY, 1);
	total_city = cv::countNonZero(labelBinary);
	labelBinary = cv::imread(labelsPath.at(1));
	cv::cvtColor(labelBinary, labelBinary, CV_BGR2GRAY, 1);
	total_field = cv::countNonZero(labelBinary);
	labelBinary = cv::imread(labelsPath.at(2));
	cv::cvtColor(labelBinary, labelBinary, CV_BGR2GRAY, 1);
	total_forest = cv::countNonZero(labelBinary);
	labelBinary = cv::imread(labelsPath.at(3));
	cv::cvtColor(labelBinary, labelBinary, CV_BGR2GRAY, 1);
	total_grass = cv::countNonZero(labelBinary);
	labelBinary = cv::imread(labelsPath.at(4));
	cv::cvtColor(labelBinary, labelBinary, CV_BGR2GRAY, 1);
	total_street = cv::countNonZero(labelBinary);

	total_none = labelBinary.rows*labelBinary.cols - (total_city + total_field + total_forest + total_grass + total_street);


	//std::cout << root_folder << " labels : city = " << total_city << " / field = " << total_field << " / total_forest = " << total_forest << " / total_grass = " << total_grass << " / total_street " << total_street << " rest = " << total_none << std::endl;


	if (total_city>numOfPoints) {
		total_city = numOfPoints;
	}
	if (total_field>numOfPoints) {
		total_field = numOfPoints;
	}
	if (total_forest>numOfPoints) {
		total_forest = numOfPoints;
	}
	if (total_grass>numOfPoints) {
		total_grass = numOfPoints;
	}
	if (total_street>numOfPoints) {
		total_street = numOfPoints;
	}
	if (total_none>numOfPoints) {
		total_none = numOfPoints;
	}

	/*for (int col = 0; col < image.cols; col++) {
		for (int row = 0; row < image.rows; row++) {*/

   std::cout << root_folder << " labels : city = " << total_city << " / field = " << total_field << " / total_forest = " << total_forest << " / total_grass = " << total_grass << " / total_street " << total_street << " rest = " << total_none << std::endl;

	while (city.size() < total_city || field.size() < total_field || forest.size() < total_forest || grass.size() < total_grass || street.size() < total_street || none.size() < total_none) {


		auto row = uniRows(rngRows);
		auto col = uniCols(rngCols);

		//std::cout << "Row " << row << " Col " << col << std::endl;
		
		std::string indexes = std::to_string(row) + "," + std::to_string(col);
		int label = (int)labels.at<uchar>(row, col);

		bInsert = false;

		switch (label)
		{
			//city
		case 1:
			it = city.find(indexes);
			if (city.size() < numOfPoints && it == city.end())
			{
				city.insert(std::pair<std::string, int>(indexes, label));
				bInsert = true;
			}
			break;
		case 2:
			it = field.find(indexes);
			if (field.size() < numOfPoints && it == field.end())
			{
				field.insert(std::pair<std::string, int>(indexes, label));
				bInsert = true;
			}
			break;
		case 3:
			it = forest.find(indexes);
			if (forest.size() < numOfPoints && it == forest.end())
			{
				forest.insert(std::pair<std::string, int>(indexes, label));
				bInsert = true;
			}
			break;
		case 4:
			it = grass.find(indexes);
			if (grass.size() < numOfPoints && it == grass.end())
			{
				grass.insert(std::pair<std::string, int>(indexes, label));
				bInsert = true;
			}
			break;
		case 5:
			it = street.find(indexes);
			if (street.size() < numOfPoints && it == street.end())
			{
				street.insert(std::pair<std::string, int>(indexes, label));
				bInsert = true;
			}
			break;
		case 0:
			it = none.find(indexes);
			if (none.size() < numOfPoints && it == none.end())
			{
				none.insert(std::pair<std::string, int>(indexes, label));
				bInsert = true;
			}
		default:
			break;
		}

		if (bInsert) {
			//std::cout << "Inserted " << lines.size() << std::endl;
			lines.push_back(std::make_pair(indexes, (int)labels.at<uchar>(row, col)));

			if (city.size() == total_city && !full_city) {
				std::cout << root_folder << " City enteries created " << total_none << std::endl;
				full_city = true;
			}
			if (field.size() == total_field && !full_field) {
				std::cout << root_folder << " Field enteries created " << total_none << std::endl;
				full_field = true;
			}
			if (forest.size() == total_forest && !full_foresty) {
				std::cout << root_folder << " Forest enteries created " << total_none << std::endl;
				full_foresty = true;
			}
			if (grass.size() == total_grass && !full_grass) {
				std::cout << root_folder << " Grass enteries created " << total_none << std::endl;
				full_grass = true;
			}
			if (street.size() == total_street && !full_street) {
				std::cout << root_folder << " Street enteries created " << total_none << std::endl;
				full_street = true;
			}
			if (none.size() == total_none && !full_none) {
				std::cout << root_folder << " Empty enteries created " << total_none << std::endl;
				full_none = true;
			}
		}
		else {
			//std::cout << "Skipped " << std::endl;
		}
	}
	/*	}
	}*/




	std::cout << lines.size() << " Lines were created from original image " << std::endl;

	//std::string indexes = lines_[lines_id_].first;
	//vector<string> strIndexes = split<string>(indexes, ",");
}


template <typename Dtype>
void ComplexImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();
  std::cout << " root_folder " << root_folder << std::endl;
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;


  //C:/Data/magnitude_7_7/full_images/oph_lexi.rat
  vector<string> strSplitSource = splitCustomString<string>(source, "/");

  if (strSplitSource.at(strSplitSource.size() - 1) == "oph_lexi.rat") {
	  m_fullImage = getMainImage(source);
  }
  else {
	  m_fullImage = cv::imread(source, cv::IMREAD_ANYCOLOR);
  }


  if (this->phase_ == TRAIN) {
	  numOfPoints = NUM_OF_TRAIN_IMAGES;
  }
  else {
	  numOfPoints = NUM_OF_TEST_IMAGES;
  }

  getImageLinesAndLabels(numOfPoints, m_fullImage, m_fullImage.cols, m_fullImage.rows, root_folder, lines_);

  LOG(INFO) << "File " << source<<" loaded";

  /*std::ifstream infile(source.c_str());
  string line;
  size_t pos;
  int label;
  while (std::getline(infile, line)) {
    pos = line.find_last_ofi(' ');
    label = atoi(line.substr(pos + 1).c_str());
    lines_.push_back(std::make_pair(line.substr(0, pos), label));
  }*/

  CHECK(!lines_.empty()) << "File is empty";

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  } else {
    if (this->phase_ == TRAIN && Caffe::solver_rank() > 0 &&
        this->layer_param_.image_data_param().rand_skip() == 0) {
      LOG(WARNING) << "Shuffling or skipping recommended for multi-GPU";
    }
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  
  std::string indexes = lines_[lines_id_].first;
  vector<string> strIndexes = splitCustomString<string>(indexes, ",");
  cv::Mat cv_img = getImageByIndex(m_fullImage, std::stoi(strIndexes.at(0)), std::stoi(strIndexes.at(1)), IMAGE_RADIUS);
 /* cv::namedWindow("entry" + std::to_string(lines_[lines_id_].first) + std::to_string(lines_[lines_id_].second), cv::WINDOW_NORMAL);
  cv::imshow("entry" + std::to_string(lines_[lines_id_].first) + std::to_string(lines_[lines_id_].second), cv_img);
  cv::waitKey(10);
*/

  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void ComplexImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ComplexImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
 /* cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
*/
  const string& source = image_data_param.source();

 /* if (this->phase_ == TRAIN) {
	  std::cout << std::endl << std::endl << " -----       TRAIN    -----        " << " line id " << lines_id_ << " from " << lines_.size() <<  std::endl << std::endl << std::endl;
  }else{
	  std::cout << std::endl << std::endl << " -----       TEST     -----        " << " line id " << lines_id_ << " from " << lines_.size() << std::endl << std::endl << std::endl;
  }*/

  if (m_fullImage.empty()) {

	  vector<string> strSplitSource = splitCustomString<string>(source, "/");
	  if (strSplitSource.at(strSplitSource.size() - 1) == "oph_lexi.rat") {
		  m_fullImage = getMainImage(source);
	  }
	  else {
		  m_fullImage = cv::imread(source, cv::IMREAD_ANYCOLOR);
	  }
  }
  std::string indexes = lines_[lines_id_].first;
  vector<string> strIndexes = splitCustomString<string>(indexes, ",");

  cv::Mat cv_img = getImageByIndex(m_fullImage, std::stoi(strIndexes.at(0)), std::stoi(strIndexes.at(1)), IMAGE_RADIUS);
 /* cv::namedWindow("entry" + std::to_string(lines_[lines_id_].first) + std::to_string(lines_[lines_id_].second), cv::WINDOW_NORMAL);
  cv::imshow("entry" + std::to_string(lines_[lines_id_].first) + std::to_string(lines_[lines_id_].second), cv_img);
  cv::waitKey();*/


  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
  /*  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        new_height, new_width, is_color);*/

	std::string indexes = lines_[lines_id_].first;
	vector<string> strIndexes = splitCustomString<string>(indexes, ",");
	cv::Mat cv_img = getImageByIndex(m_fullImage, std::stoi(strIndexes.at(0)), std::stoi(strIndexes.at(1)), IMAGE_RADIUS);

    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image  
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

	prefetch_label[item_id] = lines_[lines_id_].second;//lines_[lines_id_].second;
	

	//std::cout << "prefetch_label[item_id]   " << prefetch_label[item_id]<<std::endl;
	//cv::namedWindow(indexes+"-"+std::to_string(prefetch_label[item_id]), cv::WINDOW_NORMAL);
	//cv::imshow(indexes + "-" + std::to_string(prefetch_label[item_id]), cv_img);
	//cv::waitKey();
	//cv::destroyAllWindows();

	vector<std::string> splitedPathToSource = splitCustomString<std::string>(source, "/");

    cv::imwrite("C:/Data/db/" + splitedPathToSource.at(splitedPathToSource.size() - 2) + "/" + std::to_string(lines_[lines_id_].second) + "_label_" + indexes + "_" + splitedPathToSource.at(splitedPathToSource.size() - 2) + ".png", cv_img);

	 
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ComplexImageDataLayer);
REGISTER_LAYER_CLASS(ComplexImageData);

}  // namespace caffe
#endif  // USE_OPENCV
