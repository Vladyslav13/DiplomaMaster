#include "pch.h"
#include <recognitionLib/Yolo3.h>

namespace rclib
{
namespace yolo
{

YOLO3::YOLO3()
	: frameProcessedCallback_([](auto) {})
	, isRunning_(false)
	, netConfigured_(false)
{
	
}

YOLO3::YOLO3(const ConfigFiles& configFiles, const Settings& settings)
	: cfg_(configFiles)
	, frameProcessedCallback_([](auto) {})
	, isRunning_(false)
	, netConfigured_(false)
	, settings_(settings)
{
}

void YOLO3::DrawPred(
	int classId, float conf, int left, int top, int right, int bottom, FrameData& frame)
{
	//Get the label for the class name and its confidence
	std::string label = cv::format("%.2f", conf);
	if (!classes_.empty())
	{
		if (classId >= classes_.size()) {
			return;
		}

		const auto className = classes_[classId];

		auto it = std::find_if(
			classesToDisplay_.begin(),
			classesToDisplay_.end(),
			[&className](const std::string& name) {
				return className == name;
			});
		if(it == classesToDisplay_.end()) {
			return;
		}

		label = className + ":" + label;
	}
	else
	{
		return;
	}

	//Draw a rectangle displaying the bounding box
	rectangle(
		*frame,
		cv::Point(left, top),
		cv::Point(right, bottom),
		cv::Scalar(255, 178, 50),
		3);

	//Display the label at the top of the bounding box
	int baseLine;
	cv::Size labelSize =
		cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = std::max(top, labelSize.height);
	rectangle(
		*frame,
		cv::Point(left, top - round(1.5*labelSize.height)),
		cv::Point(left + round(1.5*labelSize.width), top + baseLine),
		cv::Scalar(255, 255, 255),
		cv::FILLED);

	putText(
		*frame,
		label,
		cv::Point(left, top),
		cv::FONT_HERSHEY_SIMPLEX,
		0.75,
		cv::Scalar(0, 0, 0),
		1);
}

std::vector<std::string> YOLO3::GetOutputsNames(const cv::dnn::Net& net)
{
	static std::vector<std::string> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		std::vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		std::vector<std::string> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

void YOLO3::PostProcess(FrameData& frame, const std::vector<cv::Mat>& outs)
{
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			cv::Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > settings_.confThreshold_)
			{
				int centerX = (int)(data[0] * frame->cols);
				int centerY = (int)(data[1] * frame->rows);
				int width = (int)(data[2] * frame->cols);
				int height = (int)(data[3] * frame->rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(cv::Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	std::vector<int> indices;
	cv::dnn::NMSBoxes(
		boxes, confidences,  settings_.confThreshold_, settings_.nmsThreshold_, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		DrawPred(
			classIds[idx],
			confidences[idx],
			box.x,
			box.y,
			box.x + box.width,
			box.y + box.height, 
			frame);
	}
}

void YOLO3::PrepareNet()
{
	processingRes_.clear();

	if (netConfigured_) {
		return;
	}

	std::ifstream ifs(cfg_.classesNamesFile_);
	std::string line;
	classes_.clear();
	while (getline(ifs, line)) {
		classes_.push_back(line);
	}

	// Load the network
	net_ =
		cv::dnn::readNetFromDarknet(cfg_.moduleCfgFile_, cfg_.modelWeights_);
	net_.setPreferableBackend(settings_.preferableBackend_);
	net_.setPreferableTarget(settings_.preferableTarget_);

	netConfigured_ = true;
}

std::string YOLO3::Process(
	const DataType processingDataType,
	const std::string& fileToProcess)
{
	// TODO: It can be false but processing loop will be in the last iteration.
	if (isRunning_) {
		return "Can't start new processing, the previous one is in progress";
	}

	// TODO: add mutex synchronization while accessing this parameter?
	isRunning_ = true;
	std::string error;
	
	try
	{
		switch (processingDataType)
		{
		case DataType::CaptureFromVideoCam:
			ProcessStream(0, fileToProcess); // TODO: add possibility to chose cam.
			break;
		case DataType::VideoProcessing:
			ProcessVideo(fileToProcess);
			break;
		case DataType::Unknown:
		default:
			throw std::runtime_error{ "Unknown data type" };
		}
	}
	catch (const std::exception& e)
	{
		error = "Error occured in YOLO3 work: ";
		error += e.what();
	}

	Stop();

	return error;
}

void YOLO3::ProcessStream(
	const int deviceInd, const std::string& fileToProcess)
{
	PrepareNet();

	cv::VideoCapture cap;
	cap.open(deviceInd);

	// Add normal logging.
	std::cout << "Start processing the object" << std::endl;

	ProcessVideoImpl(cap);

	cap.release();
}

void YOLO3::ProcessVideo(const std::string& fileToProcess)
{
	PrepareNet();
	
	cv::VideoCapture cap;
	cap.open(fileToProcess);

	// Add normal logging.
	std::cout << "Start processing the object" << std::endl;

	ProcessVideoImpl(cap);

	cap.release();
}

void YOLO3::ProcessVideoImpl(cv::VideoCapture& cap)
{
	frameSize_ = cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));

	while (cv::waitKey(1) < 0 && isRunning_)
	{
		auto frame = std::make_shared<cv::Mat>();
		auto blob = std::make_shared<cv::Mat>();

		// Get frame from the video
		cap >> *frame;

		// Stop the program if reached end of video
		if (frame->empty()) {
			break;
		}

		// Create a 4D blob from a frame.
		cv::dnn::blobFromImage(
			*frame,
			*blob,
			1/255.0,
			cv::Size(settings_.inpWidth_, settings_.inpHeight_),
			cv::Scalar(0,0,0),
			true,
			false);

		// Sets the input to the network
		net_.setInput(*blob);

		// Runs the forward pass to get output of the output layers
		std::vector<cv::Mat> outs;
		net_.forward(outs, GetOutputsNames(net_));

		// Remove the bounding boxes with low confidence
		PostProcess(frame, outs);

		// Write the frame with the detection boxes
		processingRes_.push_back(frame);

		frameProcessedCallback_(frame);
	}
}

bool YOLO3::IsRunning() const
{
	return isRunning_;
}

bool YOLO3::SetFrameProcessedCallback(const FrameProcessedCallback& callback)
{
	if (!callback) {
		return false;
	}

	frameProcessedCallback_ = callback;
	return true;
}

void YOLO3::SetConfigs(const ConfigFiles& cfg)
{
	cfg_ = cfg;
	netConfigured_ = false;
}

void YOLO3::SetClassesToDisplay(const std::vector<std::string>& classes)
{
	classesToDisplay_ = classes;
}

void YOLO3::SetSettings(const Settings& settings)
{
	settings_ = settings;
	netConfigured_ = false;
}

YOLO3::ConfigFiles YOLO3::GetConfigs() const
{
	return cfg_;
}

cv::Size YOLO3::GetFrameSize() const
{
	return frameSize_;
}

YOLO3::Settings YOLO3::GetSettings() const
{
	return settings_;
}

std::vector<YOLO3::FrameData> YOLO3::GetProcessedData() const
{
	if (isRunning_) {
		return {};
	}

	return processingRes_;
}

void YOLO3::Stop()
{
	isRunning_ = false;
}

} // namespace yolo
} // namespace rclib