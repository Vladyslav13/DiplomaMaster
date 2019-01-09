#pragma once

/*
https://www.learnopencv.com/deep-learning-based-object-detection-and-instance-segmentation-using-mask-r-cnn-in-opencv-python-c/
*/

#include <recognitionLib/Types.h>
#include <recognitionLib/NeroAlgorithm.h>

namespace rclib
{
namespace rcnn
{

class MaskRcnn
	: public NeroAlgorithm
{
	//
	// Public types.
	//
public:
	//! MaskRcnn model config files.
	struct ConfigFiles
	{
		//! Model classes names.
		std::string classesNamesFile_;
		//! File with colors.
		std::string colorsFile_;
		//! Path to file with model weights.
		std::string textGraph_;
		//! Path to module config file.
		std::string modelWeights_;
	};
	//! MaskRcnn algorithm settings.
	struct Settings
	{
		//! True if detected object will be draw with mask.
		bool drawRectangle_ = true;
		//! True if detected object will be draw with mask.
		bool drawMask_ = false;
		//! Width of network's input image (320, 416, 620).
		int inpWidth_ = 320;
		//! Height of network's input image (320, 416, 620).
		int inpHeight_ = 320;
		//! Confidence threshold.
		float confThreshold_ = 0.5f;
		//! Suppression threshold.
		float maskThreshold_ = 0.3f;
		//! Preferable computation backend for Net.
		int preferableBackend_ = cv::dnn::DNN_BACKEND_OPENCV;
		//! Preferable computation target device.
		int preferableTarget_ = cv::dnn::DNN_TARGET_CPU;
	};

	//
	// Construction and destruction.
	//
public:
	//! Construction.
	MaskRcnn();
	//! Construction.
	MaskRcnn(
		const ConfigFiles& configFiles, const Settings& settings = Settings());

	//
	// Public interface.
	//
public:
	//! Configure algorithm with default parameters.
	void InitDefaultConfiguration() override;
	//! Starts algorithm with given parameters. Return error message if its occurred.
	std::string Process(
		const DataType processingDataType,
		const std::string& fileToProcess,
		const int deviceInd = 0) override;

	//
	// TODO: Temporary moved into private section because of troubles with asynchronous calls.
	//
private:
	//! Processes video stream.
	void ProcessStream(
		const int deviceInd,
		const std::string& fileToProcess);
	//! Starts processing video.
	void ProcessVideo(const std::string& fileToProcess);

	//
	// Setters and getters.
	//
public:
	//! Returns busy status.
	bool IsRunning() const override;
	//! Set configs.
	void SetConfigs(const ConfigFiles& cfg);
	//! Sets handle function for processing frames. Returns true in success.
	bool SetFrameProcessedCallback(const FrameProcessedCallback& callback) override;
	//! Set settings.
	void SetSettings(const Settings& settings);
	//! Set is running status on false. Work will be stopped on the next iteration.
	void Stop() override;
	//! Get configs.
	ConfigFiles GetConfigs() const;
	//! Returns data frame size.
	cv::Size GetFrameSize() const;
	//! Get settings.
	Settings GetSettings() const;
	//! Returns processed data.
	std::vector<FrameData> GetProcessedData() const;

	//
	// Private functions.
	//
private:
	//! Draw the predicted bounding box
	void DrawBox(FrameData& frame, int classId, float conf, cv::Rect box, FrameData& objectMask);
	//! Get the names of the output layers
	std::vector<std::string> GetOutputsNames(const cv::dnn::Net& net);
	//! Removes the bounding boxes with low confidence using non-maxima suppression.
	void PostProcess(FrameData& frame, const std::vector<cv::Mat>& out);
	//! Loads classes names from file.
	void PrepareNet();
	//! Implementation of video processing loop.
	void ProcessVideoImpl(cv::VideoCapture& cap);

	//
	// Private data members.
	//
private:
	//! Model config files.
	ConfigFiles cfg_;
	//! Colors.
	std::vector<cv::Scalar> colors_;
	//! Classes names.
	std::vector<std::string> classes_;
	//! Function to call for internal handle of processed frame.
	FrameProcessedCallback frameProcessedCallback_;
	//! SIze od data frame.
	cv::Size frameSize_;
	//! True if processing in progress.
	std::atomic_bool isRunning_;
	//! Neural network.
	cv::dnn::Net net_;
	//! True if algorithm parameters was loaded.
	std::atomic_bool netConfigured_;
	//! Result of processing.
	std::vector<FrameData> processingRes_;
	//! Algorithm settings.
	Settings settings_;
};



int RunRcnn();

} // namespace rcnn
} // namespace rclib
