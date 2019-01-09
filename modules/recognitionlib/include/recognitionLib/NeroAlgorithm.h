#pragma once

namespace rclib
{

class NeroAlgorithm
{
	//
	// Public construction and destruction.
	//
public:
	//! Constructor.
	NeroAlgorithm() = default;
	//! Destructor.
	virtual ~NeroAlgorithm() = default;

	//
	// Public types.
	//
public:
	//! Type of data that algorithm will be working with.
	enum class DataType
	{
		Unknown = 0,
		VideoFile,
		CaptureFromVideoCam
	};

	//
	// Public type aliases.
	//
public:
	//! Data type.
	using FrameData = std::shared_ptr<cv::Mat>;
	//! Type of call back function to call after frame was processed.
	using FrameProcessedCallback = std::function<void(FrameData)>;

	//
	// Public interface.
	//
public:
	//! Configure algorithm with default parameters.
	virtual void InitDefaultConfiguration() = 0;
	//! Starts algorithm with given parameters. Return error message if its occurred.
	virtual std::string Process(
		const DataType processingDataType,
		const std::string& fileToProcess,
		const int deviceInd = 0) = 0;

	//
	// Setters and getters.
	//
public:
	//! Returns busy status.
	virtual bool IsRunning() const = 0;
	//! Sets classes that  allowed to display.
	virtual void SetClassesToDisplay(const std::vector<std::string>& classes);
	//! Sets handle function for processing frames. Returns true in success.
	virtual bool SetFrameProcessedCallback(const FrameProcessedCallback& callback) = 0;
	//! Set is running status on false. Work will be stopped on the next iteration.
	virtual void Stop() = 0;
	//! Returns data frame size.
	virtual cv::Size GetFrameSize() const = 0;
	//! Returns processed data.
	virtual std::vector<FrameData> GetProcessedData() const = 0;

	//
	// Protected data members.
	//
protected:
	//! Classes to display.
	std::vector<std::string> classesToDisplay_;
};

inline void NeroAlgorithm::SetClassesToDisplay(const std::vector<std::string>& classes)
{
	classesToDisplay_ = classes;
}

} // namespace rclib
