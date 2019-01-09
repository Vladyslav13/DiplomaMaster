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
		VideoProcessing,
		CaptureFromVideoCam
	};

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
};

//! Supported algorithm types.
enum class NeroAlgoTypes
{
	Unknown = 0,
	Yolo
};

} // namespace rclib
