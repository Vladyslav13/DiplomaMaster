#pragma once

namespace rclib
{

enum class ProcessingType
{
	Unknown = 0,
	ImageProcessing,
	VideoProcessing,
	CaptureFromVideoCam // Currently unsupported
};


} // namespace rclib
