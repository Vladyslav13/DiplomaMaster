#pragma once

namespace rclib
{

class NeroAlgorithm;

//! Supported algorithm types.
enum class NeroAlgoTypes
{
	Unknown = 0,
	Yolo
};

//! Type of shared pointer on NeroAlgorithm.
using NeroAlgoPtr = std::shared_ptr<NeroAlgorithm>;

} // namespace rclib
