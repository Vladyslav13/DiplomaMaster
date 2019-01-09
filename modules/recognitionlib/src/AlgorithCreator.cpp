#include "pch.h"

#include <recognitionLib/Yolo3.h>

namespace rclib
{

NeroAlgoPtr CreateNeroAlgorithm(const NeroAlgoTypes type)
{
	switch(type)
	{
	case NeroAlgoTypes::Yolo:
		return std::make_shared<yolo::YOLO3>();
	default:
		throw std::runtime_error{
			"Cant create nero algorithm: Unsupported algorithm type" };
	}
}

} // namespace rclib
