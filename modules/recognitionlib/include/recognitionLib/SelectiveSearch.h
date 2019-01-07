#pragma once

/*
https://www.learnopencv.com/selective-search-for-object-detection-cpp-python/
*/

namespace rclib
{
namespace search
{

enum class SelectiveSearchType
{
	Unknown = 0,
	Fast,
	Quality
};

int SelectiveSearch(const std::string imPath, const SelectiveSearchType searchType);

} // naemspace search
} // namespace rclib
