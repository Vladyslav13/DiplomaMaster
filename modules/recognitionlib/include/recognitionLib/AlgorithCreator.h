#pragma once

#include <recognitionLib/Types.h>

namespace rclib
{

//! Creates algorithm of specified type.
NeroAlgoPtr CreateNeroAlgorithm(const NeroAlgoTypes type);

} // namespace rclib
