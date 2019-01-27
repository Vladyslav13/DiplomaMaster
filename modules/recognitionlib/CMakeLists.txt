cmake_minimum_required(VERSION 3.12)

project(RecognitionLib)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed
set(CMAKE_AUTOMOC ON)

find_package(OpenCV 4.0.1 REQUIRED)
find_package(Qt5 COMPONENTS Widgets REQUIRED)

set(PCH
	src/pch.h
	src/pch.cpp
	)

set(INCLUDE_FILES
	include/recognitionLib/AlgorithCreator.h
	include/recognitionLib/Goturn.h
	include/recognitionLib/NeroAlgorithm.h
	include/recognitionLib/Rcnn.h
	include/recognitionLib/RecognitionLib.h
	include/recognitionLib/SelectiveSearch.h
	include/recognitionLib/Types.h
	include/recognitionLib/Yolo3.h
	)

set(SOURCE_FILES
	src/AlgorithCreator.cpp
	src/Goturn.cpp
	src/Rcnn.cpp
	src/SelectiveSearch.cpp
	src/Yolo3.cpp
	)

add_library(${PROJECT_NAME}
	${PCH}
	${INCLUDE_FILES}
	${SOURCE_FILES}
	)

qt5_use_modules(${PROJECT_NAME}
	Core
	Gui
	Widgets
	)

target_include_directories(${PROJECT_NAME}
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/src
		${CMAKE_CURRENT_BINARY_DIR}
		${OpenCV_INCLUDE_DIRS}
	)

target_link_libraries(${PROJECT_NAME}
	${OpenCV_LIBS}
	)
