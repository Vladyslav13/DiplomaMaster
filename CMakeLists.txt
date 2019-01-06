cmake_minimum_required(VERSION 3.12)

project(Master-Diploma-Vlad)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed
set(CMAKE_AUTOMOC ON)

find_package(OpenCV 4.0.1 REQUIRED)
#find_package(VTK 9.0 REQUIRED)
find_package(Boost REQUIRED)

find_package(Qt5 COMPONENTS Widgets REQUIRED)

set(SOURCE_FILES
	main.cpp
	)

add_executable(${PROJECT_NAME}
	${SOURCE_FILES}
	)

qt5_use_modules(${PROJECT_NAME}
	Core
	Gui
	Widgets
	)

target_include_directories(${PROJECT_NAME}
	PRIVATE
		${CMAKE_CURRENT_BINARY_DIR}
		${OpenCV_INCLUDE_DIRS}
		${Boost_INCLUDE_DIRS}
	)

message("!!!!!!!!!!!!!!!!!!!!!!" ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
	${Boost_LIBRARIES}
	${OpenCV_LIBS}
#	${VTK_LIBRARIES}
	)