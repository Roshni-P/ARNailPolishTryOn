cmake_minimum_required(VERSION 4.2)

project(ARNailPolishTryOnC++)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR "D:/3DFace_Dependencies/build_opencv")
# Find OpenCV
find_package(OpenCV REQUIRED)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h")

# Create executable
add_executable(ARNailPolishTryOnC++ ${SOURCES} ${HEADERS})

source_group(TREE ${PROJECT_SOURCE_DIR} FILES ${SOURCES} ${HEADERS})

# Link OpenCV
target_link_libraries(ARNailPolishTryOnC++ PRIVATE 
	${OpenCV_LIBS})

# Target paths to include
target_include_directories(ARNailPolishTryOnC++ PUBLIC
	${CMAKE_SOURCE_DIR}/src 
	${CMAKE_SOURCE_DIR}/include
)

set(OpenCV_BIN_DIR "D:/3DFace_Dependencies/build_opencv/bin")

add_custom_command(TARGET ARNailPolishTryOnC++ POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${OpenCV_BIN_DIR}/$<$<CONFIG:Debug>:debug>$<$<CONFIG:Release>:release>/opencv_world4140$<$<CONFIG:Debug>:d>.dll"
    $<TARGET_FILE_DIR:ARNailPolishTryOnC++>
)

