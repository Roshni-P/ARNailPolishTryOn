#pragma once
#include "common.h"

enum class ModelType
{
	HandDetector,
	HandLandmark,
	Segmentation
};

// Type traits mapping
template<ModelType> struct ModelMap;

// Specialization of template
template<> struct ModelMap<ModelType::HandDetector>
{
	using type = HandDetector;
};

template<> struct ModelMap<ModelType::HandLandmark>
{
	using type = HandLandmarksDetector;
};

class ModelFactory
{
public: 

	template<ModelType T> 
	static std::unique_ptr<typename ModelMap<T>::type> Create()
	{
		using modelObj = typename ModelMap<T>::type;
		return std::make_unique<modelObj>();
	}
};