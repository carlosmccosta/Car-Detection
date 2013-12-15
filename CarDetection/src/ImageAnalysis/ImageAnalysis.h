#pragma once

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// std includes
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

// project includes
#include "../GUI/GUIUtils.h"

// namespace specific imports to avoid namespace pollution
using std::string;
using std::stringstream;
using std::vector;
using std::map;
using std::pair;
using cv::Mat;
using cv::Rect;
using cv::RotatedRect;
using cv::Scalar;
using cv::Vec3f;
using cv::Point;
using cv::Point2f;
using cv::Size;
using cv::VideoCapture;
using cv::imread;
using cv::waitKey;
using cv::imshow;
using cv::namedWindow;
using cv::moveWindow;
using cv::resizeWindow;
using cv::circle;
using cv::ellipse;
using cv::rectangle;
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <constants definitions> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#define WINDOW_NAME_MAIN "0. Original image"
#define WINDOW_NAME_OPTIONS "Parameterization"
#define WINDOW_NAME_BILATERAL_FILTER "1. Bilateral filter"
#define WINDOW_NAME_BILATERAL_FILTER_OPTIONS "1.1. Bilateral filter options"
#define WINDOW_NAME_HISTOGRAM_EQUALIZATION "2. Global histogram equalization (not used)"
#define WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE "2. Histogram equalization CLAHE"
#define WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE_OPTIONS "2.1. Histogram equalization CLAHE options"
#define WINDOW_NAME_CONTRAST_AND_BRIGHTNESS "3. Contrast, brightness and bilateral filtering (2nd pass)"
#define WINDOW_NAME_CONTRAST_AND_BRIGHTNESS_OPTIONS "3.1. Contrast and brightness options"

#define TRACK_BAR_NAME_BILATERAL_FILTER_DIST "1Dist"
#define TRACK_BAR_NAME_BILATERAL_FILTER_COLOR_SIG "1Color Sig"
#define TRACK_BAR_NAME_BILATERAL_FILTER_SPACE_SIG "1Space Sig"
#define TRACK_BAR_NAME_CLAHE_CLIP "2Clip"
#define TRACK_BAR_NAME_CLAHE_TILE_X "2Tile X"
#define TRACK_BAR_NAME_CLAHE_TILE_Y "2Tile Y"
#define TRACK_BAR_NAME_CONTRAST "3Contr*10"
#define TRACK_BAR_NAME_BRIGHTNESS "3Brigh*10"

#define ESC_KEYCODE 27
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </constants definitions> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <Image analysis>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

/// Image analysis class that detects speed limits signs and recognizes the speed limit number
class ImageAnalysis {
	public:
		
		/// Constructor with initialization of parameters with default value		 		 
		ImageAnalysis();
		
		/// ImageAnalysis destructor that performs cleanup of OpenCV HighGUI windows (in case they are used)		 
		virtual ~ImageAnalysis();			


		/*!
		 * \brief Processes the image from the specified path
		 * \param path Full path to image
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 * \return true if image was successfully processed
		 */
		bool processImage(string path, bool useCVHighGUI = true);


		/*!
		 * \brief Processes the image already loaded
		 * \param image Image loaded and ready to be processed
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 * \return true if image was successfully processed
		 */
		bool processImage(Mat& image, bool useCVHighGUI = true);


		/*!
		 * \brief Preprocesses the image by applying bilateral filtering, histogram equalization, contrast and brightness correction and bilateral filtering again
		 * \param image Image to be preprocessed
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 */
		void preprocessImage(Mat& image, bool useCVHighGUI = true);


		/*!
		 * \brief Applies histogram equalization to the specified image
		 * \param image Image to equalize
		 * \param useCLAHE If true, uses the contrast limited adaptive histogram equalization (CLAHE)
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 * \return 
		 */
		void histogramEqualization(Mat& image, bool useCLAHE = true, bool useCVHighGUI = true);		
		

		/*!
		 * \brief Computes the probability of the digitImageTemplate is present in the textColorSegmentationDigitROI
		 * \param textColorSegmentationDigitROI Image were the digit might be
		 * \param digitImageTemplate Template of the digit that is going to be searched
		 * \param feacturePointsGoodMatches Image were the good matches between the textColorSegmentationDigitROI and digitImageTemplate will be drawn if useCVHighGUI is true
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 * \return Probability of the digitImageTemplate is present in the textColorSegmentationDigitROI
		 */
		float recognizeDigitWithFeatureMatching(Mat& textColorSegmentationDigitROI, Mat& digitImageTemplate, Mat& feacturePointsGoodMatches, bool useCVHighGUI = true);		
				

		/*!
		 * \brief Processes the image to reflect any internal parameter change		 
		 * \return True if processing finished successfully
		 */
		bool updateImage();
		

		/*!
		 * \brief Processes a video from a file, analyzing the presence of speed limit signs
		 * \param path Full path to video
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 * \return True if processing finished successfully
		 */
		bool processVideo(string path, bool useCVHighGUI = true);

		/*!
		 * \brief Processes a video from a camera, analyzing the presence of speed limit signs
		 * \param cameraDeviceNumber Camera device number
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 * \return True if processing finished successfully
		 */
		bool processVideo(int cameraDeviceNumber, bool useCVHighGUI = true);


		/*!
		 * \brief Processes a video from a VideoCapture source, analyzing the presence of speed limit signs
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 * \return True if processing finished successfully
		 */
		bool processVideo(VideoCapture videoCapture, bool useCVHighGUI = true);

		
		/// brief Setups the HighGUI window were the original image is going to be drawn		 		 
		void setupMainWindow();


		/*!
		 * \brief Setups the windows were the results will be presented
		 * \param optionsOneWindow Flag to indicate to group the track bars in one window		 
		 */
		void setupResultsWindows(bool optionsOneWindow = false);			


		// ------------------------------------------------------------------------------  <gets | sets> -------------------------------------------------------------------------------	
		int getScreenWidth() const { return _screenWidth; }
		void setScreenWidth(int val) { _screenWidth = val; }

		int getScreenHeight() const { return _screenHeight; }
		void setScreenHeight(int val) { _screenHeight = val; }

		bool getOptionsOneWindow() const { return _optionsOneWindow; }
		void setOptionsOneWindow(bool val) { _optionsOneWindow = val; }
		// ------------------------------------------------------------------------------  </gets | sets> ------------------------------------------------------------------------------

	private:		
		Mat _originalImage;
		Mat _preprocessedImage;
		Mat _processedImage;
		bool _useCVHiGUI;
		bool _windowsInitialized;
		bool _optionsOneWindow;
		
		int _frameRate;
		int _screenWidth;		
		int _screenHeight;

		int _claehClipLimit;
		int _claehTileXSize;
		int _claehTileYSize;

		int _bilateralFilterDistance;
		int _bilateralFilterSigmaColor;
		int _bilateralFilterSigmaSpace;

		int _contrast;
		int _brightness;		
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <Image analysis>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
