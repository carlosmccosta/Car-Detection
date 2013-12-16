#pragma once

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

#define IMAGE_TOKEN ".image.png"
#define MASK_TOKEN ".mask."
#define MASK_EXTENSION ".png"

#define IMGS_DIRECTORY "imgs/devImgs/cars/"
#define VOCABULARY_IMG_LIST "imgs/devImgs/cars_train.txt"
#define CLASSIFIER_IMG_LIST "imgs/devImgs/cars_train.txt"

#define VOCABULARY_WORD_COUNT 1000
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </constants definitions> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<