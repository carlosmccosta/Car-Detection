

  INRIA Annotations for Graz-02 (IG02)


  INRIA Annotations for Graz-02 (IG02) is a reedition of the popular
  natural-scene object category dataset prepared at Graz University of
  Technology. The new annotations created at INRIA are aimed to be
  object-oriented and more precise. This dataset includes photos with objects
  of high complexity and high intra-class variability on cluttered
  backgrounds.

  The images were collected by Andreas Opelt and Axel Pinz. The annotation was
  led by Marcin Marszalek and Cordelia Schmid. We thank Axel Pinz for his
  support and comments.


  Details

  Our team re-annotated cars, bicycles and people on the original set of
  images. Only some portrait images incorrectly saved as landscape images were
  corrected. For each object a segmentation mask was drawn, which includes
  occluded object parts (marked with a different color). Each object was
  marked "truncated" when it was cut by the image edge, "multiple" when it
  could not be separated from other objects of this class and "difficult" if
  it was hard to notice or segment. The images were then considered as
  suitable for training (when there was at least one non-truncated and
  non-multiple object in the image) and testing (if all the objects in the
  image could be individually segmented). As those lists had some overlap, we
  have randomly partitioned it to create a suggested balanced split into a
  training set and a test set.                                                                 


  Statistics

  +--------------------------------------------------------------------------+
  |       |                                |         Images         |        |
  | Class |              File              |------------------------|Objects |
  |       |                                |total|training |testing |        |
  |-------+--------------------------------+-----+---------+--------+--------|
  |Bikes  |ig02-v1.0-bikes.zip (208 MB)    |365  |162      |162     |511     |
  |-------+--------------------------------+-----+---------+--------+--------|
  |Cars   |ig02-v1.0-cars.zip (205 MB)     |420  |177      |177     |770     |
  |-------+--------------------------------+-----+---------+--------+--------|
  |People |ig02-v1.0-people.zip (152 MB)   |311  |140      |140     |785     |
  +--------------------------------------------------------------------------+


  Related publications

  To cite the dataset please use:

  Marcin Marszalek and Cordelia Schmid. Accurate Object Localization with
  Shape Masks. IEEE Conference on Computer Vision & Pattern Recognition, 2007.

  A. Opelt, A. Pinz, M. Fussenegger, P. Auer. Generic Object Recognition with
  Boosting. IEEE Transactions on Pattern Recognition and Machine Intelligence
  (PAMI), Vol.28, No. 3, March 2006.

  Please let us know about publications in which you use our dataset, so we
  can refer to them here.


  Bugs and updates

  If you spot any errors in the annotations, please report them to Marcin
  Marszalek. Corrections of the segmentations are especially welcome. When the
  number of corrections becomes significant, we will release an update.

