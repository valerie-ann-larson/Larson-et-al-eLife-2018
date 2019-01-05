README for Identify_Axons code

This code is designed to work with Anaconda (Python 2.7 version). The PyFits package is no longer required.

Put original files and threshold files in the input_files directory (must be in .tif format). Threshold files should be named '[Original File Name]_threshold.tif'. Cartoon example files are included here.

make_font.py and font_240_120_128.fits are necessary for the code to run, but do not need to be opened or run independently by the user.

To run the code, run Identify_Axons.py

Output is generated in the input_files folder. 
Axon areas (in pixels) are found in column B. These values can be used to calculate axon diameter. Convert pixels to units of length based on the parameters of your input images.
Column F of the output file is a test of non-circularity. If a region has a value>2, it is determined to not represent an axon and the value in column G is marked as '0' and it is colored red in the output image. If the value is <2, it is assumed to be an axon and the value in column G is marked as '1' and it is colored green in the output image.
