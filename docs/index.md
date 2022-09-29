# Sudoku AI Solver

In this project, a Sudoku Artificial Intelligence (AI) puzzle solver is built using python, OpenCV, Deep Learning (DL) and OCR (Optical Character Recognition) methods to solve puzzles obtained from images. The steps required to implement the solver will be outlined after a brief overview of Sudoku puzzles.

(Work in progress)

## Sudoku
Sudoku is logic-based puzzle where the objective is to fill up a 9 x 9 grid with numbers from 1-9 in each row, each column and each mini 3 x 3 grid/block in such a way that each number does not appear more than once in a row, column or mini grid. Each puzzle contains prefilled numbers with the empty spaces are to be logically filled in with the Sudoku rules in mind. An example of a typical sudoku puzzle is shown below.

<p align='center'>
    <img src='images/sudoku.jpg' width=400>
</p>

## Sudoku AI Solver Steps

The steps needed to build the sudoku AI solver are outlined in the following flow chart (adapted from [Pyimagesearch](https://pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/)).

<p align='center'>
    <img src='images/sudoku_ai_steps.jpg' width=800>
</p>

The sudoku AI solver starts out by accepting an input image containing a sudoku puzzle. Next, OpenCV is applied to locate and extract the sudoku board from the image. After this, each cell of the board is located then checked if there is a digit in each cell or not. If there is a digit present, a Deep Learning trained Optical Character Recognition (OCR) model is employed to identify it. At this point, given the cell locations and digits, a python script is run to solve the sudoku puzzle. Finally, the solved puzzle is displayed as an image to the user.

Most of these steps can be accomplished using OpenCV, however, training the OCR model involves using the Keras and Tensorflow libraries. The packages, libraries and frameworks used in this project are listed below:

- [OpenCV](https://opencv.org/) - Open source library that provides real-time computer vision tools, functions and hardware.
- [JupyterLab](https://jupyter.org/) - Web-based interactive development environment for notebooks, code and data.
- [Tensorflow](https://www.tensorflow.org/) - An Artificial Intelligence library that is used to build, train and deploy Machine Learning and Deep Learning models.
- [Keras](https://keras.io/) - A Deep Learning library that provides an interface for Tensorflow.
- [Numpy](https://numpy.org/doc/stable/index.html) - A Python library used for multidimensional array manipulation and calculations, basic linear algebra, statistical operations and more. It is utilized by OpenCV for array operations. 
- [Matplotlib](https://matplotlib.org/) - Library used for visualizations in Python.
- [Scikit](https://scikit-image.org/) - Library used for image processing in Python.
- [Imutils](https://pypi.org/project/imutils/) - Python package used for basic image processing operations.
- [Pytest](https://docs.pytest.org/en/7.1.x/) - Python testing framework used to write tests for applications and libraries.

Python is one of the main prerequisites for the project and can be downloaded from [here](https://www.python.org/downloads/). The python package manager, pip, is used to install the python packages in the list above -- pip can be installed from [here](https://pip.pypa.io/en/stable/installation/). Afterwards, the python packages are installed by executing this command in a terminal:

```
pip install numpy matplotlib imutils jupyter jupyterlab scikit-image tensorflow pytest
```

Keras is automatically installed with Tensorflow. OpenCV is installed separately and can be downloaded [here](https://docs.opencv.org/4.x/da/df6/tutorial_py_table_of_contents_setup.html) with options for Windows, Fedora and Ubuntu operating systems -- Ubuntu 20.04 is used for this project.

The `sudoku_puzzle_extractor.ipynb` jupyter notebook in this repository is the main file used to build the sudoku AI solver. The file was based on the steps used in this [Pyimagesearch article](https://pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/) with modifications to the functions in the jupyter notebook, the OCR model, the script that solves the extracted puzzle and more. 

A docker container, based on a custom image built on Ubuntu 20.04, which mirrors the project repository with all the necessary packages, libraries and frameworks, can also be used to run the main jupyter notebook to solve image extracted sudoku puzzles. The docker build is detailed in subsequent subsection.

Now that the project software environment has been setup, it is time to build the Sudoku AI Solver.

### Load input image and extract sudoku puzzle

The input image is loaded into the `sudoku_puzzle_extractor.ipynb` notebook in the first line below:

```
img = cv2.imread('sudoku_images/sudoku.jpg')
img = imutils.resize(img, width=600)
```

In the second line, the image is resized to aid with image processing. The image needs to be processed before the puzzle can be extracted from it. The image is processed in the following steps and these are found in the `find_puzzle(img)` function in the notebook:

- <strong>Convert the resized image to grayscale</strong>
 
    This is achieved using the following OpenCV function:

    ```
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ```

    This is a requirement for the other processing steps to work.

- <strong>Filter noise from the image</strong>
  
    The [Gaussian Blur](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1) is used to filter noise from the image. The general form of the `GaussianBlur()` function syntax is as follows:

    <p align='center'><big>GaussianBlur(src, dst, ksize, sigmaX, sigmaY)</big></p>

    - <strong>src</strong> - Input image
    - <strong>dst</strong> - Output image of same size and type as src.
    - <strong>ksize</strong> - Gaussian kernel size[height width ]. The kernel is a group of pixels that move along the src image pixel being worked on by the filter. Height and width must be odd numbers and can have different values. If ksize is set to [0,0], then ksize is computed from sigma value.
    - <strong>sigmaX</strong> - Kernel standard derivation along X-axis.(horizontal direction).
    - <strong>sigmaY</strong> - Kernel standard derivation along Y-axis (vertical direction). If sigmaY = 0 then sigmaX value is taken for sigmaY.

    In the notebook, GaussianBlur is applied on the gray image with the following kernel size and sigmaX value:

    ```
    blurred = cv2.GaussianBlur(gray, (7,7), 3)
    ```
    More information on GaussianBlur and other OpenCV filtering and blurring techniques can be read about [here](https://datacarpentry.org/image-processing/06-blurring/) and [here](https://www.javatpoint.com/opencv-blur).

- <strong>Image thresholding</strong>

    The next image process step is <strong><em>thresholding</em></strong>. In attempting to find the puzzle from an image, being able to detect the edges and shapes within the image are important. Thresholding is a method of segmenting an image into different regions or contours. 
    
    A binary threshold is applied on the blurred grayscale image to convert it to consist of only two values, 0 or 255 - black or white respectively.  

   
### Localize each cell

Lorem ipsum

<!-- Display images of a few extracted digits, like was done in the pyimagesearch -->

### Step 4

Lorem ipsum

### Step 5

Lorem ipsum

### Step 6
<p align='center'>
    <img src='images/extracted_gray.jpg' width=400>
</p>

Lorem ipsum

<p align='center'>
    <img src='images/solved_puzzle.jpg' width=400>
</p>

<!-- Need to circle back to introduce the main cell of the notebook, or do that here? -->
## Docker Image Build
Lorem ipsum

<!-- ```
docker pull thenoobinventor/sudoku-ai-solver:latest
``` -->

## Observations
Lorem ipsum


## Future work/suggestions
- Live stream video solver

- Mask solutions on original skewed image

## References

- [Base Sudoku Solver](https://www.youtube.com/watch?v=tvP_FZ-D9Ng)

- [Sudoku Solver using Computer Vision & Deep Learning](https://aakashjhawar.medium.com/sudoku-solver-using-opencv-and-dl-part-1-490f08701179)

- [Image Processing Sudoku AI](https://becominghuman.ai/image-processing-sudokuai-opencv-45380715a629)

- [OpenCV Sudoku Solver and OCR](https://pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/)

- [Using OpenCV to solve a sudoku](https://golsteyn.com/writing/sudoku)
  
- [OpenCV Docs](https://docs.opencv.org/4.x/d1/dfb/intro.html)
 
- [Understanding OpenCV getperspective transform](https://theailearner.com/tag/cv2-getperspectivetransform/)

- [Dockerfile setup reference](https://github.com/elehcimd/jupyter-opencv)
 
- [Add non-root user in Dockerfile](https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user)

