#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include <cmath>
#include <iostream>

using namespace std;
using namespace cv;

const int BLOCK = 40;
const int NEIGH_R = 40;
const int NEIGH_C = 40;

Point calnPoint(Mat& image, Mat& kernel, int indexR, int indexC) {
  int maxValue = 0;
  Point ret;

  for (int r = indexR * BLOCK - NEIGH_R; r <= indexR * BLOCK + NEIGH_R; r++) {
    for (int c = indexC * BLOCK - NEIGH_C; c <= indexC * BLOCK + NEIGH_C; c++) {
      if (0 <= r && r + BLOCK < image.rows && 0 <= c && c + BLOCK < image.cols) {
        int value = 0;
        int total = 0;
        for (int kr = 0; kr < BLOCK; kr++) {
          for (int kc = 0; kc < BLOCK; kc++) {
            total += (int)image.at<uchar>(r + kr, c + kc);
            value += (int)image.at<uchar>(r + kr, c + kc) * kernel.at<uchar>(kr, kc);
          }
        }
        value /= total;
        if (value > maxValue) {
          maxValue = value;
          ret = Point(c + BLOCK / 2, r + BLOCK / 2);
        }
      }
    }
  }

  return ret;
}

int main()
{
  Mat origin = imread("1.jpg", 0);
  Mat frame = imread("2.jpg", 0);
  assert(origin.rows == frame.rows && origin.cols == frame.cols);

  Mat blurredFrame;
  GaussianBlur(frame, blurredFrame, Size(3, 3), 0);

  int R = origin.rows / BLOCK;
  int C = origin.cols / BLOCK;

    for (int r = 0; r < R; r++) {
      for (int c = 0; c < C; c++) {
        Mat kernel = origin(Range(r * BLOCK, (r + 1) * BLOCK), Range(c * BLOCK, (c + 1) * BLOCK));
        Point point = calnPoint(blurredFrame, kernel, r, c);
        circle(frame, point, 3, 255, 2);
      }
    }

  imshow("window", frame);
  waitKey(0);

  return 0;
}
