#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include <cmath>
#include <iostream>

using namespace std;
using namespace cv;

const int MAXN = 100;
const int MAXM = 1500;
const int BLOCK = 40;
const int NEIGH_R = 30;
const int NEIGH_C = 30;
const int BLUR_SIZE = 3;
const float KERNEL_KEY_COE = 0.9;
const int EXIST_THRESHOLD = 150;

Mat origin, frame, blurredFrame;
vector<Point> kernelKeys[MAXN][MAXN];
int R, C;
int n, m;
int sum2[MAXM][MAXM];

int calnSum(int r, int c) {
  int sum = sum2[r + BLOCK - 1][c + BLOCK - 1];
  if (r - 1 >= 0) {
    sum -= sum2[r - 1][c + BLOCK - 1];
  }
  if (c - 1 >= 0) {
    sum -= sum2[r + BLOCK - 1][c - 1];
  }
  if (r - 1 >= 0 && c - 1 >= 0) {
    sum += sum2[r - 1][c - 1];
  }
  return sum;
}

Point3_<int> calnFeaturePoint(int indexR, int indexC) {
  int rBegin = indexR * BLOCK - NEIGH_R, rEnd = indexR * BLOCK + NEIGH_R;
  int cBegin = indexC * BLOCK - NEIGH_C, cEnd = indexC * BLOCK + NEIGH_C;
  vector<Point>& keys = kernelKeys[indexR][indexC];

  float maxAve = 0;
  Point3_<int> ret;

  for (int r = rBegin; r <= rEnd; r++) {
    for (int c = cBegin; c <= cEnd; c++) {
      if (0 <= r && r + BLOCK < n && 0 <= c && c + BLOCK < m) {
        float value = 0;
        for (int i = 0; i < keys.size(); i++) {
          int kr = keys[i].y;
          int kc = keys[i].x;
          value += blurredFrame.at<uchar>(r + kr, c + kc);
        }
        int sum = calnSum(r, c);
        
        if (value / sum > maxAve) {
          maxAve = value / sum;
          ret = Point3_<int>(c + BLOCK / 2, r + BLOCK / 2, value / keys.size());
        }
      }
    }
  }

  return ret;
}

vector<Point> getKernelKeys(int indexR, int indexC) {
  int rBegin = indexR * BLOCK, rEnd = rBegin + BLOCK;
  int cBegin = indexC * BLOCK, cEnd = cBegin + BLOCK;

  uchar maxValue = 0;
  for (int r = rBegin; r < rEnd; r++) {
    for (int c = cBegin; c < cEnd; c++) {
      maxValue = max(maxValue, origin.at<uchar>(r, c));
    }
  }
  int threshold = maxValue * KERNEL_KEY_COE;
  vector<Point> kernelKeys;
  for (int r = 0; r < BLOCK; r++) {
    for (int c = 0; c < BLOCK; c++) {
      int value = origin.at<uchar>(rBegin + r, cBegin + c);
      if (value >= threshold) {
        if (r - 1 >= 0 && origin.at<uchar>(r - 1, c) > value) continue;
        if (r + 1 <  n && origin.at<uchar>(r + 1, c) > value) continue;
        if (c - 1 >= 0 && origin.at<uchar>(r, c - 1) > value) continue;
        if (c + 1 <  m && origin.at<uchar>(r, c + 1) > value) continue;
        kernelKeys.push_back(Point(c, r));
      }
    }
  }
  return kernelKeys;
}

void initOrigin() {
  n = origin.rows;
  m = origin.cols;
  R = n / BLOCK;
  C = m / BLOCK;
  for (int r = 0; r < R; r++) {
    for (int c = 0; c < C; c++) {
      kernelKeys[r][c] = getKernelKeys(r, c);
    }
  }
}

void initFrame() {
  for (int r = 0; r < n; r++) {
    int sum1 = 0;
    for (int c = 0; c < m; c++) {
      sum1 += blurredFrame.at<uchar>(r, c);
      sum2[r][c] = sum1;
      if (r - 1 >= 0) {
        sum2[r][c] += sum2[r - 1][c];
      }
    }
  }
}

int main()
{
  origin = imread("1.jpg", 0);
  frame = imread("2.jpg", 0);
  assert(origin.rows == frame.rows && origin.cols == frame.cols);
  initOrigin();
  
  GaussianBlur(frame, blurredFrame, Size(BLUR_SIZE, BLUR_SIZE), 0);
  initFrame();
  for (int r = 0; r < R; r++) {
    for (int c = 0; c < C; c++) {
      Point3_<int> point = calnFeaturePoint(r, c);
      if (point.z >= EXIST_THRESHOLD) {
        circle(frame, Point(point.x, point.y), 3, 255, 2);
      }
    }
  }

  imshow("window", frame);
  waitKey(0);

  return 0;
}
