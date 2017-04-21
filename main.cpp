#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include <cmath>
#include <iostream>
#include <thread>

using namespace std;
using namespace cv;

const int MAXN = 50;
const int BLOCK = 20;
const int KERNEL_DIST = 2;
const int THRESHOLD = 100;

Mat frame;
int n, m, R, C;
int result[MAXN][MAXN];
vector<Point> keys;

vector<Point> getKernelKeys(Mat& image, int threshold) {
  vector<Point> kernelKeys;

  for (int r = 0; r < n; r++) {
    for (int c = 0; c < m; c++) {
      int value = image.at<uchar>(r, c);
      if (value >= threshold) {
        bool flag = true;
        for (int dr = -KERNEL_DIST; dr <= KERNEL_DIST; dr++) {
          for (int dc = -KERNEL_DIST; dc <= KERNEL_DIST; dc++) {
            if (dr == 0 && dc == 0) {
              continue;
            }
            if (0 <= r + dr && r + dr < n && 0 <= c + dc && c + dc < m) {
              if (dr < 0 || (dr == 0 && dc < 0)) {
                if (image.at<uchar>(r + dr, c + dc) >= value) {
                  flag = false;
                }
              } else {
                if (image.at<uchar>(r + dr, c + dc) > value) {
                  flag = false;
                }
              }
            }
          }
        }
        if (flag) {
          kernelKeys.push_back(Point(c, r));
        }
      }
    }
  }

  return kernelKeys;
}

Mat getFrame(bool color = false) {
  return imread("frame3.jpg", color);
}

void output() {
  Mat output = frame;
  for (int r = 0; r < n; r++) {
    for (int c = 0; c < m; c++) {
      output.at<uchar>(r, c) = 0;
    }
  }

  for (int i = 0; i < keys.size(); i++) {
    int c = keys[i].x;
    int r = keys[i].y;
    circle(output, keys[i], 0, 255, 2);
  }

  imshow("window", output);
  waitKey(0);
}

void update() {
  frame = getFrame();

  n = frame.rows;
  m = frame.cols;
  R = n / BLOCK;
  C = m / BLOCK;

  for (int r = 0; r < R; r++) {
    for (int c = 0; c < C; c++) {
      result[r][c] = 0;
    }
  }

  keys = getKernelKeys(frame, THRESHOLD);
  for (int i = 0; i < keys.size(); i++) {
    int c = keys[i].x;
    int r = keys[i].y;
    result[r / BLOCK][c / BLOCK]++;
  }

  for (int r = 0; r < R; r++) {
    for (int c = 0; c < C; c++) {
      cout << result[r][c] << " ";
    }
    cout << endl;
  }

  output();
}

int main()
{
  float beginClock = getTickCount();
  update();
  cout << (float)(getTickCount() - beginClock) / 1e9 << endl;

  return 0;
}
