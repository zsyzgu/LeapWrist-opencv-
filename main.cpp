#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include <cmath>
#include <iostream>
#include <thread>

using namespace std;
using namespace cv;

const int THREAD_NUMBER = 16;
const int MAXN = 100;
const int MAXM = 1000;
const int BLOCK = 40;
const int NEIGH_R = 30;
const int NEIGH_C = 30;
const int BLUR_SIZE = 3;
const float KERNEL_KEY_COE = 0.9;
const int EXIST_THRESHOLD = 150;

int completedThread;
mutex threadMutex;
Mat origin, frame;
vector<Point> kernelKeys[MAXN][MAXN];
int R, C;
int n, m;
int blockSum[MAXM][MAXM];
Point result[MAXN][MAXN];

Point3_<int> calnFeaturePoint(int indexR, int indexC) {
  int rBegin = indexR * BLOCK - NEIGH_R, rEnd = indexR * BLOCK + NEIGH_R;
  int cBegin = indexC * BLOCK - NEIGH_C, cEnd = indexC * BLOCK + NEIGH_C;
  vector<Point>& keys = kernelKeys[indexR][indexC];

  float maxValue = 0;
  Point3_<int> ret;

  for (int r = rBegin; r <= rEnd; r++) {
    for (int c = cBegin; c <= cEnd; c++) {
      if (0 <= r && r + BLOCK < n && 0 <= c && c + BLOCK < m) {
        float value = 0;
        for (int i = 0; i < keys.size(); i++) {
          int kr = keys[i].y;
          int kc = keys[i].x;
          value += frame.at<uchar>(r + kr, c + kc);
        }
        int z = value / keys.size();
        value = value / blockSum[r][c];
        
        if (value > maxValue) {
          maxValue = value;
          ret = Point3_<int>(c + BLOCK / 2, r + BLOCK / 2, z);
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

void calnBlockSum() {
  for (int r = 0; r < n; r++) {
    blockSum[r + 1][0] = 0;
    for (int c = 0; c < m; c++) {
      blockSum[r + 1][c + 1] = blockSum[r + 1][c] + frame.at<uchar>(r, c);
    }
    for (int c = 0; c + BLOCK <= m; c++) {
      blockSum[r + 1][c] = blockSum[r + 1][c + BLOCK] - blockSum[r + 1][c];
    }
  }

  for (int c = 0; c < m; c++) {
    blockSum[0][c] = 0;
    for (int r = 0; r < n; r++) {
      blockSum[r + 1][c] += blockSum[r][c];
    }
    for (int r = 0; r + BLOCK <= n; r++) {
      blockSum[r][c] = blockSum[r + BLOCK][c] - blockSum[r][c];
    }
  }
}

void subThread(int tid) {
  for (int r = 0; r < R; r++) {
    for (int c = tid; c < C; c += THREAD_NUMBER) {
      Point3_<int> point = calnFeaturePoint(r, c);
      if (point.z >= EXIST_THRESHOLD) {
        result[r][c] = Point(point.x, point.y);
      } else {
        result[r][c] = Point(0, 0);
      }
    }
  }

  threadMutex.lock();
  completedThread++;
  threadMutex.unlock();
}

Mat getOrigin() {
  return imread("origin.jpg", 0);
}

Mat getFrame() {
  return imread("frame.jpg", 0);
}

void start() {
  origin = getOrigin();
  
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

void update() {
  frame = getFrame();

  GaussianBlur(frame, frame, Size(BLUR_SIZE, BLUR_SIZE), 0);
  calnBlockSum();
  completedThread = 0;
  for (int tid = 0; tid < THREAD_NUMBER; tid++) {
    thread t(&subThread, tid);
    if (tid == THREAD_NUMBER - 1) {
      t.join();
    } else {
      t.detach();
    }
  }
  while(true) {
    threadMutex.lock();
    if (completedThread == THREAD_NUMBER) {
      threadMutex.unlock();
      break;
    }
    threadMutex.unlock();
  }
}

void output() {
  for (int r = 0; r < R; r++) {
    for (int c = 0; c < C; c++) {
      circle(frame, result[r][c], 3, 255, 2);
    }
  }

  imshow("window", frame);
  waitKey(0);
}

int main()
{
  start();
  float beginClock = getTickCount();
  update();
  cout << (float)(getTickCount() - beginClock) / 1e9 << endl;
  output();

  return 0;
}
