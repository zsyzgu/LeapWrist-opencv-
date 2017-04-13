#include <iostream>
using namespace std;

extern "C" __declspec(dllexport) void hello() {
  cout << "Hello" << endl;
}
