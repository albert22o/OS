#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
using namespace std;
typedef std::chrono::milliseconds ms;
typedef std::chrono::nanoseconds ns;

int main()
{
  for (long n = 10; n <= 100000000; n *=10){
    cout << endl << "n = " << n << endl;
    vector<float> a(n), b(n), c(n);
  chrono::time_point<chrono::system_clock> start, end;

  for (int i = 0; i < n; ++i)
  {
    a[i] = i;
    b[i] = i * 2;
  }

  start = chrono::system_clock::now();
  for (int i = 0; i < n; ++i)
  {
    c[i] = a[i] + b[i];
  }
  end = chrono::system_clock::now();

  cout << "Wasted time: " << chrono::duration_cast<ms>(end - start).count() << "ms" << endl
       << chrono::duration_cast<ns>(end - start).count() << "ns" << endl;
}
  return 0;
}
