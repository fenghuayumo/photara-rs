#include "../src/realtime/navigation.h"
#include <cstdlib>
#include <iostream>

using photara::realtime::Navigation;
static void check(bool ok) { if (!ok) std::abort(); }
static float travel(int fps, bool repeat)
{
  Navigation n;
  n.advance(0, true);
  n.held[2] = true;
  float z = 0;
  for (int i = 1; i <= fps; ++i) {
    z += n.advance(double(i) / fps, true)[2];
    if (repeat) n.held[2] = true;
  }
  return z;
}
int main()
{
  for (int fps : {30, 60, 144}) {
    check(std::abs(travel(fps, false) - 2) < 1e-5);
    check(std::abs(travel(fps, true) - 2) < 1e-5);
  }
  Navigation n;
  n.advance(0, true);
  n.held[1] = n.held[2] = true;
  auto d = n.advance(.05, true);
  check(std::abs(std::sqrt(d[0]*d[0]+d[2]*d[2]) - .1f) < 1e-6);
  n.held[0] = n.held[3] = true;
  d = n.advance(.1, true);
  check(d[0] == 0 && d[2] == 0);
  n.held.fill(false);
  check(!n.active());
  n.held[2] = true;
  check(n.advance(20, false)[2] == 0);
  check(n.advance(20.01, true)[2] < .021f);
  check(n.advance(100, true)[2] <= .201f);
  n.held.fill(false);
  check(n.advance(100.01, true)[2] == 0);
  std::cout << "Navigation timing, repeat, diagonals, release and pause checks passed\n";
}
