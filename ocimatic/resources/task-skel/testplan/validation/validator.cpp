#include "testlib.h"

typedef long long llong;

std::tuple<llong, llong> read() {
  llong a = inf.readLong(-3'000'000'000ll, 3'000'000'000ll, "a");
  inf.readSpace();
  llong b = inf.readLong(-3'000'000'000ll, 3'000'000'000ll, "b");
  inf.readEoln();
  inf.readEof();

  return {a, b};
}

int main(int argc, char *argv[]) {
  registerValidation();
  auto [a, b] = read();

  if (!strcmp(argv[1], "st1")) {
    ensure(a >= -1'000'000'000ll && a <= 1'000'000'000ll);
    ensure(b >= -1'000'000'000ll && b <= 1'000'000'000ll);
  }
}
