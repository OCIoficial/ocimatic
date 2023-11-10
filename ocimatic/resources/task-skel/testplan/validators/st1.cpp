#include "testlib.h"

int main() {
  registerValidation();
  inf.readInt(-1e9, 1e9, "a");
  inf.readSpace();
  inf.readInt(-1e9, 1e9, "b");
  inf.readEoln();
  inf.readEof();
}
