#include <string>
#include <cstdio>
#include "lmv_seq.h"
#include "lmv_vcl.h"

#define REQUIRE(expr) Require(expr, __LINE__)

using namespace hasc;

bool Approx(double a, double b)
{
    return (std::abs(a-b) < std::numeric_limits<double>::epsilon());
}

void Require(bool expr, int line)
{
    if (!expr) {
        std::string str = "test failure at line number " + std::to_string(line);
        std::exit(1);
    } else {
        std::string str = "test success at line number " + std::to_string(line);
        std::printf("%s\n", str.c_str());
    }
}

int main()
{
  //TODO: tests
    return 0;

}
