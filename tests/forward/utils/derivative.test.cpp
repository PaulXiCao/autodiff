//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2018-2022 Allan Leal
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// C++ includes
#include <vector>

// Catch includes
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// autodiff includes
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

TEST_CASE("testing forward derivative module", "[forward][utils][derivative][seed]")
{
    SECTION("testing seed operations for higher-order cross derivatives...")
    {
        dual4th x, y;

        seed(wrt(x, y, x, y));

        CHECK(val(x.grad) == 1.0);
        CHECK(val(x.val.val.grad) == 1.0);

        CHECK(val(y.val.grad) == 1.0);
        CHECK(val(y.val.val.val.grad) == 1.0);
    }

    SECTION("testing seed operations for higher-order directional derivatives using real4th...")
    {
        real4th x, y;

        seed(at(x, y), along(2, 3));

        CHECK(x[1] == 2.0);
        CHECK(y[1] == 3.0);
    }

    SECTION("testing seed operations for higher-order directional derivatives using dual4th...")
    {
        dual4th x, y;

        seed(at(x, y), along(2, 3));

        CHECK(derivative<1>(x) == 2.0);
        CHECK(derivative<1>(y) == 3.0);
    }

    SECTION("testing seed operations for higher-order directional derivatives using std::vector...")
    {
        std::vector<real4th> x(4);

        real4th y;

        std::vector<double> v = {2.0, 3.0, 4.0, 5.0};

        seed(at(x, y), along(v, 7.0));

        CHECK(x[0][1] == 2.0);
        CHECK(x[1][1] == 3.0);
        CHECK(x[2][1] == 4.0);
        CHECK(x[3][1] == 5.0);
        CHECK(y[1] == 7.0);

        unseed(at(x, y));

        CHECK(x[0][1] == 0.0);
        CHECK(x[1][1] == 0.0);
        CHECK(x[2][1] == 0.0);
        CHECK(x[3][1] == 0.0);
        CHECK(y[1] == 0.0);
    }
}

TEST_CASE("testing forward derivative module", "[forward][utils][derivative][gradient]")
{
    constexpr auto atol = 1e-16;

    constexpr auto f = [](const real& x, const real& y, const real& z) { return x + 2 * y * z * z; };
    constexpr auto dfdx = [](const auto x, const auto y, const auto z) { return 1; };
    constexpr auto dfdy = [](const auto x, const auto y, const auto z) { return 2 * z * z; };
    constexpr auto dfdz = [](const auto x, const auto y, const auto z) { return 4 * y * z; };

    SECTION("wrt, at: array of 3 separate scalars")
    {
        using Catch::Matchers::WithinAbs;

        constexpr auto x0 = 1, y0 = -2, z0 = 5;
        real x = x0, y = y0, z = z0;

        const auto fx = gradient(f, wrt(x), at(x, y, z)); // = [df/dx]
        CHECK_THAT(fx[0], WithinAbs(dfdx(x0, y0, z0), atol));

        const auto fz = gradient(f, wrt(z), at(x, y, z)); // = [df/dz]
        CHECK_THAT(fz[0], WithinAbs(dfdz(x0, y0, z0), atol));

        const auto fyxz = gradient(f, wrt(y, x, z), at(x, y, z)); // = [df/dy, df/dx, df/dz]
        CHECK_THAT(fyxz[0], WithinAbs(dfdy(x0, y0, z0), atol));
        CHECK_THAT(fyxz[1], WithinAbs(dfdx(x0, y0, z0), atol));
        CHECK_THAT(fyxz[2], WithinAbs(dfdz(x0, y0, z0), atol));
    }

    SECTION("wrt, at: variables: (2-dim, 1-dim)")
    {
        using Catch::Matchers::WithinAbs;

        constexpr auto g = [f](const Array2real& xy, const real& z) { return f(xy[0], xy[1], z); };

        constexpr auto x0 = 1, y0 = -2, z0 = 5;
        Array2real xy(2);
        xy << x0, y0;
        real z = z0;

        const auto gxy = gradient(g, wrt(xy), at(xy, z)); // = [df/dx, df/dy]
        CHECK_THAT(gxy[0], WithinAbs(dfdx(x0, y0, z0), atol));
        CHECK_THAT(gxy[1], WithinAbs(dfdy(x0, y0, z0), atol));

        const auto gz = gradient(g, wrt(z), at(xy, z)); // = [df/dz]
        CHECK_THAT(gz[0], WithinAbs(dfdz(x0, y0, z0), atol));

        const auto gzxy = gradient(g, wrt(z, xy), at(xy, z)); // = [df/dz, df/dx, df/dy]
        CHECK_THAT(gzxy[0], WithinAbs(dfdz(x0, y0, z0), atol));
        CHECK_THAT(gzxy[1], WithinAbs(dfdx(x0, y0, z0), atol));
        CHECK_THAT(gzxy[2], WithinAbs(dfdy(x0, y0, z0), atol));

        const auto gxyz = gradient(g, wrt(xy, z), at(xy, z)); // = [df/dx, df/dy, df/dz]
        CHECK_THAT(gxyz[0], WithinAbs(dfdx(x0, y0, z0), atol));
        CHECK_THAT(gxyz[1], WithinAbs(dfdy(x0, y0, z0), atol));
        CHECK_THAT(gxyz[2], WithinAbs(dfdz(x0, y0, z0), atol));
    }
}
