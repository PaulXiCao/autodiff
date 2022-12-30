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

#include <exception>

// autodiff includes
#include <autodiff/common/eigen.hpp>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <tests/utils/catch.hpp>
using namespace autodiff;

void compareArrays(const detail::VectorX<double>& lhs, const detail::VectorX<double>& rhs)
{
    CHECK(lhs.size() == rhs.size());
}

template<typename Number, typename Array, typename Fun>
void checkGradient(Fun&& f)
{
    /* Define vector x */
    Array x(5);
    x << 2, 3, 5, 7, 9;

    /* Define functions g(x) = f(x)/f(x) - 1 == 0 */
    const auto g = [f](const Array& x) -> Number { return f(x) / f(x) - 1; };

    /* Compute dgdx which is identical to zero by construction */
    {
        const auto dgdx = gradient(g, wrt(x), at(x));
        CHECK(dgdx.size() == 5);
        CHECK(dgdx.squaredNorm() == approx(0.0));
    }

    /* Compute dfdx to be used as reference when checking against dfdw below for different orderings of x entries in w */
    const auto dfdx = gradient(f, wrt(x), at(x));

    /* Compute gradient using pre-allocated storage. */
    {
        double dfdxPre[5];
        Eigen::Map<Eigen::VectorXd> map_5(dfdxPre, 5);
        Number u;
        gradient(f, wrt(x), at(x), u, map_5);
        CHECK(dfdxPre[0] == approx(dfdx[0]));
        CHECK(dfdxPre[1] == approx(dfdx[1]));
        CHECK(dfdxPre[2] == approx(dfdx[2]));
        CHECK(dfdxPre[3] == approx(dfdx[3]));
        CHECK(dfdxPre[4] == approx(dfdx[4]));
    }

    /* Compute dfdw where w = (x1, x2, x3, x4, x0) */
    {
        const auto dfdw = gradient(f, wrt(x.tail(4), x[0]), at(x));
        CHECK(dfdw.size() == 5);
        CHECK(dfdw[0] == approx(dfdx[1]));
        CHECK(dfdw[1] == approx(dfdx[2]));
        CHECK(dfdw[2] == approx(dfdx[3]));
        CHECK(dfdw[3] == approx(dfdx[4]));
        CHECK(dfdw[4] == approx(dfdx[0]));
    }

    /* Compute dfdw where w = (x3, x0, x4) */
    {
        const auto dfdw = gradient(f, wrt(x[3], x[0], x[4]), at(x));
        CHECK(dfdw.size() == 3);
        CHECK(dfdw[0] == approx(dfdx[3]));
        CHECK(dfdw[1] == approx(dfdx[0]));
        CHECK(dfdw[2] == approx(dfdx[4]));
    }

    /* Compute dfdw where w = (x3) */
    {
        const auto dfdw = gradient(f, wrt(x[3]), at(x));
        CHECK(dfdw.size() == 1);
        CHECK(dfdw[0] == approx(dfdx[3]));
    }

    /* Compute dfdw where w = (x0, x1, x2, x3, x4, x0, x1, x2, x3, x4) */
    {
        const auto dfdw = gradient(f, wrt(x, x), at(x));
        CHECK(dfdw.size() == 10);
        CHECK(dfdw[0] == approx(dfdx[0]));
        CHECK(dfdw[1] == approx(dfdx[1]));
        CHECK(dfdw[2] == approx(dfdx[2]));
        CHECK(dfdw[3] == approx(dfdx[3]));
        CHECK(dfdw[4] == approx(dfdx[4]));
        CHECK(dfdw[5] == approx(dfdx[0]));
        CHECK(dfdw[6] == approx(dfdx[1]));
        CHECK(dfdw[7] == approx(dfdx[2]));
        CHECK(dfdw[8] == approx(dfdx[3]));
        CHECK(dfdw[9] == approx(dfdx[4]));
    }
}

template<typename Fun>
void checkRealGradient(Fun&& f)
{
    checkGradient<real, ArrayXreal>(f);
}

template<typename Fun>
void checkDualGradient(Fun&& f)
{
    checkGradient<dual, ArrayXdual>(f);
}

TEST_CASE("testing forward gradient module", "[forward][utils][gradient]")
{
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    SECTION("testing gradient computations")
    {
        checkRealGradient([](const auto& x) { return x.sum(); });
        checkRealGradient([](const auto& x) { return x.exp().sum(); });
        checkRealGradient([](const auto& x) { return x.log().sum(); });
        checkRealGradient([](const auto& x) { return x.tan().sum(); });
        checkRealGradient([](const auto& x) { return (x * x).sum(); });
        checkRealGradient([](const auto& x) { return (x.sin() * x.exp()).sum(); });
        checkRealGradient([](const auto& x) { return (x * x.log()).sum(); });
        checkRealGradient([](const auto& x) { return (x.sin() * x.cos()).sum(); });

        checkDualGradient([](const auto& x) { return x.sum(); });
        // checkDualGradient([](const auto& x) { return x.exp().sum(); });
        // checkDualGradient([](const auto& x) { return x.log().sum(); });
        // checkDualGradient([](const auto& x) { return x.tan().sum(); });
        // checkDualGradient([](const auto& x) { return (x * x).sum(); });
        // checkDualGradient([](const auto& x) { return (x.sin() * x.exp()).sum(); });
        // checkDualGradient([](const auto& x) { return (x * x.log()).sum(); });
        // checkDualGradient([](const auto& x) { return (x.sin() * x.cos()).sum(); });
    }
}

//     SECTION("testing hessian computations")
//     {
//         CHECK_HESSIAN( dual2nd, x.sum() );
//         // CHECK_HESSIAN( dual2nd, x.exp().sum() );
//         // CHECK_HESSIAN( dual2nd, x.log().sum() );
//         // CHECK_HESSIAN( dual2nd, x.tan().sum() );
//         // CHECK_HESSIAN( dual2nd, (x * x).sum() );
//         // CHECK_HESSIAN( dual2nd, (x.sin() * x.exp()).sum() );
//         // CHECK_HESSIAN( dual2nd, (x * x.log()).sum() );
//         // CHECK_HESSIAN( dual2nd, (x.sin() * x.cos()).sum() );
//     }

//     SECTION("testing jacobian computations")
//     {
//         CHECK_JACOBIAN( real, x );
//         CHECK_JACOBIAN( real, x.exp() );
//         CHECK_JACOBIAN( real, x.log() );
//         CHECK_JACOBIAN( real, x.tan() );
//         CHECK_JACOBIAN( real, (x * x) );
//         CHECK_JACOBIAN( real, (x.sin() * x.exp()) );
//         CHECK_JACOBIAN( real, (x * x.log()) );
//         CHECK_JACOBIAN( real, (x.sin() * x.cos()) );

//         CHECK_JACOBIAN( dual, x );
//         CHECK_JACOBIAN( dual, x.exp() );
//         CHECK_JACOBIAN( dual, x.log() );
//         CHECK_JACOBIAN( dual, x.tan() );
//         CHECK_JACOBIAN( dual, (x * x) );
//         CHECK_JACOBIAN( dual, (x.sin() * x.exp()) );
//         CHECK_JACOBIAN( dual, (x * x.log()) );
//         CHECK_JACOBIAN( dual, (x.sin() * x.cos()) );
//     }
// }

// #define CHECK_HESSIAN(type, expr) \
// { \
//     /* Define vector x and matrix y */ \
//     double y[25]; \
//     ArrayX##type x(5); \
//     x << 2.0, 3.0, 5.0, 7.0, 9.0; \
//     /* Define functions f(x) = expr/expr - 1 == 0 and g(x) = expr */ \
//     std::function<type(const ArrayX##type&)> f, g; \
//     f = [](const ArrayX##type& x) -> type { return expr/expr - 1.0; }; \
//     g = [](const ArrayX##type& x) -> type { return expr; }; \
//     /* Auxiliary matrices fxx, gxx, gww where w is a vector with a combination of x entries */ \
//     Eigen::MatrixXd fxx, gxx, gww; \
//     Eigen::Map<Eigen::MatrixXd> map_5x5(y, 5, 5); \
//     /* The indices of the x variables in the w vector */ \
//     std::vector<size_t> iw; \
//     /* Compute fxx which is identical to zero by construction */ \
//     fxx = hessian(f, wrt(x), at(x)); \
//     CHECK( fxx.rows() == 5 ); \
//     CHECK( fxx.cols() == 5 ); \
//     CHECK( fxx.squaredNorm() == approx(0.0) ); \
//     /* Compute gxx to be used as reference when checking againts gww below for different orderings of x entries in w */ \
//     gxx = hessian(g, wrt(x), at(x)); \
//     /* Compute hessian using pre-allocated storage */ \
//     type u; \
//     VectorXd grad; \
//     hessian(g, wrt(x), at(x), u, grad, map_5x5); \
//     for(size_t i = 0; i < gxx.rows(); ++i) \
//         for(size_t j = 0; j < gxx.cols(); ++j) \
//             CHECK( gxx(i, j) == approx(map_5x5(i, j)) ); \
//     /* Compute gww where w = (x1, x2, x3, x4, x0) */ \
//     gww = hessian(g, wrt(x.tail(4), x[0]), at(x)); \
//     iw = {1, 2, 3, 4, 0}; \
//     CHECK( gww.rows() == iw.size() ); \
//     CHECK( gww.rows() == gww.cols() ); \
//     for(size_t i = 0; i < iw.size(); ++i) \
//         for(size_t j = 0; j < iw.size(); ++j) \
//             CHECK( gxx(iw[i], iw[j]) == approx(gww(i, j)) ); \
//     /* Compute gww where w = (x3, x0, x4) */ \
//     gww = hessian(g, wrt(x[3], x[0], x[4]), at(x)); \
//     iw = {3, 0, 4}; \
//     CHECK( gww.rows() == iw.size() ); \
//     CHECK( gww.rows() == gww.cols() ); \
//     for(size_t i = 0; i < iw.size(); ++i) \
//         for(size_t j = 0; j < iw.size(); ++j) \
//             CHECK( gxx(iw[i], iw[j]) == approx(gww(i, j)) ); \
//     /* Compute gww where w = (x3) */ \
//     gww = hessian(g, wrt(x[3]), at(x)); \
//     iw = {3}; \
//     CHECK( gww.rows() == iw.size() ); \
//     CHECK( gww.rows() == gww.cols() ); \
//     for(size_t i = 0; i < iw.size(); ++i) \
//         for(size_t j = 0; j < iw.size(); ++j) \
//             CHECK( gxx(iw[i], iw[j]) == approx(gww(i, j)) ); \
//     /* Compute gww where w = (x0, x1, x2, x3, x4, x0, x1, x2, x3, x4) */ \
//     gww = hessian(g, wrt(x, x), at(x)); \
//     iw = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4}; \
//     CHECK( gww.rows() == iw.size() ); \
//     CHECK( gww.rows() == gww.cols() ); \
//     for(size_t i = 0; i < iw.size(); ++i) \
//         for(size_t j = 0; j < iw.size(); ++j) \
//             CHECK( gxx(iw[i], iw[j]) == approx(gww(i, j)) ); \
// }

// #define CHECK_JACOBIAN(type, expr) \
// { \
//     /* Define vector x and matrix y */ \
//     double y[25]; \
//     ArrayX##type x(5); \
//     x << 2.0, 3.0, 5.0, 7.0, 9.0; \
//     /* Define functions f(x) = expr/expr - 1 == 0 and g(x) = expr */ \
//     std::function<ArrayX##type(const ArrayX##type&)> f, g; \
//     f = [](const ArrayX##type& x) -> ArrayX##type { return expr/expr - 1.0; }; \
//     g = [](const ArrayX##type& x) -> ArrayX##type { return expr; }; \
//     /* Auxiliary matrices dfdx, dgdx, dgdw where w is a vector with a combination of x entries */ \
//     Eigen::MatrixXd dfdx, dgdx, dgdw; \
//     Eigen::Map<Eigen::MatrixXd> map_5x5(y, 5, 5); \
//     Eigen::Map<Eigen::MatrixXd> map_5x3(y, 5, 3); \
//     /* Compute dfdx which is identical to zero by construction */ \
//     dfdx = jacobian(f, wrt(x), at(x)); \
//     CHECK( dfdx.rows() == 5 ); \
//     CHECK( dfdx.cols() == 5 ); \
//     CHECK( dfdx.squaredNorm() == approx(0.0) ); \
//     /* Compute dgdx to be used as reference when checking againts dgdw below for different orderings of x entries in w */ \
//     dgdx = jacobian(g, wrt(x), at(x)); \
//     /* Compute square jacobian using pre-allocated storage */ \
//     VectorX##type Gval; \
//     jacobian(g, wrt(x), at(x), Gval, map_5x5); \
//     CHECK( dgdx.col(0).isApprox(map_5x5.col(0)) ); \
//     CHECK( dgdx.col(1).isApprox(map_5x5.col(1)) ); \
//     CHECK( dgdx.col(2).isApprox(map_5x5.col(2)) ); \
//     CHECK( dgdx.col(3).isApprox(map_5x5.col(3)) ); \
//     CHECK( dgdx.col(4).isApprox(map_5x5.col(4)) ); \
//     /* Compute rectangular jacobian using pre-allocated storage */ \
//     jacobian(g, wrt(x[0], x[1], x[2]), at(x), Gval, map_5x3); \
//     CHECK( dgdx.col(0).isApprox(map_5x3.col(0)) ); \
//     CHECK( dgdx.col(1).isApprox(map_5x3.col(1)) ); \
//     CHECK( dgdx.col(2).isApprox(map_5x3.col(2)) ); \
//     /* Compute dgdw where w = (x1, x2, x3, x4, x0) */ \
//     dgdw = jacobian(g, wrt(x.tail(4), x[0]), at(x)); \
//     CHECK( dgdw.rows() == 5 ); \
//     CHECK( dgdw.cols() == 5 ); \
//     CHECK( dgdw.col(0).isApprox(dgdx.col(1)) ); \
//     CHECK( dgdw.col(1).isApprox(dgdx.col(2)) ); \
//     CHECK( dgdw.col(2).isApprox(dgdx.col(3)) ); \
//     CHECK( dgdw.col(3).isApprox(dgdx.col(4)) ); \
//     CHECK( dgdw.col(4).isApprox(dgdx.col(0)) ); \
//     /* Compute dgdw where w = (x3, x0, x4) */ \
//     dgdw = jacobian(g, wrt(x[3], x[0], x[4]), at(x)); \
//     CHECK( dgdw.rows() == 5 ); \
//     CHECK( dgdw.cols() == 3 ); \
//     CHECK( dgdw.col(0).isApprox(dgdx.col(3)) ); \
//     CHECK( dgdw.col(1).isApprox(dgdx.col(0)) ); \
//     CHECK( dgdw.col(2).isApprox(dgdx.col(4)) ); \
//     /* Compute dgdw where w = (x3) */ \
//     dgdw = jacobian(g, wrt(x[3]), at(x)); \
//     CHECK( dgdw.rows() == 5 ); \
//     CHECK( dgdw.cols() == 1 ); \
//     CHECK( dgdw.col(0).isApprox(dgdx.col(3)) ); \
//     /* Compute dgdw where w = (x0, x1, x2, x3, x4, x0, x1, x2, x3, x4) */ \
//     dgdw = jacobian(g, wrt(x, x), at(x)); \
//     CHECK( dgdw.rows() == 5 ); \
//     CHECK( dgdw.cols() == 10 ); \
//     CHECK( dgdw.col(0).isApprox(dgdx.col(0)) ); \
//     CHECK( dgdw.col(1).isApprox(dgdx.col(1)) ); \
//     CHECK( dgdw.col(2).isApprox(dgdx.col(2)) ); \
//     CHECK( dgdw.col(3).isApprox(dgdx.col(3)) ); \
//     CHECK( dgdw.col(4).isApprox(dgdx.col(4)) ); \
//     CHECK( dgdw.col(5).isApprox(dgdx.col(0)) ); \
//     CHECK( dgdw.col(6).isApprox(dgdx.col(1)) ); \
//     CHECK( dgdw.col(7).isApprox(dgdx.col(2)) ); \
//     CHECK( dgdw.col(8).isApprox(dgdx.col(3)) ); \
//     CHECK( dgdw.col(9).isApprox(dgdx.col(4)) ); \
// }
