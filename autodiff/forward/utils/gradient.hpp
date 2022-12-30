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

#pragma once

// autodiff includes
#include <autodiff/common/classtraits.hpp>
#include <autodiff/common/eigen.hpp>
#include <autodiff/common/meta.hpp>
#include <autodiff/forward/utils/derivative.hpp>

namespace autodiff {
namespace detail {

/// Return the length of an item in a `wrt(...)` list.
template<typename Item>
auto wrt_item_length(const Item& item) -> size_t
{
    if constexpr(isVector<Item>)
        return item.size(); // if item is a vector, return its size
    else
        return 1; // if not a vector, say, a number, return 1 for its length
}

/// Return the sum of lengths of all itens in a `wrt(...)` list.
template<typename... Vars>
auto wrt_total_length(const Wrt<Vars...>& wrt) -> size_t
{
    return Reduce(
        wrt.args, [&](auto&& item) constexpr {
            return wrt_item_length(item);
        });
}

// Loop through each variable in a wrt list and apply a function f(i, x) that
// accepts an index i and the variable x[i], where i is the global index of the
// variable in the list.
template<typename Function, typename... Vars>
constexpr auto ForEachWrtVar(const Wrt<Vars...>& wrt, Function&& f)
{
    auto i = 0; // the current index of the variable in the wrt list
    ForEach(
        wrt.args, [&](auto& item) constexpr {
            using T = decltype(item);
            static_assert(isVector<T> || Order<T> > 0, "Expecting a wrt list with either vectors or individual autodiff numbers.");
            if constexpr(isVector<T>) {
                for(auto j = 0; j < item.size(); ++j)
                    // call given f with current index and variable from item (a vector)
                    if constexpr(detail::has_operator_bracket<T>())
                        f(i++, item[j]);
                    else
                        f(i++, item(j));
            } else
                f(i++, item); // call given f with current index and variable from item (a number, not a vector)
        });
}

/// Return the gradient of scalar function *f* with respect to some or all variables *x*.
template<typename Fun, typename... Vars, typename... Args, typename FunRet, typename Vec>
void gradient(const Fun& f, const Wrt<Vars...>& wrt, const At<Args...>& at, FunRet& u, Vec& g)
{
    static_assert(sizeof...(Vars) >= 1);
    static_assert(sizeof...(Args) >= 1);

    const size_t n = wrt_total_length(wrt);

    g.resize(n);

    if(n == 0)
        return;

    ForEachWrtVar(
        wrt, [&](auto&& i, auto&& xi) constexpr {
            static_assert(!isConst<decltype(xi)>, "Expecting non-const autodiff numbers in wrt list because these need to be seeded, and thus altered!");
            u = eval(f, at, detail::wrt(xi)); // evaluate u with xi seeded so that du/dxi is also computed
            g[i] = derivative(u);
        });
}

/// Return the gradient of scalar function *f* with respect to some or all variables *x*.
template<typename Fun, typename... Vars, typename... Args, typename FunRet>
auto gradient(const Fun& f, const Wrt<Vars...>& wrt, const At<Args...>& at, FunRet& u)
{
    using T = NumericType<FunRet>; // the underlying numeric floating point type in the autodiff number u, e.g. double
    using Vec = VectorX<T>;        // the gradient vector type with floating point values (not autodiff numbers!)

    Vec g;
    gradient(f, wrt, at, u, g);
    return g;
}

/// Return the gradient of scalar function *f* with respect to some or all variables *x*.
template<typename Fun, typename... Vars, typename... Args>
auto gradient(const Fun& f, const Wrt<Vars...>& wrt, const At<Args...>& at)
{
    using FunRet = ReturnType<Fun, Args...>;
    static_assert(!std::is_same_v<FunRet, void>, "The function needs a non-void return type.");

    FunRet u;
    return gradient(f, wrt, at, u);
}

/// Return the Jacobian matrix of a function *f* with respect to some or all variables.
template<typename Fun, typename... Vars, typename... Args, typename FunRet, typename Mat>
void jacobian(const Fun& f, const Wrt<Vars...>& wrt, const At<Args...>& at, FunRet& F, Mat& J)
{
    static_assert(sizeof...(Vars) >= 1);
    static_assert(sizeof...(Args) >= 1);

    const auto n = wrt_total_length(wrt);
    size_t m = 0;

    ForEachWrtVar(
        wrt, [&](auto&& i, auto&& xi) constexpr {
            static_assert(!isConst<decltype(xi)>, "Expecting non-const autodiff numbers in wrt list because these need to be seeded, and thus altered!");
            F = eval(f, at, detail::wrt(xi)); // evaluate F=f(x) with xi seeded so that dF/dxi is also computed
            if(m == 0) {
                m = F.size();
                J.resize(m, n);
            };
            for(size_t row = 0; row < m; ++row)
                J(row, i) = derivative<1>(F[row]);
        });
}

/// Return the Jacobian matrix of a function *f* with respect to some or all variables.
template<typename Fun, typename... Vars, typename... Args, typename FunRet>
auto jacobian(const Fun& f, const Wrt<Vars...>& wrt, const At<Args...>& at, FunRet& F)
{
    using U = VectorValueType<FunRet>; // the type of the autodiff numbers in vector F
    using T = NumericType<U>;          // the underlying numeric floating point type in the autodiff number U
    using Mat = MatrixX<T>;            // the jacobian matrix type with floating point values (not autodiff numbers!)

    Mat J;
    jacobian(f, wrt, at, F, J);
    return J;
}

/// Return the Jacobian matrix of a function *f* with respect to some or all variables.
template<typename Fun, typename... Vars, typename... Args>
auto jacobian(const Fun& f, const Wrt<Vars...>& wrt, const At<Args...>& at)
{
    using FunRet = ReturnType<Fun, Args...>;
    static_assert(!std::is_same_v<FunRet, void>, "The function needs a non-void return type.");

    FunRet F;
    return jacobian(f, wrt, at, F);
}

/// Return the Hessian matrix of scalar function *f* with respect to some or all variables *x*.
template<typename Fun, typename... Vars, typename... Args, typename FunRet, typename Vec, typename Mat>
void hessian(const Fun& f, const Wrt<Vars...>& wrt, const At<Args...>& at, FunRet& u, Vec& v, Mat& H)
{
    static_assert(sizeof...(Vars) >= 1);
    static_assert(sizeof...(Args) >= 1);

    const auto n = wrt_total_length(wrt);

    v.resize(n);
    H.resize(n, n);

    ForEachWrtVar(
        wrt, [&](auto&& i, auto&& xi) constexpr {
            ForEachWrtVar(
                wrt, [&](auto&& j, auto&& xj) constexpr {
                    static_assert(!isConst<decltype(xi)> && !isConst<decltype(xj)>, "Expecting non-const autodiff numbers in wrt list because these need to be seeded, and thus altered!");
                    if(j >= i) {                              // this take advantage of the fact the Hessian matrix is symmetric
                        u = eval(f, at, detail::wrt(xi, xj)); // evaluate u with xi and xj seeded to produce u0, du/dxi, d2u/dxidxj
                        v[i] = derivative<1>(u);              // get du/dxi from u
                        H(i, j) = H(j, i) = derivative<2>(u); // get d2u/dxidxj from u
                    }
                });
        });
}

/// Return the Hessian matrix of scalar function *f* with respect to some or all variables *x*.
template<typename Fun, typename... Vars, typename... Args, typename FunRet, typename Vec>
auto hessian(const Fun& f, const Wrt<Vars...>& wrt, const At<Args...>& at, FunRet& u, Vec& v)
{
    using T = NumericType<FunRet>; // the underlying numeric floating point type in the autodiff number u
    using Mat = MatrixX<T>;        // the Hessian matrix type with floating point values (not autodiff numbers!)

    Mat H;
    hessian(f, wrt, at, u, v, H);
    return H;
}

/// Return the Hessian matrix of scalar function *f* with respect to some or all variables *x*.
template<typename Fun, typename... Vars, typename... Args>
auto hessian(const Fun& f, const Wrt<Vars...>& wrt, const At<Args...>& at)
{
    using FunRet = ReturnType<Fun, Args...>;
    static_assert(!std::is_same_v<FunRet, void>, "The function needs a non-void return type.");
    using T = NumericType<FunRet>;
    using Vec = VectorX<T>;

    FunRet u;
    Vec v;
    return hessian(f, wrt, at, u, v);
}

} // namespace detail

using detail::gradient;
using detail::hessian;
using detail::jacobian;

} // namespace autodiff
