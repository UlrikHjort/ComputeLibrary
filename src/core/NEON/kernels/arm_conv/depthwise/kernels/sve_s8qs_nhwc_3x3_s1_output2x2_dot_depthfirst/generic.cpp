/*
 * Copyright (c) 2021-2023 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#if defined(ARM_COMPUTE_ENABLE_SVE)

#include "arm_gemm.hpp"
#include <cstdint>

namespace arm_conv {
namespace depthwise {

void sve_s8qs_nhwc_3x3_s1_output2x2_dot_depthfirst_impl(
  const unsigned int n_channels,
  const int8_t *const *const inptrs,
  const int8_t *params,
  const int32_t *,  // Bias, should be wrapped into the parameters
  const arm_gemm::Requantize32& qp,
  const int32_t *, const int32_t *,  // Requant parameters, also wrapped
  int8_t *const *const outptrs
)
{
  __asm__ __volatile__(
    "mov x13, #0x0\n"
    "whilelt p2.b, x13, %x[n_channels]\n"
    "ldp x12, x11, [%x[inptrs], #0x0]\n"
    "ldp x10, x9, [%x[inptrs], #0x10]\n"
    "ldp x28, x27, [%x[inptrs], #0x20]\n"
    "ldp x26, x25, [%x[inptrs], #0x30]\n"
    "ptrue p1.b\n"
    "mov x24, #0x0\n"
    "ldp x23, x22, [%x[outptrs], #0x0]\n"
    "ldp x21, x20, [%x[outptrs], #0x10]\n"
    "ld1b { z9.b }, p2/Z, [x12, x13]\n"
    "ld1b { z8.b }, p2/Z, [x11, x13]\n"
    "ldp x12, x11, [%x[inptrs], #0x40]\n"
    "ld1b { z7.b }, p2/Z, [x10, x13]\n"
    "zip2 z6.b, z9.b, z7.b\n"
    "zip1 z9.b, z9.b, z7.b\n"
    "ld1b { z5.b }, p2/Z, [x9, x13]\n"
    "ldp x10, x9, [%x[inptrs], #0x50]\n"
    "zip1 z7.b, z8.b, z5.b\n"
    "zip2 z5.b, z8.b, z5.b\n"
    "ld1b { z4.b }, p2/Z, [x28, x13]\n"
    "ld1b { z3.b }, p2/Z, [x27, x13]\n"
    "zip2 z8.b, z9.b, z7.b\n"
    "zip1 z9.b, z9.b, z7.b\n"
    "ldp x28, x27, [%x[inptrs], #0x60]\n"
    "ld1b { z2.b }, p2/Z, [x26, x13]\n"
    "zip1 z7.b, z6.b, z5.b\n"
    "zip2 z5.b, z6.b, z5.b\n"
    "ld1b { z1.b }, p2/Z, [x25, x13]\n"
    "ldp x26, x25, [%x[inptrs], #0x70]\n"
    "zip2 z0.b, z4.b, z2.b\n"
    "zip1 z4.b, z4.b, z2.b\n"
    "ld1b { z31.b }, p2/Z, [x12, x13]\n"
    "ld1b { z30.b }, p2/Z, [x11, x13]\n"
    "zip1 z2.b, z3.b, z1.b\n"
    "zip2 z1.b, z3.b, z1.b\n"
    "ld1b { z29.b }, p2/Z, [x10, x13]\n"
    "ld1b { z28.b }, p2/Z, [x9, x13]\n"
    "zip2 z27.b, z31.b, z29.b\n"
    "zip1 z31.b, z31.b, z29.b\n"
    "ld1b { z26.b }, p2/Z, [x28, x13]\n"
    "ld1b { z25.b }, p2/Z, [x27, x13]\n"
    "zip1 z29.b, z30.b, z28.b\n"
    "zip2 z28.b, z30.b, z28.b\n"
    "ld1b { z24.b }, p2/Z, [x26, x13]\n"
    "ld1b { z23.b }, p2/Z, [x25, x13]\n"
    "zip2 z22.b, z26.b, z24.b\n"
    "zip1 z26.b, z26.b, z24.b\n"
    "zip1 z24.b, z25.b, z23.b\n"
    "zip2 z23.b, z25.b, z23.b\n"
    "ld1w { z6.s }, p1/Z, [%x[params]]\n"
    "ld1rw { z21.s }, p1/Z, [%x[qp], %[offsetof_Requantize32_minval]]\n"
    "ld1rw { z20.s }, p1/Z, [%x[qp], %[offsetof_Requantize32_maxval]]\n"
    "ld1rw { z19.s }, p1/Z, [%x[qp], %[offsetof_Requantize32_c_offset]]\n"
    "zip2 z3.b, z4.b, z2.b\n"
    "zip1 z4.b, z4.b, z2.b\n"
    "ldp x12, x11, [%x[inptrs], #0x0]\n"
    "ldp x10, x9, [%x[inptrs], #0x10]\n"
    "zip1 z2.b, z0.b, z1.b\n"
    "zip2 z1.b, z0.b, z1.b\n"
    "ldp x28, x27, [%x[inptrs], #0x20]\n"
    "ldp x26, x25, [%x[inptrs], #0x30]\n"
    "zip2 z30.b, z31.b, z29.b\n"
    "zip1 z31.b, z31.b, z29.b\n"
    "zip1 z29.b, z27.b, z28.b\n"
    "zip2 z28.b, z27.b, z28.b\n"
    "ld1b { z18.b }, p1/Z, [%x[params], #1, MUL VL]\n"
    "ld1b { z17.b }, p1/Z, [%x[params], #2, MUL VL]\n"
    "zip2 z25.b, z26.b, z24.b\n"
    "zip1 z26.b, z26.b, z24.b\n"
    "ld1b { z16.b }, p1/Z, [%x[params], #3, MUL VL]\n"
    "addvl %x[params], %x[params], #4\n"
    "zip1 z24.b, z22.b, z23.b\n"
    "zip2 z23.b, z22.b, z23.b\n"
    "mov z0.d, z6.d\n"
    "mov z27.d, z6.d\n"
    "mov z22.d, z6.d\n"
    "1:"  // Loop
    "sdot z6.s, z18.b, z9.b\n"
    "sdot z27.s, z18.b, z4.b\n"
    "ext z9.b, z9.b, z9.b, #0x1\n"
    "whilelt p0.s, x24, %x[n_channels]\n"
    "sdot z6.s, z17.b, z4.b\n"
    "ext z4.b, z4.b, z4.b, #0x1\n"
    "sdot z0.s, z18.b, z9.b\n"
    "ld1w { z9.s }, p1/Z, [%x[params]]\n"
    "sdot z22.s, z18.b, z4.b\n"
    "sdot z27.s, z17.b, z31.b\n"
    "incw x13, ALL, MUL #4\n"
    "sdot z6.s, z16.b, z31.b\n"
    "ext z31.b, z31.b, z31.b, #0x1\n"
    "sdot z0.s, z17.b, z4.b\n"
    "ld1w { z4.s }, p1/Z, [%x[params], #1, MUL VL]\n"
    "sdot z22.s, z17.b, z31.b\n"
    "sdot z27.s, z16.b, z26.b\n"
    "ext z26.b, z26.b, z26.b, #0x1\n"
    ".inst 0x04a974c6  // sqrdmulh z6.s, z6.s, z9.s\n"
    "sdot z0.s, z16.b, z31.b\n"
    "sdot z22.s, z16.b, z26.b\n"
    "and z18.d, z6.d, z4.d\n"
    "asr z18.s, z18.s, #0x1f\n"
    ".inst 0x04a97400  // sqrdmulh z0.s, z0.s, z9.s\n"
    ".inst 0x04a9777b  // sqrdmulh z27.s, z27.s, z9.s\n"
    ".inst 0x04a976d6  // sqrdmulh z22.s, z22.s, z9.s\n"
    "sqadd z6.s, z6.s, z18.s\n"
    ".inst 0x44828486  // srshl z6.s, p1/M, z6.s, z4.s\n"
    "ld1w { z9.s }, p1/Z, [%x[params], #6, MUL VL]\n"
    "and z17.d, z0.d, z4.d\n"
    "and z16.d, z27.d, z4.d\n"
    "and z18.d, z22.d, z4.d\n"
    "asr z17.s, z17.s, #0x1f\n"
    "asr z16.s, z16.s, #0x1f\n"
    "asr z18.s, z18.s, #0x1f\n"
    "sqadd z0.s, z0.s, z17.s\n"
    "sqadd z27.s, z27.s, z16.s\n"
    ".inst 0x44828480  // srshl z0.s, p1/M, z0.s, z4.s\n"
    ".inst 0x4482849b  // srshl z27.s, p1/M, z27.s, z4.s\n"
    "sqadd z22.s, z22.s, z18.s\n"
    "add z6.s, z6.s, z19.s\n"
    ".inst 0x44828496  // srshl z22.s, p1/M, z22.s, z4.s\n"
    "smax z6.s, p1/M, z6.s, z21.s\n"
    "add z0.s, z0.s, z19.s\n"
    "add z27.s, z27.s, z19.s\n"
    "smin z6.s, p1/M, z6.s, z20.s\n"
    "smax z0.s, p1/M, z0.s, z21.s\n"
    "add z22.s, z22.s, z19.s\n"
    "smax z27.s, p1/M, z27.s, z21.s\n"
    "smax z22.s, p1/M, z22.s, z21.s\n"
    "st1b { z6.s }, p0, [x23, x24]\n"
    "ld1w { z6.s }, p1/Z, [%x[params], #2, MUL VL]\n"
    "ld1b { z18.b }, p1/Z, [%x[params], #3, MUL VL]\n"
    "smin z0.s, p1/M, z0.s, z20.s\n"
    "smin z27.s, p1/M, z27.s, z20.s\n"
    "smin z22.s, p1/M, z22.s, z20.s\n"
    "st1b { z0.s }, p0, [x22, x24]\n"
    "mov z0.d, z6.d\n"
    "ld1b { z17.b }, p1/Z, [%x[params], #4, MUL VL]\n"
    "st1b { z27.s }, p0, [x21, x24]\n"
    "mov z27.d, z6.d\n"
    "sdot z27.s, z18.b, z3.b\n"
    "ld1b { z16.b }, p1/Z, [%x[params], #5, MUL VL]\n"
    "st1b { z22.s }, p0, [x20, x24]\n"
    "mov z22.d, z6.d\n"
    "sdot z6.s, z18.b, z8.b\n"
    "sdot z6.s, z17.b, z3.b\n"
    "ext z8.b, z8.b, z8.b, #0x1\n"
    "ext z3.b, z3.b, z3.b, #0x1\n"
    "sdot z0.s, z18.b, z8.b\n"
    "ld1w { z4.s }, p1/Z, [%x[params], #7, MUL VL]\n"
    "sdot z22.s, z18.b, z3.b\n"
    "sdot z27.s, z17.b, z30.b\n"
    "incw x24\n"
    "whilelt p0.s, x24, %x[n_channels]\n"
    "sdot z6.s, z16.b, z30.b\n"
    "ext z30.b, z30.b, z30.b, #0x1\n"
    "sdot z0.s, z17.b, z3.b\n"
    "addvl %x[params], %x[params], #16\n"
    "sdot z22.s, z17.b, z30.b\n"
    "sdot z27.s, z16.b, z25.b\n"
    "ext z25.b, z25.b, z25.b, #0x1\n"
    ".inst 0x04a974c6  // sqrdmulh z6.s, z6.s, z9.s\n"
    "sdot z0.s, z16.b, z30.b\n"
    "sdot z22.s, z16.b, z25.b\n"
    "and z18.d, z6.d, z4.d\n"
    "asr z18.s, z18.s, #0x1f\n"
    ".inst 0x04a97400  // sqrdmulh z0.s, z0.s, z9.s\n"
    ".inst 0x04a9777b  // sqrdmulh z27.s, z27.s, z9.s\n"
    ".inst 0x04a976d6  // sqrdmulh z22.s, z22.s, z9.s\n"
    "sqadd z6.s, z6.s, z18.s\n"
    ".inst 0x44828486  // srshl z6.s, p1/M, z6.s, z4.s\n"
    "ld1w { z9.s }, p1/Z, [%x[params], #-4, MUL VL]\n"
    "and z17.d, z0.d, z4.d\n"
    "and z16.d, z27.d, z4.d\n"
    "and z18.d, z22.d, z4.d\n"
    "asr z17.s, z17.s, #0x1f\n"
    "asr z16.s, z16.s, #0x1f\n"
    "asr z18.s, z18.s, #0x1f\n"
    "sqadd z0.s, z0.s, z17.s\n"
    "sqadd z27.s, z27.s, z16.s\n"
    ".inst 0x44828480  // srshl z0.s, p1/M, z0.s, z4.s\n"
    ".inst 0x4482849b  // srshl z27.s, p1/M, z27.s, z4.s\n"
    "sqadd z22.s, z22.s, z18.s\n"
    "add z6.s, z6.s, z19.s\n"
    ".inst 0x44828496  // srshl z22.s, p1/M, z22.s, z4.s\n"
    "smax z6.s, p1/M, z6.s, z21.s\n"
    "add z0.s, z0.s, z19.s\n"
    "add z27.s, z27.s, z19.s\n"
    "smin z6.s, p1/M, z6.s, z20.s\n"
    "smax z0.s, p1/M, z0.s, z21.s\n"
    "add z22.s, z22.s, z19.s\n"
    "smax z27.s, p1/M, z27.s, z21.s\n"
    "smax z22.s, p1/M, z22.s, z21.s\n"
    "st1b { z6.s }, p0, [x23, x24]\n"
    "ld1w { z6.s }, p1/Z, [%x[params], #-8, MUL VL]\n"
    "ld1b { z18.b }, p1/Z, [%x[params], #-7, MUL VL]\n"
    "smin z0.s, p1/M, z0.s, z20.s\n"
    "smin z27.s, p1/M, z27.s, z20.s\n"
    "smin z22.s, p1/M, z22.s, z20.s\n"
    "st1b { z0.s }, p0, [x22, x24]\n"
    "mov z0.d, z6.d\n"
    "ld1b { z17.b }, p1/Z, [%x[params], #-6, MUL VL]\n"
    "st1b { z27.s }, p0, [x21, x24]\n"
    "mov z27.d, z6.d\n"
    "sdot z27.s, z18.b, z2.b\n"
    "ld1b { z16.b }, p1/Z, [%x[params], #-5, MUL VL]\n"
    "st1b { z22.s }, p0, [x20, x24]\n"
    "mov z22.d, z6.d\n"
    "sdot z6.s, z18.b, z7.b\n"
    "sdot z6.s, z17.b, z2.b\n"
    "ext z7.b, z7.b, z7.b, #0x1\n"
    "ext z2.b, z2.b, z2.b, #0x1\n"
    "sdot z0.s, z18.b, z7.b\n"
    "ld1w { z4.s }, p1/Z, [%x[params], #-3, MUL VL]\n"
    "sdot z22.s, z18.b, z2.b\n"
    "sdot z27.s, z17.b, z29.b\n"
    "incw x24\n"
    "whilelt p0.s, x24, %x[n_channels]\n"
    "sdot z6.s, z16.b, z29.b\n"
    "ext z29.b, z29.b, z29.b, #0x1\n"
    "sdot z0.s, z17.b, z2.b\n"
    "sdot z22.s, z17.b, z29.b\n"
    "sdot z27.s, z16.b, z24.b\n"
    "ext z24.b, z24.b, z24.b, #0x1\n"
    ".inst 0x04a974c6  // sqrdmulh z6.s, z6.s, z9.s\n"
    "sdot z0.s, z16.b, z29.b\n"
    "sdot z22.s, z16.b, z24.b\n"
    "and z18.d, z6.d, z4.d\n"
    "asr z18.s, z18.s, #0x1f\n"
    ".inst 0x04a97400  // sqrdmulh z0.s, z0.s, z9.s\n"
    ".inst 0x04a9777b  // sqrdmulh z27.s, z27.s, z9.s\n"
    ".inst 0x04a976d6  // sqrdmulh z22.s, z22.s, z9.s\n"
    "sqadd z6.s, z6.s, z18.s\n"
    ".inst 0x44828486  // srshl z6.s, p1/M, z6.s, z4.s\n"
    "ld1w { z9.s }, p1/Z, [%x[params], #2, MUL VL]\n"
    "and z17.d, z0.d, z4.d\n"
    "and z16.d, z27.d, z4.d\n"
    "and z18.d, z22.d, z4.d\n"
    "asr z17.s, z17.s, #0x1f\n"
    "asr z16.s, z16.s, #0x1f\n"
    "asr z18.s, z18.s, #0x1f\n"
    "sqadd z0.s, z0.s, z17.s\n"
    "sqadd z27.s, z27.s, z16.s\n"
    ".inst 0x44828480  // srshl z0.s, p1/M, z0.s, z4.s\n"
    ".inst 0x4482849b  // srshl z27.s, p1/M, z27.s, z4.s\n"
    "sqadd z22.s, z22.s, z18.s\n"
    "add z6.s, z6.s, z19.s\n"
    ".inst 0x44828496  // srshl z22.s, p1/M, z22.s, z4.s\n"
    "smax z6.s, p1/M, z6.s, z21.s\n"
    "add z0.s, z0.s, z19.s\n"
    "add z27.s, z27.s, z19.s\n"
    "smin z6.s, p1/M, z6.s, z20.s\n"
    "smax z0.s, p1/M, z0.s, z21.s\n"
    "add z22.s, z22.s, z19.s\n"
    "smax z27.s, p1/M, z27.s, z21.s\n"
    "smax z22.s, p1/M, z22.s, z21.s\n"
    "st1b { z6.s }, p0, [x23, x24]\n"
    "ld1w { z6.s }, p1/Z, [%x[params], #-2, MUL VL]\n"
    "ld1b { z18.b }, p1/Z, [%x[params], #-1, MUL VL]\n"
    "smin z0.s, p1/M, z0.s, z20.s\n"
    "smin z27.s, p1/M, z27.s, z20.s\n"
    "smin z22.s, p1/M, z22.s, z20.s\n"
    "st1b { z0.s }, p0, [x22, x24]\n"
    "mov z0.d, z6.d\n"
    "ld1b { z17.b }, p1/Z, [%x[params]]\n"
    "st1b { z27.s }, p0, [x21, x24]\n"
    "mov z27.d, z6.d\n"
    "sdot z27.s, z18.b, z1.b\n"
    "ld1b { z16.b }, p1/Z, [%x[params], #1, MUL VL]\n"
    "st1b { z22.s }, p0, [x20, x24]\n"
    "mov z22.d, z6.d\n"
    "sdot z6.s, z18.b, z5.b\n"
    "sdot z6.s, z17.b, z1.b\n"
    "ext z5.b, z5.b, z5.b, #0x1\n"
    "ext z1.b, z1.b, z1.b, #0x1\n"
    "sdot z0.s, z18.b, z5.b\n"
    "ld1w { z4.s }, p1/Z, [%x[params], #3, MUL VL]\n"
    "sdot z22.s, z18.b, z1.b\n"
    "sdot z27.s, z17.b, z28.b\n"
    "incw x24\n"
    "whilelt p0.s, x24, %x[n_channels]\n"
    "sdot z6.s, z16.b, z28.b\n"
    "ext z28.b, z28.b, z28.b, #0x1\n"
    "sdot z0.s, z17.b, z1.b\n"
    "whilelt p2.b, x13, %x[n_channels]\n"
    "sdot z22.s, z17.b, z28.b\n"
    "sdot z27.s, z16.b, z23.b\n"
    "ext z23.b, z23.b, z23.b, #0x1\n"
    "ld1b { z8.b }, p2/Z, [x11, x13]\n"
    ".inst 0x04a974c6  // sqrdmulh z6.s, z6.s, z9.s\n"
    "sdot z0.s, z16.b, z28.b\n"
    "sdot z22.s, z16.b, z23.b\n"
    "ld1b { z7.b }, p2/Z, [x10, x13]\n"
    "and z18.d, z6.d, z4.d\n"
    "asr z18.s, z18.s, #0x1f\n"
    "ld1b { z5.b }, p2/Z, [x9, x13]\n"
    "ld1b { z3.b }, p2/Z, [x27, x13]\n"
    ".inst 0x04a97400  // sqrdmulh z0.s, z0.s, z9.s\n"
    ".inst 0x04a9777b  // sqrdmulh z27.s, z27.s, z9.s\n"
    "ld1b { z2.b }, p2/Z, [x26, x13]\n"
    "ld1b { z1.b }, p2/Z, [x25, x13]\n"
    ".inst 0x04a976d6  // sqrdmulh z22.s, z22.s, z9.s\n"
    "sqadd z6.s, z6.s, z18.s\n"
    ".inst 0x44828486  // srshl z6.s, p1/M, z6.s, z4.s\n"
    "ld1b { z9.b }, p2/Z, [x12, x13]\n"
    "and z17.d, z0.d, z4.d\n"
    "and z16.d, z27.d, z4.d\n"
    "ldp x12, x11, [%x[inptrs], #0x40]\n"
    "ldp x10, x9, [%x[inptrs], #0x50]\n"
    "and z18.d, z22.d, z4.d\n"
    "asr z17.s, z17.s, #0x1f\n"
    "ld1b { z31.b }, p2/Z, [x12, x13]\n"
    "ld1b { z30.b }, p2/Z, [x11, x13]\n"
    "asr z16.s, z16.s, #0x1f\n"
    "asr z18.s, z18.s, #0x1f\n"
    "ld1b { z29.b }, p2/Z, [x10, x13]\n"
    "ld1b { z28.b }, p2/Z, [x9, x13]\n"
    "sqadd z0.s, z0.s, z17.s\n"
    "sqadd z27.s, z27.s, z16.s\n"
    ".inst 0x44828480  // srshl z0.s, p1/M, z0.s, z4.s\n"
    ".inst 0x4482849b  // srshl z27.s, p1/M, z27.s, z4.s\n"
    "sqadd z22.s, z22.s, z18.s\n"
    "add z6.s, z6.s, z19.s\n"
    ".inst 0x44828496  // srshl z22.s, p1/M, z22.s, z4.s\n"
    "smax z6.s, p1/M, z6.s, z21.s\n"
    "add z0.s, z0.s, z19.s\n"
    "add z27.s, z27.s, z19.s\n"
    "ld1b { z4.b }, p2/Z, [x28, x13]\n"
    "ldp x28, x27, [%x[inptrs], #0x60]\n"
    "add z22.s, z22.s, z19.s\n"
    "ldp x26, x25, [%x[inptrs], #0x70]\n"
    "smin z6.s, p1/M, z6.s, z20.s\n"
    "smax z0.s, p1/M, z0.s, z21.s\n"
    "smax z27.s, p1/M, z27.s, z21.s\n"
    "smax z22.s, p1/M, z22.s, z21.s\n"
    "st1b { z6.s }, p0, [x23, x24]\n"
    "ld1b { z26.b }, p2/Z, [x28, x13]\n"
    "ld1b { z25.b }, p2/Z, [x27, x13]\n"
    "ld1b { z24.b }, p2/Z, [x26, x13]\n"
    "zip2 z6.b, z9.b, z7.b\n"
    "zip1 z9.b, z9.b, z7.b\n"
    "ld1b { z23.b }, p2/Z, [x25, x13]\n"
    "zip1 z7.b, z8.b, z5.b\n"
    "zip2 z5.b, z8.b, z5.b\n"
    "smin z0.s, p1/M, z0.s, z20.s\n"
    "smin z27.s, p1/M, z27.s, z20.s\n"
    "smin z22.s, p1/M, z22.s, z20.s\n"
    "st1b { z0.s }, p0, [x22, x24]\n"
    "zip2 z8.b, z9.b, z7.b\n"
    "st1b { z27.s }, p0, [x21, x24]\n"
    "zip1 z9.b, z9.b, z7.b\n"
    "zip1 z7.b, z6.b, z5.b\n"
    "ldp x12, x11, [%x[inptrs], #0x0]\n"
    "st1b { z22.s }, p0, [x20, x24]\n"
    "zip2 z5.b, z6.b, z5.b\n"
    "zip2 z0.b, z4.b, z2.b\n"
    "ld1w { z6.s }, p1/Z, [%x[params], #4, MUL VL]\n"
    "zip1 z4.b, z4.b, z2.b\n"
    "zip1 z2.b, z3.b, z1.b\n"
    "incw x24\n"
    "ldp x10, x9, [%x[inptrs], #0x10]\n"
    "zip2 z1.b, z3.b, z1.b\n"
    "zip2 z27.b, z31.b, z29.b\n"
    "ldp x28, x27, [%x[inptrs], #0x20]\n"
    "ldp x26, x25, [%x[inptrs], #0x30]\n"
    "zip1 z31.b, z31.b, z29.b\n"
    "zip1 z29.b, z30.b, z28.b\n"
    "ld1b { z18.b }, p1/Z, [%x[params], #5, MUL VL]\n"
    "ld1b { z17.b }, p1/Z, [%x[params], #6, MUL VL]\n"
    "zip2 z28.b, z30.b, z28.b\n"
    "zip2 z22.b, z26.b, z24.b\n"
    "ld1b { z16.b }, p1/Z, [%x[params], #7, MUL VL]\n"
    "addvl %x[params], %x[params], #8\n"
    "zip1 z26.b, z26.b, z24.b\n"
    "zip1 z24.b, z25.b, z23.b\n"
    "zip2 z23.b, z25.b, z23.b\n"
    "zip2 z3.b, z4.b, z2.b\n"
    "zip1 z4.b, z4.b, z2.b\n"
    "zip1 z2.b, z0.b, z1.b\n"
    "zip2 z1.b, z0.b, z1.b\n"
    "zip2 z30.b, z31.b, z29.b\n"
    "zip1 z31.b, z31.b, z29.b\n"
    "zip1 z29.b, z27.b, z28.b\n"
    "zip2 z28.b, z27.b, z28.b\n"
    "zip2 z25.b, z26.b, z24.b\n"
    "zip1 z26.b, z26.b, z24.b\n"
    "zip1 z24.b, z22.b, z23.b\n"
    "zip2 z23.b, z22.b, z23.b\n"
    "mov z0.d, z6.d\n"
    "mov z27.d, z6.d\n"
    "mov z22.d, z6.d\n"
    "b.any 1b\n"
    : [params] "+&r" (params)
    : [inptrs] "r" (inptrs), [n_channels] "r" (n_channels), [offsetof_Requantize32_c_offset] "I" (offsetof(arm_gemm::Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(arm_gemm::Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(arm_gemm::Requantize32, minval)), [outptrs] "r" (outptrs), [qp] "r" (&qp)
    : "cc", "memory", "p0", "p1", "p2", "x9", "x10", "x11", "x12", "x13", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
