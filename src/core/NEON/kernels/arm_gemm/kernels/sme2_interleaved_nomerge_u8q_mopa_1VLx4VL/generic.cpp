/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#ifdef __ARM_FEATURE_SVE
#ifdef ARM_COMPUTE_ENABLE_SME2

#include "arm_gemm.hpp"

#include <cstdint>
#include "../../asmlib.hpp"
#include "../../utils.hpp"

namespace arm_gemm {

void sme2_interleaved_nomerge_u8q_mopa_1VLx4VL(const uint8_t *const A, const uint8_t *const B, uint8_t *const C, int ldc, const int M, const int N, const int K, const int32_t *const bias, const Requantize32 &rq, const int n_0, bool accumulate, int32_t *const accumulator_buffer)
{
  struct KernelArgs
  {
    KernelArgs(
      const uint8_t *const A,
      const uint8_t *const B,
      uint8_t *const C, const int ldc,
      const int M, const int N, const int K,
      const int32_t *const bias,
      const Requantize32 &rq,
      const int n_0,
      bool accumulate,
      int32_t *const accumulator_buffer
    ) : A(A),
        B(B), kstride_bytes(roundup(K, 4) * sizeof(uint8_t)),
        C(C), ldcb(ldc * sizeof(uint8_t)),
        M(M), N(N), K(K),
        n_loops(((K / 4) - 1) / 2), n_tail_iters(((K / 4) - 1) % 2),

        bias(bias), n_0(n_0),
        accumulator_buffer(accumulator_buffer),
        flags(0x0)
    {
      if (accumulate)
      {
        flags |= 1 << 0;  // FILL_ACCUMULATORS_FROM_BUFFER
      }
      if (C == nullptr)
      {
        flags |= 1 << 1;  // STORE_ACCUMULATORS_TO_BUFFER
      }
      if (rq.per_channel_requant)
      {
        flags |= 1 << 2;  // PER_CHANNEL_QUANTISATION
      }
      }

    const uint8_t *const A;
    const uint8_t *const B;
    const long kstride_bytes;
    uint8_t *const C;
    const long ldcb;
    const long M, N, K, n_loops, n_tail_iters;
    int32_t min = std::numeric_limits<uint8_t>::min();
    int32_t max = std::numeric_limits<uint8_t>::max();

    const int32_t *const bias;
    const int n_0;

    int32_t *const accumulator_buffer;
    uint64_t flags;
  };

  // Construct arguments for this kernel
  KernelArgs args(A, B, C, ldc, M, N, K, bias, rq, n_0, accumulate, accumulator_buffer);

  __asm__ __volatile__(
      "ldr x14, [%x[args], %[offsetof_flags]]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ptrue p1.b\n"
      ".inst 0x25207811  // ptrue pn9.b\n"
      "ldr x13, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "ldr x11, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "tbz x14, #0, 2f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "1:"  // Initial accumulator load from buffer: Loop
      ".inst 0xa040c5ac  // ld1w { z12.s-z15.s }, pn9.b/Z, [x13]\n"
      ".inst 0xc0840580  // mova za0h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xa041c5bc  // ld1w { z28.s-z31.s }, pn9.b/Z, [x13, #0x4, MUL VL]\n"
      ".inst 0xc0840781  // mova za1h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xa042c5bc  // ld1w { z28.s-z31.s }, pn9.b/Z, [x13, #0x8, MUL VL]\n"
      ".inst 0xc0840782  // mova za2h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xa043c5a4  // ld1w { z4.s-z7.s }, pn9.b/Z, [x13, #0xc, MUL VL]\n"
      ".inst 0xc0840483  // mova za3h.s[x12], { z4.s-z7.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "addvl x13, x13, #16\n"
      "blt 1b\n"
      "2:"  // Initial accumulator load from buffer: End
      "ldr w10, [%x[args], %[offsetof_M]]\n"
      "mov x9, #0x0\n"
      "mov x28, #0x0\n"
      "ldr w27, [%x[args], %[offsetof_N]]\n"
      "ldr x26, [%x[args], %[offsetof_A]]\n"
      "3:"  // M and N loop
      "mov x25, x26\n"
      ".inst 0x25bb6790  // whilelt pn8.s, x28, x27, VLx4\n"
      "tbnz x14, #0, 4f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x20, 5f\n"
      ".inst 0xa01cc299  // ldnt1w { z24.s-z27.s }, p8/Z, [x20, x28, LSL #2]\n"
      ".inst 0xc0902700  // addha za0.s, p1/M, p1/M, z24.s\n"
      ".inst 0xc0902721  // addha za1.s, p1/M, p1/M, z25.s\n"
      ".inst 0xc0902742  // addha za2.s, p1/M, p1/M, z26.s\n"
      ".inst 0xc0902763  // addha za3.s, p1/M, p1/M, z27.s\n"
      "4:"  // Prepare accumulators: Test for last block
      "mov x20, x28\n"
      "mov x21, x9\n"
      "incw x20, ALL, MUL #4\n"
      "incw x21\n"
      "cmp x20, x27\n"
      "csel x21, x9, x21, LT\n"
      "mov x20, x14\n"
      "bfm x14, XZR, #0x0, #0x0  // bfc x14, #0x0, #0x1\n"
      "cmp x21, x10\n"
      "csel x14, x20, x14, LT\n"
      "5:"  // Prepare accumulators: End
      "ldr x20, [%x[args], %[offsetof_K]]\n"
      "add x20, x20, #0x3\n"
      "lsr x20, x20, #0x2\n"
      "ldr x23, [%x[args], %[offsetof_B]]\n"
      "lsr x22, x20, #0x2\n"
      "and x21, x20, #0x3\n"
      "ldr x20, [%x[args], %[offsetof_kstride_bytes]]\n"
      "madd x23, x28, x20, x23\n"  // bptr = B + n * kstride_bytes
      "cbz x22, 8f\n"
      "subs x22, x22, #0x1\n"
      "ld1b { z10.b }, p1/Z, [x25]\n"
      ".inst 0xa04086fd  // ldnt1b { z28.b-z31.b }, pn9.b/Z, [x23]\n"
      "ld1b { z16.b }, p1/Z, [x25, #1, MUL VL]\n"
      ".inst 0xa04186ed  // ldnt1b { z12.b-z15.b }, pn9.b/Z, [x23, #0x4, MUL VL]\n"
      "ld1b { z21.b }, p1/Z, [x25, #2, MUL VL]\n"
      ".inst 0xa04286f9  // ldnt1b { z24.b-z27.b }, pn9.b/Z, [x23, #0x8, MUL VL]\n"
      "ld1b { z19.b }, p1/Z, [x25, #3, MUL VL]\n"
      "addvl x25, x25, #4\n"
      ".inst 0xa04386e1  // ldnt1b { z0.b-z3.b }, pn9.b/Z, [x23, #0xc, MUL VL]\n"
      "addvl x23, x23, #16\n"
      "ble 7f\n"
      "6:"  // K loop
      ".inst 0xa1bc2540  // umopa za0.s, p1/M, p1/M, z10.b, z28.b\n"
      "subs x22, x22, #0x1\n"
      ".inst 0xa1bd2541  // umopa za1.s, p1/M, p1/M, z10.b, z29.b\n"
      ".inst 0xa1be2542  // umopa za2.s, p1/M, p1/M, z10.b, z30.b\n"
      ".inst 0xa1bf2543  // umopa za3.s, p1/M, p1/M, z10.b, z31.b\n"
      "ld1b { z10.b }, p1/Z, [x25]\n"
      ".inst 0xa1ac2600  // umopa za0.s, p1/M, p1/M, z16.b, z12.b\n"
      ".inst 0xa04086fd  // ldnt1b { z28.b-z31.b }, pn9.b/Z, [x23]\n"
      ".inst 0xa1ad2601  // umopa za1.s, p1/M, p1/M, z16.b, z13.b\n"
      ".inst 0xa1ae2602  // umopa za2.s, p1/M, p1/M, z16.b, z14.b\n"
      ".inst 0xa1af2603  // umopa za3.s, p1/M, p1/M, z16.b, z15.b\n"
      "ld1b { z16.b }, p1/Z, [x25, #1, MUL VL]\n"
      ".inst 0xa1b826a0  // umopa za0.s, p1/M, p1/M, z21.b, z24.b\n"
      ".inst 0xa04186ed  // ldnt1b { z12.b-z15.b }, pn9.b/Z, [x23, #0x4, MUL VL]\n"
      ".inst 0xa1b926a1  // umopa za1.s, p1/M, p1/M, z21.b, z25.b\n"
      ".inst 0xa1ba26a2  // umopa za2.s, p1/M, p1/M, z21.b, z26.b\n"
      ".inst 0xa1bb26a3  // umopa za3.s, p1/M, p1/M, z21.b, z27.b\n"
      "ld1b { z21.b }, p1/Z, [x25, #2, MUL VL]\n"
      ".inst 0xa04286f9  // ldnt1b { z24.b-z27.b }, pn9.b/Z, [x23, #0x8, MUL VL]\n"
      ".inst 0xa1a02660  // umopa za0.s, p1/M, p1/M, z19.b, z0.b\n"
      ".inst 0xa1a12661  // umopa za1.s, p1/M, p1/M, z19.b, z1.b\n"
      ".inst 0xa1a22662  // umopa za2.s, p1/M, p1/M, z19.b, z2.b\n"
      ".inst 0xa1a32663  // umopa za3.s, p1/M, p1/M, z19.b, z3.b\n"
      "ld1b { z19.b }, p1/Z, [x25, #3, MUL VL]\n"
      "addvl x25, x25, #4\n"
      ".inst 0xa04386e1  // ldnt1b { z0.b-z3.b }, pn9.b/Z, [x23, #0xc, MUL VL]\n"
      "addvl x23, x23, #16\n"
      "bgt 6b\n"
      "7:"  // K loop tail
      ".inst 0xa1bc2540  // umopa za0.s, p1/M, p1/M, z10.b, z28.b\n"
      ".inst 0xa1bd2541  // umopa za1.s, p1/M, p1/M, z10.b, z29.b\n"
      ".inst 0xa1be2542  // umopa za2.s, p1/M, p1/M, z10.b, z30.b\n"
      ".inst 0xa1bf2543  // umopa za3.s, p1/M, p1/M, z10.b, z31.b\n"
      ".inst 0xa1ac2600  // umopa za0.s, p1/M, p1/M, z16.b, z12.b\n"
      ".inst 0xa1ad2601  // umopa za1.s, p1/M, p1/M, z16.b, z13.b\n"
      ".inst 0xa1ae2602  // umopa za2.s, p1/M, p1/M, z16.b, z14.b\n"
      ".inst 0xa1af2603  // umopa za3.s, p1/M, p1/M, z16.b, z15.b\n"
      ".inst 0xa1b826a0  // umopa za0.s, p1/M, p1/M, z21.b, z24.b\n"
      ".inst 0xa1b926a1  // umopa za1.s, p1/M, p1/M, z21.b, z25.b\n"
      ".inst 0xa1ba26a2  // umopa za2.s, p1/M, p1/M, z21.b, z26.b\n"
      ".inst 0xa1bb26a3  // umopa za3.s, p1/M, p1/M, z21.b, z27.b\n"
      ".inst 0xa1a02660  // umopa za0.s, p1/M, p1/M, z19.b, z0.b\n"
      ".inst 0xa1a12661  // umopa za1.s, p1/M, p1/M, z19.b, z1.b\n"
      ".inst 0xa1a22662  // umopa za2.s, p1/M, p1/M, z19.b, z2.b\n"
      ".inst 0xa1a32663  // umopa za3.s, p1/M, p1/M, z19.b, z3.b\n"
      "8:"  // K oddments
      "cbz x21, 10f\n"
      "9:"  // K oddments: Loop
      "ld1b { z10.b }, p1/Z, [x25]\n"
      "subs x21, x21, #0x1\n"
      "addvl x25, x25, #1\n"
      ".inst 0xa04086fc  // ld1b { z28.b-z31.b }, pn9.b/Z, [x23]\n"
      "addvl x23, x23, #4\n"
      ".inst 0xa1bc2540  // umopa za0.s, p1/M, p1/M, z10.b, z28.b\n"
      ".inst 0xa1bd2541  // umopa za1.s, p1/M, p1/M, z10.b, z29.b\n"
      ".inst 0xa1be2542  // umopa za2.s, p1/M, p1/M, z10.b, z30.b\n"
      ".inst 0xa1bf2543  // umopa za3.s, p1/M, p1/M, z10.b, z31.b\n"
      "bgt 9b\n"
      "10:"  // K oddments: End
      "ld1w { z14.s }, p1/Z, [x25]\n"
      "addvl x25, x25, #1\n"
      ".inst 0xc09125c0  // addva za0.s, p1/M, p1/M, z14.s\n"
      ".inst 0xc09125c1  // addva za1.s, p1/M, p1/M, z14.s\n"
      ".inst 0xc09125c2  // addva za2.s, p1/M, p1/M, z14.s\n"
      ".inst 0xc09125c3  // addva za3.s, p1/M, p1/M, z14.s\n"
      "tbz x14, #1, 14f\n"
      "tbz x14, #0, 12f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "11:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c5b8  // ld1w { z24.s-z27.s }, pn9.b/Z, [x13]\n"
      ".inst 0xc086041c  // mova { z28.s-z31.s }, za0h.s[x12]\n"
      ".inst 0xc0840700  // mova za0h.s[x12], { z24.s-z27.s }\n"
      ".inst 0xc0860428  // mova { z8.s-z11.s }, za1h.s[x12]\n"
      ".inst 0xa041c5b0  // ld1w { z16.s-z19.s }, pn9.b/Z, [x13, #0x4, MUL VL]\n"
      ".inst 0xc0840601  // mova za1h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xc0860458  // mova { z24.s-z27.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xa042c5a4  // ld1w { z4.s-z7.s }, pn9.b/Z, [x13, #0x8, MUL VL]\n"
      ".inst 0xc0840482  // mova za2h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xa043c5a4  // ld1w { z4.s-z7.s }, pn9.b/Z, [x13, #0xc, MUL VL]\n"
      ".inst 0xc0840483  // mova za3h.s[x12], { z4.s-z7.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      ".inst 0xa060c57c  // st1w { z28.s-z31.s }, pn9.b, [x11]\n"
      "addvl x13, x13, #16\n"
      ".inst 0xa061c568  // st1w { z8.s-z11.s }, pn9.b, [x11, #0x4, MUL VL]\n"
      ".inst 0xa062c578  // st1w { z24.s-z27.s }, pn9.b, [x11, #0x8, MUL VL]\n"
      ".inst 0xa063c56c  // st1w { z12.s-z15.s }, pn9.b, [x11, #0xc, MUL VL]\n"
      "addvl x11, x11, #16\n"
      "blt 11b\n"
      "b 21f\n"
      "12:"  // Store to partial result buffer: Store only
      "mov x12, #0x0\n"
      "cntw x20\n"
      "13:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc086041c  // mova { z28.s-z31.s }, za0h.s[x12]\n"
      ".inst 0xc0860420  // mova { z0.s-z3.s }, za1h.s[x12]\n"
      ".inst 0xa060c57c  // st1w { z28.s-z31.s }, pn9.b, [x11]\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
      ".inst 0xa061c560  // st1w { z0.s-z3.s }, pn9.b, [x11, #0x4, MUL VL]\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      ".inst 0xa062c568  // st1w { z8.s-z11.s }, pn9.b, [x11, #0x8, MUL VL]\n"
      ".inst 0xa063c570  // st1w { z16.s-z19.s }, pn9.b, [x11, #0xc, MUL VL]\n"
      "addvl x11, x11, #16\n"
      "blt 13b\n"
      "b 21f\n"
      "14:"  // Store to output array
      "ldr x24, [%x[args], %[offsetof_C]]\n"
      "add x24, x24, x28\n"  // C += n
      "sub x23, x10, x9\n"
      "ld1rw { z12.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_mul]]\n"
      "ldr x22, [%x[args], %[offsetof_ldcb]]\n"
      "madd x24, x9, x22, x24\n"  // C += m * ldc
      "ld1rw { z13.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_mul]]\n"
      "ld1rw { z14.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_mul]]\n"
      "ld1rw { z15.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_mul]]\n"
      "ld1rw { z4.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_right_shift]]\n"
      "ld1rw { z5.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_right_shift]]\n"
      "ld1rw { z6.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_right_shift]]\n"
      "ld1rw { z7.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_right_shift]]\n"
      "ld1rw { z1.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_c_offset]]\n"
      "ld1rw { z21.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_minval]]\n"
      "ld1rw { z20.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_maxval]]\n"
      "tbz x14, #2, 15f\n"
      "ldr w21, [%x[args], %[offsetof_n_0]]\n"
      "add x21, x21, x28\n"
      "ldr x20, [%x[rq], %[offsetof_Requantize32_per_channel_muls]]\n"
      "add x20, x20, x21, LSL #2\n"
      ".inst 0xa040c28c  // ld1w { z12.s-z15.s }, p8/Z, [x20]\n"
      "ldr x20, [%x[rq], %[offsetof_Requantize32_per_channel_right_shifts]]\n"
      "add x20, x20, x21, LSL #2\n"
      ".inst 0xa040c284  // ld1w { z4.s-z7.s }, p8/Z, [x20]\n"
      "15:"  // Store to output array: Load per-channel parameters: End
      "cntw x20\n"
      "whilelt p0.b, x28, x27\n"
      "cmp x23, x20\n"
      "csel x20, x23, x20, LT\n"
      "lsr x21, x20, #0x1\n"
      "mov x12, #0x0\n"
      "and x20, x20, #0x1\n"
      "cbz x21, 17f\n"
      "16:"  // Store to output array: Accumulator row 0 loop
      ".inst 0xc086001a  // mova { z26.s-z27.s }, za0h.s[x12, 0:1]\n"
      ".inst 0xc086005c  // mova { z28.s-z29.s }, za1h.s[x12, 0:1]\n"
      ".inst 0xc1aca41a  // sqdmulh { z26.s-z27.s }, { z26.s-z27.s }, z12.s\n"
      ".inst 0xc0860096  // mova { z22.s-z23.s }, za2h.s[x12, 0:1]\n"
      ".inst 0xc08600d0  // mova { z16.s-z17.s }, za3h.s[x12, 0:1]\n"
      ".inst 0xc1ada41c  // sqdmulh { z28.s-z29.s }, { z28.s-z29.s }, z13.s\n"
      ".inst 0xc1aea416  // sqdmulh { z22.s-z23.s }, { z22.s-z23.s }, z14.s\n"
      "add x12, x12, #0x2\n"
      "cmp x12, x21, LSL #1\n"
      ".inst 0xc1afa410  // sqdmulh { z16.s-z17.s }, { z16.s-z17.s }, z15.s\n"
      ".inst 0xc1a4a23a  // srshl { z26.s-z27.s }, { z26.s-z27.s }, z4.s\n"
      ".inst 0xc1a5a23c  // srshl { z28.s-z29.s }, { z28.s-z29.s }, z5.s\n"
      ".inst 0xc1a6a236  // srshl { z22.s-z23.s }, { z22.s-z23.s }, z6.s\n"
      ".inst 0xc1a7a230  // srshl { z16.s-z17.s }, { z16.s-z17.s }, z7.s\n"
      ".inst 0xc1a1a31a  // add { z26.s-z27.s }, { z26.s-z27.s }, z1.s\n"
      ".inst 0xc1a1a31c  // add { z28.s-z29.s }, { z28.s-z29.s }, z1.s\n"
      ".inst 0xc1a1a316  // add { z22.s-z23.s }, { z22.s-z23.s }, z1.s\n"
      ".inst 0xc1a1a310  // add { z16.s-z17.s }, { z16.s-z17.s }, z1.s\n"
      ".inst 0xc1b4c6ba  // sclamp { z26.s-z27.s }, z21.s, z20.s\n"
      ".inst 0xc1b4c6bc  // sclamp { z28.s-z29.s }, z21.s, z20.s\n"
      "uzp1 z19.b, z26.b, z28.b\n"
      ".inst 0xc1b4c6b6  // sclamp { z22.s-z23.s }, z21.s, z20.s\n"
      ".inst 0xc1b4c6b0  // sclamp { z16.s-z17.s }, z21.s, z20.s\n"
      "uzp1 z16.b, z22.b, z16.b\n"
      "uzp1 z18.b, z27.b, z29.b\n"
      "uzp1 z17.b, z23.b, z17.b\n"
      "uzp1 z16.b, z19.b, z16.b\n"
      "st1b { z16.b }, p0, [x24]\n"
      "add x24, x24, x22\n"
      "uzp1 z16.b, z18.b, z17.b\n"
      "st1b { z16.b }, p0, [x24]\n"
      "add x24, x24, x22\n"
      "blt 16b\n"
      "17:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 18f\n"
      ".inst 0xc0860002  // mova { z2.s-z3.s }, za0h.s[x12, 0:1]\n"
      ".inst 0xc0860058  // mova { z24.s-z25.s }, za1h.s[x12, 0:1]\n"
      ".inst 0xc1aca402  // sqdmulh { z2.s-z3.s }, { z2.s-z3.s }, z12.s\n"
      ".inst 0xc0860090  // mova { z16.s-z17.s }, za2h.s[x12, 0:1]\n"
      ".inst 0xc08600ca  // mova { z10.s-z11.s }, za3h.s[x12, 0:1]\n"
      ".inst 0xc1ada418  // sqdmulh { z24.s-z25.s }, { z24.s-z25.s }, z13.s\n"
      ".inst 0xc1aea410  // sqdmulh { z16.s-z17.s }, { z16.s-z17.s }, z14.s\n"
      ".inst 0xc1afa40a  // sqdmulh { z10.s-z11.s }, { z10.s-z11.s }, z15.s\n"
      ".inst 0xc1a4a222  // srshl { z2.s-z3.s }, { z2.s-z3.s }, z4.s\n"
      ".inst 0xc1a5a238  // srshl { z24.s-z25.s }, { z24.s-z25.s }, z5.s\n"
      ".inst 0xc1a6a230  // srshl { z16.s-z17.s }, { z16.s-z17.s }, z6.s\n"
      ".inst 0xc1a7a22a  // srshl { z10.s-z11.s }, { z10.s-z11.s }, z7.s\n"
      ".inst 0xc1a1a302  // add { z2.s-z3.s }, { z2.s-z3.s }, z1.s\n"
      ".inst 0xc1a1a318  // add { z24.s-z25.s }, { z24.s-z25.s }, z1.s\n"
      ".inst 0xc1a1a310  // add { z16.s-z17.s }, { z16.s-z17.s }, z1.s\n"
      ".inst 0xc1a1a30a  // add { z10.s-z11.s }, { z10.s-z11.s }, z1.s\n"
      ".inst 0xc1b4c6a2  // sclamp { z2.s-z3.s }, z21.s, z20.s\n"
      ".inst 0xc1b4c6b8  // sclamp { z24.s-z25.s }, z21.s, z20.s\n"
      "uzp1 z23.b, z2.b, z24.b\n"
      ".inst 0xc1b4c6b0  // sclamp { z16.s-z17.s }, z21.s, z20.s\n"
      ".inst 0xc1b4c6aa  // sclamp { z10.s-z11.s }, z21.s, z20.s\n"
      "uzp1 z16.b, z16.b, z10.b\n"
      "uzp1 z16.b, z23.b, z16.b\n"
      "st1b { z16.b }, p0, [x24]\n"
      "18:"  // Store to output array: Accumulator row 0 oddments: End
      "19:"  // Store to output array: End
      "tbz x14, #0, 21f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "20:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c5b0  // ld1w { z16.s-z19.s }, pn9.b/Z, [x13]\n"
      ".inst 0xc0840600  // mova za0h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xa041c5ac  // ld1w { z12.s-z15.s }, pn9.b/Z, [x13, #0x4, MUL VL]\n"
      ".inst 0xc0840581  // mova za1h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xa042c5b0  // ld1w { z16.s-z19.s }, pn9.b/Z, [x13, #0x8, MUL VL]\n"
      ".inst 0xc0840602  // mova za2h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xa043c5ac  // ld1w { z12.s-z15.s }, pn9.b/Z, [x13, #0xc, MUL VL]\n"
      ".inst 0xc0840583  // mova za3h.s[x12], { z12.s-z15.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "addvl x13, x13, #16\n"
      "blt 20b\n"
      "21:"  // End block
      "incw x28, ALL, MUL #4\n"
      "cmp x28, x27\n"
      "blt 3b\n"
      "incw x9\n"
      "cmp x9, x10\n"
      "mov x28, #0x0\n"
      "mov x26, x25\n"
      "blt 3b\n"
      ".inst 0xd503467f  // SMSTOP\n"
      :
      : [args] "r" (&args), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_Requantize32_c_offset] "I" (offsetof(Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(Requantize32, minval)), [offsetof_Requantize32_per_channel_muls] "I" (offsetof(Requantize32, per_channel_muls)), [offsetof_Requantize32_per_channel_right_shifts] "I" (offsetof(Requantize32, per_channel_right_shifts)), [offsetof_Requantize32_per_layer_mul] "I" (offsetof(Requantize32, per_layer_mul)), [offsetof_Requantize32_per_layer_right_shift] "I" (offsetof(Requantize32, per_layer_right_shift)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_kstride_bytes] "I" (offsetof(KernelArgs, kstride_bytes)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb)), [offsetof_n_0] "I" (offsetof(KernelArgs, n_0)), [rq] "r" (&rq)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace arm_gemm

#endif  // ARM_COMPUTE_ENABLE_SME2
#endif  // __ARM_FEATURE_SVE
