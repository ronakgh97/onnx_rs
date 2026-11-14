//! Kernel implementations for ONNX operators.
use crate::ops::{
    AttributesDataValue, Operator, take_attr_i64, take_attr_i64s, take_attr_i64s_n, take_attr_str,
};
use crate::tensor::{Tensor, take_tensor};
use anyhow::{Result, anyhow};
use blas_rs::GLOBAL_THREAD_POOL;
use blas_rs::lvl3;
use blas_rs::rayon::prelude::*;
use std::arch::x86_64::{
    __m256, _CMP_LT_OQ, _MM_FROUND_CUR_DIRECTION, _mm256_add_epi32, _mm256_add_ps,
    _mm256_blendv_ps, _mm256_castsi256_ps, _mm256_cmp_ps, _mm256_cvtps_epi32, _mm256_div_ps,
    _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps,
    _mm256_permute2f128_ps, _mm256_round_ps, _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_shuffle_ps, _mm256_slli_epi32, _mm256_storeu_ps, _mm256_sub_ps, _mm256_unpackhi_ps,
    _mm256_unpacklo_ps,
};

impl Operator {
    #[inline(always)]
    pub fn compute(
        self,
        output_names: &[&str],
        inputs: &[Option<&Tensor>],
        attrs: &[(String, AttributesDataValue)],
    ) -> Result<Vec<Tensor>> {
        // decode_model checks this already
        if output_names.is_empty() {
            return Err(anyhow!(
                "operator '{:?}' received empty output names list",
                self,
            ));
        }
        // according to the ONNX spec, empty output names valid, means optional
        // decode_model already provides us with minimum outputs
        // required by the operator for execution
        if output_names.iter().any(|n| n.is_empty()) {
            return Err(anyhow!("operator '{:?}' has an empty output name", self));
        }

        match self {
            Operator::Clip => Ok(vec![clip(output_names[0], inputs)?]),
            Operator::Relu => Ok(vec![relu(output_names[0], inputs)?]),
            Operator::PRelu => Ok(vec![prelu(output_names[0], inputs)?]),
            Operator::DepthToSpace => Ok(vec![depth_to_space(output_names[0], inputs, attrs)?]),
            Operator::Conv => Ok(vec![conv(output_names[0], inputs, attrs)?]),
            Operator::Add => Ok(vec![add(output_names[0], inputs)?]),
            Operator::Sub => Ok(vec![sub(output_names[0], inputs)?]),
            Operator::Mul => Ok(vec![mul(output_names[0], inputs)?]),
            Operator::Div => Ok(vec![div(output_names[0], inputs)?]),
            Operator::Sigmoid => Ok(vec![sigmoid(output_names[0], inputs)?]),
            Operator::Concat => Ok(vec![concat(output_names[0], inputs, attrs)?]),
            Operator::SoftPlus => Ok(vec![softplus(output_names[0], inputs)?]),
            _ => Err(anyhow!(
                "operator '{:?}' has no kernel implemented yet",
                self
            )),
        }
    }
}

// TODO: Need reference inputs/outputs for correctness testing
// TODO: Overflow & underflow bugs are hiding in plane sight
// TODO: SIMD path has weird behavior with NaN/Inf, need to check against ONNX spec

#[inline]
fn clip(output: &str, inputs: &[Option<&Tensor>]) -> Result<Tensor> {
    let x = take_tensor(inputs, 0, "Clip")?;
    let min = if let Some(t) = inputs.get(1).copied().flatten() {
        if t.data.len() != 1 {
            return Err(anyhow!(
                "Clip: min (input 1) must be a scalar (empty tensor), got elements {}",
                t.data.len()
            ));
        }
        Some(t.data[0])
    } else {
        None
    };
    let max = if let Some(t) = inputs.get(2).copied().flatten() {
        if t.data.len() != 1 {
            return Err(anyhow!(
                "Clip: max (input 2) must be a scalar (empty tensor), got elements {}",
                t.data.len()
            ));
        }
        Some(t.data[0])
    } else {
        None
    };

    // if min is greater than max, then fill input with max
    if let (Some(lo), Some(hi)) = (min, max)
        && lo > hi
    {
        return Tensor::create(output, &x.dims, vec![hi; x.len()]);
    }

    // basically do this cmp::min(max, cmp::max(min, x))
    let min = min.unwrap_or(f32::MIN);
    let max = max.unwrap_or(f32::MAX);
    let mut out_data = vec![0f32; x.len()];

    let (input_chunk32, input_rem32) = x.data.as_chunks::<32>();
    let (output_chunk32, output_rem32) = out_data.as_chunks_mut::<32>();

    unsafe {
        let vmin = _mm256_set1_ps(min);
        let vmax = _mm256_set1_ps(max);

        // unroll 4x for ILP, 32 floats per iteration
        for (in_chunk, out_chunk) in input_chunk32.iter().zip(output_chunk32.iter_mut()) {
            let v0 = _mm256_loadu_ps(in_chunk.as_ptr());
            let v1 = _mm256_loadu_ps(in_chunk.as_ptr().add(8));
            let v2 = _mm256_loadu_ps(in_chunk.as_ptr().add(16));
            let v3 = _mm256_loadu_ps(in_chunk.as_ptr().add(24));

            // min then max, interleaved across the 4 lanes for ILP
            let v0 = _mm256_max_ps(_mm256_min_ps(v0, vmax), vmin);
            let v1 = _mm256_max_ps(_mm256_min_ps(v1, vmax), vmin);
            let v2 = _mm256_max_ps(_mm256_min_ps(v2, vmax), vmin);
            let v3 = _mm256_max_ps(_mm256_min_ps(v3, vmax), vmin);

            _mm256_storeu_ps(out_chunk.as_mut_ptr(), v0);
            _mm256_storeu_ps(out_chunk.as_mut_ptr().add(8), v1);
            _mm256_storeu_ps(out_chunk.as_mut_ptr().add(16), v2);
            _mm256_storeu_ps(out_chunk.as_mut_ptr().add(24), v3);
        }

        // simd tail (< 32 floats)
        let (input_chunk8, input_rem8) = input_rem32.as_chunks::<8>();
        let (output_chunk8, output_rem8) = output_rem32.as_chunks_mut::<8>();
        for (in_chunk, out_chunk) in input_chunk8.iter().zip(output_chunk8.iter_mut()) {
            let v = _mm256_loadu_ps(in_chunk.as_ptr());
            let v = _mm256_max_ps(_mm256_min_ps(v, vmax), vmin);
            _mm256_storeu_ps(out_chunk.as_mut_ptr(), v);
        }

        // scalar (< 8 floats)
        for (src, dst) in input_rem8.iter().zip(output_rem8.iter_mut()) {
            *dst = src.max(min).min(max);
        }
    }

    Tensor::create(output, &x.dims, out_data)
}

#[inline]
fn relu(output: &str, inputs: &[Option<&Tensor>]) -> Result<Tensor> {
    let x = take_tensor(inputs, 0, "Relu")?;

    let mut data = vec![0f32; x.data.len()];
    if x.data.is_empty() {
        return Tensor::create(output, &x.dims, data);
    }

    if data.len() >= 8 {
        // simd path
        unsafe {
            let zero = _mm256_setzero_ps();

            // unroll by 4 ymm per iteration so the scheduler has
            // four independent `maxps` chains to overlap
            let (input_chunk32, input_rem32) = x.data.as_chunks::<32>();
            let (output_chunk32, output_rem32) = data.as_chunks_mut::<32>();

            for (input, output) in input_chunk32.iter().zip(output_chunk32.iter_mut()) {
                let v0 = _mm256_loadu_ps(input.as_ptr());
                let v1 = _mm256_loadu_ps(input.as_ptr().add(8));
                let v2 = _mm256_loadu_ps(input.as_ptr().add(16));
                let v3 = _mm256_loadu_ps(input.as_ptr().add(24));

                let v0 = _mm256_max_ps(v0, zero);
                let v1 = _mm256_max_ps(v1, zero);
                let v2 = _mm256_max_ps(v2, zero);
                let v3 = _mm256_max_ps(v3, zero);

                _mm256_storeu_ps(output.as_mut_ptr(), v0);
                _mm256_storeu_ps(output.as_mut_ptr().add(8), v1);
                _mm256_storeu_ps(output.as_mut_ptr().add(16), v2);
                _mm256_storeu_ps(output.as_mut_ptr().add(24), v3);
            }

            // simd tail (< 32 floats)
            let (input_chunk8, input_rem8) = input_rem32.as_chunks::<8>();
            let (output_chunk8, output_rem8) = output_rem32.as_chunks_mut::<8>();
            for (input, output) in input_chunk8.iter().zip(output_chunk8.iter_mut()) {
                let v = _mm256_loadu_ps(input.as_ptr());
                let result = _mm256_max_ps(v, zero);
                _mm256_storeu_ps(output.as_mut_ptr(), result);
            }

            // scalar (< 8 floats)
            for (input, output) in input_rem8.iter().zip(output_rem8.iter_mut()) {
                *output = input.max(0.0);
            }
        }
    } else {
        // scalar path
        for (&v, out) in x.data.iter().zip(data.iter_mut()) {
            *out = v.max(0.0);
        }
    }

    Tensor::create(output, &x.dims, data)
}

#[inline]
fn prelu(output: &str, inputs: &[Option<&Tensor>]) -> Result<Tensor> {
    let x = take_tensor(inputs, 0, "PRelu")?;
    let slope = take_tensor(inputs, 1, "PRelu")?;
    let x_dims = &x.dims;
    let s_dims = &slope.dims;

    // if slope has more dimensions than x, it cannot be broadcasted to x
    if s_dims.len() > x_dims.len() {
        return Err(anyhow!(
            "PRelu: slope dims {:?} is less than X dims {:?} and cannot be broadcasted",
            slope.dims,
            x.dims
        ));
    }

    // check that slope is broadcastable to x, trailing axes must match or be 1 (right-aligned)
    // e.g: x: [1, 3, 224, 224], slope: [3, 1, 1] is valid,
    // or [1, 1, 1], or [3, 224, 224], but not [2, 1, 1] or [3, 2, 224]
    let offset = x_dims.len() - s_dims.len();
    for (i, &sd) in s_dims.iter().enumerate() {
        let xd = x_dims[offset + i];
        if sd != 1 && sd != xd {
            return Err(anyhow!(
                "PRelu: slope dims {:?} has incompatible axis {} with X dims {:?}",
                slope.dims,
                i,
                x.dims
            ));
        }
    }

    let data_len = x.data.len();
    if x.data.is_empty() {
        return Tensor::create(output, &x.dims, vec![]);
    }

    // TODO: Add faster path for scalar slope

    // calculate the number of elements in the last contiguous block
    // of x that is covered by a single slope value
    let mut block: usize = 1;
    let rank = x_dims.len();
    for k in (0..rank).rev() {
        if k < offset || s_dims[k - offset] == 1 {
            block *= x_dims[k] as usize;
        } else {
            break;
        }
    }

    let mut data = Vec::with_capacity(data_len);
    let s_strides: Vec<usize> = slope.strides().iter().map(|&s| s as usize).collect();

    if block >= 8 {
        // simd path
        let in_ptr: *const f32 = x.data.as_ptr();
        let out_ptr: *mut f32 = data.as_mut_ptr();
        let x_strides: Vec<usize> = x.strides().iter().map(|&s| s as usize).collect();

        unsafe {
            let mut base = 0;
            while base < data_len {
                let end = base + block; // block is exactly divisible by data.len()
                let mut s_idx = 0; // element index into slope tensor for this block
                // compute the slope index for this block, using broadcasting rules
                // TODO: this part is still confusing to me, will come back later
                for j in 0..s_dims.len() {
                    if s_dims[j] != 1 {
                        let k = offset + j;
                        let coord = (base / x_strides[k]) % (x_dims[k] as usize);
                        s_idx += coord * s_strides[j];
                    }
                }
                // broadcast slope value to the entire block

                let in_ptr = in_ptr.add(base);
                let out_ptr = out_ptr.add(base);
                let vs = _mm256_set1_ps(slope.data[s_idx]);

                // unroll by 4 ymm
                let mut i = 0;
                let len = block;
                let zero = _mm256_setzero_ps();
                while i + 32 <= len {
                    let v0 = _mm256_loadu_ps(in_ptr.add(i));
                    let v1 = _mm256_loadu_ps(in_ptr.add(i + 8));
                    let v2 = _mm256_loadu_ps(in_ptr.add(i + 16));
                    let v3 = _mm256_loadu_ps(in_ptr.add(i + 24));

                    let m0 = _mm256_cmp_ps::<{ _CMP_LT_OQ }>(v0, zero);
                    let m1 = _mm256_cmp_ps::<{ _CMP_LT_OQ }>(v1, zero);
                    let m2 = _mm256_cmp_ps::<{ _CMP_LT_OQ }>(v2, zero);
                    let m3 = _mm256_cmp_ps::<{ _CMP_LT_OQ }>(v3, zero);

                    let o0 = _mm256_blendv_ps(v0, _mm256_mul_ps(v0, vs), m0);
                    let o1 = _mm256_blendv_ps(v1, _mm256_mul_ps(v1, vs), m1);
                    let o2 = _mm256_blendv_ps(v2, _mm256_mul_ps(v2, vs), m2);
                    let o3 = _mm256_blendv_ps(v3, _mm256_mul_ps(v3, vs), m3);

                    _mm256_storeu_ps(out_ptr.add(i), o0);
                    _mm256_storeu_ps(out_ptr.add(i + 8), o1);
                    _mm256_storeu_ps(out_ptr.add(i + 16), o2);
                    _mm256_storeu_ps(out_ptr.add(i + 24), o3);
                    i += 32;
                }
                // 8 float tail
                while i + 8 <= len {
                    let v = _mm256_loadu_ps(in_ptr.add(i));
                    let m = _mm256_cmp_ps::<{ _CMP_LT_OQ }>(v, zero);
                    let v = _mm256_blendv_ps(v, _mm256_mul_ps(v, vs), m);
                    _mm256_storeu_ps(out_ptr.add(i), v);
                    i += 8;
                }
                // 1 float tail
                let s = slope.data[s_idx];
                while i < len {
                    let v = *in_ptr.add(i);
                    *out_ptr.add(i) = v + (s - 1.0) * v.min(0.0);
                    i += 1;
                }

                base = end;
            }
            data.set_len(data_len);
        }
    } else {
        // scalar path
        let mut coords = vec![0usize; rank];
        for &v in x.data.iter() {
            let out = if v < 0.0 {
                let mut s_idx = 0usize;
                for j in 0..s_dims.len() {
                    if s_dims[j] != 1 {
                        s_idx += coords[offset + j] * s_strides[j];
                    }
                }
                v * slope.data[s_idx]
            } else {
                v
            };
            data.push(out);

            for k in (0..rank).rev() {
                coords[k] += 1;
                if (coords[k] as i64) < x_dims[k] {
                    break;
                }
                coords[k] = 0;
            }
        }
    }

    Tensor::create(output, &x.dims, data)
}

#[inline]
fn depth_to_space(
    output: &str,
    inputs: &[Option<&Tensor>],
    attrs: &[(String, AttributesDataValue)],
) -> Result<Tensor> {
    let x = take_tensor(inputs, 0, "DepthToSpace")?;
    let mode = take_attr_str(attrs, "mode")?;

    let dims = &x.dims;
    if dims.len() != 4 {
        return Err(anyhow!(
            "DepthToSpace: wants rank-4 NCHW, got rank {}",
            dims.len()
        ));
    }

    let bs = if let Some(bs) = take_attr_i64(attrs, "blocksize")? {
        if *bs < 1 {
            return Err(anyhow!("DepthToSpace: blocksize must be >= 1, got {bs}"));
        } else {
            *bs as usize
        }
    } else {
        return Err(anyhow!(
            "DepthToSpace: missing required attribute 'blocksize'"
        ));
    };

    let is_default = match mode {
        Some("DCR") | None => true,
        Some("CRD") => false,
        _ => {
            return Err(anyhow!(
                "DepthToSpace: unsupported mode '{:?}', only 'DCR' and 'CRD' are supported",
                mode
            ));
        }
    };

    let [ni, cin, h, w] = [
        dims[0] as usize,
        dims[1] as usize,
        dims[2] as usize,
        dims[3] as usize,
    ];

    let bs_sq = bs.checked_mul(bs).ok_or_else(|| {
        anyhow!(
            "DepthToSpace: blocksize {} is too large, would overflow",
            bs
        )
    })?;

    if cin % bs_sq != 0 {
        return Err(anyhow!(
            "DepthToSpace: channels {cin} are not divisible by blocksize^2({})",
            bs * bs
        ));
    }
    let co = cin / (bs_sq); // output channels
    let (oh, ow) = (h * bs, w * bs); //  output height and width
    let out_dims = [ni as i64, co as i64, oh as i64, ow as i64];

    let total = x.data.len();
    if ni * cin * h * w != total {
        return Err(anyhow!(
            "DepthToSpace: data length {} does not match dims {:?}",
            total,
            dims
        ));
    }

    // empty tensor, or blocksize 1 (identity in both modes)
    if total == 0 || bs == 1 {
        return Tensor::create(output, &out_dims, x.data.to_vec());
    }

    let plane = h * w;
    let src = x.data.as_ptr();
    let mut data: Vec<f32> = Vec::with_capacity(total);

    unsafe {
        // `out` only ever moves forward, the output is written once, in order
        let mut out = data.as_mut_ptr();
        for n in 0..ni {
            let src_n = src.add(n * cin * plane);
            for c in 0..co {
                for hi in 0..h {
                    for i in 0..bs {
                        // output row (hi*bs + i) interleaves bs input rows
                        // column offset j reads input channel ci0 + j*cstep
                        let (ci0, cstep) = if is_default {
                            (i * bs * co + c, co) // DCR
                        } else {
                            (c * bs_sq + i * bs, 1) // CRD
                        };
                        let r0 = src_n.add(ci0 * plane + hi * w);
                        let rs = cstep * plane;

                        match bs {
                            2 => interleave2(r0, r0.add(rs), out, w),
                            4 => {
                                interleave4(r0, r0.add(rs), r0.add(2 * rs), r0.add(3 * rs), out, w)
                            }
                            _ => {
                                for j in 0..bs {
                                    let s = r0.add(j * rs);
                                    for k in 0..w {
                                        *out.add(k * bs + j) = *s.add(k);
                                    }
                                }
                            }
                        }
                        out = out.add(ow);
                    }
                }
            }
        }
        // ni * co * (h*bs) rows of ow floats = total, all written above
        data.set_len(total);
    }

    Tensor::create(output, &out_dims, data)
}

// dst[2k] = a[k], dst[2k+1] = b[k] for k in 0..w
#[inline(always)]
fn interleave2(a: *const f32, b: *const f32, dst: *mut f32, w: usize) {
    #[inline(always)]
    fn x8(a: *const f32, b: *const f32, dst: *mut f32) {
        unsafe {
            let va = _mm256_loadu_ps(a);
            let vb = _mm256_loadu_ps(b);
            let lo = _mm256_unpacklo_ps(va, vb); // a0 b0 a1 b1 | a4 b4 a5 b5
            let hi = _mm256_unpackhi_ps(va, vb); // a2 b2 a3 b3 | a6 b6 a7 b7
            _mm256_storeu_ps(dst, _mm256_permute2f128_ps(lo, hi, 0x20));
            _mm256_storeu_ps(dst.add(8), _mm256_permute2f128_ps(lo, hi, 0x31));
        }
    }

    unsafe {
        if w < 8 {
            for k in 0..w {
                *dst.add(2 * k) = *a.add(k);
                *dst.add(2 * k + 1) = *b.add(k);
            }
            return;
        }
        let mut k = 0;
        while k + 8 <= w {
            x8(a.add(k), b.add(k), dst.add(2 * k));
            k += 8;
        }
        // tail, ovewrite last 8 floats,
        // even if w < 8, because we already wrote the first part
        if k < w {
            let k = w - 8;
            x8(a.add(k), b.add(k), dst.add(2 * k));
        }
    }
}

// dst[4k + j] = (a, b, c, d)[j][k] for k in 0..w
#[inline(always)]
fn interleave4(
    a: *const f32,
    b: *const f32,
    c: *const f32,
    d: *const f32,
    dst: *mut f32,
    w: usize,
) {
    #[inline(always)]
    fn x8(a: *const f32, b: *const f32, c: *const f32, d: *const f32, dst: *mut f32) {
        unsafe {
            let va = _mm256_loadu_ps(a);
            let vb = _mm256_loadu_ps(b);
            let vc = _mm256_loadu_ps(c);
            let vd = _mm256_loadu_ps(d);

            let t0 = _mm256_unpacklo_ps(va, vb); // a0 b0 a1 b1 | a4 b4 a5 b5
            let t1 = _mm256_unpackhi_ps(va, vb); // a2 b2 a3 b3 | a6 b6 a7 b7
            let t2 = _mm256_unpacklo_ps(vc, vd); // c0 d0 c1 d1 | c4 d4 c5 d5
            let t3 = _mm256_unpackhi_ps(vc, vd); // c2 d2 c3 d3 | c6 d6 c7 d7

            let u0 = _mm256_shuffle_ps(t0, t2, 0x44); // a0 b0 c0 d0 | a4 b4 c4 d4
            let u1 = _mm256_shuffle_ps(t0, t2, 0xEE); // a1 b1 c1 d1 | a5 b5 c5 d5
            let u2 = _mm256_shuffle_ps(t1, t3, 0x44); // a2 b2 c2 d2 | a6 b6 c6 d6
            let u3 = _mm256_shuffle_ps(t1, t3, 0xEE); // a3 b3 c3 d3 | a7 b7 c7 d7

            _mm256_storeu_ps(dst, _mm256_permute2f128_ps(u0, u1, 0x20));
            _mm256_storeu_ps(dst.add(8), _mm256_permute2f128_ps(u2, u3, 0x20));
            _mm256_storeu_ps(dst.add(16), _mm256_permute2f128_ps(u0, u1, 0x31));
            _mm256_storeu_ps(dst.add(24), _mm256_permute2f128_ps(u2, u3, 0x31));
        }
    }

    unsafe {
        if w < 8 {
            for k in 0..w {
                *dst.add(4 * k) = *a.add(k);
                *dst.add(4 * k + 1) = *b.add(k);
                *dst.add(4 * k + 2) = *c.add(k);
                *dst.add(4 * k + 3) = *d.add(k);
            }
            return;
        }
        let mut k = 0;
        while k + 8 <= w {
            x8(a.add(k), b.add(k), c.add(k), d.add(k), dst.add(4 * k));
            k += 8;
        }
        if k < w {
            let k = w - 8;
            x8(a.add(k), b.add(k), c.add(k), d.add(k), dst.add(4 * k));
        }
    }
}

// TODO: I dont understand convolution properly, will come back to this later
// TODO: According to ONNX spec, Conv has some more optionally stuffs, which are ignored for now
fn conv(
    output: &str,
    inputs: &[Option<&Tensor>],
    attrs: &[(String, AttributesDataValue)],
) -> Result<Tensor> {
    // take two required inputs: X and W, and optional bias
    let x = take_tensor(inputs, 0, "Conv")?;
    let w = take_tensor(inputs, 1, "Conv")?;

    // handle attributes: auto_pad, group, strides, dilations, pads, kernel_shape

    let valid_auto = ["NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"];
    let auto_pad = match take_attr_str(attrs, "auto_pad")? {
        Some(ap) if valid_auto.contains(&ap) => ap, // was `_ => "NOTSET"`, which threw valid values away
        Some(ap) => {
            return Err(anyhow!(
                "Conv: auto_pad must be one of {valid_auto:?}, got '{ap}'"
            ));
        }
        None => "NOTSET",
    };
    if auto_pad != "NOTSET" && take_attr_i64s(attrs, "pads")?.is_some() {
        return Err(anyhow!(
            "Conv: 'pads' attribute cannot be used with auto_pad '{auto_pad}'"
        ));
    }

    let group = match take_attr_i64(attrs, "group")?.copied() {
        Some(g) if g < 1 => {
            return Err(anyhow!("Conv: group must be >= 1, got {g}"));
        }
        Some(g) => g,
        None => 1,
    };
    let group = group as usize;

    let [sh, sw] = take_attr_i64s_n::<2>(attrs, "strides")?.unwrap_or([1, 1]);
    let [dh, dw] = take_attr_i64s_n::<2>(attrs, "dilations")?.unwrap_or([1, 1]);
    if sh <= 0 || sw <= 0 || dh <= 0 || dw <= 0 {
        return Err(anyhow!("Conv: strides/dilations must be positive"));
    }
    let [sh, sw, dh, dw] = [sh as usize, sw as usize, dh as usize, dw as usize];

    let (xd, wd) = (&x.dims, &w.dims);
    if xd.len() != 4 || wd.len() != 4 {
        return Err(anyhow!(
            "Conv: wants rank-4 NCHW X and weights, got ranks {}/{}",
            xd.len(),
            wd.len()
        ));
    }

    let [n, cin, hi, wi] = [
        xd[0] as usize,
        xd[1] as usize,
        xd[2] as usize,
        xd[3] as usize,
    ];
    let [m, wcin, kh, kw] = [
        wd[0] as usize,
        wd[1] as usize,
        wd[2] as usize,
        wd[3] as usize,
    ];

    if kh == 0 || kw == 0 {
        return Err(anyhow!("Conv: kernel dims must be >= 1, weights {wd:?}"));
    }

    if cin != wcin * group {
        return Err(anyhow!(
            "Conv: X channels {cin} != weights channels {wcin} * group {group}"
        ));
    }
    if m % group != 0 {
        return Err(anyhow!(
            "Conv: output channels {m} not divisible by group {group}"
        ));
    }
    if let Some(ks) = take_attr_i64s(attrs, "kernel_shape")?
        && ks != [kh as i64, kw as i64]
    {
        return Err(anyhow!(
            "Conv: kernel_shape {ks:?} disagrees with weights [{kh}, {kw}]"
        ));
    }

    let bias: Option<&[f32]> = match inputs.get(2).copied().flatten() {
        Some(t) if t.dims.as_ref() == [m as i64] => Some(t.data.as_ref()),
        Some(t) => {
            return Err(anyhow!(
                "Conv: bias must have shape [{m}], got dims {:?}",
                t.dims
            ));
        }
        None => None,
    };

    let (eff_kh, eff_kw) = ((kh - 1) * dh + 1, (kw - 1) * dw + 1);
    let [pt, pl, pb, pr]: [usize; 4] = match auto_pad {
        "NOTSET" => {
            let pads = take_attr_i64s_n::<4>(attrs, "pads")?.unwrap_or([0; 4]);
            if pads.iter().any(|&p| p < 0) {
                return Err(anyhow!(
                    "Conv: pads must be non-negative when auto_pad is NOTSET"
                ));
            }
            [
                pads[0] as usize,
                pads[1] as usize,
                pads[2] as usize,
                pads[3] as usize,
            ]
        }
        "VALID" => [0; 4],
        _ => {
            let total = |inp: usize, s: usize, eff: usize| -> usize {
                let out = inp.div_ceil(s);
                ((out as i64 - 1) * s as i64 + eff as i64 - inp as i64).max(0) as usize
            };
            let (th, tw) = (total(hi, sh, eff_kh), total(wi, sw, eff_kw));
            // odd total is extra pixel at the end for SAME_UPPER, at the start for SAME_LOWER
            let (b_h, b_w) = if auto_pad == "SAME_UPPER" {
                // extra padding at the end (bottom/right)
                (th / 2, tw / 2)
            } else {
                // extra padding at the start (top/left)
                (th - th / 2, tw - tw / 2)
            };
            [b_h, b_w, th - b_h, tw - b_w]
        }
    };

    // compute output dims from explicit/computed pads
    let oh_ext = hi as i64 + pt as i64 + pb as i64 - eff_kh as i64;
    let ow_ext = wi as i64 + pl as i64 + pr as i64 - eff_kw as i64;
    if oh_ext < 0 || ow_ext < 0 {
        return Err(anyhow!(
            "Conv: kernel [{kh}×{kw}] does not fit into [{hi}×{wi}] with pads [{pt},{pl},{pb},{pr}]"
        ));
    }
    let (oh, ow) = (
        (oh_ext / sh as i64 + 1) as usize,
        (ow_ext / sw as i64 + 1) as usize,
    );
    let p = oh
        .checked_mul(ow)
        .ok_or_else(|| anyhow!("Conv: output dims {oh}*{ow} overflow element count"))?;

    // group convolutions
    let m_g = m / group; // output channels per group
    let c_g = cin / group; // input channels per group
    let k_g = c_g * kh * kw; // K per group

    let out_len = n
        .checked_mul(m)
        .and_then(|t| t.checked_mul(p))
        .ok_or_else(|| anyhow!("Conv: output buffer {n}*{m}*{p} overflows"))?;

    // tile over output rows so the im2col buffer stays bounded
    // ~12 MB tile stays mostly L3-resident while the gemm reads it back
    const COL_TILE_BUDGET: usize = 12 << 20;
    let budget_f32 = COL_TILE_BUDGET / size_of::<f32>();
    let tile_oh = match k_g.checked_mul(ow) {
        Some(k_ow) => (budget_f32 / k_ow).max(1).min(oh),
        None => 1,
    };
    let tile_p_max = tile_oh * ow;

    let col_len = k_g
        .checked_mul(tile_p_max)
        .ok_or_else(|| anyhow!("Conv: im2col buffer {k_g}*{tile_p_max} overflows"))?;
    let mut col = vec![0f32; col_len]; // [p_tile, K_g] col-major col[k * p_tile + pix]

    let mut out = vec![0f32; out_len];
    // prefill bias so gemm can run with beta = 1.0 (no separate bias pass)
    if let Some(b) = bias {
        for (i, plane) in out.chunks_mut(p).enumerate() {
            plane.fill(b[i % m]);
        }
    }

    // compute the number of pixels in a kernel,
    // used for indexing into the input channels
    let khw = kh * kw;
    let xdata: &[f32] = &x.data;

    for ni in 0..n {
        for g in 0..group {
            let wg = &w.data[g * m_g * k_g..(g + 1) * m_g * k_g]; // col-major [K_g, M_g]
            let c_start = g * c_g;
            let outbase_g = ni * m * p + g * m_g * p;

            // tile over output rows so the im2col buffer stays bounded
            for tile in (0..oh).step_by(tile_oh) {
                let oh_end = (tile + tile_oh).min(oh);
                let p_tile = (oh_end - tile) * ow;

                // TODO: this is god awful, but it works
                // I don't understand convolution well enough to do better yet

                // fill one row of the im2col buffer for a given kernel index k
                let fill_row = |k: usize, dst: &mut [f32]| {
                    let ci = k / khw;
                    let (khi, kwi) = ((k % khw) / kw, (k % khw) % kw);
                    let xchan = &xdata[(ni * cin + c_start + ci) * hi * wi..][..hi * wi];

                    // valid output-col range where the input col is inside the image
                    let off = (kwi * dw) as i64 - pl as i64;
                    let low_b = if off >= 0 {
                        0
                    } else {
                        ((-off) as usize).div_ceil(sw)
                    }
                    .min(ow);
                    let high_b = ((wi as i64 - off).max(0) as usize)
                        .div_ceil(sw)
                        .min(ow)
                        .max(low_b);

                    for ohi in tile..oh_end {
                        let row = &mut dst[(ohi - tile) * ow..][..ow];
                        let ih = (ohi * sh + khi * dh) as i64 - pt as i64;
                        if ih < 0 || ih >= hi as i64 {
                            row.fill(0.0);
                            continue;
                        }
                        row[..low_b].fill(0.0);
                        row[high_b..].fill(0.0);
                        if high_b > low_b {
                            let src = &xchan[ih as usize * wi..][..wi];
                            let s0 = (low_b as i64 * sw as i64 + off) as usize;
                            if sw == 1 {
                                row[low_b..high_b].copy_from_slice(&src[s0..s0 + high_b - low_b]);
                            } else {
                                for (t, d) in row[low_b..high_b].iter_mut().enumerate() {
                                    *d = src[s0 + t * sw];
                                }
                            }
                        }
                    }
                };

                // fill im2col, one task per k row, each row is contiguous copies
                const PAR_IM2COL_THRESHOLD: usize = 1 << 17;
                let col_tile = &mut col[..k_g * p_tile];
                if k_g * p_tile < PAR_IM2COL_THRESHOLD {
                    col_tile
                        .chunks_mut(p_tile)
                        .enumerate()
                        .for_each(|(k, dst)| fill_row(k, dst));
                } else {
                    GLOBAL_THREAD_POOL.install(|| {
                        col_tile
                            .par_chunks_mut(p_tile)
                            .enumerate()
                            .for_each(|(k, dst)| fill_row(k, dst));
                    });
                }

                // C[P_tile x M_g] = col[P_tile x K_g] * Wg^T[K_g x M_g]
                // C is column-major with ldc = p, which is exactly the NCHW plane layout
                lvl3::gemm(
                    p_tile,
                    m_g,
                    k_g,
                    1.0,
                    &col[..k_g * p_tile],
                    p_tile,
                    wg,
                    k_g,
                    1.0, // bias already in out
                    &mut out[outbase_g + tile * ow..],
                    p,
                    false,
                    false,
                );
            }
        }
    }

    Tensor::create(output, &[n as i64, m as i64, oh as i64, ow as i64], out)
}

#[inline]
fn add(output: &str, inputs: &[Option<&Tensor>]) -> Result<Tensor> {
    let x = take_tensor(inputs, 0, "Add")?;
    let y = take_tensor(inputs, 1, "Add")?;

    let x_dims = &x.dims;
    let y_dims = &y.dims;
    let n = x.data.len();

    // fast simd path for same shape tensors
    if x_dims == y_dims {
        let mut data = Vec::with_capacity(n);
        let xp = x.data.as_ptr();
        let yp = y.data.as_ptr();
        let op: *mut f32 = data.as_mut_ptr();

        unsafe {
            let mut i = 0usize;
            while i + 32 <= n {
                let a0 = _mm256_loadu_ps(xp.add(i));
                let a1 = _mm256_loadu_ps(xp.add(i + 8));
                let a2 = _mm256_loadu_ps(xp.add(i + 16));
                let a3 = _mm256_loadu_ps(xp.add(i + 24));
                let b0 = _mm256_loadu_ps(yp.add(i));
                let b1 = _mm256_loadu_ps(yp.add(i + 8));
                let b2 = _mm256_loadu_ps(yp.add(i + 16));
                let b3 = _mm256_loadu_ps(yp.add(i + 24));
                _mm256_storeu_ps(op.add(i), _mm256_add_ps(a0, b0));
                _mm256_storeu_ps(op.add(i + 8), _mm256_add_ps(a1, b1));
                _mm256_storeu_ps(op.add(i + 16), _mm256_add_ps(a2, b2));
                _mm256_storeu_ps(op.add(i + 24), _mm256_add_ps(a3, b3));
                i += 32;
            }
            while i + 8 <= n {
                let a = _mm256_loadu_ps(xp.add(i));
                let b = _mm256_loadu_ps(yp.add(i));
                _mm256_storeu_ps(op.add(i), _mm256_add_ps(a, b));
                i += 8;
            }
            while i < n {
                *op.add(i) = *xp.add(i) + *yp.add(i);
                i += 1;
            }
            data.set_len(n);
        }

        return Tensor::create(output, x_dims, data);
    }

    broadcast_binary_op(x, y, output, |a, b| a + b)
}

#[inline]
fn sub(output: &str, inputs: &[Option<&Tensor>]) -> Result<Tensor> {
    let x = take_tensor(inputs, 0, "Sub")?;
    let y = take_tensor(inputs, 1, "Sub")?;

    let x_dims = &x.dims;
    let y_dims = &y.dims;
    let n = x.data.len();

    // fast simd path for same shape tensors
    if x_dims == y_dims {
        let mut data = Vec::with_capacity(n);
        let xp = x.data.as_ptr();
        let yp = y.data.as_ptr();
        let op: *mut f32 = data.as_mut_ptr();

        unsafe {
            let mut i = 0usize;
            while i + 32 <= n {
                let a0 = _mm256_loadu_ps(xp.add(i));
                let a1 = _mm256_loadu_ps(xp.add(i + 8));
                let a2 = _mm256_loadu_ps(xp.add(i + 16));
                let a3 = _mm256_loadu_ps(xp.add(i + 24));
                let b0 = _mm256_loadu_ps(yp.add(i));
                let b1 = _mm256_loadu_ps(yp.add(i + 8));
                let b2 = _mm256_loadu_ps(yp.add(i + 16));
                let b3 = _mm256_loadu_ps(yp.add(i + 24));
                _mm256_storeu_ps(op.add(i), _mm256_sub_ps(a0, b0));
                _mm256_storeu_ps(op.add(i + 8), _mm256_sub_ps(a1, b1));
                _mm256_storeu_ps(op.add(i + 16), _mm256_sub_ps(a2, b2));
                _mm256_storeu_ps(op.add(i + 24), _mm256_sub_ps(a3, b3));
                i += 32;
            }
            while i + 8 <= n {
                let a = _mm256_loadu_ps(xp.add(i));
                let b = _mm256_loadu_ps(yp.add(i));
                _mm256_storeu_ps(op.add(i), _mm256_sub_ps(a, b));
                i += 8;
            }
            while i < n {
                *op.add(i) = *xp.add(i) - *yp.add(i);
                i += 1;
            }
            data.set_len(n);
        }

        return Tensor::create(output, x_dims, data);
    }

    broadcast_binary_op(x, y, output, |a, b| a - b)
}

#[inline]
fn mul(output: &str, inputs: &[Option<&Tensor>]) -> Result<Tensor> {
    let x = take_tensor(inputs, 0, "Mul")?;
    let y = take_tensor(inputs, 1, "Mul")?;

    let x_dims = &x.dims;
    let y_dims = &y.dims;
    let n = x.data.len();

    // fast simd path for same shape tensors
    if x_dims == y_dims {
        let mut data = Vec::with_capacity(n);
        let xp = x.data.as_ptr();
        let yp = y.data.as_ptr();
        let op: *mut f32 = data.as_mut_ptr();

        unsafe {
            let mut i = 0usize;
            while i + 32 <= n {
                let a0 = _mm256_loadu_ps(xp.add(i));
                let a1 = _mm256_loadu_ps(xp.add(i + 8));
                let a2 = _mm256_loadu_ps(xp.add(i + 16));
                let a3 = _mm256_loadu_ps(xp.add(i + 24));
                let b0 = _mm256_loadu_ps(yp.add(i));
                let b1 = _mm256_loadu_ps(yp.add(i + 8));
                let b2 = _mm256_loadu_ps(yp.add(i + 16));
                let b3 = _mm256_loadu_ps(yp.add(i + 24));
                _mm256_storeu_ps(op.add(i), _mm256_mul_ps(a0, b0));
                _mm256_storeu_ps(op.add(i + 8), _mm256_mul_ps(a1, b1));
                _mm256_storeu_ps(op.add(i + 16), _mm256_mul_ps(a2, b2));
                _mm256_storeu_ps(op.add(i + 24), _mm256_mul_ps(a3, b3));
                i += 32;
            }
            while i + 8 <= n {
                let a = _mm256_loadu_ps(xp.add(i));
                let b = _mm256_loadu_ps(yp.add(i));
                _mm256_storeu_ps(op.add(i), _mm256_mul_ps(a, b));
                i += 8;
            }
            while i < n {
                *op.add(i) = *xp.add(i) * *yp.add(i);
                i += 1;
            }
            data.set_len(n);
        }

        return Tensor::create(output, x_dims, data);
    }

    broadcast_binary_op(x, y, output, |a, b| a * b)
}

#[inline]
fn div(output: &str, inputs: &[Option<&Tensor>]) -> Result<Tensor> {
    let x = take_tensor(inputs, 0, "Div")?;
    let y = take_tensor(inputs, 1, "Div")?;

    let x_dims = &x.dims;
    let y_dims = &y.dims;
    let n = x.data.len();

    // fast simd path for same shape tensors
    if x_dims == y_dims {
        let mut data = Vec::with_capacity(n);
        let xp = x.data.as_ptr();
        let yp = y.data.as_ptr();
        let op: *mut f32 = data.as_mut_ptr();

        unsafe {
            let mut i = 0usize;
            while i + 32 <= n {
                let a0 = _mm256_loadu_ps(xp.add(i));
                let a1 = _mm256_loadu_ps(xp.add(i + 8));
                let a2 = _mm256_loadu_ps(xp.add(i + 16));
                let a3 = _mm256_loadu_ps(xp.add(i + 24));
                let b0 = _mm256_loadu_ps(yp.add(i));
                let b1 = _mm256_loadu_ps(yp.add(i + 8));
                let b2 = _mm256_loadu_ps(yp.add(i + 16));
                let b3 = _mm256_loadu_ps(yp.add(i + 24));
                _mm256_storeu_ps(op.add(i), _mm256_div_ps(a0, b0));
                _mm256_storeu_ps(op.add(i + 8), _mm256_div_ps(a1, b1));
                _mm256_storeu_ps(op.add(i + 16), _mm256_div_ps(a2, b2));
                _mm256_storeu_ps(op.add(i + 24), _mm256_div_ps(a3, b3));
                i += 32;
            }
            while i + 8 <= n {
                let a = _mm256_loadu_ps(xp.add(i));
                let b = _mm256_loadu_ps(yp.add(i));
                _mm256_storeu_ps(op.add(i), _mm256_div_ps(a, b));
                i += 8;
            }
            while i < n {
                *op.add(i) = *xp.add(i) / *yp.add(i);
                i += 1;
            }
            data.set_len(n);
        }

        return Tensor::create(output, x_dims, data);
    }

    broadcast_binary_op(x, y, output, |a, b| a / b)
}

#[inline(always)]
fn broadcast_binary_op<OP>(x: &Tensor, y: &Tensor, output: &str, scalar_op: OP) -> Result<Tensor>
where
    OP: Fn(f32, f32) -> f32,
{
    let x_dims = &x.dims;
    let y_dims = &y.dims;

    let x_strides: Vec<usize> = x.strides().iter().map(|&s| s as usize).collect();
    let y_strides: Vec<usize> = y.strides().iter().map(|&s| s as usize).collect();

    let rank = x_dims.len().max(y_dims.len());
    let x_pad = rank - x_dims.len();
    let y_pad = rank - y_dims.len();

    let mut out_dims = Vec::with_capacity(rank);
    let mut bx_strides = vec![0usize; rank];
    let mut by_strides = vec![0usize; rank];

    for k in 0..rank {
        let xd = if k < x_pad { 1 } else { x_dims[k - x_pad] };
        let yd = if k < y_pad { 1 } else { y_dims[k - y_pad] };

        let od = if xd == yd {
            xd
        } else if xd == 1 {
            yd
        } else if yd == 1 {
            xd
        } else {
            return Err(anyhow!(
                "{}: cannot broadcast X dims {:?} with Y dims {:?}",
                output,
                x_dims,
                y_dims
            ));
        };

        if k >= x_pad && xd != 1 {
            bx_strides[k] = x_strides[k - x_pad];
        }
        if k >= y_pad && yd != 1 {
            by_strides[k] = y_strides[k - y_pad];
        }

        out_dims.push(od);
    }

    let total: usize = out_dims
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d as usize))
        .ok_or_else(|| anyhow!("{}: dims overflow", output))?;

    // zero-stride which mean any size-1 output
    // axis so it merges cleanly below
    for k in 0..rank {
        if out_dims[k] == 1 {
            bx_strides[k] = 0;
            by_strides[k] = 0;
        }
    }

    // collapse adjacent axes from the right where both tensors
    let mut axes: Vec<(usize, usize, usize)> = Vec::with_capacity(rank);
    for k in (0..rank).rev() {
        let d = out_dims[k] as usize;
        let xb = bx_strides[k];
        let yb = by_strides[k];

        let merged = match axes.last_mut() {
            Some(last) => {
                let (ld, lx, ly) = *last;
                let x_ok = (xb == 0 && lx == 0) || (xb != 0 && lx != 0 && xb == lx * ld);
                let y_ok = (yb == 0 && ly == 0) || (yb != 0 && ly != 0 && yb == ly * ld);
                if x_ok && y_ok {
                    *last = (ld * d, lx, ly);
                    true
                } else {
                    false
                }
            }
            None => false,
        };
        if !merged {
            axes.push((d, xb, yb));
        }
    }
    axes.reverse();

    let d_rank = axes.len();
    let mut data = Vec::with_capacity(total);
    let data_ptr: *mut f32 = data.as_mut_ptr();

    let xp = x.data.as_ptr();
    let yp = y.data.as_ptr();

    let mut coords = vec![0usize; d_rank];
    let (mut xi, mut yi) = (0usize, 0usize);

    unsafe {
        for idx in 0..total {
            *data_ptr.add(idx) = scalar_op(*xp.add(xi), *yp.add(yi));
            for k in (0..d_rank).rev() {
                coords[k] += 1;
                let (d, bx, by) = axes[k];
                if coords[k] < d {
                    xi += bx;
                    yi += by;
                    break;
                }
                coords[k] = 0;
                xi -= (d - 1) * bx;
                yi -= (d - 1) * by;
            }
        }
        data.set_len(total);
    }

    Tensor::create(output, &out_dims, data)
}

#[inline]
fn sigmoid(output: &str, inputs: &[Option<&Tensor>]) -> Result<Tensor> {
    let x = take_tensor(inputs, 0, "Sigmoid")?;
    let n = x.data.len();

    let mut data = Vec::with_capacity(n);
    let dst_ptr = data.spare_capacity_mut().as_mut_ptr() as *mut f32;
    let src_ptr = x.data.as_ptr();

    // y = 1 / (1 + exp(-x))
    unsafe {
        let mut i = 0;
        while i + 32 <= n {
            let x0 = _mm256_loadu_ps(src_ptr.add(i));
            let neg_x0 = _mm256_sub_ps(_mm256_set1_ps(0.0), x0);
            let x1 = _mm256_loadu_ps(src_ptr.add(i + 8));
            let neg_x1 = _mm256_sub_ps(_mm256_set1_ps(0.0), x1);
            let x2 = _mm256_loadu_ps(src_ptr.add(i + 16));
            let neg_x2 = _mm256_sub_ps(_mm256_set1_ps(0.0), x2);
            let x3 = _mm256_loadu_ps(src_ptr.add(i + 24));
            let neg_x3 = _mm256_sub_ps(_mm256_set1_ps(0.0), x3);

            let exp_neg_x0 = avx_exp(neg_x0);
            let exp_neg_x1 = avx_exp(neg_x1);
            let exp_neg_x2 = avx_exp(neg_x2);
            let exp_neg_x3 = avx_exp(neg_x3);

            let denom0 = _mm256_add_ps(_mm256_set1_ps(1.0), exp_neg_x0);
            let y0 = _mm256_div_ps(_mm256_set1_ps(1.0), denom0);
            let denom1 = _mm256_add_ps(_mm256_set1_ps(1.0), exp_neg_x1);
            let y1 = _mm256_div_ps(_mm256_set1_ps(1.0), denom1);
            let denom2 = _mm256_add_ps(_mm256_set1_ps(1.0), exp_neg_x2);
            let y2 = _mm256_div_ps(_mm256_set1_ps(1.0), denom2);
            let denom3 = _mm256_add_ps(_mm256_set1_ps(1.0), exp_neg_x3);
            let y3 = _mm256_div_ps(_mm256_set1_ps(1.0), denom3);

            _mm256_storeu_ps(dst_ptr.add(i), y0);
            _mm256_storeu_ps(dst_ptr.add(i + 16), y2);
            _mm256_storeu_ps(dst_ptr.add(i + 8), y1);
            _mm256_storeu_ps(dst_ptr.add(i + 24), y3);
            i += 32
        }
        while i + 8 <= n {
            let x0 = _mm256_loadu_ps(src_ptr.add(i));
            let neg_x0 = _mm256_sub_ps(_mm256_set1_ps(0.0), x0);
            let exp_neg_x0 = avx_exp(neg_x0);
            let denom0 = _mm256_add_ps(_mm256_set1_ps(1.0), exp_neg_x0);
            let y0 = _mm256_div_ps(_mm256_set1_ps(1.0), denom0);
            _mm256_storeu_ps(dst_ptr.add(i), y0);
            i += 8;
        }
        while i < n {
            let x_val = *src_ptr.add(i);
            let exp_neg_x = (-x_val).exp();
            let y_val = 1.0 / (1.0 + exp_neg_x);
            *dst_ptr.add(i) = y_val;
            i += 1;
        }

        data.set_len(n);
    }

    Tensor::create(output, &x.dims, data)
}

#[inline(always)]
fn avx_exp(x: __m256) -> __m256 {
    unsafe {
        // clamp to avoid overflow/underflow in exp
        let max_val = _mm256_set1_ps(f32::MAX.ln()); // 88.72
        let min_val = _mm256_set1_ps(f32::MIN_POSITIVE.ln()); // -87.34
        let x = _mm256_min_ps(_mm256_max_ps(x, min_val), max_val);

        // consts
        let v_log2e = _mm256_set1_ps(std::f32::consts::LOG2_E); // 1.44269504
        let v_ln2_hi = _mm256_set1_ps(std::f32::consts::LN_2); // 0.69314718
        let v_ln2_lo = _mm256_set1_ps(1.428_606_8e-6);

        // fx = round(x * log2(e))
        let fx = _mm256_round_ps(_mm256_mul_ps(x, v_log2e), _MM_FROUND_CUR_DIRECTION);
        let k = _mm256_cvtps_epi32(fx);

        // r = x - fx * ln2_hi - fx * ln2_lo
        let r = _mm256_sub_ps(
            _mm256_sub_ps(x, _mm256_mul_ps(fx, v_ln2_hi)),
            _mm256_mul_ps(fx, v_ln2_lo),
        );

        // degree-6 minimax polynomial for exp(r) via Horner's method
        let mut p = _mm256_set1_ps(0.00138889);
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.00833675));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.04166397));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.16666655));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));

        // reconstruct 2^k by bit-shifting directly into IEEE-754 exponent bits
        let bias = _mm256_set1_epi32(127);
        let ik = _mm256_add_epi32(k, bias);
        let pow2k = _mm256_castsi256_ps(_mm256_slli_epi32(ik, 23));

        // multiply polynomial result by 2^k to get exp(x)
        _mm256_mul_ps(p, pow2k)
    }
}

#[inline]
fn concat(
    output: &str,
    inputs: &[Option<&Tensor>],
    attrs: &[(String, AttributesDataValue)],
) -> Result<Tensor> {
    // gather the inputs, an absent input is an error for Concat
    let mut tensors: Vec<&Tensor> = Vec::with_capacity(inputs.len());
    for (i, t) in inputs.iter().enumerate() {
        let Some(t) = t else {
            return Err(anyhow!("Concat: input #{i} is absent"));
        };
        tensors.push(t);
    }
    if tensors.is_empty() {
        return Err(anyhow!("Concat: needs at least one input"));
    }

    let first = tensors[0];
    let rank = first.dims.len();

    let axis = match take_attr_i64(attrs, "axis")? {
        Some(a) => *a,
        None => return Err(anyhow!("Concat: missing required attribute 'axis'")),
    };

    let axis = if axis < 0 { axis + rank as i64 } else { axis };
    if axis < 0 || axis >= rank as i64 {
        return Err(anyhow!(
            "Concat: axis {axis} is out of range for a rank-{rank} input"
        ));
    }
    let axis = axis as usize;

    // every input must have the same rank, and matching dims outside 'axis'
    let mut axis_total: i64 = 0;
    for t in tensors.iter() {
        if t.dims.len() != rank {
            return Err(anyhow!(
                "Concat: all inputs must have the same rank, got {rank} and {}",
                t.dims.len()
            ));
        }
        for d in 0..rank {
            if d != axis && t.dims[d] != first.dims[d] {
                return Err(anyhow!(
                    "Concat: input dims {:?} does not match {:?} outside axis {axis}",
                    t.dims,
                    first.dims
                ));
            }
        }
        axis_total += t.dims[axis];
    }

    // update the output dims to reflect the concatenation along 'axis'
    let mut out_dims = first.dims.to_vec();
    out_dims[axis] = axis_total;

    // the tensor is seen as [outer, axis_len, inner], where outer is the
    // product of the dims before 'axis' and inner the product of those after
    let outer: usize = first.dims[..axis].iter().map(|&d| d as usize).product();
    let inner: usize = first.dims[axis + 1..].iter().map(|&d| d as usize).product();

    let total: usize = out_dims
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d as usize))
        .ok_or_else(|| anyhow!("Concat: output dims {out_dims:?} overflow element count"))?;
    if total == 0 {
        return Tensor::create(output, &out_dims, Vec::new());
    }

    let mut data = Vec::with_capacity(total);
    let dst: *mut f32 = data.as_mut_ptr();
    let axis_total_usize = axis_total as usize;
    let mut axis_offset = 0usize;

    for t in tensors.iter() {
        let a_len = t.dims[axis] as usize;
        if a_len == 0 {
            continue;
        }
        let chunk = a_len * inner; // one contiguous run per (input, outer)
        let src = t.data.as_ptr();
        for o in 0..outer {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.add(o * chunk),
                    dst.add((o * axis_total_usize + axis_offset) * inner),
                    chunk,
                );
            }
        }
        axis_offset += a_len;
    }
    unsafe {
        data.set_len(total);
    }

    Tensor::create(output, &out_dims, data)
}

#[inline]
fn softplus(output: &str, inputs: &[Option<&Tensor>]) -> Result<Tensor> {
    let x = take_tensor(inputs, 0, "SoftPlus")?;
    let n = x.data.len();

    // TODO: AVX2 has no native exp or ln instruction
    let mut data = Vec::with_capacity(n);
    for (dst, &v) in data.spare_capacity_mut().iter_mut().zip(x.data.iter()) {
        dst.write((v.exp() + 1.0).ln());
    }
    unsafe {
        data.set_len(n);
    }

    Tensor::create(output, &x.dims, data)
}
