//! Kernel implementations for ONNX operators.
use crate::ops::{AttributesDataValue, Operator};
use crate::tensor::Tensor;
use anyhow::{Result, anyhow};
use blas_rs::*;
use std::arch::x86_64::{
    _CMP_LT_OQ, _mm256_blendv_ps, _mm256_cmp_ps, _mm256_load_ps, _mm256_loadu_ps, _mm256_max_ps,
    _mm256_min_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_store_ps,
    _mm256_storeu_ps,
};

impl Operator {
    pub fn compute(
        self,
        out_name: &str,
        inputs: &[Option<&Tensor>],
        attrs: &[(String, AttributesDataValue)],
    ) -> Result<Tensor> {
        match self {
            Operator::Clip => clip(out_name, inputs),
            Operator::PRelu => prelu(out_name, inputs),
            Operator::DepthToSpace => depth_to_space(out_name, inputs, attrs),
            Operator::Conv => conv(out_name, inputs, attrs),
            Operator::Gemm => gemm_kernel(out_name, inputs, attrs),
            _ => Err(anyhow!(
                "operator '{}' has no kernel implemented yet",
                self.static_name()
            )),
        }
    }
}

#[inline(always)]
fn slot<'s>(inputs: &'s [Option<&'s Tensor>], index: usize, op_name: &str) -> Result<&'s Tensor> {
    let Some(t) = inputs.get(index).copied().flatten() else {
        return Err(anyhow!("{op_name}: required input slot #{index} is absent"));
    };
    Ok(t)
}

#[inline(always)]
fn attr_f32<'s>(attrs: &'s [(String, AttributesDataValue)], name: &str) -> Result<Option<&'s f32>> {
    for (k, v) in attrs {
        if k == name {
            let AttributesDataValue::F32(v) = v else {
                return Err(anyhow!("attribute '{name}': wrong payload kind"));
            };
            return Ok(Some(v));
        }
    }
    Ok(None)
}

#[inline(always)]
fn attr_i64<'s>(attrs: &'s [(String, AttributesDataValue)], name: &str) -> Result<Option<&'s i64>> {
    for (k, v) in attrs {
        if k == name {
            let AttributesDataValue::I64(v) = v else {
                return Err(anyhow!("attribute '{name}': wrong payload kind"));
            };
            return Ok(Some(v));
        }
    }
    Ok(None)
}

#[inline(always)]
fn attr_str<'s>(attrs: &'s [(String, AttributesDataValue)], name: &str) -> Result<Option<&'s str>> {
    for (k, v) in attrs {
        if k == name {
            let AttributesDataValue::Str(s) = v else {
                return Err(anyhow!("attribute '{name}': wrong payload kind"));
            };
            return Ok(Some(s));
        }
    }
    Ok(None)
}

#[inline(always)]
fn attr_ints<'s>(
    attrs: &'s [(String, AttributesDataValue)],
    name: &str,
) -> Result<Option<&'s [i64]>> {
    for (k, v) in attrs {
        if k == name {
            let AttributesDataValue::Ints(v) = v else {
                return Err(anyhow!("attribute '{name}': wrong payload kind"));
            };
            return Ok(Some(v));
        }
    }
    Ok(None)
}

fn read_two_int_attr(
    attrs: &[(String, AttributesDataValue)],
    name: &str,
    default: [i64; 2],
) -> Result<[usize; 2]> {
    let Some(v) = attr_ints(attrs, name)? else {
        return Ok([default[0] as usize, default[1] as usize]);
    };
    if v.len() != 2 {
        return Err(anyhow!("attribute '{name}': wants 2 ints, got {}", v.len()));
    }
    if v.iter().any(|&x| x < 0) {
        return Err(anyhow!("attribute '{name}': negative value"));
    }
    Ok([v[0] as usize, v[1] as usize])
}

fn read_four_int_attr(
    attrs: &[(String, AttributesDataValue)],
    name: &str,
    default: [i64; 4],
) -> Result<[usize; 4]> {
    let Some(v) = attr_ints(attrs, name)? else {
        return Ok([
            default[0] as usize,
            default[1] as usize,
            default[2] as usize,
            default[3] as usize,
        ]);
    };
    if v.len() != 4 {
        return Err(anyhow!("attribute '{name}': wants 4 ints, got {}", v.len()));
    }
    if v.iter().any(|&x| x < 0) {
        return Err(anyhow!("attribute '{name}': negative value"));
    }
    Ok([v[0] as usize, v[1] as usize, v[2] as usize, v[3] as usize])
}

// cmp::min(max, cmp::max(min, x))
// if min is greater than max, then fill input with max
fn clip(out_name: &str, inputs: &[Option<&Tensor>]) -> Result<Tensor> {
    let x = slot(inputs, 0, "Clip")?;
    let take = |slot: Option<&Tensor>, bound: &str| -> Result<Option<f32>> {
        let Some(t) = slot else {
            return Ok(None);
        };
        if t.data.len() != 1 {
            return Err(anyhow!(
                "Clip: {bound} must be a scalar (empty tensor), got {} elements",
                t.data.len()
            ));
        }
        Ok(Some(t.data[0]))
    };
    let min = take(inputs.get(1).copied().flatten(), "min (input 1)")?;
    let max = take(inputs.get(2).copied().flatten(), "max (input 2)")?;
    if let (Some(lo), Some(hi)) = (min, max) {
        if lo > hi {
            return Ok(Tensor::create(
                out_name.to_string(),
                x.dims.to_vec(),
                vec![hi; x.len()],
            ));
        }
    }

    let min = min.unwrap_or(f32::MIN);
    let max = max.unwrap_or(f32::MAX);
    let mut out_data = vec![0f32; x.len()];
    let (input_chunk, input_remainder) = x.data.as_chunks::<8>();
    let (output_chunk, output_remainder) = out_data.as_chunks_mut::<8>();

    unsafe {
        let vmin = _mm256_set1_ps(min);
        let vmax = _mm256_set1_ps(max);

        for (in_chunk, out_chunk) in input_chunk.iter().zip(output_chunk.iter_mut()) {
            // TODO; handle NAN/Inf behavior according to ONNX spec
            let v = _mm256_load_ps(in_chunk.as_ptr());
            let v = _mm256_min_ps(v, vmax);
            let v = _mm256_max_ps(v, vmin);
            _mm256_store_ps(out_chunk.as_mut_ptr(), v);
        }

        for (src, dst) in input_remainder.iter().zip(output_remainder.iter_mut()) {
            *dst = src.max(min).min(max);
        }
    }

    Ok(Tensor::create(
        out_name.to_string(),
        x.dims.to_vec(),
        out_data,
    ))
}

fn prelu(out_name: &str, inputs: &[Option<&Tensor>]) -> Result<Tensor> {
    let x = slot(inputs, 0, "PRelu")?;
    let slope = slot(inputs, 1, "PRelu")?;
    let x_dims = &x.dims;
    let s_dims = &slope.dims;
    // if slope has more dimensions than x, it cannot be broadcasted to x
    if s_dims.len() > x_dims.len() {
        return Err(anyhow!(
            "PRelu: slope dims {:?} is less than X dims {:?} and cannot be broadcasted",
            &slope.dims,
            &x.dims
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
                &slope.dims,
                &i,
                &x.dims
            ));
        }
    }
    let rank = x_dims.len();
    let mut data = vec![0f32; x.data.len()];
    if x.data.is_empty() {
        return Ok(Tensor::create(out_name.to_string(), x_dims.to_vec(), data));
    }

    // calculate the number of elements in the last contiguous block
    // of x that is covered by a single slope value
    let mut block: usize = 1;
    for k in (0..rank).rev() {
        if k < offset || s_dims[k - offset] == 1 {
            block *= x_dims[k] as usize;
        } else {
            break;
        }
    }

    let s_strides: Vec<usize> = if s_dims.is_empty() {
        vec![]
    } else {
        slope.strides().iter().map(|&s| s as usize).collect()
    };

    if block >= 8 {
        // simd path
        let x_strides: Vec<usize> = x.strides().iter().map(|&s| s as usize).collect();
        let mut base = 0;
        while base < data.len() {
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
            unsafe {
                let input = &x.data[base..end];
                let output = &mut data[base..end];
                let slope = slope.data[s_idx];

                let vs = _mm256_set1_ps(slope);
                let zero = _mm256_setzero_ps();

                let (input_chunk, input_remainder) = input.as_chunks::<8>();
                let (output_chunk, output_remainder) = output.as_chunks_mut::<8>();

                for (in_chunk, out_chunk) in input_chunk.iter().zip(output_chunk.iter_mut()) {
                    // TODO: Handle NaN/Inf behavior according to ONNX spec
                    let v = _mm256_loadu_ps(in_chunk.as_ptr());
                    let mask = _mm256_cmp_ps::<{ _CMP_LT_OQ }>(v, zero);
                    let v_neg = _mm256_mul_ps(v, vs);
                    let v_out = _mm256_blendv_ps(v, v_neg, mask);
                    _mm256_storeu_ps(out_chunk.as_mut_ptr(), v_out);
                }
                // leftover tail (fewer than 8 floats)
                for (a, b) in input_remainder.iter().zip(output_remainder.iter_mut()) {
                    *b = if *a < 0.0 { *a * slope } else { *a };
                }
            }
            base = end; // move to the next block
        }
    } else {
        // scalar path
        let mut coords = vec![0usize; rank];
        for (&v, out) in x.data.iter().zip(data.iter_mut()) {
            if v < 0.0 {
                let mut s_idx = 0usize;
                for j in 0..s_dims.len() {
                    if s_dims[j] != 1 {
                        s_idx += coords[offset + j] * s_strides[j];
                    }
                }
                *out = v * slope.data[s_idx];
            } else {
                *out = v;
            }
            for k in (0..rank).rev() {
                coords[k] += 1;
                if (coords[k] as i64) < x_dims[k] {
                    break;
                }
                coords[k] = 0;
            }
        }
    }

    Ok(Tensor::create(out_name.to_string(), x_dims.to_vec(), data))
}

fn depth_to_space(
    out_name: &str,
    inputs: &[Option<&Tensor>],
    attrs: &[(String, AttributesDataValue)],
) -> Result<Tensor> {
    let x = slot(inputs, 0, "DepthToSpace")?;
    let mode = attr_str(attrs, "mode")?;

    let dims = &x.dims;
    if dims.len() != 4 {
        return Err(anyhow!(
            "DepthToSpace: wants rank-4 NCHW, got rank {}",
            dims.len()
        ));
    }

    let blocksize = if let Some(bs) = attr_i64(attrs, "blocksize")? {
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
    if cin % (blocksize * blocksize) != 0 {
        return Err(anyhow!(
            "DepthToSpace: channels {cin} are not divisible by blocksize^2({})",
            blocksize * blocksize
        ));
    }
    let co = cin / (blocksize * blocksize); // output channels
    let (oh, ow) = (h * blocksize, w * blocksize); //  output height and width

    let mut data = vec![0f32; x.data.len()];
    for n in 0..ni {
        for c in 0..co {
            for i in 0..blocksize {
                for j in 0..blocksize {
                    // which input slice feeds output offset (i, j)
                    // inside each blocksize*blocksize block
                    let ci = if is_default {
                        (i * blocksize + j) * co + c
                    } else {
                        c * blocksize * blocksize + i * blocksize + j
                    };
                    for hi in 0..h {
                        let src = &x.data[((n * cin + ci) * h + hi) * w..][..w];
                        let dst_row = ((n * co + c) * oh + hi * blocksize + i) * ow;
                        for wi in 0..w {
                            data[dst_row + wi * blocksize + j] = src[wi];
                        }
                    }
                }
            }
        }
    }

    Ok(Tensor::create(
        out_name.to_string(),
        vec![ni as i64, co as i64, oh as i64, ow as i64],
        data,
    ))
}
