//! Kernel implementations for ONNX operators.
use crate::ops::{
    AttributesDataValue, Operator, take_attr_i64, take_attr_ints, take_attr_str,
    take_four_attr_int, take_two_attr_int,
};
use crate::tensor::{Tensor, take_tensor};
use anyhow::{Result, anyhow};
use blas_rs::*;
use std::arch::x86_64::{
    _CMP_LT_OQ, _mm256_blendv_ps, _mm256_cmp_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_min_ps,
    _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
};

impl Operator {
    /// Compute the output tensor for this operator, given the input tensors and attributes
    #[inline(always)]
    pub fn compute(
        self,
        output_name: &str,
        inputs: &[Option<&Tensor>],
        attrs: &[(String, AttributesDataValue)],
    ) -> Result<Tensor> {
        match self {
            Operator::Clip => clip(output_name, inputs),
            Operator::Relu => relu(output_name, inputs),
            Operator::PRelu => prelu(output_name, inputs),
            Operator::DepthToSpace => depth_to_space(output_name, inputs, attrs),
            Operator::Conv => conv(output_name, inputs, attrs),
            Operator::Add => add(output_name, inputs),
            _ => Err(anyhow!(
                "operator '{:?}' has no kernel implemented yet",
                self
            )),
        }
    }
}

fn clip(name: &str, inputs: &[Option<&Tensor>]) -> Result<Tensor> {
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
        return Tensor::create(name, &x.dims, &vec![hi; x.len()]);
    }

    // basically do this cmp::min(max, cmp::max(min, x))
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
            let v = _mm256_loadu_ps(in_chunk.as_ptr());
            let v = _mm256_min_ps(v, vmax);
            let v = _mm256_max_ps(v, vmin);
            _mm256_storeu_ps(out_chunk.as_mut_ptr(), v);
        }

        for (src, dst) in input_remainder.iter().zip(output_remainder.iter_mut()) {
            *dst = src.max(min).min(max);
        }
    }

    Tensor::create(name, &x.dims, &out_data)
}

fn relu(name: &str, inputs: &[Option<&Tensor>]) -> Result<Tensor> {
    let x = take_tensor(inputs, 0, "Relu")?;

    let mut data = vec![0f32; x.data.len()];
    if x.data.is_empty() {
        return Tensor::create(name, &x.dims, &data);
    }

    if data.len() >= 8 {
        // simd path
        unsafe {
            let zero = _mm256_setzero_ps();

            let (input_chunks, input_remainder) = x.data.as_chunks::<8>();
            let (output_chunks, output_remainder) = data.as_chunks_mut::<8>();

            for (input, output) in input_chunks.iter().zip(output_chunks.iter_mut()) {
                // TODO: Handle NaN/Inf behavior according to ONNX spec
                let v = _mm256_loadu_ps(input.as_ptr());
                let result = _mm256_max_ps(v, zero);
                _mm256_storeu_ps(output.as_mut_ptr(), result);
            }

            // leftover tail (fewer than 8 floats)
            for (input, output) in input_remainder.iter().zip(output_remainder.iter_mut()) {
                *output = input.max(0.0);
            }
        }
    } else {
        // scalar path
        for (&v, out) in x.data.iter().zip(data.iter_mut()) {
            *out = v.max(0.0);
        }
    }

    Tensor::create(name, &x.dims, &data)
}

fn prelu(name: &str, inputs: &[Option<&Tensor>]) -> Result<Tensor> {
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
    let rank = x_dims.len();
    let mut data = vec![0f32; x.data.len()];
    if x.data.is_empty() {
        return Tensor::create(name, &x.dims, &data);
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

    let s_strides: Vec<usize> = slope.strides().iter().map(|&s| s as usize).collect();

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

    Tensor::create(name, &x.dims, &data)
}

fn depth_to_space(
    name: &str,
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

    let blocksize = if let Some(bs) = take_attr_i64(attrs, "blocksize")? {
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

    Tensor::create(name, &[ni as i64, co as i64, oh as i64, ow as i64], &data)
}

// TODO: I dont understand convolution properly, will come back to this later
// TODO: According to ONNX spec, Conv has some more optionally stuffs, which are ignored for now
fn conv(
    name: &str,
    inputs: &[Option<&Tensor>],
    attrs: &[(String, AttributesDataValue)],
) -> Result<Tensor> {
    // take two required inputs: X and W, and optional bias
    let x = take_tensor(inputs, 0, "Conv")?;
    let w = take_tensor(inputs, 1, "Conv")?;

    // handle attributes: auto_pad, group, strides, dilations, pads, kernel_shape

    let valid_auto = ["NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"];
    let auto_pad = match take_attr_str(attrs, "auto_pad")? {
        Some(ap) if !valid_auto.contains(&ap) => {
            return Err(anyhow!(
                "Conv: auto_pad must be one of {valid_auto:?}, got '{ap}'"
            ));
        }
        _ => "NOTSET",
    };

    if auto_pad != "NOTSET" && take_attr_ints(attrs, "pads")?.is_some() {
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

    let [sh, sw] = take_two_attr_int(attrs, "strides", [1, 1])?;
    let [dh, dw] = take_two_attr_int(attrs, "dilations", [1, 1])?;
    if sh == 0 || sw == 0 || dh == 0 || dw == 0 {
        return Err(anyhow!("Conv: strides/dilations must be positive"));
    }

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
    if let Some(ks) = take_attr_ints(attrs, "kernel_shape")?
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

    let [pt, pl, pb, pr]: [usize; 4] = if auto_pad == "NOTSET" {
        take_four_attr_int(attrs, "pads", [0, 0, 0, 0])?
    } else {
        // kernel_shape (spatial) from weights
        let k_h = kh;
        let k_w = kw;
        // effective kernel sizes
        let eff_kh = (k_h - 1) * dh + 1;
        let eff_kw = (k_w - 1) * dw + 1;
        // output sizes per SAME/VALID rule
        let out_h = hi.div_ceil(sh); // ceil(hi / sh)
        let out_w = wi.div_ceil(sw); // ceil(wi / sw)
        if auto_pad == "VALID" {
            [0usize, 0usize, 0usize, 0usize]
        } else {
            // SAME_UPPER or SAME_LOWER
            let total_pad_h =
                ((out_h as i64 - 1) * sh as i64 + eff_kh as i64 - hi as i64).max(0) as usize;
            let total_pad_w =
                ((out_w as i64 - 1) * sw as i64 + eff_kw as i64 - wi as i64).max(0) as usize;
            if auto_pad == "SAME_UPPER" {
                // extra padding at the end (bottom/right)
                let pt = total_pad_h / 2;
                let pb = total_pad_h - pt;
                let pl = total_pad_w / 2;
                let pr = total_pad_w - pl;
                [pt, pl, pb, pr]
            } else {
                // SAME_LOWER: extra at beginning (top/left)
                let pb = total_pad_h / 2;
                let pt = total_pad_h - pb;
                let pr = total_pad_w / 2;
                let pl = total_pad_w - pr;
                [pt, pl, pb, pr]
            }
        }
    };

    // compute output dims from explicit/computed pads
    let oh_ext = hi as i64 + pt as i64 + pb as i64 - ((kh as i64 - 1) * dh as i64 + 1);
    let ow_ext = wi as i64 + pl as i64 + pr as i64 - ((kw as i64 - 1) * dw as i64 + 1);
    if oh_ext < 0 || ow_ext < 0 {
        return Err(anyhow!(
            "Conv: kernel [{kh}×{kw}] does not fit into [{hi}×{wi}] with pads [{pt},{pl},{pb},{pr}]"
        ));
    }
    let (oh, ow) = (oh_ext / sh as i64 + 1, ow_ext / sw as i64 + 1);
    let (oh, ow) = (oh as usize, ow as usize);
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
    let col_len = k_g
        .checked_mul(p)
        .ok_or_else(|| anyhow!("Conv: im2col buffer {k_g}*{p} overflows"))?;
    let ycm_len = m_g
        .checked_mul(p)
        .ok_or_else(|| anyhow!("Conv: gemm buffer {m_g}*{p} overflows"))?;

    let mut out = vec![0f32; out_len];

    // We reuse a single im2col buffer for all groups (max K per group = k_g <= k_total)
    let mut col = vec![0f32; col_len]; // [K_g, P] column-major
    let mut ycm = vec![0f32; ycm_len]; // [M_g, P] column-major

    for ni in 0..n {
        let xbase_n = ni * cin * hi * wi;
        for g in 0..group {
            // weight slice for this group: W[g*m_g .. (g+1)*m_g, :] -> [M_g, K_g]
            let w_start = g * m_g * k_g;
            let w_end = w_start + m_g * k_g;
            let wg = &w.data[w_start..w_end];

            // input channel range for this group
            let c_start = g * c_g;
            // c_end = c_start + c_g

            // im2col for this group
            let mut pcol = 0usize;
            for ohi in 0..oh {
                for owi in 0..ow {
                    let coloff = pcol * k_g;
                    for ci in 0..c_g {
                        let rowoff = coloff + ci * kh * kw;
                        let xchan = xbase_n + (c_start + ci) * hi * wi;
                        for khi in 0..kh {
                            let ih = ohi * sh + khi * dh;
                            let ih = ih as i64 - pt as i64;
                            if !(0..hi as i64).contains(&ih) {
                                continue;
                            }
                            for kwi in 0..kw {
                                let iw = owi * sw + kwi * dw;
                                let iw = iw as i64 - pl as i64;
                                if !(0..wi as i64).contains(&iw) {
                                    continue;
                                }
                                col[rowoff + khi * kw + kwi] =
                                    x.data[xchan + ih as usize * wi + iw as usize];
                            }
                        }
                    }
                    pcol += 1;
                }
            }

            // gemm: Wg^T[M_g, K_g] · col[K_g, P] = Yg[M_g, P]
            // Wg is row-major [M_g, K_g] -> treat as col-major [K_g, M_g] with trans_a
            lvl3::gemm(
                m_g,
                p,
                k_g,
                1.0,
                wg,
                k_g, // lda = K_g for trans_a
                col.as_slice(),
                k_g,
                0.0,
                ycm.as_mut_slice(),
                m_g,
                true,
                false,
            );

            // scatter + bias
            let bias_g = bias.map(|b| &b[g * m_g..(g + 1) * m_g]);
            let outbase_g = ni * m * p + g * m_g * p;
            for mi in 0..m_g {
                let add = bias_g.map_or(0.0, |b| b[mi]);
                for j in 0..p {
                    let (ohi, owi) = (j / ow, j % ow);
                    out[outbase_g + (mi * oh + ohi) * ow + owi] = ycm[j * m_g + mi] + add;
                }
            }
        }
    }

    Tensor::create(name, &[n as i64, m as i64, oh as i64, ow as i64], &out)
}

fn add(name: &str, inputs: &[Option<&Tensor>]) -> Result<Tensor> {
    let x = take_tensor(inputs, 0, "Add")?;
    let y = take_tensor(inputs, 1, "Add")?;

    let x_dims = &x.dims;
    let y_dims = &y.dims;

    // compute strides
    let x_strides: Vec<usize> = x.strides().iter().map(|&s| s as usize).collect();
    let y_strides: Vec<usize> = y.strides().iter().map(|&s| s as usize).collect();

    // compute output dimensions with broadcasting
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
                "Add: cannot broadcast X dims {:?} with Y dims {:?}",
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
        .ok_or_else(|| anyhow!("Add: output dims {out_dims:?} overflow element count"))?;
    let mut data = vec![0f32; total];

    let mut coords = vec![0usize; rank];
    let (mut xi, mut yi) = (0usize, 0usize);
    for dst in &mut data {
        *dst = x.data[xi] + y.data[yi];

        for k in (0..rank).rev() {
            coords[k] += 1;
            if coords[k] < out_dims[k] as usize {
                xi += bx_strides[k];
                yi += by_strides[k];
                break;
            }
            coords[k] = 0;
            xi -= (out_dims[k] as usize - 1) * bx_strides[k];
            yi -= (out_dims[k] as usize - 1) * by_strides[k];
        }
    }

    Tensor::create(name, &out_dims, &data)
}
