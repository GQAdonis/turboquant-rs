//! Compact bit-packing for 2-, 3-, and 4-bit unsigned indices.
//!
//! | bits | values per byte | packed bytes for 128 values |
//! |------|-----------------|------------------------------|
//! |  2   |        4        |             32               |
//! |  3   |     8 per 3 B   |             48               |
//! |  4   |        2        |             64               |
//!
//! All routines are branchless hot-path friendly and operate on contiguous
//! byte slices.

/// Number of packed bytes needed for `count` values at `bits` bits each.
#[inline]
pub const fn packed_byte_size(count: usize, bits: u8) -> usize {
    (count * bits as usize + 7) / 8
}

/// Pack `indices` (each in 0..2^bits) into a compact byte buffer.
pub fn pack(indices: &[u8], bits: u8) -> crate::error::Result<Vec<u8>> {
    match bits {
        2 => Ok(pack2(indices)),
        3 => Ok(pack3(indices)),
        4 => Ok(pack4(indices)),
        _ => Err(crate::error::TurboQuantError::UnsupportedBitWidth { bits }),
    }
}

/// Unpack `count` indices from `data` at `bits` bits each.
pub fn unpack(data: &[u8], count: usize, bits: u8) -> crate::error::Result<Vec<u8>> {
    match bits {
        2 => Ok(unpack2(data, count)),
        3 => Ok(unpack3(data, count)),
        4 => Ok(unpack4(data, count)),
        _ => Err(crate::error::TurboQuantError::UnsupportedBitWidth { bits }),
    }
}

// ── 2-bit: 4 values per byte ───────────────────────────────────────────────

fn pack2(idx: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; (idx.len() + 3) / 4];
    for (i, &v) in idx.iter().enumerate() {
        out[i / 4] |= (v & 0x3) << ((i % 4) * 2);
    }
    out
}

fn unpack2(data: &[u8], count: usize) -> Vec<u8> {
    (0..count)
        .map(|i| (data[i / 4] >> ((i % 4) * 2)) & 0x3)
        .collect()
}

// ── 3-bit: 8 values per 3 bytes (24 bits) ─────────────────────────────────

fn pack3(idx: &[u8]) -> Vec<u8> {
    let groups = (idx.len() + 7) / 8;
    let mut out = vec![0u8; groups * 3];
    for (g, chunk) in idx.chunks(8).enumerate() {
        let mut word: u32 = 0;
        for (i, &v) in chunk.iter().enumerate() {
            word |= (v as u32 & 0x7) << (i * 3);
        }
        let base = g * 3;
        out[base]     =  word        as u8;
        out[base + 1] = (word >>  8) as u8;
        out[base + 2] = (word >> 16) as u8;
    }
    out
}

fn unpack3(data: &[u8], count: usize) -> Vec<u8> {
    (0..count)
        .map(|i| {
            let g    = i / 8;
            let pos  = i % 8;
            let base = g * 3;
            let word = (data[base] as u32)
                | ((data[base + 1] as u32) << 8)
                | ((data[base + 2] as u32) << 16);
            ((word >> (pos * 3)) & 0x7) as u8
        })
        .collect()
}

// ── 4-bit: 2 values per byte ───────────────────────────────────────────────

fn pack4(idx: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; (idx.len() + 1) / 2];
    for (i, &v) in idx.iter().enumerate() {
        out[i / 2] |= (v & 0xF) << ((i % 2) * 4);
    }
    out
}

fn unpack4(data: &[u8], count: usize) -> Vec<u8> {
    (0..count)
        .map(|i| (data[i / 2] >> ((i % 2) * 4)) & 0xF)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(bits: u8, count: usize) {
        let max  = (1u8 << bits) - 1;
        let idx: Vec<u8> = (0..count).map(|i| (i as u8) % (max + 1)).collect();
        let packed   = pack(&idx, bits).unwrap();
        let unpacked = unpack(&packed, count, bits).unwrap();
        assert_eq!(idx, unpacked, "{bits}-bit roundtrip failed for count={count}");
    }

    #[test] fn rt_2bit_128() { roundtrip(2, 128); }
    #[test] fn rt_3bit_128() { roundtrip(3, 128); }
    #[test] fn rt_4bit_128() { roundtrip(4, 128); }
    #[test] fn rt_3bit_odd()  { roundtrip(3, 10);  }
    #[test] fn rt_2bit_1()    { roundtrip(2, 1);   }

    #[test]
    fn packed_sizes() {
        assert_eq!(packed_byte_size(128, 2), 32);
        assert_eq!(packed_byte_size(128, 3), 48);
        assert_eq!(packed_byte_size(128, 4), 64);
    }

    #[test]
    fn invalid_bit_width_returns_error() {
        assert!(pack(&[0, 1, 2], 5).is_err());
        assert!(unpack(&[0u8; 4], 4, 5).is_err());
    }
}
