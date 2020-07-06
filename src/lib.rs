use core::{
    fmt::{Binary, Debug, Formatter},
    marker::PhantomData,
};

/// Provides a predefined set of the standard set-set operations as bit chunk combinators,
/// for use creating ChunkReadCombined structures
pub mod combinators;

/// A contiguous machine-word sized storage of bits
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Chunk(pub usize);

/// Characterizes any owned storage of bits. Allows creation from iterators that either:
/// 1. visit all set bits
/// 2. visit all chunks in index order [0*Chunk::BITS, 1*Chunk::BITS, 2*Chunk::BITS, ...]
/// 3. bisit all bits in index order [0, 1, 2, ...]
pub trait BitCollect: BitWrite + Default {
    fn from_chunk_iter<I: IntoIterator<Item = Chunk>>(iter: I) -> Self;
    fn from_bit_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self;
    fn from_bit_indices<I: IntoIterator<Item = usize>>(iter: I) -> Self;
}

/// Same functionality as std::convert::{Into, From}, but combined,
/// and given a new name such that it can be implemented for
/// upstream types such as Vec without clashing with From/Into.
pub trait FromIntoChunks: Sized {
    type Chunks: Sized;
    fn into_chunks(self) -> Self::Chunks;
    fn from_chunks(chunks: Self::Chunks) -> Self;
}

pub type BitSet = Vec<Chunk>;

/// Characterizes any readable storage of Chunks.
/// Allows for querying, visiting set indices, etc.
/// Allows random access to indexed chunks, associating every `usize` with some chunk's bit.
pub trait ChunkRead: Sized {
    fn chunks_len(&self) -> usize;
    fn get_chunk(&self, chunk_index: usize) -> Chunk;
    fn get_bit(&self, bit_index: usize) -> bool;
    fn iter_set_bits(&self) -> BitIndexIter<Self> {
        BitIndexIter::new(self)
    }
    fn count_set_bits(&self) -> usize {
        chunk_iter(self).map(|chunk| chunk.count_set_bits() as usize).sum()
    }
    fn combine_with<'a, 'b, C: ChunkCombinator, B: ChunkRead>(
        &'a self,
        other: &'b B,
    ) -> ChunkReadCombined<&'a Self, &'b B, C> {
        ChunkReadCombined::new(self, other)
    }
    fn to_owned(&self) -> Vec<Chunk> {
        let mut vec = Vec::with_capacity(self.chunks_len());
        for chunk_index in 0..vec.capacity() {
            vec.push(self.get_chunk(chunk_index));
        }
        vec
    }
}

/// Characterizes owned, writable storage for indexed bits.
pub trait BitWrite: Sized {
    fn insert_bit(&mut self, bit_index: usize) -> bool;
    fn remove_bit(&mut self, bit_index: usize) -> bool;
    fn clear(&mut self);
}

/// Defines functions for combinng chunks of bits.
/// Default implementation assumes that functions are pure,
/// and that the index of a bit is independent of its treatment.
pub trait ChunkCombinator {
    fn combine_chunks_len<A: ChunkRead, B: ChunkRead>(a: &A, b: &B) -> usize;
    fn combine_chunk(a: Chunk, b: Chunk) -> Chunk;
    fn combine_bit(a: bool, b: bool) -> bool {
        let a = if a { Chunk::FULL } else { Chunk::EMPTY };
        let b = if b { Chunk::FULL } else { Chunk::EMPTY };
        !Self::combine_chunk(a, b).is_empty()
    }
    fn combine<A: ChunkRead, B: ChunkRead>(a: A, b: B) -> ChunkReadCombined<A, B, Self>
    where
        Self: Sized,
    {
        ChunkReadCombined::new(a, b)
    }
}
/// Iterates over some ChunkRead type, emitting indexes of set bits in ascending order.
pub struct BitIndexIter<'a, T: ChunkRead> {
    current_chunk_index: usize,
    chunks_len: usize,
    next_bit_index: usize,
    cached_chunk: Chunk,
    bit_read: &'a T,
}
/// A ChunkRead type whose chunks are defined by a combinator, and two inner ChunkRead types.
/// Used to query and traverse bit sets defined in terms of other bitsets, E.g., union.
#[derive(Clone, Copy)]
pub struct ChunkReadCombined<A: ChunkRead, B: ChunkRead, C: ChunkCombinator> {
    c: PhantomData<C>,
    a: A,
    b: B,
}
struct BitAddress {
    chunk_index: usize,
    index_in_chunk: usize,
}
//////////////////////
impl<'a, T: ChunkRead> BitIndexIter<'a, T> {
    pub fn new(bit_read: &'a T) -> Self {
        Self {
            current_chunk_index: 0,
            chunks_len: bit_read.chunks_len(),
            next_bit_index: 0,
            cached_chunk: bit_read.get_chunk(0),
            bit_read,
        }
    }
}
pub fn chunk_iter(cr: impl ChunkRead) -> impl Iterator<Item = Chunk> {
    (0..cr.chunks_len()).map(move |chunk_index| cr.get_chunk(chunk_index))
}
impl FromIntoChunks for Vec<usize> {
    type Chunks = Vec<Chunk>;
    fn into_chunks(self) -> Self::Chunks {
        unsafe { std::mem::transmute(self) }
    }
    fn from_chunks(chunks: Self::Chunks) -> Self {
        unsafe { std::mem::transmute(chunks) }
    }
}
impl<'a> FromIntoChunks for &'a [usize] {
    type Chunks = &'a [Chunk];
    fn into_chunks(self) -> Self::Chunks {
        unsafe { std::mem::transmute(self) }
    }
    fn from_chunks(chunks: Self::Chunks) -> Self {
        unsafe { std::mem::transmute(chunks) }
    }
}
impl Chunk {
    pub const BITS: usize = std::mem::size_of::<Self>() * 8;
    pub const EMPTY: Self = Chunk(0);
    pub const FULL: Self = Chunk(!0);
    pub fn count_set_bits(self) -> u32 {
        self.0.count_ones()
    }
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
    pub fn get_bit(&self, bit_index: usize) -> bool {
        self.0 & (1 << bit_index) > 0
    }
    fn insert_bit(&mut self, bit_index: usize) -> bool {
        debug_assert!(bit_index < Self::BITS);
        let was = self.0;
        self.0 |= 1 << bit_index;
        was != self.0
    }
    pub fn remove_bit(&mut self, bit_index: usize) -> bool {
        let was = self.0;
        self.0 &= !(1 << bit_index);
        was != self.0
    }
    pub fn clear(&mut self) {
        *self = Self::EMPTY
    }
}
impl BitWrite for Vec<Chunk> {
    fn insert_bit(&mut self, bit_index: usize) -> bool {
        let BitAddress { chunk_index, index_in_chunk } = BitAddress::from_bit_index(bit_index);
        self.resize((chunk_index + 1).max(self.as_slice().chunks_len()), Chunk::EMPTY);
        let chunk = unsafe { self.get_unchecked_mut(chunk_index) };
        chunk.insert_bit(index_in_chunk)
    }
    fn remove_bit(&mut self, bit_index: usize) -> bool {
        let BitAddress { chunk_index, index_in_chunk } = BitAddress::from_bit_index(bit_index);
        if let Some(chunk) = self.get_mut(chunk_index) {
            let ret = chunk.remove_bit(index_in_chunk);
            'trunc_null_suffix: while let Some(suffix_chunk) = self.pop() {
                if !suffix_chunk.is_empty() {
                    // oops! put it back
                    self.push(suffix_chunk);
                    break 'trunc_null_suffix;
                }
            }
            ret
        } else {
            false
        }
    }
    fn clear(&mut self) {
        self.clear();
    }
}
impl<'a> ChunkRead for &'a [Chunk] {
    fn chunks_len(&self) -> usize {
        self.len()
    }
    fn get_chunk(&self, chunk_index: usize) -> Chunk {
        self.get(chunk_index).copied().unwrap_or(Chunk::EMPTY)
    }
    fn get_bit(&self, bit_index: usize) -> bool {
        let BitAddress { chunk_index, index_in_chunk } = BitAddress::from_bit_index(bit_index);
        self.get(chunk_index).copied().map(|chunk| chunk.get_bit(index_in_chunk)).unwrap_or(false)
    }
}
impl ChunkRead for Vec<Chunk> {
    fn chunks_len(&self) -> usize {
        self.as_slice().chunks_len()
    }
    fn get_chunk(&self, chunk_index: usize) -> Chunk {
        self.as_slice().get_chunk(chunk_index)
    }
    fn get_bit(&self, bit_index: usize) -> bool {
        self.as_slice().get_bit(bit_index)
    }
}
impl<A: ChunkRead, B: ChunkRead, C: ChunkCombinator> ChunkReadCombined<A, B, C> {
    pub fn new(a: A, b: B) -> Self {
        ChunkReadCombined { a, b, c: Default::default() }
    }
}
impl<T: ChunkRead> ChunkRead for &T {
    fn get_bit(&self, bit_index: usize) -> bool {
        <T as ChunkRead>::get_bit(self, bit_index)
    }
    fn get_chunk(&self, chunk_index: usize) -> Chunk {
        <T as ChunkRead>::get_chunk(self, chunk_index)
    }
    fn chunks_len(&self) -> usize {
        <T as ChunkRead>::chunks_len(self)
    }
}
impl<T: ChunkRead> Clone for BitIndexIter<'_, T> {
    fn clone(&self) -> Self {
        Self {
            chunks_len: self.chunks_len,
            next_bit_index: self.next_bit_index,
            cached_chunk: self.cached_chunk,
            current_chunk_index: self.current_chunk_index,
            bit_read: self.bit_read,
        }
    }
}
impl<A: ChunkRead, B: ChunkRead, C: ChunkCombinator> ChunkRead for ChunkReadCombined<A, B, C> {
    fn get_bit(&self, bit_index: usize) -> bool {
        C::combine_bit(self.a.get_bit(bit_index), self.b.get_bit(bit_index))
    }
    fn get_chunk(&self, chunk_index: usize) -> Chunk {
        C::combine_chunk(self.a.get_chunk(chunk_index), self.b.get_chunk(chunk_index))
    }
    fn chunks_len(&self) -> usize {
        C::combine_chunks_len(&self.a, &self.b)
    }
}
impl BitAddress {
    #[inline(always)]
    fn from_bit_index(bit_index: usize) -> Self {
        Self { chunk_index: bit_index / Chunk::BITS, index_in_chunk: (bit_index % Chunk::BITS) }
    }
}
impl<T: ChunkRead> Iterator for BitIndexIter<'_, T> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        while self.cached_chunk.is_empty() {
            // advance to the next chunk until the next chunk yields some element
            self.current_chunk_index += 1;
            if self.current_chunk_index >= self.chunks_len {
                return None;
            }
            self.cached_chunk = self.bit_read.get_chunk(self.current_chunk_index);
            self.next_bit_index = ((self.next_bit_index / Chunk::BITS) + 1) * Chunk::BITS;
        }
        // self.cached_chunk contains the next bit whose index we will return
        debug_assert!(!self.cached_chunk.is_empty());
        // shift the cached chunk until the rightmost 1 bit is in the last position,
        // incrementing next_bit_index appropriately
        let mut mask_ones = Chunk::BITS / 2;
        // with optimizations, this loop is indeed unrolled
        while mask_ones > 0 {
            const ALL_ONES: usize = !0; // 11111...1111 chunk
            let mask = ALL_ONES >> (Chunk::BITS - mask_ones);
            if self.cached_chunk.0 & mask == 0 {
                self.next_bit_index += mask_ones;
                self.cached_chunk.0 >>= mask_ones;
            }
            mask_ones /= 2;
        }
        debug_assert_eq!(self.cached_chunk.0 & 1, 1);
        // self.next_bit_index is now the value we return
        let ret = self.next_bit_index;
        // jump over this bit for next time
        self.cached_chunk.0 >>= 1;
        self.next_bit_index += 1;
        Some(ret)
    }
}
impl<T: ChunkRead> Debug for BitIndexIter<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(<Self as Clone>::clone(&self)).finish()
    }
}
impl BitCollect for Vec<Chunk> {
    fn from_bit_indices<I: IntoIterator<Item = usize>>(into_iter: I) -> Self {
        let mut vec = Self::default();
        for bit_index in into_iter {
            vec.insert_bit(bit_index);
        }
        vec
    }
    fn from_chunk_iter<I: IntoIterator<Item = Chunk>>(into_iter: I) -> Self {
        into_iter.into_iter().collect()
    }
    fn from_bit_iter<I: IntoIterator<Item = bool>>(into_iter: I) -> Self {
        let mut vec: Vec<Chunk> = Default::default();
        let mut chunk = Chunk::EMPTY;
        let mut bit_index = 0;
        for b in into_iter {
            if bit_index == Chunk::BITS {
                vec.push(chunk);
                bit_index = 0;
                chunk = Chunk::EMPTY;
            }
            if b {
                chunk.insert_bit(bit_index);
            }
            bit_index += 1;
        }
        if !chunk.is_empty() {
            vec.push(chunk)
        }
        vec
    }
}
impl Debug for Chunk {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:01$b}", self.0, Chunk::BITS)
    }
}
impl<A: ChunkRead, B: ChunkRead, C: ChunkCombinator> Debug for ChunkReadCombined<A, B, C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.iter_set_bits()).finish()
    }
}

impl<A: ChunkRead, B: ChunkRead, C: ChunkCombinator> Binary for ChunkReadCombined<A, B, C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(chunk_iter(self)).finish()
    }
}

#[test]
fn zoop() {
    // let c = &[0b100110, 0b111].into_chunks();
    // assert_eq!(c.count_set_bits(), 6);
    let bs = BitSet::from_bit_iter(
        std::iter::repeat(false)
            .take(5)
            .chain([true, false, false].iter().copied().cycle().take(21)),
    );
    println!("{:?}", bs);
}
