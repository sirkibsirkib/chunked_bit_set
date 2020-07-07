use core::{
    fmt::{Binary, Debug, Formatter},
    marker::PhantomData,
    ops::Range,
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
pub trait ChunkOwn: ChunkWrite + Default {
    fn from_chunk_iter<I: IntoIterator<Item = Chunk>>(into_iter: I) -> Self;
    fn from_bit_iter<I: IntoIterator<Item = bool>>(into_iter: I) -> Self;
    fn from_set_bits_iter<I: IntoIterator<Item = usize>>(into_iter: I) -> Self {
        let mut me = Self::default();
        me.extend_from_set_bits_iter(into_iter);
        me
    }
    fn extend_from_set_bits_iter<I: IntoIterator<Item = usize>>(&mut self, into_iter: I) {
        for bit_index in into_iter {
            self.insert_bit(bit_index);
        }
    }
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

pub trait ChunkRead: Sized + Clone {
    fn empty_chunks_start(&self) -> usize; // promises that all chunks in range empty_chunks_start().. will be Chunk::EMPTY
    fn read_chunk(&self, chunk_index: usize) -> Chunk;

    fn set_cmp(&self) -> SetCmp<&Self> {
        SetCmp(self)
    }

    fn read_bit(&self, bit_index: usize) -> bool {
        let BitAddress { chunk_index, index_in_chunk } = BitAddress::from_bit_index(bit_index);
        let chunk = self.read_chunk(chunk_index);
        chunk.read_bit(index_in_chunk)
    }
    fn iter_set_bits(&self) -> BitIndexIter<&Self> {
        let mut chunk_iter = self.chunk_iter();
        let cached_chunk = chunk_iter.next().unwrap_or(Chunk::EMPTY);
        BitIndexIter { chunk_iter, cached_chunk }
    }
    fn count_set_bits(&self) -> usize {
        self.chunk_iter().map(|chunk| chunk.count_set_bits() as usize).sum()
    }
    fn combine_with<'a, 'b, C: ChunkCombinator, B: ChunkRead>(
        &'a self,
        other: &'b B,
    ) -> ChunkReadCombined<&'a Self, &'b B, C> {
        ChunkReadCombined::new(self, other)
    }
    fn to_owned<T: ChunkOwn>(&self) -> T {
        T::from_chunk_iter(self.chunk_iter())
    }
    fn is_empty(&self) -> bool {
        self.chunk_iter().all(Chunk::is_empty)
    }
    fn chunk_iter(&self) -> ChunkIter<&Self> {
        ChunkIter { chunk_index_range: 0..self.empty_chunks_start(), r: self }
    }
}

/// Characterizes owned, writable storage for indexed bits.
pub trait ChunkWrite: Sized + ChunkRead {
    fn insert_bit(&mut self, bit_index: usize) -> bool;
    fn remove_bit(&mut self, bit_index: usize) -> bool;
    fn clear_chunks(&mut self);
    fn pop_set_bit(&mut self) -> Option<usize> {
        let bit_index = self.iter_set_bits().next()?;
        self.remove_bit(bit_index);
        Some(bit_index)
    }
}

/// Defines functions for combinng chunks of bits.
/// Default implementation assumes that functions are pure,
/// and that the index of a bit is independent of its treatment.
pub trait ChunkCombinator {
    fn combine_empty_chunks_start<A: ChunkRead, B: ChunkRead>(a: &A, b: &B) -> usize;
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
/// A ChunkRead type whose chunks are defined by a combinator, and two inner ChunkRead types.
/// Used to query and traverse bit sets defined in terms of other bitsets, E.g., union.
pub struct ChunkReadCombined<A: ChunkRead, B: ChunkRead, C: ChunkCombinator> {
    c: PhantomData<C>,
    a: A,
    b: B,
}
pub struct BitAddress {
    chunk_index: usize,
    index_in_chunk: usize, // invariant: index_in_chunk < Chunk::BITS
}
#[derive(Clone, Debug)]
pub struct ChunkIter<R> {
    r: R,
    chunk_index_range: Range<usize>,
}
/// Iterates over some ChunkRead type, emitting indexes of set bits in ascending order.
#[derive(Clone)]
pub struct BitIndexIter<R: ChunkRead> {
    chunk_iter: ChunkIter<R>,
    cached_chunk: Chunk,
}
#[derive(Debug, Copy, Clone)]
pub struct SetCmp<A: ChunkRead>(pub A);
////////////////////////////////////////////////////////////////////////////////////////////////////
impl<A: ChunkRead, B: ChunkRead> PartialEq<SetCmp<B>> for SetCmp<A> {
    fn eq(&self, other: &SetCmp<B>) -> bool {
        combinators::SymmetricDifference::combine(&self.0, &other.0).is_empty()
    }
}
impl<A: ChunkRead, B: ChunkRead> PartialOrd<SetCmp<B>> for SetCmp<A> {
    fn partial_cmp(&self, other: &SetCmp<B>) -> Option<core::cmp::Ordering> {
        use core::cmp::Ordering;
        let a_subset = combinators::Difference::combine(&self.0, &other.0).is_empty();
        let b_subset = combinators::Difference::combine(&other.0, &self.0).is_empty();
        match [a_subset, b_subset] {
            [false, false] => None,
            [true, false] => Some(Ordering::Less),
            [true, true] => Some(Ordering::Equal),
            [false, true] => Some(Ordering::Greater),
        }
    }
}
impl<A: ChunkRead> Eq for SetCmp<A> {}
impl<A: ChunkRead, B: ChunkRead, C: ChunkCombinator> Clone for ChunkReadCombined<A, B, C> {
    fn clone(&self) -> Self {
        Self { a: self.a.clone(), b: self.b.clone(), c: Default::default() }
    }
}
impl<A: ChunkRead + Copy, B: ChunkRead + Copy, C: ChunkCombinator> Copy
    for ChunkReadCombined<A, B, C>
{
}
impl<R: ChunkRead> Iterator for ChunkIter<R> {
    type Item = Chunk;
    fn next(&mut self) -> Option<Self::Item> {
        if self.chunk_index_range.start == self.chunk_index_range.end {
            None
        } else {
            self.chunk_index_range.start += 1;
            Some(self.r.read_chunk(self.chunk_index_range.start - 1))
        }
    }
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
    pub fn read_bit(self, bit_index: usize) -> bool {
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
    pub fn trailing_zeroes(mut self) -> usize {
        // TODO replace with intrinsic when it stabilizes
        let mut index_in_chunk = 0;
        let mut mask_ones = Chunk::BITS / 2;
        while mask_ones > 0 {
            const ALL_ONES: usize = !0; // 11111...1111 chunk
            let mask = ALL_ONES >> (Chunk::BITS - mask_ones);
            if self.0 & mask == 0 {
                index_in_chunk += mask_ones;
                self.0 >>= mask_ones;
            }
            mask_ones /= 2;
        }
        if self.0 & 1 > 0 {
            index_in_chunk
        } else {
            Chunk::BITS
        }
    }
}
impl ChunkWrite for Vec<Chunk> {
    fn insert_bit(&mut self, bit_index: usize) -> bool {
        let BitAddress { chunk_index, index_in_chunk } = BitAddress::from_bit_index(bit_index);
        self.resize((chunk_index + 1).max(self.as_slice().empty_chunks_start()), Chunk::EMPTY);
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
    fn clear_chunks(&mut self) {
        self.clear();
    }
}
impl<'a> ChunkRead for &'a [Chunk] {
    fn empty_chunks_start(&self) -> usize {
        self.len()
    }
    fn read_chunk(&self, chunk_index: usize) -> Chunk {
        self.get(chunk_index).copied().unwrap_or(Chunk::EMPTY)
    }
}
impl ChunkRead for Vec<Chunk> {
    fn empty_chunks_start(&self) -> usize {
        self.as_slice().empty_chunks_start()
    }
    fn read_chunk(&self, chunk_index: usize) -> Chunk {
        self.as_slice().read_chunk(chunk_index)
    }
}
impl<A: ChunkRead, B: ChunkRead, C: ChunkCombinator> ChunkReadCombined<A, B, C> {
    pub fn new(a: A, b: B) -> Self {
        ChunkReadCombined { a, b, c: Default::default() }
    }
}
impl<T: ChunkRead> ChunkRead for &T {
    fn read_chunk(&self, chunk_index: usize) -> Chunk {
        <T as ChunkRead>::read_chunk(self, chunk_index)
    }
    fn empty_chunks_start(&self) -> usize {
        <T as ChunkRead>::empty_chunks_start(self)
    }
}
impl<A: ChunkRead, B: ChunkRead, C: ChunkCombinator> ChunkRead for ChunkReadCombined<A, B, C> {
    fn read_chunk(&self, chunk_index: usize) -> Chunk {
        C::combine_chunk(self.a.read_chunk(chunk_index), self.b.read_chunk(chunk_index))
    }
    fn empty_chunks_start(&self) -> usize {
        C::combine_empty_chunks_start(&self.a, &self.b)
    }
}
impl BitAddress {
    #[inline(always)]
    pub fn into_bit_index(self) -> usize {
        self.chunk_index * Chunk::BITS + self.index_in_chunk
    }
    #[inline(always)]
    pub fn from_bit_index(bit_index: usize) -> Self {
        Self { chunk_index: bit_index / Chunk::BITS, index_in_chunk: bit_index % Chunk::BITS }
    }
}
impl<T: ChunkRead> Iterator for BitIndexIter<T> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        while self.cached_chunk.is_empty() {
            self.cached_chunk = self.chunk_iter.next()?;
        }
        debug_assert!(!self.cached_chunk.is_empty());
        let index_in_chunk = self.cached_chunk.trailing_zeroes();
        debug_assert!(index_in_chunk < Chunk::BITS);
        self.cached_chunk.remove_bit(index_in_chunk);
        let ba =
            BitAddress { index_in_chunk, chunk_index: self.chunk_iter.chunk_index_range.start - 1 };
        Some(ba.into_bit_index())
    }
}
impl<T: ChunkRead> Debug for BitIndexIter<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(<Self as Clone>::clone(&self)).finish()
    }
}
impl ChunkOwn for Vec<Chunk> {
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
        f.debug_list().entries(self.chunk_iter()).finish()
    }
}

#[cfg(smallvec_impl)]
mod smallvec_impl {
    use crate::*;
    use smallvec::{Array, SmallVec};
    impl<T: Array<Item = Chunk>> ChunkRead for SmallVec<T> {
        fn read_chunk(&self, chunk_index: usize) -> Chunk {
            self.get(chunk_index).copied().unwrap_or(Chunk::EMPTY)
        }
        fn empty_chunks_start(&self) -> usize {
            self.len()
        }
    }
    impl<T: Array<Item = Chunk>> ChunkWrite for SmallVec<T> {
        fn insert_bit(&mut self, bit_index: usize) -> bool {
            let BitAddress { chunk_index, index_in_chunk } = BitAddress::from_bit_index(bit_index);
            self.resize((chunk_index + 1).max(self.as_slice().empty_chunks_start()), Chunk::EMPTY);
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
        fn clear_chunks(&mut self) {
            self.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{distributions::Distribution as _, rngs::SmallRng, Rng, SeedableRng as _};
    use std::{collections::HashSet, iter};

    fn num_stream<R: Rng>(rng: &mut R) -> impl Iterator<Item = usize> + '_ {
        let dist = rand::distributions::Uniform::new(0usize, 1_000);
        iter::repeat_with(move || dist.sample(rng))
    }

    #[test]
    pub fn index_iter() {
        let mut rng = SmallRng::from_seed([0; 16]);
        let mut indices: Vec<usize> = num_stream(&mut rng).take(250).collect();
        let bs = BitSet::from_set_bits_iter(indices.iter().copied());
        let indices2: Vec<usize> = bs.iter_set_bits().collect();
        indices.sort();
        indices.dedup();
        assert_eq!(indices, indices2);
    }

    #[test]
    pub fn cmp_union() {
        let mut rng = SmallRng::from_seed([0; 16]);
        let a: HashSet<usize> = num_stream(&mut rng).take(200).collect();
        let b: HashSet<usize> = num_stream(&mut rng).take(200).collect();
        let bs_a = BitSet::from_set_bits_iter(a.iter().copied());
        let bs_b = BitSet::from_set_bits_iter(b.iter().copied());
        assert_eq!(
            a.iter().chain(b.iter()).copied().collect::<HashSet<_>>(),
            combinators::Union::combine(&bs_a, &bs_b).iter_set_bits().collect()
        );
    }

    #[test]
    pub fn traits() {
        let usizes: Vec<usize> = vec![0b1100000, 0b1];
        let chunks: Vec<Chunk> = usizes.into_chunks();
        let bs: BitSet = chunks; // BitSet is just an alias for Vec<Chunk>
        assert!(bs.read_bit(5));

        let usize_slice: &'static [usize] = &[0b1001];
        let chunk_slice: &'static [Chunk] = usize_slice.into_chunks();
        assert!(chunk_slice.read_bit(3));
    }

    #[test]
    pub fn more_traits() {
        let a = [0, 1, 2].into_chunks();
        let b = [2, 3, 4].into_chunks();
        let _aub = combinators::Union::combine(a, b);
    }

    #[test]
    pub fn eq_test() {
        let mut rng = SmallRng::from_seed([0; 16]);
        let a = BitSet::from_set_bits_iter(num_stream(&mut rng).take(200));
        assert_eq!(SetCmp(&a), SetCmp(&a));

        let mut b = a.clone();
        assert_eq!(SetCmp(&a), SetCmp(&b));

        b.pop_set_bit();
        assert_ne!(SetCmp(&a), SetCmp(&b));
    }

    #[test]
    pub fn cmp_test() {
        let a = BitSet::from_set_bits_iter(0..5);
        let b = BitSet::from_set_bits_iter(0..4);
        assert!(SetCmp(&a) > SetCmp(&b));
        assert!(!(SetCmp(&a) < SetCmp(&b)));
        assert!(SetCmp(&a) != SetCmp(&b));

        let c = BitSet::from_set_bits_iter(40..41);
        assert!(!(SetCmp(&a) < SetCmp(&c)));
        assert!(!(SetCmp(&a) > SetCmp(&c)));
        assert!(!(SetCmp(&a) == SetCmp(&c)));
        assert!(!(SetCmp(&a) <= SetCmp(&c)));
        assert!(!(SetCmp(&a) >= SetCmp(&c)));
        assert!(a.set_cmp().partial_cmp(&c.set_cmp()).is_none());
    }
}
