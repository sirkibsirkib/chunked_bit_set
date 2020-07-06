use core::{
    fmt::{Binary, Debug, Formatter},
    marker::PhantomData,
};

pub mod combinators;

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Chunk(pub usize);

pub trait BitCollect: Sized {
    fn from_bit_indices<I: IntoIterator<Item = usize>>(iter: I) -> Self;
    fn from_bit_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self;
}

pub trait BitRead: Sized {
    fn chunks_len(&self) -> usize;
    fn get_chunk(&self, chunk_index: usize) -> Chunk;
    fn get_bit(&self, bit_index: usize) -> bool;
    fn iter_set_bits(&self) -> BitSetIter<Self> {
        BitSetIter {
            current_chunk_index: 0,
            chunks_len: self.chunks_len(),
            next_bit_index: 0,
            cached_chunk: self.get_chunk(0),
            bit_read: self,
        }
    }
    fn count_set_bits(&self) -> usize {
        let mut count = 0;
        for chunk_index in 0..self.chunks_len() {
            count += self.get_chunk(chunk_index).count_ones() as usize;
        }
        count
    }
    fn combine_with<'a, 'b, C: BitCombinator, B: BitRead>(
        &'a self,
        other: &'b B,
    ) -> BitReadCombined<&'a Self, &'b B, C> {
        BitReadCombined::new(self, other)
    }
    fn to_owned(&self) -> Vec<Chunk> {
        let mut vec = Vec::with_capacity(self.chunks_len());
        for chunk_index in 0..vec.capacity() {
            vec.push(self.get_chunk(chunk_index));
        }
        vec
    }
}
pub trait BitWrite: Sized {
    fn insert_bit(&mut self, bit_index: usize) -> bool;
    fn remove_bit(&mut self, bit_index: usize) -> bool;
    fn clear(&mut self);
}
pub trait BitCombinator {
    fn combine_chunks_len<A: BitRead, B: BitRead>(a: &A, b: &B) -> usize;
    fn combine_chunk(a: Chunk, b: Chunk) -> Chunk;
    fn combine_bit(a: bool, b: bool) -> bool {
        let a = if a { Chunk::FULL } else { Chunk::EMPTY };
        let b = if b { Chunk::FULL } else { Chunk::EMPTY };
        !Self::combine_chunk(a, b).is_empty()
    }
    fn combine<A: BitRead, B: BitRead>(a: A, b: B) -> BitReadCombined<A, B, Self>
    where
        Self: Sized,
    {
        BitReadCombined::new(a, b)
    }
}
pub struct BitSetIter<'a, T: BitRead> {
    current_chunk_index: usize,
    chunks_len: usize,
    next_bit_index: usize,
    cached_chunk: Chunk,
    bit_read: &'a T,
}
pub struct BitReadCombined<A: BitRead, B: BitRead, C: BitCombinator> {
    c: PhantomData<C>,
    a: A,
    b: B,
}
pub trait FromIntoChunks: Sized {
    type Chunks: Sized;
    fn into_chunks(self) -> Self::Chunks;
    fn from_chunks(chunks: Self::Chunks) -> Self;
}
struct BitAddress {
    chunk_index: usize,
    index_in_chunk: usize,
}
//////////////////////
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
    pub fn count_ones(self) -> u32 {
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
impl<'a> BitRead for &'a [Chunk] {
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
impl BitRead for Vec<Chunk> {
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
impl<A: BitRead, B: BitRead, C: BitCombinator> BitReadCombined<A, B, C> {
    pub fn new(a: A, b: B) -> Self {
        BitReadCombined { a, b, c: Default::default() }
    }
}
impl<T: BitRead> BitRead for &T {
    fn get_bit(&self, bit_index: usize) -> bool {
        <T as BitRead>::get_bit(self, bit_index)
    }
    fn get_chunk(&self, chunk_index: usize) -> Chunk {
        <T as BitRead>::get_chunk(self, chunk_index)
    }
    fn chunks_len(&self) -> usize {
        <T as BitRead>::chunks_len(self)
    }
}
impl<T: BitRead> Clone for BitSetIter<'_, T> {
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
impl<A: BitRead, B: BitRead, C: BitCombinator> BitRead for BitReadCombined<A, B, C> {
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
impl<T: BitRead> Iterator for BitSetIter<'_, T> {
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
impl<T: BitRead> Debug for BitSetIter<'_, T> {
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
    fn from_bit_iter<I: IntoIterator<Item = bool>>(into_iter: I) -> Self {
        let mut vec: Vec<Chunk> = Default::default();
        let mut chunk = Chunk::EMPTY;
        let mut bit_index = 0;
        for b in into_iter {
            if bit_index as usize == Chunk::BITS {
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
impl<A: BitRead, B: BitRead, C: BitCombinator> Debug for BitReadCombined<A, B, C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.iter_set_bits()).finish()
    }
}

impl<A: BitRead, B: BitRead, C: BitCombinator> Binary for BitReadCombined<A, B, C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let chunk_iter = (0..self.chunks_len()).map(|chunk_index| self.get_chunk(chunk_index));
        f.debug_list().entries(chunk_iter).finish()
    }
}

#[test]
fn zoop() {
    let c = &[0b100110, 0b111].into_chunks();
    println!("{:?}", &c);
    println!("{:?}", c.iter_set_bits());
    let mut v = vec![0b0110].into_chunks();
    v.insert_bit(66);
    println!("{:?}", &v);
    println!("{:?}", v.iter_set_bits());

    println!("{:?}", combinators::SymmetricDifference::combine(&c, &v));

    // let q: BitSet = std::iter::repeat(false).take(16).chain(std::iter::once(true)).collect();
    // println!("{:?}", q);
    // println!("{:?}", q.iter_set_bits());
}
