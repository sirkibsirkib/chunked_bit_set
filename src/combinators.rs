use crate::{BitCombinator, BitRead, Chunk};

#[derive(Debug)]
pub enum Union {}
#[derive(Debug)]
pub enum SymmetricDifference {}
#[derive(Debug)]
pub enum Difference {}
#[derive(Debug)]
pub enum Intersection {}

impl BitCombinator for Union {
    fn combine_chunks_len<A: BitRead, B: BitRead>(a: &A, b: &B) -> usize {
        a.chunks_len().max(b.chunks_len())
    }
    fn combine_chunk(a: Chunk, b: Chunk) -> Chunk {
        Chunk(a.0 | b.0)
    }
}
impl BitCombinator for SymmetricDifference {
    fn combine_chunks_len<A: BitRead, B: BitRead>(a: &A, b: &B) -> usize {
        a.chunks_len().max(b.chunks_len())
    }
    fn combine_chunk(a: Chunk, b: Chunk) -> Chunk {
        Chunk(a.0 ^ b.0)
    }
}
impl BitCombinator for Difference {
    fn combine_chunks_len<A: BitRead, B: BitRead>(a: &A, _b: &B) -> usize {
        a.chunks_len()
    }
    fn combine_chunk(a: Chunk, b: Chunk) -> Chunk {
        Chunk(a.0 & !b.0)
    }
}
impl BitCombinator for Intersection {
    fn combine_chunks_len<A: BitRead, B: BitRead>(a: &A, b: &B) -> usize {
        a.chunks_len().min(b.chunks_len())
    }
    fn combine_chunk(a: Chunk, b: Chunk) -> Chunk {
        Chunk(a.0 & b.0)
    }
}
