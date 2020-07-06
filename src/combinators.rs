use crate::{Chunk, ChunkCombinator, ChunkRead};

/// Set union (elements in A or B), applied per chunk of bits
#[derive(Debug)]
pub enum Union {}

/// Symmetric set difference (elements in A xor B), applied per chunk of bits
#[derive(Debug)]
pub enum SymmetricDifference {}

/// Set difference (elements of A not in B) applied per chunk of bits
#[derive(Debug)]
pub enum Difference {}

/// Symmetric set intersection (elements in A and B), applied per chunk of bits
#[derive(Debug)]
pub enum Intersection {}

impl ChunkCombinator for Union {
    fn combine_chunks_len<A: ChunkRead, B: ChunkRead>(a: &A, b: &B) -> usize {
        a.chunks_len().max(b.chunks_len())
    }
    fn combine_chunk(a: Chunk, b: Chunk) -> Chunk {
        Chunk(a.0 | b.0)
    }
}
impl ChunkCombinator for SymmetricDifference {
    fn combine_chunks_len<A: ChunkRead, B: ChunkRead>(a: &A, b: &B) -> usize {
        a.chunks_len().max(b.chunks_len())
    }
    fn combine_chunk(a: Chunk, b: Chunk) -> Chunk {
        Chunk(a.0 ^ b.0)
    }
}
impl ChunkCombinator for Difference {
    fn combine_chunks_len<A: ChunkRead, B: ChunkRead>(a: &A, _b: &B) -> usize {
        a.chunks_len()
    }
    fn combine_chunk(a: Chunk, b: Chunk) -> Chunk {
        Chunk(a.0 & !b.0)
    }
}
impl ChunkCombinator for Intersection {
    fn combine_chunks_len<A: ChunkRead, B: ChunkRead>(a: &A, b: &B) -> usize {
        a.chunks_len().min(b.chunks_len())
    }
    fn combine_chunk(a: Chunk, b: Chunk) -> Chunk {
        Chunk(a.0 & b.0)
    }
}
