# Chunked Bit Set
Defines traits for treating an indexed storage of bit chunks as a set of bit-indexes (i.e. similar to a `HashSet<usize>`). However, the focus is on (a) compact storage of bits, and (b) efficient iteration over elements in order.

```rust
let mut bitset = BitSet::default();
bitset.insert_bit(0);
bitset.insert_bit(1);
bitset.insert_bit(2);
bitset.insert_bit(3);
assert!(bitset.read_bit(1)   == true );
assert!(bitset.read_bit(99)  == false);
assert!(bitset.remove_bit(1) == true );
assert!(bitset.read_bit(1)   == false);
for _ in bitset.iter_set_indices() {} // [0, 2, 3]
```
These traits revolve around the random access of indexed bits, packed into machine-word sized `Chunk`s. Methods for reading and writing bits and chunks are separated into two traits, and implementations are provided for standard library slice and vector types as one would expect. 

```rust
let chunks: &'static [Chunk] = &[Chunk(0b0000)];
assert!(chunks.read_bit(0) == false); // &[Chunk] implements ChunkRead only

let chunks: Vec<Chunk> = vec![(Chunk(0b111))]; // {0, 1, 2}
let mut chunks: BitSet = chunks; // BitSet is just a type alias for Vec<Chunk>!
assert!(chunks.remove_bit(2) == true); // BitSet implements ChunkWrite too
```
Traverse or query combinations or readble sets with implementors of `ChunkCombinator`. The canonical use cases of union, intersection, difference and symmetric difference are provided. Iteration of these composite sets cache chunks internally for efficiency.

```rust
let a = [0b011].into_chunks(); // {0, 1}
let b = [0b110].into_chunks(); // {1, 2}
let aub = combinators::Union::combine(a, b); // {0, 1} U {1, 2} = {0, 1, 2}
assert_eq!(3, bitset.count_set_bits());
``` 
