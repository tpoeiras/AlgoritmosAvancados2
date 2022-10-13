use itertools::Itertools;
use rand::distributions::Uniform;
use rand::prelude::*;
use rand::rngs::StdRng;

use std::marker::PhantomData;
use std::ops::Add;
use std::time::Instant;

trait Operation: Sized {
    type Type: Copy;

    fn op(val1: &Self::Type, val2: &Self::Type) -> Self::Type;
}

#[derive(Debug)]
struct Min<T: Ord + Clone> {
    _phantom: PhantomData<T>,
}

impl<T: Ord + Copy> Operation for Min<T> {
    type Type = T;

    fn op(val1: &T, val2: &T) -> T {
        *val1.min(val2)
    }
}

impl<T: Ord + Copy> IdempotentOperation for Min<T> {}

#[derive(Debug)]
struct Sum<T: Add<T, Output = T>> {
    _phantom: PhantomData<T>,
}

impl<T: Add<T, Output = T> + Copy> Operation for Sum<T> {
    type Type = T;

    fn op(val1: &T, val2: &T) -> T {
        *val1 + *val2
    }
}

impl<T: Add<T, Output = T> + Copy> SparseTableOperation for Sum<T> {}

#[derive(Debug)]
struct SparseTable<O: Operation> {
    table: Vec<Vec<O::Type>>,
}

impl<'a, O: SparseTableOperation> SparseTable<O> {
    fn new(array: &'a [O::Type]) -> SparseTable<O> {
        let length = (array.len() as f64).log2().floor() as usize + 1;

        let mut table = vec![Vec::new(); length];
        for val in array {
            table[0].push(*val);
        }

        for i in 1..length {
            for j in 0..=(array.len() - (1 << i)) {
                let next_entry = O::op(&table[i - 1][j], &table[i - 1][j + (1 << (i - 1))]);
                table[i].push(next_entry);
            }
        }

        SparseTable { table }
    }

    fn query(&self, left: usize, right: usize) -> O::Type {
        O::sparse_table_query(self, left, right)
    }
}

trait SparseTableOperation: Operation {
    fn sparse_table_query(
        sparse_table: &SparseTable<Self>,
        mut left: usize,
        right: usize,
    ) -> Self::Type {
        let mut log = (0usize.leading_zeros() - (right - left + 1).leading_zeros() - 1) as usize;

        let mut acc = sparse_table.table[log][left];
        left += 1 << log;

        while left <= right {
            let block_size = 1 << log;
            if block_size <= right - left + 1 {
                acc = Self::op(&acc, &sparse_table.table[log][left]);
                left += block_size;
            }
            log -= 1;
        }

        acc
    }
}

trait IdempotentOperation: Operation {}

impl<O: IdempotentOperation> SparseTableOperation for O {
    fn sparse_table_query(
        sparse_table: &SparseTable<Self>,
        left: usize,
        right: usize,
    ) -> Self::Type {
        let log = (0usize.leading_zeros() - (right - left + 1).leading_zeros() - 1) as usize;

        Self::op(
            &sparse_table.table[log][left],
            &sparse_table.table[log][right - (1 << log) + 1],
        )
    }
}

#[derive(Debug)]
struct SqrtTree<'a, O: Operation> {
    sqrt: usize,
    array: &'a [O::Type],
    between: Vec<Vec<O::Type>>,
    prefix: Vec<Vec<O::Type>>,
    suffix: Vec<Vec<O::Type>>,
    block_trees: Vec<SqrtTree<'a, O>>,
}

impl<'a, O: Operation> SqrtTree<'a, O>
where
    <O as Operation>::Type: Default,
{
    fn new(array: &'a [O::Type]) -> SqrtTree<'a, O> {
        let sqrt = (array.len() as f64).sqrt().floor() as usize;
        let rem = array.len() % sqrt;
        let n_blocks = array.len() / sqrt + if rem == 0 { 0 } else { 1 };

        let mut prefix = vec![vec![O::Type::default(); sqrt]; n_blocks - 1];
        let mut suffix = vec![vec![O::Type::default(); sqrt]; n_blocks - 1];
        if rem == 0 {
            prefix.push(vec![O::Type::default(); sqrt]);
            suffix.push(vec![O::Type::default(); sqrt]);
        } else {
            prefix.push(vec![O::Type::default(); rem]);
            suffix.push(vec![O::Type::default(); rem]);
        }

        let mut i = 0;
        for block in 0..n_blocks {
            let mut acc = array[i];
            prefix[block][0] = acc;
            i += 1;

            for j in 1..prefix[block].len() {
                acc = O::op(&acc, &array[i]);
                prefix[block][j] = acc;
                i += 1;
            }
        }

        let mut i = 0;
        for block in 0..n_blocks {
            i += prefix[block].len() - 1;
            let mut acc = array[i];
            suffix[block][prefix[block].len() - 1] = acc;

            for j in (0..(prefix[block].len() - 1)).rev() {
                i -= 1;
                acc = O::op(&acc, &array[i]);
                suffix[block][j] = acc;
            }
            i += sqrt;
        }

        let mut between = Vec::new();
        for block_s in 0..n_blocks {
            between.push(Vec::new());

            let mut acc = suffix[block_s][0];
            between[block_s].push(acc);
            for block_f in (block_s + 1)..n_blocks {
                acc = O::op(&acc, &suffix[block_f][0]);
                between[block_s].push(acc);
            }
        }

        let mut block_trees = Vec::new();
        if sqrt > 2 {
            for chunk in array.chunks(sqrt) {
                let subtree = SqrtTree::<O>::new(chunk);
                block_trees.push(subtree);
            }
        }

        SqrtTree {
            sqrt,
            array,
            between,
            suffix,
            prefix,
            block_trees,
        }
    }

    fn query(&self, left: usize, right: usize) -> O::Type {
        let block_left = left / self.sqrt;
        let rem_left = left % self.sqrt;
        let block_right = right / self.sqrt;
        let rem_right = right % self.sqrt;

        if block_left != block_right {
            let mut acc = self.suffix[block_left][rem_left];

            if block_right > block_left + 1 {
                acc = O::op(
                    &acc,
                    &self.between[block_left + 1][block_right - block_left - 2],
                )
            }

            acc = O::op(&acc, &self.prefix[block_right][rem_right]);

            acc
        } else if self.sqrt <= 2 {
            let mut acc = self.array[left];

            if left < right {
                acc = O::op(&acc, &self.array[right]);
            }

            acc
        } else {
            self.block_trees[block_left].query(rem_left, rem_right)
        }
    }
}

#[derive(Debug)]
struct Rmq<'a, T: Copy + Ord, const B: usize = 30> {
    array: &'a [T],
    table: Option<SparseTable<Min<T>>>,
    mask: Vec<u32>,
}

impl<'a, const B: usize, T: Copy + Ord> Rmq<'a, T, B> {
    fn new(array: &'a [T]) -> Rmq<T, B> {
        let mut current_mask = 0u32;
        let mut mask = Vec::new();

        for i in 0..array.len() {
            current_mask = (current_mask << 1) & ((1 << B) - 1);

            while current_mask > 0 && array[i] <= array[i - current_mask.trailing_zeros() as usize]
            {
                current_mask ^= 1 << current_mask.trailing_zeros();
            }

            current_mask |= 1;
            mask.push(current_mask);
        }

        let mut rmq = Rmq {
            array,
            mask,
            table: None,
        };

        let small_array = (0..array.len())
            .step_by(B)
            .into_iter()
            .map(|i| rmq.small_query(i, i + B - 1))
            .collect::<Vec<_>>();
        rmq.table = Some(SparseTable::<Min<T>>::new(&small_array));
        rmq
    }

    fn small_query(&self, left: usize, right: usize) -> T {
        let dist = msb(self.mask[right] & ((1 << (right - left + 1)) - 1));
        self.array[right - dist as usize]
    }

    fn query(&self, left: usize, right: usize) -> T {
        if right - left + 1 <= B {
            return self.small_query(left, right);
        }

        let mut acc = Min::op(
            &self.small_query(left, left + B - 1),
            &self.small_query(right - B + 1, right),
        );

        let block_left = left / B + 1;
        let block_right = right / B - 1;

        if block_left <= block_right {
            acc = Min::op(
                &acc,
                &self.table.as_ref().unwrap().query(block_left, block_right),
            );
        }

        acc
    }
}

fn msb(val: u32) -> u32 {
    31 - val.leading_zeros()
}

fn main() {
    let mut rng = StdRng::seed_from_u64(131254153214);

    for _ in 0..15 {
        for size in (30000..300000)
            .step_by(30000)
            .into_iter()
            .chain((300000..10000000).step_by(300000))
        {
            let (build_time, query_time) = test_sparse_table::<false>(&mut rng, size, 100);
            //let (build_time, query_time) = test_sqrt_tree::<false>(&mut rng, size, 100);
            //let (build_time, query_time) = test_rmq::<false>(&mut rng, size, 100);
            println!("{size},{build_time},{query_time}");
        }
    }
}

fn test_sparse_table<const A: bool>(rng: &mut StdRng, size: usize, ops: usize) -> (u128, u128) {
    let dist = Uniform::from(1..size);
    let array = rng.sample_iter(dist).take(size).collect::<Vec<_>>();

    let start = Instant::now();
    let sparse_table = SparseTable::<Min<_>>::new(&array);
    let build_time = start.elapsed().as_nanos();

    let mut query_time = 0;
    for (mut l, mut r) in rng.sample_iter(dist).tuples().take(ops) {
        if r < l {
            (l, r) = (r, l);
        }

        let start = Instant::now();
        let sparse_table_result = sparse_table.query(l, r);
        query_time += start.elapsed().as_nanos();

        if A {
            let query_result = array[l..=r].iter().min().unwrap();
            assert_eq!(*query_result, sparse_table_result);
        }
    }

    (build_time, query_time)
}

fn test_sqrt_tree<const A: bool>(rng: &mut StdRng, size: usize, ops: usize) -> (u128, u128) {
    let dist = Uniform::from(1..size);
    let array = rng.sample_iter(dist).take(size).collect::<Vec<_>>();

    let start = Instant::now();
    let rmq = SqrtTree::<Min<_>>::new(&array);
    let build_time = start.elapsed().as_nanos();

    let mut query_time = 0;
    for (mut l, mut r) in rng.sample_iter(dist).tuples().take(ops) {
        if r < l {
            (l, r) = (r, l);
        }

        let start = Instant::now();
        let sparse_table_result = rmq.query(l, r);
        query_time += start.elapsed().as_nanos();

        if A {
            let query_result = array[l..=r].iter().min().unwrap();
            assert_eq!(*query_result, sparse_table_result);
        }
    }

    (build_time, query_time)
}

fn test_rmq<const A: bool>(rng: &mut StdRng, size: usize, ops: usize) -> (u128, u128) {
    let size = size - (size % 30);
    let dist = Uniform::from(1..size);
    let array = rng.sample_iter(dist).take(size).collect::<Vec<_>>();

    let start = Instant::now();
    let rmq = Rmq::<_>::new(&array);
    let build_time = start.elapsed().as_nanos();

    let mut query_time = 0;
    for (mut l, mut r) in rng.sample_iter(dist).tuples().take(ops) {
        if r < l {
            (l, r) = (r, l);
        }

        let start = Instant::now();
        let rmq_result = rmq.query(l, r);
        query_time += start.elapsed().as_nanos();

        if A {
            let query_result = array[l..=r].iter().min().unwrap();
            assert_eq!(*query_result, rmq_result);
        }
    }

    (build_time, query_time)
}
