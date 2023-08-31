use crate::graphs::p3_calculate_coordinates::{VDir, HDir};

pub(crate) mod layers;
pub(crate) mod traits;
pub(crate) mod lookup_maps;

// todo: Refactor this into trait
pub(crate) fn iterate(dir: IterDir, length: usize) -> impl Iterator<Item = usize> {
    let (mut start, step) = match dir {
        IterDir::Forward => (usize::MAX, 1), // up corresponds to left to right
        IterDir::Backward => (length, usize::MAX),
    };
    std::iter::repeat_with(move || {
            start = start.wrapping_add(step);
            start
        }).take(length)
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) enum IterDir {
    Forward,
    Backward,
}

impl From<VDir> for IterDir {
    fn from(val: VDir) -> Self {
        match val {
            VDir::Up => Self::Backward,
            VDir::Down => Self::Forward,
        }
    }
}

impl From<HDir> for IterDir {
    fn from(val: HDir) -> Self {
        match val {
            HDir::Left => Self::Backward,
            HDir::Right => Self::Forward,
        }
    }
}

// TODO: implement this with binary and see if it is faster

pub(crate) fn radix_sort(mut input: Vec<usize>, key_length: usize) -> Vec<usize> {
    let mut output = vec![0; input.len()];

    let mut key = 1;
    for _ in 0..key_length {
        counting_sort(&mut input, key, &mut output);
        key *= 10;
    }
    input
}

#[inline(always)]
fn counting_sort(input: &mut [usize], key: usize, output: &mut [usize]) {
    let mut count = [0; 10]; 
    // insert initial counts
    for i in input.iter().map(|n| self::key(key, *n)) {
        count[i] += 1;
    }

    // built accumulative sum
    for i in 1..10 {
        count[i] += count[i - 1];
    }

    for i in (0..input.len()).rev() {
        let k = self::key(key, input[i]);
        count[k] = count[k] - 1;
        output[count[k]] = input[i];
    }
    for i in 0..input.len() {
        input[i] = output[i];
    }
}

#[inline(always)]
fn key(key: usize, n: usize) -> usize {
    (n / key) % 10
}

#[test]
fn test_counting_sort_first_digit() {
    let mut input = [10, 0, 1, 5, 4, 22, 12];
    let mut output = [0; 7];
    counting_sort(&mut input, 1, &mut output);
    assert_eq!(output, [10, 0, 1, 22, 12, 4, 5])
}

#[test]
fn test_counting_sort_second_digit() {
    let mut input = [10, 0, 1, 5, 4, 22, 12];
    let mut output = [0; 7];
    counting_sort(&mut input, 10, &mut output);
    assert_eq!(output, [0, 1, 5, 4, 10, 12, 22]);
}

#[test]
fn test_radix_sort() {
    let input = [10, 0, 1, 5, 4, 22, 12];
    let output = radix_sort(input.to_vec(), 2);
    assert_eq!(output, [0, 1, 4, 5, 10, 12, 22]);
}
