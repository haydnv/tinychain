use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::ops::Deref;

use log::debug;

use tc_error::*;
use tc_transact::fs::Dir;
use tc_transact::Transaction;
use tcgeneric::Tuple;

use super::{TensorAccess, TensorMath, TensorReduce, TensorTransform};

type Label = Vec<char>;

const VALID_LABELS: [char; 52] = [
    'a', 'A', 'b', 'B', 'c', 'C', 'd', 'D', 'e', 'E', 'f', 'F', 'g', 'G', 'h', 'H', 'i', 'I', 'j',
    'J', 'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S',
    't', 'T', 'u', 'U', 'v', 'V', 'w', 'W', 'x', 'X', 'y', 'Y', 'z', 'Z',
];

fn parse_format(format: &str) -> TCResult<(Vec<Label>, Label)> {
    if !format.contains("->") {
        return Err(TCError::bad_request(
            "invalid format for einsum (missing '->')",
            format,
        ));
    }

    let mut parts: VecDeque<&str> = format.split("->").collect();
    if parts.is_empty() || parts.len() > 2 {
        return Err(TCError::bad_request("invalid format for einsum", format));
    }

    let f_inputs = parts
        .pop_front()
        .unwrap()
        .split(',')
        .map(|f_input| f_input.chars().collect())
        .collect::<Vec<Label>>();

    let f_output = parts.pop_back().unwrap_or("");
    if f_output.chars().collect::<HashSet<_>>().len() != f_output.len() {
        return Err(TCError::bad_request(
            "einsum output cannot include repeated subscripts",
            f_output,
        ));
    }

    let f_output = f_output.chars().collect::<Label>();

    let mut present_labels = HashSet::<char>::new();
    let valid_labels: HashSet<char> = VALID_LABELS.iter().cloned().collect();
    for f_input in &f_inputs {
        let mut invalid_labels = f_input
            .iter()
            .filter(|l| !valid_labels.contains(l))
            .peekable();

        if invalid_labels.peek().is_some() {
            return Err(TCError::bad_request(
                "invalid labels in einsum format",
                invalid_labels.collect::<Tuple<&char>>(),
            ));
        }

        present_labels.extend(f_input);
    }

    let mut invalid_labels = f_output
        .iter()
        .filter(|l| !valid_labels.contains(l))
        .peekable();

    if invalid_labels.peek().is_some() {
        return Err(TCError::bad_request(
            "invalid labels in einsum format",
            invalid_labels.collect::<Tuple<&char>>(),
        ));
    }

    for l in &f_output {
        if !present_labels.contains(l) {
            return Err(TCError::bad_request("no such input dimension", l));
        }
    }

    Ok((f_inputs, f_output))
}

fn validate_args<T: TensorAccess>(
    f_inputs: &[Label],
    tensors: &[T],
) -> TCResult<BTreeMap<char, u64>> {
    if f_inputs.len() != tensors.len() {
        return Err(TCError::bad_request(
            "number of Tensors passed to einsum does not match number of format strings",
            format!("{} != {}", tensors.len(), f_inputs.len()),
        ));
    } else if tensors.is_empty() {
        return Err(TCError::bad_request(
            "no Tensor was provided to einsum",
            "[]",
        ));
    }

    let mut dimensions = BTreeMap::new();

    for (f_input, tensor) in f_inputs.iter().zip(tensors.iter()) {
        if f_input.len() != tensor.ndim() {
            return Err(TCError::unsupported(format!(
                "tensor with {} dimensions does not match format string {}",
                tensor.ndim(),
                f_input.iter().cloned().collect::<String>()
            )));
        }

        for (label, dim) in f_input.iter().zip(tensor.shape().to_vec().iter()) {
            if let Some(known_dim) = dimensions.get(label) {
                if *dim != *known_dim {
                    return Err(TCError::unsupported(format!(
                        "einsum got inconsistent dimension for axis {}: {} vs {}",
                        label, dim, known_dim
                    )));
                }
            } else {
                dimensions.insert(*label, *dim);
            }
        }
    }

    Ok(dimensions)
}

fn normalize<D, Txn, T>(
    tensor: T,
    f_input: &[char],
    f_output: &[char],
    dimensions: &BTreeMap<char, u64>,
) -> TCResult<T>
where
    D: Dir,
    Txn: Transaction<D>,
    T: TensorAccess
        + TensorReduce<D, Txn = Txn, Reduce = T>
        + TensorTransform<Broadcast = T, Expand = T, Transpose = T>
        + Clone,
{
    assert_eq!(tensor.ndim(), f_input.len());

    debug!(
        "normalize tensor with shape {} from {:?} -> {:?}",
        tensor.shape(),
        f_input,
        f_output
    );

    if f_input == f_output {
        return Ok(tensor);
    }

    let mut permutation: Vec<usize> = (0..f_input.len()).collect();
    permutation.sort_by(|x1, x2| {
        let l1 = &f_input[*x1];
        let l2 = &f_input[*x2];
        f_output
            .iter()
            .position(|l| l == l1)
            .cmp(&f_output.iter().position(|l| l == l2))
    });

    debug!("permutation of {:?} is {:?}", f_input, permutation);
    let mut tensor = tensor.transpose(Some(permutation.clone()))?;

    // sum over repeated subscripts
    // the assumption that they have been transposed together is valid because subscripts
    // in f_output are required to be unique
    debug!(
        "sum over repeated labels of {:?}",
        permutation.iter().map(|x| f_input[*x]).collect::<Vec<char>>()
    );

    let mut x = 0;
    let mut labels = Vec::with_capacity(f_output.len());
    for i in 0..(f_input.len() - 1) {
        let subscript = f_input[permutation[i]];
        if subscript == f_input[permutation[i + 1]] {
            tensor = tensor.sum(x)?;
        } else {
            labels.push(subscript);
            x += 1;
        }
    }

    labels.push(f_input[*permutation.last().unwrap()]);
    debug!("after summation: {:?}", labels);
    assert_eq!(tensor.ndim(), labels.len());

    let mut i = 0;
    while i < dimensions.len() {
        if i == labels.len() || labels[i] != f_output[i] {
            tensor = tensor.expand_dims(i)?;
            labels.insert(i, f_output[i]);
        } else {
            i += 1;
        }
    }

    debug!(
        "input tensor with shape {} has labels {:?}",
        tensor.shape(),
        labels
    );

    let shape = f_output
        .iter()
        .map(|l| dimensions.get(l).expect("tensor dimension"))
        .cloned()
        .collect::<Vec<u64>>();

    if tensor.shape().deref() == &shape {
        Ok(tensor)
    } else {
        debug!("broadcast {} into {:?}", tensor.shape(), shape);
        tensor.broadcast(shape.into())
    }
}

fn outer_product<D, Txn, T>(
    f_inputs: &[Label],
    dimensions: &BTreeMap<char, u64>,
    tensors: Vec<T>,
) -> TCResult<T>
where
    D: Dir,
    Txn: Transaction<D>,
    T: TensorAccess
        + TensorMath<D, T, LeftCombine = T>
        + TensorReduce<D, Txn = Txn, Reduce = T>
        + TensorTransform<Broadcast = T, Expand = T, Transpose = T>
        + Clone,
{
    assert_eq!(f_inputs.len(), tensors.len());
    assert!(!tensors.is_empty());

    let f_output = dimensions.keys().cloned().collect::<Label>();

    let mut normalized = tensors
        .into_iter()
        .zip(f_inputs.iter())
        .map(|(tensor, f_input)| normalize(tensor, f_input, &f_output, &dimensions))
        .collect::<TCResult<VecDeque<T>>>()?;

    let mut op = normalized.pop_front().unwrap();
    while let Some(tensor) = normalized.pop_front() {
        op = op.mul(tensor)?;
    }

    Ok(op)
}

fn contract<D, T>(mut op: T, dimensions: BTreeMap<char, u64>, f_output: Label) -> TCResult<T>
where
    D: Dir,
    T: TensorAccess + TensorReduce<D, Reduce = T> + TensorTransform<Transpose = T>,
{
    let mut f_input = dimensions.keys().cloned().collect::<Label>();
    let mut axis = 0;
    while op.ndim() > f_output.len() {
        assert_eq!(f_input.len(), op.ndim());

        if !f_output.contains(&f_input[axis]) {
            debug!("einsum will contract over axis {}", f_input[axis]);
            op = op.sum(axis)?;
            f_input.remove(axis);
        } else {
            assert!(f_input.contains(&f_output[axis]));
            axis += 1;
        }
    }

    if f_input == f_output {
        Ok(op)
    } else {
        debug!(
            "transpose outer product with shape {} from {:?} -> {:?}",
            op.shape(),
            f_input,
            f_output
        );

        let source: HashMap<char, usize> = f_input.iter().cloned().zip(0..f_input.len()).collect();
        let permutation: Vec<usize> = f_output.iter().map(|l| *source.get(l).unwrap()).collect();
        op.transpose(Some(permutation))
    }
}

pub fn einsum<D, T>(format: &str, tensors: Vec<T>) -> TCResult<T>
where
    D: Dir,
    T: TensorAccess
        + TensorMath<D, T, LeftCombine = T>
        + TensorTransform<Broadcast = T, Expand = T, Transpose = T>
        + TensorReduce<D, Reduce = T>
        + Clone,
{
    let (f_inputs, f_output) = parse_format(format)?;

    debug!(
        "einsum with input labels: {:?}, output label {:?}",
        f_inputs, f_output
    );

    let dimensions = validate_args(&f_inputs, &tensors)?;
    let op = outer_product(&f_inputs, &dimensions, tensors)?;

    debug_assert_eq!(
        op.shape().as_slice(),
        dimensions
            .values()
            .cloned()
            .collect::<Vec<u64>>()
            .as_slice()
    );

    contract(op, dimensions, f_output)
}
