use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::ops::Deref;

use log::debug;

use tc_error::*;
use tc_transact::fs::Dir;

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

    let f_output = parts.pop_back().unwrap_or("").chars().collect::<Label>();

    let valid_labels: HashSet<char> = VALID_LABELS.iter().cloned().collect();
    for f_input in &f_inputs {
        let labels: HashSet<char> = f_input.iter().cloned().collect();
        if labels.len() != f_input.len() {
            return Err(TCError::bad_request(
                "duplicate label in einsum format",
                f_input.iter().cloned().collect::<String>(),
            ));
        }

        let invalid_labels = labels.difference(&valid_labels).cloned().collect::<Label>();
        if !invalid_labels.is_empty() {
            return Err(TCError::bad_request(
                "invalid labels in einsum format",
                invalid_labels.into_iter().collect::<String>(),
            ));
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
                    return Err(TCError::bad_request(
                        "einsum got inconsistent dimension for axis",
                        label,
                    ));
                }
            } else {
                dimensions.insert(*label, *dim);
            }
        }
    }

    Ok(dimensions)
}

fn normalize<
    T: TensorAccess + TensorTransform<Broadcast = T, Expand = T, Transpose = T> + Clone,
>(
    tensor: T,
    f_input: &[char],
    f_output: &[char],
    dimensions: &BTreeMap<char, u64>,
) -> TCResult<T> {
    debug!(
        "normalize tensor with shape {} from {:?} -> {:?}",
        tensor.shape(),
        f_input,
        f_output
    );
    if f_input == f_output {
        return Ok(tensor);
    }

    let source: HashMap<char, usize> = f_input.iter().cloned().zip(0..f_input.len()).collect();
    let permutation: Vec<usize> = f_output
        .iter()
        .filter_map(|l| source.get(l))
        .cloned()
        .collect();

    let mut labels = Vec::with_capacity(f_output.len());
    for axis in &permutation {
        labels.push(f_input[*axis]);
    }

    let mut tensor = tensor.transpose(Some(permutation))?;

    let mut i = 0;
    while i < dimensions.len() {
        if i == labels.len() || labels[i] != f_output[i] {
            tensor = tensor.expand_dims(i)?;
            labels.insert(i, f_output[i]);
        } else {
            i += 1;
        }
    }

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

fn outer_product<D, T>(
    f_inputs: &[Label],
    dimensions: &BTreeMap<char, u64>,
    tensors: Vec<T>,
) -> TCResult<T>
where
    D: Dir,
    T: TensorAccess
        + TensorMath<D, T, LeftCombine = T>
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
        .collect::<TCResult<Vec<T>>>()?;

    let mut op = normalized.pop().unwrap();
    while let Some(tensor) = normalized.pop() {
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
            op = op.sum(axis)?;
            f_input.remove(axis);
        } else {
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
