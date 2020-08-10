use std::collections::{BTreeMap, HashMap, HashSet};

use crate::class::TCResult;
use crate::error;

use super::{TensorMath, TensorReduce, TensorTransform, TensorView};

const VALID_LABELS: [char; 52] = [
    'a', 'A', 'b', 'B', 'c', 'C', 'd', 'D', 'e', 'E', 'f', 'F', 'g', 'G', 'h', 'H', 'i', 'I', 'j',
    'J', 'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S',
    't', 'T', 'u', 'U', 'v', 'V', 'w', 'W', 'x', 'X', 'y', 'Y', 'z', 'Z',
];

fn parse_format(format: &str) -> TCResult<(Vec<Vec<char>>, Vec<char>)> {
    if !format.contains("->") {
        return Err(error::bad_request("Invalid format for einsum", format));
    }

    let mut parts: Vec<&str> = format.split("->").collect();
    if parts.is_empty() || parts.len() > 2 {
        return Err(error::bad_request("Invalid format for einsum", format));
    }

    let f_inputs: Vec<Vec<char>> = parts
        .pop()
        .unwrap()
        .split(',')
        .map(|f_input| f_input.chars().collect())
        .collect();
    let f_output: Vec<char> = parts.pop().unwrap_or("").chars().collect();

    let valid_labels: HashSet<char> = VALID_LABELS.iter().cloned().collect();
    for f_input in &f_inputs {
        let labels: HashSet<char> = f_input.iter().cloned().collect();
        if labels.len() != f_input.len() {
            return Err(error::bad_request(
                "Duplicate label in einsum format",
                f_input.iter().cloned().collect::<String>(),
            ));
        }

        let invalid_labels: Vec<char> = labels.difference(&valid_labels).cloned().collect();
        if !invalid_labels.is_empty() {
            return Err(error::bad_request(
                "Invalid labels in einsum format",
                invalid_labels.into_iter().collect::<String>(),
            ));
        }
    }

    Ok((f_inputs, f_output))
}

fn validate_args<T: TensorView>(
    f_inputs: &[Vec<char>],
    tensors: &[T],
) -> TCResult<BTreeMap<char, u64>> {
    if f_inputs.len() != tensors.len() {
        return Err(error::bad_request(
            "Number of tensors passed to einsum does not match number of format strings",
            format!("{} != {}", tensors.len(), f_inputs.len()),
        ));
    } else if tensors.is_empty() {
        return Err(error::bad_request("No Tensor was provided to einsum", "[]"));
    }

    let mut dimensions = BTreeMap::new();

    for (f_input, tensor) in f_inputs.iter().zip(tensors.iter()) {
        if f_input.len() != tensor.ndim() {
            return Err(error::bad_request(
                "Wrong tensor shape found for format string",
                f_input.iter().collect::<String>(),
            ));
        }

        for (label, dim) in f_input.iter().zip(tensor.shape().to_vec().iter()) {
            if let Some(known_dim) = dimensions.get(label) {
                if *dim != *known_dim {
                    return Err(error::bad_request(
                        "einsum found inconsistent dimension for axis",
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

fn normalize<T: Clone + TensorTransform>(
    tensor: &T,
    f_input: &[char],
    f_output: &[char],
    dimensions: &BTreeMap<char, u64>,
) -> TCResult<T> {
    if f_input == f_output {
        return Ok(tensor.clone());
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

    Ok(tensor)
}

fn outer_product<T: Clone + TensorMath + TensorTransform>(
    f_inputs: &[Vec<char>],
    dimensions: &BTreeMap<char, u64>,
    tensors: Vec<T>,
) -> TCResult<T> {
    assert!(f_inputs.len() == tensors.len());
    assert!(!tensors.is_empty());

    let f_output: Vec<char> = dimensions.keys().cloned().collect();

    let mut normalized = tensors
        .iter()
        .zip(f_inputs.iter())
        .map(|(tensor, f_input)| normalize(tensor, f_input, &f_output, &dimensions))
        .collect::<TCResult<Vec<T>>>()?;

    let mut op = normalized.pop().unwrap();
    while let Some(tensor) = normalized.pop() {
        op = op.multiply(&tensor)?;
    }

    Ok(op)
}

fn contract<T: TensorReduce + TensorTransform>(
    mut op: T,
    dimensions: BTreeMap<char, u64>,
    f_output: Vec<char>,
) -> TCResult<T> {
    let mut f_input: Vec<char> = dimensions.keys().cloned().collect();
    let mut axis = 0;
    while op.ndim() > f_output.len() {
        assert!(f_input.len() == op.ndim());

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
        let source: HashMap<char, usize> = f_input.iter().cloned().zip(0..f_input.len()).collect();
        let permutation: Vec<usize> = f_output.iter().map(|l| *source.get(l).unwrap()).collect();
        op.transpose(Some(permutation))
    }
}

pub fn einsum<T: Clone + TensorMath + TensorTransform + TensorReduce>(
    format: &str,
    tensors: Vec<T>,
) -> TCResult<T> {
    let (f_inputs, f_output) = parse_format(format)?;
    let dimensions = validate_args(&f_inputs, &tensors)?;

    let op = outer_product(&f_inputs, &dimensions, tensors)?;
    assert!(op.shape().to_vec() == dimensions.values().cloned().collect::<Vec<u64>>());

    contract(op, dimensions, f_output)
}
