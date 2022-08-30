use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use log::debug;

use tc_error::*;
use tc_tensor::{TensorAccess, TensorMath, TensorReduce, TensorTransform};
use tc_transact::fs::DirLock;
use tc_transact::Transaction;
use tcgeneric::Tuple;

type Label = Vec<char>;

const DOT: char = '.';
const ELLIPSIS: [char; 3] = [DOT, DOT, DOT];
const VALID_SUBSCRIPTS: [char; 52] = [
    'a', 'A', 'b', 'B', 'c', 'C', 'd', 'D', 'e', 'E', 'f', 'F', 'g', 'G', 'h', 'H', 'i', 'I', 'j',
    'J', 'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S',
    't', 'T', 'u', 'U', 'v', 'V', 'w', 'W', 'x', 'X', 'y', 'Y', 'z', 'Z',
];

struct Dimensions {
    shape: HashMap<char, u64>,
    order: Vec<char>,
}

impl Dimensions {
    fn with_capacity(size: usize) -> Self {
        Self {
            shape: HashMap::with_capacity(size),
            order: Vec::with_capacity(size),
        }
    }

    fn extend(&mut self, labels: &[char], shape: &[u64]) -> TCResult<()> {
        if labels.len() != shape.len() {
            return Err(TCError::unsupported(format!(
                "tensor with {} dimensions does not match format string {}",
                shape.len(),
                labels.iter().copied().collect::<String>()
            )));
        }

        for (label, dim) in labels.iter().zip(shape) {
            if let Some(known) = self.shape.get(label) {
                if dim != known {
                    return Err(TCError::unsupported(format!(
                        "einsum found inconsistent dimensions for axis {}: {:?} vs {:?}",
                        label, known, dim
                    )));
                }
            } else {
                self.shape.insert(*label, *dim);
                self.order.push(*label);
            }
        }

        Ok(())
    }

    fn get(&self, subscript: &char) -> Option<&u64> {
        self.shape.get(subscript)
    }

    fn len(&self) -> usize {
        self.order.len()
    }

    fn order(&self) -> &[char] {
        &self.order
    }
}

fn parse_format<T: TensorAccess>(inputs: &[T], format: &str) -> TCResult<(Vec<Label>, Label)> {
    debug!("einsum format string is {}", format);

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

    let f_inputs = parts[0].split(',');
    match f_inputs.count() {
        count if count == inputs.len() => Ok(()),
        count => Err(TCError::unsupported(format!(
            "einsum got {} tensors with {} format strings",
            inputs.len(),
            count
        ))),
    }?;

    let f_output = parts.pop_back().ok_or_else(|| {
        TCError::unsupported(
            "einsum requires > 0 output dimensions; use sum to sum over an entire Tensor",
        )
    })?;

    let valid_subscripts: HashSet<char> = VALID_SUBSCRIPTS.iter().cloned().collect();

    let mut elided = None;
    let mut present_subscripts = HashSet::<char>::with_capacity(parts[0].len());
    for (tensor, f_input) in inputs.iter().zip(parts[0].split(',')) {
        if f_input.starts_with(&ELLIPSIS[..]) {
            if !f_input[ELLIPSIS.len()..]
                .chars()
                .all(|c| valid_subscripts.contains(&c))
            {
                return Err(TCError::bad_request(
                    "einsum got invalid subscript",
                    f_input,
                ));
            }

            let num_elided = tensor.ndim() - (f_input.len() - ELLIPSIS.len());
            if let Some(elided) = &mut elided {
                *elided = Ord::max(*elided, num_elided);
            } else {
                elided = Some(num_elided)
            };

            present_subscripts.extend(f_input[ELLIPSIS.len()..].chars());
        } else if f_input.contains(DOT) {
            return Err(TCError::bad_request("invalid format for einsum", f_input));
        } else {
            if !f_input.chars().all(|c| valid_subscripts.contains(&c)) {
                return Err(TCError::bad_request(
                    "einsum got invalid subscript",
                    f_input,
                ));
            }

            present_subscripts.extend(f_input.chars());
        }
    }

    let elided = if let Some(num_elided) = elided {
        let elided: String = VALID_SUBSCRIPTS
            .iter()
            .filter(|c| !present_subscripts.contains(c))
            .take(num_elided)
            .collect();

        if elided.len() == num_elided {
            Ok(Some(elided))
        } else {
            Err(TCError::unsupported(
                "einsum got too many dimensions to elide",
            ))
        }
    } else {
        Ok(None)
    }?;

    let f_inputs = parts
        .pop_front()
        .unwrap()
        .split(',')
        .zip(inputs)
        .map(|(f_input, tensor)| {
            if f_input.starts_with(&ELLIPSIS[..]) {
                let elided = elided.as_ref().expect("elided subscripts");
                let num_elided = tensor.ndim() - (f_input.len() - ELLIPSIS.len());
                assert!(elided.len() >= num_elided);
                let i = elided.len() - num_elided;
                format!("{}{}", &elided[i..], &f_input[ELLIPSIS.len()..])
            } else {
                debug_assert!(!f_input.contains(DOT));
                f_input.to_string()
            }
        })
        .map(|f_input| f_input.chars().collect())
        .collect::<Vec<Label>>();

    for f_input in &f_inputs {
        if f_input.iter().collect::<HashSet<_>>().len() != f_input.len() {
            return Err(TCError::not_implemented(
                "repeated subscripts in einsum input",
            ));
        }
    }

    if f_output.chars().collect::<HashSet<_>>().len() != f_output.len() {
        return Err(TCError::bad_request(
            "einsum output cannot include repeated subscripts",
            f_output,
        ));
    }

    let f_output = f_output.chars().collect::<Label>();

    let mut invalid_subscripts = f_output
        .iter()
        .filter(|l| !valid_subscripts.contains(l))
        .peekable();

    if invalid_subscripts.peek().is_some() {
        return Err(TCError::bad_request(
            "invalid subscripts in einsum format",
            invalid_subscripts.collect::<Tuple<&char>>(),
        ));
    }

    for l in &f_output {
        if !present_subscripts.contains(l) {
            return Err(TCError::bad_request(
                "subscript in output but not in input",
                l,
            ));
        }
    }

    let f_output = if let Some(elided) = elided {
        elided.chars().chain(f_output).collect()
    } else {
        f_output
    };

    Ok((f_inputs, f_output))
}

fn validate_args<T: TensorAccess>(f_inputs: &[Label], tensors: &[T]) -> TCResult<Dimensions> {
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

    let mut dimensions = Dimensions::with_capacity(f_inputs.iter().map(|f| f.len()).sum());

    for (f_input, tensor) in f_inputs.iter().zip(tensors.iter()) {
        dimensions.extend(f_input, tensor.shape())?;
    }

    Ok(dimensions)
}

fn normalize<D, Txn, T>(tensor: T, f_input: &[char], dimensions: &Dimensions) -> TCResult<T>
where
    D: DirLock,
    Txn: Transaction<D>,
    T: TensorAccess
        + TensorReduce<D, Txn = Txn, Reduce = T>
        + TensorTransform<Broadcast = T, Expand = T, Transpose = T>
        + Clone,
{
    assert_eq!(tensor.ndim(), f_input.len());

    let f_output = dimensions.order().to_vec();

    debug!(
        "normalize tensor with shape {} from {:?} -> {:?}",
        tensor.shape(),
        f_input,
        f_output,
    );

    if f_input == f_output {
        debug!(
            "{:?} is already normalized to {:?}, returning...",
            f_input, f_output
        );

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

    let mut subscripts = permutation
        .iter()
        .map(|x| f_input[*x])
        .collect::<Vec<char>>();

    debug!("permutation of {:?} is {:?}", f_input, permutation);
    let mut tensor = tensor.transpose(Some(permutation.clone()))?;

    let mut i = 0;
    while i < dimensions.len() {
        if i == subscripts.len() || subscripts[i] != f_output[i] {
            tensor = tensor.expand_dims(i)?;
            subscripts.insert(i, f_output[i]);
        } else {
            i += 1;
        }
    }

    debug!(
        "input tensor with shape {} has subscripts {:?}",
        tensor.shape(),
        subscripts
    );

    let shape = f_output
        .iter()
        .map(|l| dimensions.get(l).expect("tensor dimension"))
        .cloned()
        .collect::<Vec<u64>>();

    if &**tensor.shape() == &shape {
        Ok(tensor)
    } else {
        debug!("broadcast {} into {:?}", tensor.shape(), shape);
        tensor.broadcast(shape.into())
    }
}

fn outer_product<D, Txn, T>(
    f_inputs: &[Label],
    dimensions: &Dimensions,
    tensors: Vec<T>,
) -> TCResult<T>
where
    D: DirLock,
    Txn: Transaction<D>,
    T: TensorAccess
        + TensorMath<D, T, LeftCombine = T>
        + TensorReduce<D, Txn = Txn, Reduce = T>
        + TensorTransform<Broadcast = T, Expand = T, Transpose = T>
        + Clone
        + fmt::Display,
{
    assert_eq!(f_inputs.len(), tensors.len());
    assert!(!tensors.is_empty());

    let mut normalized = tensors
        .into_iter()
        .zip(f_inputs.iter())
        .map(|(tensor, f_input)| normalize(tensor, f_input, &dimensions))
        .collect::<TCResult<VecDeque<T>>>()?;

    let mut op = normalized.pop_front().unwrap();
    while let Some(tensor) = normalized.pop_front() {
        debug!("outer product: {} *= {}", op, tensor);
        op = op.mul(tensor)?;
    }

    Ok(op)
}

fn contract<D, T>(mut op: T, dimensions: Dimensions, f_output: Label) -> TCResult<T>
where
    D: DirLock,
    T: TensorAccess + TensorReduce<D, Reduce = T> + TensorTransform<Transpose = T>,
{
    let mut f_input = dimensions.order().to_vec();
    let mut axis = 0;
    while op.ndim() > f_output.len() {
        assert_eq!(f_input.len(), op.ndim());

        if !f_output.contains(&f_input[axis]) {
            debug!("einsum will contract over axis {}", f_input[axis]);
            op = op.sum(axis, false)?;
            f_input.remove(axis);
        } else {
            assert!(f_input.contains(&f_output[axis]));
            axis += 1;
        }
    }

    if f_input == f_output {
        debug!(
            "outer product already has shape {:?}, no need to transpose",
            f_output
        );

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
    D: DirLock,
    T: TensorAccess
        + TensorMath<D, T, LeftCombine = T>
        + TensorTransform<Broadcast = T, Expand = T, Transpose = T>
        + TensorReduce<D, Reduce = T>
        + Clone
        + fmt::Display,
{
    let (f_inputs, f_output) = parse_format(&tensors, format)?;

    debug!(
        "einsum with input labels: {:?}, output label {:?}",
        f_inputs, f_output
    );

    let dimensions = validate_args(&f_inputs, &tensors)?;
    let op = outer_product(&f_inputs, &dimensions, tensors)?;
    contract(op, dimensions, f_output)
}
