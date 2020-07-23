use std::collections::HashSet;

use crate::error;
use crate::value::TCResult;

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
        let labels: HashSet<char> = f_input.into_iter().cloned().collect();
        if labels.len() != f_input.len() {
            return Err(error::bad_request(
                "Duplicate label in einsum format",
                f_input.into_iter().cloned().collect::<String>(),
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
