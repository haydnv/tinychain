use tc_error::*;
use tc_tensor::TensorAccess;

pub async fn svd<T>(matrix: T) -> TCResult<T>
where
    T: TensorAccess,
{
    if matrix.ndim() != 2 {
        return Err(TCError::bad_request(
            "SVD requires exactly two dimensions, found",
            matrix.ndim(),
        ));
    }

    Err(TCError::not_implemented("singular value decomposition"))
}
