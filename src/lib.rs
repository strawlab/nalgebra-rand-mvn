//! Random multi-variate normal generation using nalgebra
use nalgebra::{OMatrix, OVector};
use rand::distributions::Distribution;
use rand_distr::Normal;

use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, DimName, DimSub, Dynamic, RealField};

/// An error
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
}

impl Error {
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.kind)
    }
}

/// Kind of error
#[derive(Debug, Clone)]
pub enum ErrorKind {
    NotDefinitePositive,
}

fn standard_normal<Real: RealField, R: Dim, C: Dim>(nrows: R, ncols: C) -> OMatrix<Real, R, C>
where
    DefaultAllocator: Allocator<Real, R, C>,
{
    let normal = Normal::new(0.0, 1.0).expect("creating normal");
    let mut rng = rand::thread_rng();
    OMatrix::<Real, R, C>::from_fn_generic(nrows, ncols, |_row, _col| {
        nalgebra::convert::<f64, Real>(normal.sample(&mut rng))
    })
}

/// Draw random samples from a multi-variate normal
///
/// Return `n_samples` samples from the N dimensional normal given by mean `mu` and
/// covariance `sigma`.
pub fn rand_mvn_generic<Real, Count, N>(
    n_samples: Count,
    mu: &OVector<Real, N>,
    sigma: nalgebra::OMatrix<Real, N, N>,
) -> Result<OMatrix<Real, Count, N>, Error>
where
    Real: RealField,
    Count: Dim,
    N: Dim + DimSub<Dynamic>,
    DefaultAllocator: Allocator<Real, Count, N>,
    DefaultAllocator: Allocator<Real, N, Count>,
    DefaultAllocator: Allocator<Real, N, N>,
    DefaultAllocator: Allocator<Real, N>,
{
    let ncols = N::from_usize(mu.nrows());
    let norm_data: OMatrix<Real, N, Count> = standard_normal(ncols, n_samples);
    let sigma_chol: nalgebra::OMatrix<Real, N, N> = nalgebra::linalg::Cholesky::new(sigma)
        .ok_or(Error {
            kind: ErrorKind::NotDefinitePositive,
        })?
        .l();
    Ok(broadcast_add(&(sigma_chol * norm_data).transpose(), mu))
}

/// Draw random samples from a multi-variate normal
///
/// Return `Count` samples from the N dimensional normal given by mean `mu` and
/// covariance `sigma`.
pub fn rand_mvn<Real, Count, N>(
    mu: &OVector<Real, N>,
    sigma: nalgebra::OMatrix<Real, N, N>,
) -> Result<OMatrix<Real, Count, N>, Error>
where
    Real: RealField,
    Count: DimName,
    N: DimName,
    DefaultAllocator: Allocator<Real, Count, N>,
    DefaultAllocator: Allocator<Real, N, Count>,
    DefaultAllocator: Allocator<Real, N, N>,
    DefaultAllocator: Allocator<Real, N>,
{
    let nrows = Count::name();
    rand_mvn_generic(nrows, mu, sigma)
}

/// Add `vec` to each row of `arr`, returning the result with shape of `arr`.
///
/// Inputs `arr` has shape R x C and `vec` is C dimensional. Result
/// has shape R x C.
fn broadcast_add<Real, R, C>(
    arr: &OMatrix<Real, R, C>,
    vec: &OVector<Real, C>,
) -> OMatrix<Real, R, C>
where
    Real: RealField,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<Real, R, C>,
    DefaultAllocator: Allocator<Real, C>,
{
    let ndim = arr.nrows();
    let nrows = R::from_usize(arr.nrows());
    let ncols = C::from_usize(arr.ncols());

    // TODO: remove explicit index calculation and indexing
    OMatrix::from_iterator_generic(
        nrows,
        ncols,
        arr.iter().enumerate().map(|(i, el)| {
            let vi = i / ndim; // integer div to get index into vec
            el.clone() + vec[vi].clone()
        }),
    )
}

#[cfg(test)]
mod tests {
    use crate::*;
    use approx::assert_relative_eq;
    use nalgebra as na;

    /// Calculate the sample covariance
    ///
    /// Calculates the sample covariances among N-dimensional samples with M
    /// observations each. Calculates N x N covariance matrix from observations
    /// in `arr`, which is M rows of N columns used to store M vectors of
    /// dimension N.
    fn sample_covariance<Real: RealField, M: Dim, N: Dim>(
        arr: &OMatrix<Real, M, N>,
    ) -> nalgebra::OMatrix<Real, N, N>
    where
        DefaultAllocator: Allocator<Real, M, N>,
        DefaultAllocator: Allocator<Real, N, M>,
        DefaultAllocator: Allocator<Real, N, N>,
        DefaultAllocator: Allocator<Real, N>,
    {
        let mu: OVector<Real, N> = mean_axis0(arr);
        let y = broadcast_add(arr, &-mu);
        let n: Real = nalgebra::convert(arr.nrows() as f64);
        let sigma = (y.transpose() * y) / (n - Real::one());
        sigma
    }

    /// Calculate the mean of R x C matrix along the rows and return C dim vector
    fn mean_axis0<Real, R, C>(arr: &OMatrix<Real, R, C>) -> OVector<Real, C>
    where
        Real: RealField,
        R: Dim,
        C: Dim,
        DefaultAllocator: Allocator<Real, R, C>,
        DefaultAllocator: Allocator<Real, C>,
    {
        let vec_dim: C = C::from_usize(arr.ncols());
        let mut mu = OVector::zeros_generic(vec_dim, na::Const);
        let scale: Real = Real::one() / na::convert(arr.nrows() as f64);
        for j in 0..arr.ncols() {
            let col_sum = arr
                .column(j)
                .iter()
                .fold(Real::zero(), |acc, x| acc.clone() + x.clone());
            mu[j] = col_sum * scale.clone();
        }
        mu
    }

    #[test]
    fn test_covar() {
        use nalgebra::core::dimension::{U2, U3};

        // We use the same example as https://numpy.org/doc/stable/reference/generated/numpy.cov.html

        // However, our format is transposed compared to numpy. We have
        // variables as the columns and samples as rows.
        let arr = OMatrix::<f64, U2, U3>::new(-2.1, -1.0, 4.3, 3.0, 1.1, 0.12).transpose();

        let c = sample_covariance(&arr);

        let expected = nalgebra::OMatrix::<f64, U2, U2>::new(11.71, -4.286, -4.286, 2.144133);

        assert_relative_eq!(c, expected, epsilon = 1e-3);
    }

    #[test]
    fn test_mean_axis0() {
        use nalgebra::dimension::{U2, U4};

        let a1 = OMatrix::<f64, U2, U4>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let actual1: OVector<f64, U4> = mean_axis0(&a1);
        let expected1 = &[3.0, 4.0, 5.0, 6.0];
        assert!(actual1.as_slice() == expected1);

        let a2 = OMatrix::<f64, U4, U2>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let actual2: OVector<f64, U2> = mean_axis0(&a2);
        let expected2 = &[4.0, 5.0];
        assert!(actual2.as_slice() == expected2);
    }

    #[test]
    fn test_rand() {
        use na::dimension::{U25, U4};
        let mu = OVector::<f64, U4>::new(1.0, 2.0, 3.0, 4.0);
        let sigma = nalgebra::OMatrix::<f64, U4, U4>::new(
            2.0, 0.1, 0.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        );
        let y: OMatrix<f64, U25, U4> = rand_mvn(&mu, sigma).unwrap();
        assert!(y.nrows() == 25);
        assert!(y.ncols() == 4);

        let mu2 = mean_axis0(&y);
        assert_relative_eq!(mu, mu2, epsilon = 0.5); // expect occasional failures here

        let sigma2 = sample_covariance(&y);
        assert_relative_eq!(sigma, sigma2, epsilon = 1.0); // expect occasional failures here
    }

    #[test]
    fn test_rand_dynamic() {
        use na::dimension::{Dynamic, U4};
        let mu = OVector::<f64, U4>::new(1.0, 2.0, 3.0, 4.0);
        let sigma = nalgebra::OMatrix::<f64, U4, U4>::new(
            2.0, 0.1, 0.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        );
        let nrows = Dynamic::new(1_000);
        let y: nalgebra::OMatrix<f64, Dynamic, U4> =
            rand_mvn_generic(nrows, &mu, sigma.clone()).unwrap();
        assert!(y.ncols() == 4);

        let mu2 = mean_axis0(&y);
        assert_relative_eq!(mu, mu2, epsilon = 0.2); // expect occasional failures here

        let sigma2 = sample_covariance(&y);
        assert_relative_eq!(sigma, sigma2, epsilon = 0.2); // expect occasional failures here
    }
}
