This directory contains a vendored MATLAB v7.3 fixture for MATLAB `string`
objects.

Files:

- `test_string_v73.mat`: MATLAB-generated fixture with scalar, array, and empty
  `string` variables
- `test_string_gen.m`: upstream MATLAB generator script used to create the
  fixture

Source:

- `mat-io` 0.6.3
- https://pypi.org/project/mat-io/

License:

- BSD 3-Clause
- see `LICENSE.mat-io`

Why this exists:

- MATIO 1.5.30 does not decode MATLAB `string` objects semantically; it reports
  them as unknown class values.
- This fixture gives the MAT writer tests a real MATLAB-produced HDF5 schema to
  target for future `string` support without requiring a MATLAB license in CI.
