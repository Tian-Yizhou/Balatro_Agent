# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Restructure the repository for v2: a `src/` layout with the `sim`, `replay`, `env`
  and `serve` subpackages, managed with uv.

### Removed

- The v1 implementation. It remains available at tag `v1-legacy`.
