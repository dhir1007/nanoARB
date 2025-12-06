# Contributing to NanoARB

Thank you for your interest in contributing to NanoARB! This document provides guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

Before reporting a bug:
1. Check existing issues to avoid duplicates
2. Use the bug report template
3. Include reproduction steps, expected vs actual behavior
4. Include system information (OS, Rust version, etc.)

### Suggesting Features

1. Check existing issues and discussions
2. Use the feature request template
3. Explain the use case and benefits

### Pull Requests

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Run tests: `cargo test --workspace`
5. Run lints: `cargo clippy --all-targets`
6. Format code: `cargo fmt --all`
7. Submit a PR with a clear description

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/nanoARB.git
cd nanoARB

# Build
cargo build

# Run tests
cargo test --workspace

# Run benchmarks
cargo bench --workspace
```

## Code Style

- Follow Rust idioms and conventions
- Use `rustfmt` for formatting
- Address all `clippy` warnings
- Write documentation for public APIs
- Include unit tests for new functionality

## Commit Messages

Use conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
- `feat(lob): add VPIN calculation`
- `fix(backtest): correct latency simulation`
- `docs(readme): update installation instructions`

## Testing

- Write tests for new functionality
- Maintain test coverage above 80%
- Include both unit and integration tests
- Use property-based testing where appropriate

## Questions?

Feel free to open a discussion or issue if you have questions!

