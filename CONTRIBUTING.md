# Contributing to H.E.R.A.-R

Thank you for your interest in contributing to H.E.R.A.-R! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Research Contributions](#research-contributions)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:
- Be respectful and inclusive
- Focus on constructive feedback
- Welcome diverse perspectives
- Maintain a professional environment

## Getting Started

### Prerequisites
- Python 3.9 or higher
- Git
- Basic understanding of transformer models and PyTorch

### Setting Up Development Environment

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/hera-r.git
   cd hera-r
   ```

2. **Set Up Development Environment**
   ```bash
   # Install in development mode
   pip install -e .[dev]
   
   # Set up pre-commit hooks
   pre-commit install
   
   # Run development setup script
   python scripts/setup_dev.py
   ```

3. **Verify Installation**
   ```bash
   # Run tests to verify setup
   pytest tests/ -v
   ```

## Development Workflow

### Branch Strategy
- `main`: Stable, production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features or enhancements
- `bugfix/*`: Bug fixes
- `docs/*`: Documentation updates

### Creating a New Feature

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make Your Changes**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run all checks
   python tasks.py lint
   python tasks.py test
   ```

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add amazing feature"
   ```

## Code Style

### Python Style Guide
We follow [PEP 8](https://pep8.org/) with some modifications:

#### Formatting
- Use **Black** for code formatting (88 character line length)
- Use **isort** for import sorting (Black-compatible profile)
- Use **flake8** for linting (with some exceptions)

#### Type Hints
- Use type hints for all function signatures
- Use `mypy` for type checking
- Prefer `Optional[T]` over `Union[T, None]`

#### Naming Conventions
- **Classes**: `CamelCase`
- **Functions/Methods**: `snake_case`
- **Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: `_leading_underscore`

### Example Code

```python
from typing import Optional, List, Dict
import torch
from torch import Tensor


class ExampleClass:
    """Example class demonstrating code style."""
    
    def __init__(self, param: int, optional_param: Optional[str] = None):
        self.param = param
        self.optional_param = optional_param
        self._private_var = 42
    
    def public_method(self, input_tensor: Tensor) -> Dict[str, float]:
        """Public method with type hints and docstring.
        
        Args:
            input_tensor: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Dictionary containing computed metrics
        """
        if input_tensor.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {input_tensor.dim()}D")
        
        result = self._private_method(input_tensor)
        return {"value": float(result.mean())}
    
    def _private_method(self, tensor: Tensor) -> Tensor:
        """Private method example."""
        return tensor * self.param
```

### Docstring Format
We use Google-style docstrings:

```python
def example_function(param1: str, param2: int = 42) -> bool:
    """Short description of function.
    
    Longer description explaining details, edge cases, etc.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter with default
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something goes wrong
        RuntimeError: When something else goes wrong
        
    Examples:
        >>> example_function("test", 10)
        True
    """
    if not param1:
        raise ValueError("param1 cannot be empty")
    
    return len(param1) > param2
```

## Testing

### Test Structure
- Place tests in `tests/` directory
- Name test files: `test_*.py`
- Name test classes: `Test*`
- Name test functions: `test_*`

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_engine.py

# Run with coverage
pytest tests/ --cov=hera --cov-report=html

# Run slow tests
pytest tests/ -m "slow"

# Skip slow tests
pytest tests/ -m "not slow"
```

### Writing Tests

#### Unit Tests
```python
def test_feature_extraction():
    """Test feature extraction from activations."""
    # Arrange
    activations = torch.randn(2, 10, 768)
    extractor = FeatureExtractor()
    
    # Act
    features = extractor.extract(activations)
    
    # Assert
    assert features.shape == (2, 10, 256)
    assert not torch.isnan(features).any()
```

#### Integration Tests
```python
@pytest.mark.integration
def test_full_evolution_cycle():
    """Test complete evolution cycle."""
    # Setup
    config = create_test_config()
    engine = HeraEngine(config)
    
    # Execute
    success = engine.evolve("Test prompt")
    
    # Verify
    assert isinstance(success, bool)
    if success:
        assert len(engine.registry.history) == 1
```

#### Fixtures
Use pytest fixtures for common setup:

```python
@pytest.fixture
def test_engine():
    """Provide test engine instance."""
    config = {
        "experiment": {"device": "cpu"},
        "model": {"name": "gpt2-small"},
    }
    return HeraEngine(config)
```

## Documentation

### Documentation Types
1. **User Documentation**: Installation, usage, examples
2. **API Documentation**: Module and function references
3. **Research Documentation**: Theory, experiments, results

### Building Documentation
```bash
# Install docs dependencies
pip install -e .[docs]

# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve
```

### Writing Documentation
- Use Markdown for all documentation
- Include code examples where helpful
- Link to related documentation
- Keep documentation up-to-date with code changes

## Pull Request Process

### Before Submitting
1. Ensure your code follows style guidelines
2. Run all tests and checks
3. Update documentation as needed
4. Add appropriate tests for new functionality

### Creating a Pull Request
1. **Update Your Fork**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push to Your Fork**
   ```bash
   git push origin feature/amazing-feature
   ```

3. **Create Pull Request**
   - Go to GitHub repository
   - Click "New Pull Request"
   - Select your branch
   - Fill in the PR template

### PR Template
```markdown
## Description
Brief description of changes

## Related Issues
Fixes #123, Related to #456

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Code refactoring

## Testing
- [ ] Added/updated tests
- [ ] All tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Type hints added/updated
- [ ] No new warnings
- [ ] CHANGELOG updated (if applicable)
```

### Review Process
1. **Automated Checks**: CI runs tests and linting
2. **Maintainer Review**: At least one maintainer reviews
3. **Feedback**: Address any feedback
4. **Merge**: Once approved, maintainer merges

## Research Contributions

### Experimental Results
If contributing research results:
- Include methodology description
- Provide reproducible code
- Share datasets or generation scripts
- Document hyperparameters

### New Algorithms
When proposing new algorithms:
- Include theoretical background
- Provide implementation details
- Compare with existing methods
- Include benchmark results

### Extending the Framework
For framework extensions:
- Maintain backward compatibility
- Add configuration options
- Update documentation
- Provide migration guide if needed

## Getting Help

- **Issues**: Use GitHub Issues for bug reports
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for sensitive issues

## Recognition

All contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to H.E.R.A.-R! 🧬