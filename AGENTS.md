- In all interactions, be extremely concise and sacrifice grammar for the sake of concision.

## GitHub

- Your primary method for interacting with GitHub should be the GitHub CLI

## Python

- Your primary method for interacting with anything Python related should be 'uv'.

## Build/Lint/Test Commands

- Run all tests: `./scripts/tests.sh`
- Run single test: `uv run pytest tests/test_file.py -v`
- Run CI tests: `./scripts/ci-tests.sh`
- Run simulation: `uv run python src/main_simulation.py`
- Run with options: `uv run python src/main_simulation.py --config custom.jsonc --output results.csv`

## Code Style Guidelines

- Imports: Use absolute imports from src, group stdlib, third-party, then local imports
- Formatting: Follow PEP 8, use type hints consistently
- Naming: snake_case for functions/variables, PascalCase for classes
- Error handling: Use try/except blocks, log errors with loguru
- Logging: Use loguru logger, include context in messages
- Types: Use typing module for all function signatures
- Comments: Docstrings for all classes and public functions
- Structure: Follow existing patterns in simulation/, data/, sensors/, utils/

## Documentation

- Primary documentation: https://therealsamyak.github.io/optimal-charge-security-camera/
- Project structure: src/ (main code), tests/ (test suite), scripts/ (utility scripts), docs/ (website files)
- Always update all .md files as necessary based on changes made after the changes are made

## Plans

- Make all plans multi-phase.

- At the end of each plan, give me a list of unresolved questions to answer, if any. Make the questions extremely concise. Sacrifice grammar for the sake of concision.
