# Pipeline Issues and Errors

## Run Pipeline Script Issues

### Logging Configuration Error

- **Error**: `NameError: name 'root_handler' is not defined. Did you mean: 'error_handler'?`
- **Location**: `src/logging_config.py:49` in `setup_logging()` function
- **Impact**: Pipeline cannot start due to logging system failure
- **Status**: BLOCKER
- **Root Cause**: Typo in variable name - `root_handler` should be `root_logger`

### Previous Issues (Resolved)

- **Missing Required Files**: Previously missing `results/power_profiles.json` - now found during file validation

### Additional Context

- Pipeline script: `run_pipeline.sh`
- Error occurs during Step 1: Generating training data
- Script fails immediately when trying to initialize logging system
- All required files are present and validated successfully
- Issue is in the enhanced logging configuration that was just added

## Error Details

```
Traceback (most recent call last):
  File "/Users/skakatur/Desktop/CodingStuff/optimal-charge-security-camera/generate_training_data.py", line 27, in <module>
    logger = setup_logging()
  File "/Users/skakatur/Desktop/CodingStuff/optimal-charge-security-camera/src/logging_config.py", line 49, in setup_logging
    root_handler.addHandler(error_handler)
    ^^^^^^^^^^^^
NameError: name 'root_handler' is not defined. Did you mean: 'error_handler'?
```

## Files Affected

- `src/logging_config.py` - Contains the typo
- `generate_training_data.py` - First script to call the logging setup
- All other enhanced scripts will have the same issue once this is fixed
