# Configuration

## Config file layout
- `version`: schema version
- `log_level`: logging verbosity
- `output_dir`: where exports and reports are written
- `backends`: default backend, timeouts, worker pool size
- `execution`: parallel flag, dry run, consent requirements
- `export`: format, metadata inclusion, pretty-print

## Editing config
```bash
proxima config edit
```

## Environment overrides
- `PROXIMA_CONFIG`: path to config file
- `PROXIMA_LOG_LEVEL`: logging level override
- `PROXIMA_OUTPUT_DIR`: output path override

## Validation tips
- Keep timeouts reasonable for remote backends
- Enable `execution.require_consent` when handling user data
- Set `export.pretty_print` to false for large exports
