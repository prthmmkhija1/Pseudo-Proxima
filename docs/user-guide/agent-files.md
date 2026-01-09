# Agent Files

## Layout
- `.proxima/`: workspace state
- `configs/`: default and environment-specific configs
- `reports/`: generated exports and comparison reports
- `logs/`: execution logs

## Creating agent files
```bash
proxima init --template default
```
- Generates base config and sample circuits

## Best practices
- Keep secrets out of tracked files
- Store exports in `reports/` for auditing
- Rotate logs regularly
