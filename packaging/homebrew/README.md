# ProximA Homebrew Formula

This directory contains the Homebrew formula for installing ProximA on macOS.

## Installation

### From Homebrew Tap (Recommended)

```bash
# Add the ProximA tap
brew tap your-org/proxima

# Install ProximA
brew install proxima

# Install with optional backends
brew install proxima --with-qiskit
brew install proxima --with-cirq
```

### From Local Formula

```bash
# Clone the repository
git clone https://github.com/your-org/proxima.git
cd proxima

# Install from local formula
brew install --build-from-source ./packaging/homebrew/proxima.rb
```

## Options

| Option | Description |
|--------|-------------|
| `--with-qiskit` | Include Qiskit Aer backend support |
| `--with-cirq` | Include Google Cirq backend support |
| `--with-quest` | Include QuEST backend support |
| `--HEAD` | Install from the latest development branch |

## Verification

After installation, verify ProximA is working:

```bash
# Check version
proxima --version

# List available backends
proxima backends

# Run a test simulation
proxima simulate examples/bell.qasm --shots 1000
```

## Configuration

ProximA stores its configuration at:
- Config file: `$(brew --prefix)/var/proxima/config.toml`
- Log file: `$(brew --prefix)/var/proxima/proxima.log`

## Uninstallation

```bash
brew uninstall proxima

# Optional: Remove configuration
rm -rf $(brew --prefix)/var/proxima
```

## Updating

```bash
brew update
brew upgrade proxima
```

## Troubleshooting

### Common Issues

1. **Python version mismatch**
   ```bash
   brew install python@3.11
   brew link python@3.11
   ```

2. **NumPy installation fails**
   ```bash
   brew install numpy
   ```

3. **Missing dependencies**
   ```bash
   brew reinstall proxima
   ```

### Getting Help

- Check logs: `$(brew --prefix)/var/proxima/proxima.log`
- Report issues: https://github.com/your-org/proxima/issues

## For Maintainers

### Updating the Formula

1. Update the version and SHA256 in `proxima.rb`
2. Test the formula:
   ```bash
   brew audit --strict proxima.rb
   brew install --build-from-source proxima.rb
   brew test proxima
   ```
3. Create a release on GitHub
4. Update the tap repository

### Building Bottles

```bash
brew install --build-bottle proxima
brew bottle proxima
```

Upload the generated bottle files to the release.

## License

MIT License - See the main repository for details.

