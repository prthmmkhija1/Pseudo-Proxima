# Proxima: Intelligent Quantum Simulation Orchestration Framework

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/proxima-agent.svg)](https://pypi.org/project/proxima-agent/)
[![Docker Image](https://img.shields.io/docker/v/proxima-project/proxima-agent?label=docker)](https://ghcr.io/proxima-project/proxima)
[![CI](https://github.com/proxima-project/proxima/actions/workflows/ci.yml/badge.svg)](https://github.com/proxima-project/proxima/actions/workflows/ci.yml)

Proxima is an intelligent quantum simulation orchestration framework that provides a unified interface for running quantum simulations across multiple backends with advanced features like automatic backend selection, resource monitoring, and intelligent result interpretation.

## Features

- **Multi-Backend Support**: LRET, Cirq (DensityMatrix + StateVector), Qiskit Aer (DensityMatrix + StateVector)
- **Intelligent Backend Selection**: Automatic selection with explanations
- **Execution Control**: Start, Abort, Pause, Resume, Rollback
- **Resource Awareness**: Memory and CPU monitoring with fail-safe mechanisms
- **Explicit Consent**: User confirmation for critical operations
- **LLM Integration**: Support for local and remote AI models
- **Result Interpretation**: Human-readable insights and analytics
- **Multi-Backend Comparison**: Run identical simulations across backends
- **Execution Transparency**: Real-time progress and timing display

## Installation

### From PyPI (Recommended)

```bash
# Install the base package
pip install proxima-agent

# Install with all optional dependencies (LLM, TUI, dev tools)
pip install proxima-agent[all]

# Install specific extras
pip install proxima-agent[llm]    # LLM integrations (OpenAI, Anthropic)
pip install proxima-agent[ui]     # Terminal UI (Textual)
pip install proxima-agent[dev]    # Development tools
```

### Using Docker

```bash
# Pull the latest image
docker pull ghcr.io/proxima-project/proxima:latest

# Run with Docker
docker run --rm -it ghcr.io/proxima-project/proxima:latest --help

# Run a simulation
docker run --rm -it \
  -v ~/.proxima:/home/proxima/.proxima \
  ghcr.io/proxima-project/proxima:latest \
  run --backend cirq "bell state"

# Using Docker Compose
docker-compose up -d proxima
docker-compose run proxima backends list
```

### Using Homebrew (macOS/Linux)

```bash
# Add the tap
brew tap proxima-project/proxima

# Install
brew install proxima

# Verify installation
proxima version
```

### From Source

```bash
# Clone the repository
git clone https://github.com/proxima-project/proxima.git
cd proxima

# Install in development mode
pip install -e ".[all]"

# Or use the build script
python scripts/build.py build
```

### Standalone Binaries

Download pre-built binaries from the [Releases](https://github.com/proxima-project/proxima/releases) page:

- **Linux**: `proxima-linux-x86_64`
- **macOS Intel**: `proxima-darwin-x86_64`
- **macOS Apple Silicon**: `proxima-darwin-arm64`
- **Windows**: `proxima-windows-x86_64.exe`

```bash
# Make executable (Linux/macOS)
chmod +x proxima-linux-x86_64

# Run
./proxima-linux-x86_64 --help
```

## Quick Start

```bash
# Initialize configuration
proxima init

# Show version
proxima version

# List available backends
proxima backends list

# Run a simulation
proxima run --backend cirq "bell state"

# Compare across backends
proxima compare --backends cirq,qiskit "quantum teleportation"
```

## Docker Quick Start

```bash
# Run with Docker Compose
docker-compose up -d

# Execute commands
docker-compose run proxima backends list
docker-compose run proxima run --backend cirq "entanglement"

# With local LLM support (Ollama)
docker-compose --profile llm up -d
docker-compose run proxima run --llm ollama "analyze circuit"

# Development mode
docker-compose --profile dev up -d
docker-compose exec proxima-dev pytest
```

## Project Structure

```
proxima/
├── src/proxima/          # Main package
│   ├── cli/              # Command-line interface
│   ├── core/             # Core domain logic
│   ├── backends/         # Backend adapters
│   ├── intelligence/     # AI/ML components
│   ├── resources/        # Resource management
│   ├── data/             # Data handling
│   ├── config/           # Configuration
│   └── utils/            # Utilities
├── tests/                # Test suites
├── configs/              # Configuration files
├── scripts/              # Build and release scripts
├── packaging/            # Distribution packaging
└── docs/                 # Documentation
```

## Configuration

Proxima supports multiple configuration sources (in priority order):

1. Command-line arguments
2. Environment variables (PROXIMA\_\*)
3. User config file (~/.proxima/config.yaml)
4. Project config file (./proxima.yaml)
5. Default values

## Development

```bash
# Using the build script (recommended)
python scripts/build.py all          # Run all checks
python scripts/build.py test         # Run tests
python scripts/build.py lint         # Run linting
python scripts/build.py build        # Build package
python scripts/build.py release --version 0.1.0  # Prepare release

# Or using PowerShell on Windows
.\scripts\build.ps1 all
.\scripts\build.ps1 test -Coverage

# Manual commands
pytest                               # Run tests
pytest --cov=proxima                 # With coverage
black src/ tests/                    # Format code
ruff check src/ tests/               # Lint code
mypy src/                            # Type check
mkdocs serve                         # Serve docs locally
```

## Building & Releasing

```bash
# Build Python package
python -m build

# Build Docker image
docker build -t proxima-agent:latest .

# Prepare a release (dry run)
python scripts/build.py release --version 0.2.0

# Execute release
python scripts/build.py release --version 0.2.0 --no-dry-run
```

## Architecture

Proxima follows a layered modular architecture:

1. **Presentation Layer**: CLI, TUI (future), Web API (future)
2. **Orchestration Layer**: Planner, Executor, State Manager
3. **Intelligence Layer**: LLM Router, Backend Selector, Insight Engine
4. **Resources & Safety Layer**: Memory Monitor, Consent Manager, Execution Control
5. **Backend Abstraction Layer**: Unified adapter interface
6. **Data & Output Layer**: Result storage, comparison, export

## Roadmap

- **Phase 1** (Weeks 1-4): Foundation & Core Infrastructure ✅
- **Phase 2** (Weeks 5-9): Backend Integration ✅
- **Phase 3** (Weeks 10-14): Intelligence Features ✅
- **Phase 4** (Weeks 15-18): Safety & Resource Management ✅
- **Phase 5** (Weeks 19-23): Advanced Features ✅
- **Phase 6** (Weeks 24-27): Production Ready ✅

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please read our [contributing guidelines](docs/developer-guide/contributing.md) first.

See the [CHANGELOG](CHANGELOG.md) for version history.

## Credits

Architectural inspiration from:

- [OpenCode AI](https://github.com/opencode-ai/opencode)
- [Crush (Charmbracelet)](https://github.com/charmbracelet/crush)

Proxima is an independent implementation, not a fork or derivative.

---

## 📚 Complete Usage Guide - Step by Step

> **🎯 Goal**: This guide will walk you through installing and using Proxima to run quantum simulations - designed for beginners with no technical background required!

---

### 🚀 **Step 1: Installing Proxima on Your Computer**

**What You Need:**
- A computer running Windows, macOS, or Linux
- Internet connection
- Python 3.11 or higher (check by opening terminal and typing: `python --version`)

**Installation Methods:**

**Method 1: Quick Install (Recommended)**

1. **Open your terminal/command prompt**
   - Windows: Press `Win + R`, type `cmd`, press Enter
   - Mac: Press `Cmd + Space`, type `terminal`, press Enter
   - Linux: Press `Ctrl + Alt + T`

2. **Install Proxima with all features**
   ```bash
   pip install proxima-agent[all]
   ```
   💡 **What this does**: Downloads Proxima and all its dependencies (takes 1-2 minutes)

3. **Initialize Proxima**
   ```bash
   proxima init
   ```
   💡 **What this does**: Creates configuration files in your home directory (`~/.proxima/`)
   
   **Expected output**: ✅ Configuration initialized successfully!

4. **Verify installation**
   ```bash
   proxima version
   ```
   **Expected output**: `Proxima version X.X.X`

**Method 2: Install from Source (Advanced)**

```bash
# Download the code
git clone https://github.com/prthmmkhija1/Pseudo-Proxima.git
cd Pseudo-Proxima

# Install in development mode  
pip install -e ".[all]"

# Verify
proxima version
```

**✅ Success Check**: If you see a version number, you're ready to continue!

---

### 🔍 **Step 2: Understanding Backends**

**What is a Backend?**
- A backend is like a quantum computer simulator
- Each backend has different capabilities (speed, accuracy, features)
- Proxima supports: **Cirq**, **Qiskit**, and **LRET**

**See Available Backends:**

```bash
proxima backends list
```

**What You'll See:**
- ✅ = Backend is installed and ready to use
- ❌ = Backend not installed (can be added)
- Details like max qubits, features, version

**Get Detailed Information:**

```bash
# Learn about Cirq
proxima backends info cirq

# Learn about Qiskit
proxima backends info qiskit
```

**Installing Missing Backends:**
```bash
# Install Cirq
pip install cirq

# Install Qiskit
pip install qiskit qiskit-aer
```

---

### 🎮 **Step 3: Running Your First Quantum Simulation**

**Let's Create a "Bell State" (entangled qubits):**

```bash
proxima run --backend cirq "2-qubit bell state"
```

**Breaking Down the Command:**
- `proxima run` → Run a simulation
- `--backend cirq` → Use the Cirq simulator  
- `"2-qubit bell state"` → What to simulate (plain English!)

**What Happens:**

1. **⏱️ Timer starts** - Shows elapsed time
2. **🔍 Validation** - Checks if your request makes sense
3. **💾 Resource check** - Ensures enough memory available
4. **❓ Consent** - Asks "Proceed? (y/n)" → Type `y` and press Enter
5. **🚀 Execution** - Runs the simulation
6. **📊 Results** - Shows measurements with explanations

**Example Output:**
```
⏱️ Elapsed: 2.3s
💾 Memory: 45% used
✅ Simulation complete!

Results:
|00⟩: 50.2% ████████████████
|11⟩: 49.8% ████████████████

💡 Insight: Perfect entanglement! Measuring one qubit instantly determines the other.
```

**Try More Examples:**

```bash
# 3-qubit entangled state
proxima run --backend cirq "3-qubit GHZ state"

# Let Proxima auto-select the best backend
proxima run --backend auto "quantum teleportation"

# Simple single-qubit circuit
proxima run --backend qiskit "hadamard gate on 1 qubit"
```

---

### ⚖️ **Step 4: Comparing Different Backends**

**Why Compare?**
- See which backend is fastest
- Check accuracy differences
- Find the best tool for your needs

**Compare Two Backends:**

```bash
proxima compare --backends cirq,qiskit "bell state"
```

**Output Example:**

| Backend | Execution Time | Memory | Accuracy |
|---------|---------------|--------|----------|
| Cirq    | 1.2s          | 45 MB  | Reference |
| Qiskit  | 1.5s          | 52 MB  | 99.8% match |

🏆 **Recommendation**: Cirq (faster)

**Compare All Backends:**

```bash
proxima compare --backends cirq,qiskit,lret "ghz state"
```

**More Accurate Comparison (more measurements):**

```bash
proxima compare --backends cirq,qiskit --shots 4096 "bell state"
```
💡 More shots = more accurate but slower

---

### 🖥️ **Step 5: Using the Visual Interface (TUI)**

**What is TUI?**
- TUI = Terminal User Interface
- A colorful, interactive way to use Proxima
- No need to remember commands!

**Launch TUI:**

```bash
proxima ui
```

**Navigation:**
- Press **1** → Dashboard (overview)
- Press **2** → Execution (watch simulations run)
- Press **3** → Configuration (change settings)
- Press **4** → Results (view past runs)
- Press **5** → Backends (compare features)
- Press **?** → Help
- Press **q** → Quit

**Features:**
- Real-time progress bars
- Color-coded status
- Resource monitoring
- Interactive menus

---

### 📝 **Step 6: Creating Task Lists (Agent Files)**

**What are Agent Files?**
- Write multiple tasks in one file
- Proxima executes them automatically
- Perfect for overnight experiments

**Create `my_experiment.md`:**

```markdown
# My Quantum Experiment

## Task 1: Create Bell State
- backend: cirq
- shots: 1024
- circuit: 2-qubit bell state

## Task 2: Compare Backends
- compare: cirq, qiskit
- circuit: bell state
- shots: 2048

## Task 3: Export Results
- format: json
- output: results.json
```

**Run the Agent File:**

```bash
proxima agent run my_experiment.md
```

**What Happens:**
1. Shows preview of all tasks
2. Asks for confirmation
3. Executes tasks sequentially
4. Generates final report

---

### ⏯️ **Step 7: Controlling Running Simulations**

**Start a Background Simulation:**

```bash
proxima run --backend qiskit "10-qubit circuit" &
```

**List Running Simulations:**

```bash
proxima session list
```

Output: `ID: abc123 | Status: RUNNING | Backend: qiskit`

**Pause a Simulation:**

```bash
proxima session pause abc123
```

**Resume Later:**

```bash
proxima session resume abc123
```

**Stop Completely:**

```bash
proxima session abort abc123
```

**Rollback (Undo):**

```bash
proxima session rollback abc123
```
💡 Like Ctrl+Z for quantum simulations!

---

### 🤖 **Step 8: Using AI Explanations (LLM)**

**What is LLM?**
- AI that explains quantum results in plain English
- Two options: Local (free) or Remote (powerful)

**Option 1: Local AI (Ollama - Free & Private)**

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Download model: `ollama pull llama2`
3. Run with AI:
   ```bash
   proxima run --llm ollama --backend cirq "bell state"
   ```

**What AI Explains:**
- Circuit behavior in simple terms
- Why you got specific results
- Optimization suggestions
- Real-world applications

**Option 2: Remote AI (OpenAI/Anthropic)**

1. Get API key from [OpenAI](https://openai.com)
2. Set key:
   ```bash
   export OPENAI_API_KEY="your-key-here"  # Mac/Linux
   set OPENAI_API_KEY=your-key-here       # Windows
   ```
3. Run:
   ```bash
   proxima run --llm openai --backend cirq "teleportation"
   ```

⚠️ **Privacy**: Proxima asks permission before sending data to remote AI

---

### 💾 **Step 9: Saving and Exporting Results**

**Export Formats:**

**JSON (for programmers):**
```bash
proxima run --backend cirq "bell state" --export json --output results.json
```

**CSV (opens in Excel):**
```bash
proxima run --backend cirq "bell state" --export csv --output data.csv
```

**HTML (opens in browser):**
```bash
proxima run --backend cirq "bell state" --export html --output report.html
```

**Excel (multiple sheets):**
```bash
proxima run --backend cirq "bell state" --export xlsx --output analysis.xlsx
```

**View Past Results:**

```bash
# List all past runs
proxima history list

# See specific run details
proxima history show abc123

# Export entire history
proxima history export --format json --output history.json
```

---

### ⚙️ **Step 10: Customizing Settings**

**View Current Settings:**

```bash
proxima config show
```

**Common Customizations:**

**Set Default Backend:**
```bash
proxima config set backends.default_backend cirq
```

**Adjust Memory Warnings:**
```bash
proxima config set resources.memory_threshold 0.7
```
- Lower = more cautious (old computers)
- Higher = more aggressive (powerful machines)

**Change Auto-Approval:**
```bash
proxima config set general.auto_approve false
```
- `true` = Always ask (safer)
- `false` = Auto-run (faster)

**Reset to Defaults:**
```bash
proxima config reset
```

**Config File Locations:**
- User: `~/.proxima/config.yaml`
- Project: `./proxima.yaml`

---

### 🔬 **Step 11: Advanced Features**

**Dry Run (Test Without Executing):**

```bash
proxima run --dry-run --backend cirq "bell state"
```
💡 See what would happen without running

**Force Mode (Skip Confirmations):**

```bash
proxima run --force --backend cirq "circuit"
```
⚠️ Use carefully - no safety prompts!

**Verbose Logging:**

```bash
proxima -vvv run --backend cirq "bell state"
```
- `-v` = Some details
- `-vv` = More details
- `-vvv` = Everything

**Quiet Mode:**

```bash
proxima --quiet run --backend cirq "circuit"
```

**Custom Shots:**

```bash
proxima run --backend cirq --shots 8192 "bell state"
```
- Quick tests: 256-512
- Normal: 1024-2048
- High accuracy: 4096-8192

---

### 📊 **Step 12: Resource Monitoring**

**Real-Time Monitoring:**

```bash
proxima run --backend qiskit "large circuit" --monitor
```

**Display:**
```
💾 Memory: 45% [█████-----]
🖥️ CPU: 78% [███████---]
⏱️ Time: 5.2s
```

**Automatic Warnings:**
- 60% memory: ⚠️ Warning
- 80% memory: ⚠️ Strong warning
- 95% memory: 🛑 Critical (blocked)

**If Warnings Appear:**
1. Close other programs
2. Use smaller circuits
3. Reduce shots
4. Try different backend

---

### 🔧 **Step 13: Troubleshooting**

**Problem: "Backend not found"**

Solution:
```bash
pip install cirq              # For Cirq
pip install qiskit qiskit-aer # For Qiskit
```

**Problem: "Out of memory"**

Solutions:
- Reduce circuit size
- Lower shots: `--shots 256`
- Close other programs
- Adjust threshold: `proxima config set resources.memory_threshold 0.7`

**Problem: "Permission denied"**

Solution:
```bash
chmod 755 ~/.proxima  # Mac/Linux
```

**Problem: Simulation stuck**

Solution:
```bash
proxima session list
proxima session abort abc123
```

**Check Logs:**
```bash
cat ~/.proxima/logs/proxima.log  # Mac/Linux
type %USERPROFILE%\.proxima\logs\proxima.log  # Windows
```

---

### 📖 **Step 14: Command Reference**

**Core Commands:**
```bash
proxima init            # Setup
proxima version         # Version info
proxima --help          # All commands
```

**Backends:**
```bash
proxima backends list   # List all
proxima backends info NAME  # Details
proxima backends test NAME  # Test
```

**Running:**
```bash
proxima run [OPTIONS] "DESCRIPTION"
  --backend NAME        # cirq/qiskit/lret/auto
  --shots NUMBER        # Measurements (default: 1024)
  --export FORMAT       # json/csv/html/xlsx
  --output FILE         # Output filename
  --llm NAME            # ollama/openai/anthropic
  --monitor             # Show resources
  --dry-run             # Test only
  --force               # Skip prompts
```

**Comparison:**
```bash
proxima compare --backends cirq,qiskit "DESCRIPTION"
  --shots NUMBER
  --export FORMAT
```

**Configuration:**
```bash
proxima config show         # View all
proxima config get KEY      # Get one
proxima config set KEY VAL  # Set one
proxima config reset        # Reset
```

**History:**
```bash
proxima history list       # All runs
proxima history show ID    # One run
proxima history export     # Save
proxima history clear      # Delete
```

**Sessions:**
```bash
proxima session list       # Active
proxima session status ID  # Check
proxima session pause ID   # Pause
proxima session resume ID  # Resume
proxima session abort ID   # Stop
```

**UI & Agent:**
```bash
proxima ui                    # Launch TUI
proxima agent run FILE        # Run tasks
proxima agent validate FILE   # Check file
```

---

### 🎯 **Step 15: Real-World Examples**

**Beginner: Quick Test**

```bash
proxima backends list
proxima run --backend auto "bell state"
```

**Intermediate: Backend Comparison**

```bash
proxima compare --backends cirq,qiskit "ghz state" --shots 2048
proxima run --backend cirq "ghz state" --export html --output report.html
```

**Advanced: Full Experiment**

```bash
# Create experiment.md with tasks
proxima agent preview experiment.md
proxima agent run experiment.md --llm ollama
proxima history export --format xlsx --output results.xlsx
```

**Production: Pipeline**

```bash
# Test first
proxima run --dry-run --backend cirq "circuit"

# Run with monitoring
proxima run --backend cirq "circuit" --monitor --shots 8192

# Export results
proxima run --backend cirq "circuit" --export xlsx --output data.xlsx

# Review in TUI
proxima ui
```

---

### 💡 **Quick Success Tips**

1. **Start small** - Begin with 2-3 qubit circuits
2. **Use auto mode** - `--backend auto` lets Proxima choose
3. **Save work** - Always use `--export` for important results
4. **Monitor resources** - Add `--monitor` for long runs
5. **Try dry-run** - Use `--dry-run` to test commands first
6. **Use AI** - `--llm ollama` explains results for free
7. **Keep history** - Don't clear history, it tracks progress
8. **Use agent files** - Write task lists for complex experiments
9. **Explore TUI** - Visual interface makes everything easier
10. **Ask for help** - Use `proxima COMMAND --help` anytime

---

### 🆘 **Getting Help**

**Built-in:**
```bash
proxima --help          # All commands
proxima run --help      # Specific command
proxima ui              # Visual guides
```

**Documentation:**
- Full docs: `docs/` folder
- Local: Run `mkdocs serve` → http://localhost:8000

**Community:**
- [GitHub Issues](https://github.com/prthmmkhija1/Pseudo-Proxima/issues)
- Examples: `examples/` directory
- Discussions: GitHub Discussions

**Example Files:**
- `examples/basic_circuits.md`
- `examples/comparison_workflows.md`
- `examples/advanced_features.md`

---

## 🎉 **You're Ready!**

You now have everything needed to use Proxima effectively:
- ✅ Installation completed
- ✅ Backend concepts understood
- ✅ Basic commands learned
- ✅ Advanced features available
- ✅ Troubleshooting guide ready

**Your First Command:**
```bash
proxima run --backend auto "bell state"
```

Happy quantum computing! 🚀🔬

