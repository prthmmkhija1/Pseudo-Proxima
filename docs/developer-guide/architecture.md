# Architecture

This document describes the internal architecture of the Proxima quantum simulation agent.

## Overview

Proxima follows a layered architecture with clear separation of concerns:

```

                     User Interface Layer                     
            
     CLI         TUI         API         SDK        
            

                              
                              

                    Intelligence Layer                        
            
    LLM Router      Agent Core      Insights          
            

                              
                              

                      Core Layer                              
                  
   Execution      Circuit        Result               
    Engine        Builder       Handler               
                  
                  
    Control       Session        Config               
    Manager       Manager        System               
                  

                              
                              

                     Backend Layer                            
            
     LRET        Cirq       Qiskit      QuEST       
            
                                     
     qsim     cuQuantum                                  
                                     

                              
                              

                     Data Layer                               
            
     Session         Results         Metrics          
     Storage          Cache          Database         
            

```

## Layer Descriptions

### 1. User Interface Layer

The UI layer provides multiple ways to interact with Proxima:

#### CLI (Command Line Interface)

```python
# proxima/cli/main.py
class ProximaCLI:
    """Main CLI entry point using Click."""
    
    def __init__(self):
        self.config = load_config()
        self.executor = ExecutionEngine()
```

Key commands:
- `proxima run` - Execute circuits
- `proxima compare` - Compare backends
- `proxima config` - Manage configuration
- `proxima agent` - LLM agent interaction

#### TUI (Terminal User Interface)

```python
# proxima/tui/app.py
class ProximaTUI:
    """Rich-based terminal interface."""
    
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
```

Features:
- Interactive circuit execution
- Real-time progress visualization
- Results exploration

#### API (REST/WebSocket)

```python
# proxima/api/server.py
class ProximaAPI:
    """FastAPI-based REST server."""
    
    def __init__(self):
        self.app = FastAPI()
        self.setup_routes()
```

Endpoints:
- `POST /execute` - Execute circuit
- `GET /results/{id}` - Get results
- `POST /compare` - Compare backends
- `WebSocket /stream` - Real-time execution

#### SDK (Python Library)

```python
# proxima/sdk/__init__.py
from proxima.core.execution import execute_circuit
from proxima.core.circuit import Circuit

# Direct Python usage
circuit = Circuit(2)
result = execute_circuit(circuit, backend="cirq")
```

### 2. Intelligence Layer

The intelligence layer adds AI-powered features:

#### LLM Router

```python
# proxima/intelligence/llm_router.py
class LLMRouter:
    """Routes queries to appropriate LLM providers."""
    
    def __init__(self, config: dict):
        self.providers = self._init_providers(config)
        self.fallback = LocalModel()
    
    def route(self, query: str) -> LLMResponse:
        """Route query to best available LLM."""
        intent = self.classify_intent(query)
        provider = self.select_provider(intent)
        return provider.generate(query)
```

#### Agent Core

```python
# proxima/intelligence/agent.py
class ProximaAgent:
    """Core agent for natural language interaction."""
    
    def __init__(self, agent_file: str):
        self.config = parse_agent_file(agent_file)
        self.router = LLMRouter(self.config)
        self.executor = ExecutionEngine()
    
    def process(self, query: str) -> AgentResponse:
        """Process natural language query."""
        # Parse intent
        intent = self.router.parse_intent(query)
        
        # Execute if simulation request
        if intent.is_simulation:
            result = self.executor.execute(intent.circuit)
            return self.format_response(result)
        
        # Otherwise, generate response
        return self.router.generate_response(query)
```

#### Insights Engine

```python
# proxima/intelligence/insights.py
class InsightsEngine:
    """Generates insights from execution results."""
    
    def analyze(self, result: ExecutionResult) -> Insights:
        """Analyze result and generate insights."""
        return Insights(
            performance=self._analyze_performance(result),
            accuracy=self._analyze_accuracy(result),
            recommendations=self._generate_recommendations(result)
        )
```

### 3. Core Layer

The core layer handles all simulation logic:

#### Execution Engine

```python
# proxima/core/execution.py
class ExecutionEngine:
    """Central execution coordinator."""
    
    def __init__(self):
        self.registry = BackendRegistry()
        self.control = ControlManager()
        self.monitor = ResourceMonitor()
    
    def execute(
        self,
        circuit: Circuit,
        backend: str = None,
        **options
    ) -> ExecutionResult:
        """Execute circuit with full lifecycle management."""
        # Select backend
        if backend is None:
            backend = self.auto_select_backend(circuit)
        
        adapter = self.registry.get(backend)
        
        # Pre-execution checks
        self.control.check_consent(circuit)
        self.monitor.check_resources(circuit)
        
        # Execute with monitoring
        with self.control.execution_context():
            result = adapter.execute(circuit, **options)
        
        return result
```

#### Circuit Builder

```python
# proxima/core/circuit.py
class Circuit:
    """Quantum circuit representation."""
    
    def __init__(self, qubit_count: int):
        self.qubit_count = qubit_count
        self.gates = []
        self.measurements = []
    
    def h(self, qubit: int) -> "Circuit":
        """Add Hadamard gate."""
        self.gates.append(Gate("H", [qubit]))
        return self
    
    def cnot(self, control: int, target: int) -> "Circuit":
        """Add CNOT gate."""
        self.gates.append(Gate("CNOT", [control, target]))
        return self
```

#### Result Handler

```python
# proxima/core/result.py
class ExecutionResult:
    """Standardized execution result."""
    
    def __init__(self, **kwargs):
        self.backend = kwargs.get("backend")
        self.counts = kwargs.get("counts", {})
        self.execution_time_ms = kwargs.get("execution_time_ms")
        self.qubit_count = kwargs.get("qubit_count")
        self.shot_count = kwargs.get("shot_count")
        self.metadata = kwargs.get("metadata", {})
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
```

#### Control Manager

```python
# proxima/core/control.py
class ControlManager:
    """Manages execution control operations."""
    
    def __init__(self):
        self.state = ExecutionState.IDLE
        self.checkpoint = None
    
    def pause(self) -> bool:
        """Pause current execution."""
        if self.state == ExecutionState.RUNNING:
            self.state = ExecutionState.PAUSED
            return True
        return False
    
    def resume(self) -> bool:
        """Resume paused execution."""
        if self.state == ExecutionState.PAUSED:
            self.state = ExecutionState.RUNNING
            return True
        return False
    
    def abort(self) -> bool:
        """Abort current execution."""
        self.state = ExecutionState.ABORTED
        return True
    
    def rollback(self) -> bool:
        """Rollback to last checkpoint."""
        if self.checkpoint:
            self.restore_checkpoint(self.checkpoint)
            return True
        return False
```

#### Session Manager

```python
# proxima/data/session.py
class SessionManager:
    """Manages execution sessions."""
    
    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or ".proxima/sessions"
        self.current_session = None
    
    def create_session(self) -> Session:
        """Create new execution session."""
        session = Session(
            id=generate_session_id(),
            created_at=datetime.now(),
            status="active"
        )
        self.current_session = session
        return session
    
    def save_result(self, result: ExecutionResult) -> None:
        """Save result to current session."""
        if self.current_session:
            self.current_session.add_result(result)
            self._persist_session()
```

#### Configuration System

```python
# proxima/core/config.py
class ConfigManager:
    """Hierarchical configuration management."""
    
    PRIORITY = [
        "cli_args",
        "env_vars",
        "project_config",
        "user_config",
        "defaults"
    ]
    
    def get(self, key: str, default=None):
        """Get config value with priority resolution."""
        for source in self.PRIORITY:
            value = self._get_from_source(source, key)
            if value is not None:
                return value
        return default
```

### 4. Backend Layer

The backend layer abstracts quantum simulators:

#### Backend Registry

```python
# proxima/backends/registry.py
class BackendRegistry:
    """Registry of available backends."""
    
    BACKENDS = {
        "lret": LRETAdapter,
        "cirq": CirqAdapter,
        "qiskit": QiskitAdapter,
        "quest": QuESTAdapter,
        "qsim": QsimAdapter,
        "cuquantum": CuQuantumAdapter,
    }
    
    def get(self, name: str) -> BaseBackendAdapter:
        """Get backend adapter by name."""
        if name not in self.BACKENDS:
            raise ValueError(f"Unknown backend: {name}")
        return self.BACKENDS[name]()
    
    def list_available(self) -> list:
        """List backends that are installed."""
        available = []
        for name, adapter_class in self.BACKENDS.items():
            if adapter_class.is_available():
                available.append(name)
        return available
```

#### Base Adapter

```python
# proxima/backends/base.py
class BaseBackendAdapter(ABC):
    """Abstract base for all backend adapters."""
    
    @abstractmethod
    def execute(self, circuit: Circuit, **options) -> ExecutionResult:
        """Execute quantum circuit."""
        pass
    
    @abstractmethod
    def supports_simulator(self, sim_type: str) -> bool:
        """Check simulator type support."""
        pass
    
    @abstractmethod
    def estimate_resources(self, circuit: Circuit) -> dict:
        """Estimate required resources."""
        pass
```

### 5. Data Layer

The data layer handles persistence:

#### Session Storage

```python
# proxima/data/storage.py
class SessionStorage:
    """SQLite-based session storage."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def save_session(self, session: Session) -> None:
        """Persist session to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?)",
                (session.id, session.created_at, session.status, session.to_json())
            )
```

#### Results Cache

```python
# proxima/data/cache.py
class ResultsCache:
    """LRU cache for execution results."""
    
    def __init__(self, max_size: int = 100):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key: str) -> ExecutionResult:
        """Get cached result."""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, result: ExecutionResult) -> None:
        """Cache result."""
        self.cache[key] = result
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
```

## Data Flow

### Execution Flow

```
User Request
    
    

  Parse Input    
  (CLI/API/TUI)  

    
    

 Load Config     
 (proxima.yaml)  

    
    

 Select Backend  
 (auto or manual)

    
    

 Check Consent   
 (if required)   

    
    

 Check Resources 
 (memory/qubits) 

    
    

 Convert Circuit 
 (to backend fmt)

    
    

 Execute         
 (backend sim)   

    
    

 Normalize Result
 (standard fmt)  

    
    

 Generate        
 Insights        

    
    

 Return Result   
 (JSON/display)  

```

## Extension Points

Proxima is designed for extensibility:

### 1. Custom Backends

Implement `BaseBackendAdapter` to add new simulators.

### 2. Custom Gates

Register new gate definitions in the gate registry.

### 3. Custom Normalizers

Implement `BaseResultNormalizer` for backend-specific result handling.

### 4. Custom Insights

Add analysis modules to the insights engine.

### 5. Plugins

Use the plugin system for external extensions:

```python
# proxima/plugins/manager.py
class PluginManager:
    """Manages Proxima plugins."""
    
    def load_plugin(self, path: str) -> Plugin:
        """Load plugin from path."""
        spec = importlib.util.spec_from_file_location("plugin", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.plugin
```

## Configuration Files

### proxima.yaml

Main configuration file:

```yaml
defaults:
  backend: cirq
  shots: 1024
  
backends:
  cirq:
    simulator: state_vector
  qiskit:
    optimization_level: 2
    
safety:
  require_consent: true
  max_qubits: 30
  
intelligence:
  llm_router:
    enabled: true
```

### proxima_agent.md

Agent configuration for LLM interaction.

### .proxima/

Local state directory containing sessions and cache.

## Thread Safety

Proxima components are designed for thread safety:

```python
class ExecutionEngine:
    def __init__(self):
        self._lock = threading.RLock()
    
    def execute(self, circuit, **options):
        with self._lock:
            return self._execute_internal(circuit, **options)
```

## Error Handling

Hierarchical exception system:

```python
# proxima/core/exceptions.py
class ProximaError(Exception):
    """Base exception for all Proxima errors."""
    pass

class BackendError(ProximaError):
    """Backend-related errors."""
    pass

class CircuitError(ProximaError):
    """Circuit-related errors."""
    pass

class ResourceError(ProximaError):
    """Resource limit errors."""
    pass

class ConsentError(ProximaError):
    """Consent-related errors."""
    pass
```
