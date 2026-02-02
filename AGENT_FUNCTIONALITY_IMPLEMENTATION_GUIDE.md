# Agent Functionality Implementation Guide

**Proxima Quantum Computing Platform - Advanced AI Agent System**  
**Version:** 2.0  
**Date:** February 2, 2026  
**Target Implementation:** GPT-5.1 Codex Max, GPT-5.2 Codex Max, Claude Opus 4.5

---

## Table of Contents

### Part A: Next-Generation Features (NEW)

1. [Feature 1: MCP (Model Context Protocol) Integration](#feature-1-mcp-model-context-protocol-integration)
2. [Feature 2: Agent Workspace Memory System](#feature-2-agent-workspace-memory-system)
3. [Feature 3: Multi-Agent Collaboration Framework](#feature-3-multi-agent-collaboration-framework)
4. [Feature 4: Visual Circuit Builder Integration](#feature-4-visual-circuit-builder-integration)
5. [Feature 5: Real-Time Quantum State Visualization](#feature-5-real-time-quantum-state-visualization)
6. [Feature 6: Plugin Marketplace & Extension System](#feature-6-plugin-marketplace--extension-system)

### Part B: Existing Implementation Phases

7. [Phase 1: Real-Time Execution & Results Monitoring](#phase-1-real-time-execution--results-monitoring)
8. [Phase 2: Agent UI/UX Enhancements](#phase-2-agent-uiux-enhancements)
9. [Phase 3: Terminal Integration & Multi-Process Management](#phase-3-terminal-integration--multi-process-management)
10. [Phase 4: Backend Building & Compilation System](#phase-4-backend-building--compilation-system)
11. [Phase 5: File System Operations & Administrative Access](#phase-5-file-system-operations--administrative-access)
12. [Phase 6: Natural Language Planning & Execution](#phase-6-natural-language-planning--execution)
13. [Phase 7: Git Operations Integration](#phase-7-git-operations-integration)
14. [Phase 8: Backend Code Modification with Safety](#phase-8-backend-code-modification-with-safety)
15. [Phase 9: Agent Statistics & Telemetry System](#phase-9-agent-statistics--telemetry-system)
16. [Phase 10: Integration & Testing](#phase-10-integration--testing)

---

## Executive Summary

This guide provides a comprehensive, phase-by-phase implementation plan for transforming Proxima's AI agent system into a next-generation, production-grade platform with cutting-edge AI capabilities. This version 2.0 guide includes **six new advanced features** that will make Proxima more intelligent, extensible, and user-friendly, followed by the existing implementation phases.

**New Features Overview:**

1. **MCP Integration**: Universal tool and data source connectivity
2. **Workspace Memory**: Long-term context and learning capabilities
3. **Multi-Agent Collaboration**: Coordinated agent teamwork for complex tasks
4. **Visual Circuit Builder**: Drag-and-drop quantum circuit creation
5. **Real-Time Visualization**: Live quantum state rendering and animation
6. **Plugin Marketplace**: Community extensions and custom tool sharing

The system will support three AI agents (AI Assistant, AI Analysis, AI Thinking) with full terminal access, backend compilation, file system manipulation, git operations, and code modification capabilities.

**Key Technologies:**

- **UI Framework:** Textual (Python TUI framework)
- **LLM Integration:** Multi-provider support (OpenAI, Anthropic, Google AI, xAI, Ollama, etc.)
- **Terminal Management:** asyncio-based subprocess management with PTY (pseudo-terminal) support
- **Real-time Updates:** Event-driven architecture with reactive data streams
- **Safety Systems:** Checkpoint-based rollback, consent management, audit logging

**Reference Implementation:** Charm's Crush AI Agent (https://github.com/charmbracelet/crush)

---

## PART A: NEXT-GENERATION FEATURES

---

## Feature 1: MCP (Model Context Protocol) Integration

**Objective:** Integrate the Model Context Protocol to enable Proxima agents to connect to external data sources, tools, and services through a standardized, extensible interface.

**Value Proposition:**

- **Universal Connectivity**: Access quantum cloud services (IBM Quantum, AWS Braket, Azure Quantum) through MCP servers
- **Extensibility**: Users can add custom MCP servers without modifying Proxima core
- **Intelligent Context**: MCP manages context windows and data exposure to LLMs
- **Tool Composability**: Chain multiple tools together for complex workflows
- **Provider Agnostic**: Works with any LLM (OpenAI, Anthropic, local models)

### Step 1.1: Understand MCP Architecture

**Technical Background:**

- MCP is an open protocol created by Anthropic for connecting AI agents to external systems
- Uses JSON-RPC 2.0 over stdio or Server-Sent Events (SSE) for communication
- Three core concepts: Resources (data/content), Tools (functions), Prompts (templates)
- Servers expose capabilities; clients (Proxima agents) consume them

**Key MCP Concepts:**

1. **Resources**: Read-only data sources (files, database records, API responses)
2. **Tools**: Executable functions (query database, call API, transform data)
3. **Prompts**: Template messages with variables for common workflows
4. **Transports**: Communication layer (stdio for local, SSE for remote)

**MCP Message Flow:**

```
Proxima Agent → MCP Client → [JSON-RPC] → MCP Server → External Service
                                  ↓
                         Response with results
                                  ↓
Proxima Agent ← MCP Client ← [JSON-RPC] ← MCP Server
```

**Official Specification**: Model Context Protocol uses JSON-RPC 2.0 with methods: `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, `prompts/get`

### Step 1.2: Install MCP Dependencies

**Technical Approach:**

- Use official MCP Python SDK from Anthropic
- Install MCP client library for connecting to servers
- Set up stdio transport for local servers
- Configure SSE transport for remote servers

**Required Libraries:**

- `mcp` - Official Model Context Protocol SDK (pip install mcp)
- `httpx` - HTTP client for SSE transport
- `asyncio` - Async I/O for non-blocking MCP operations
- `typing_extensions` - Type hints for MCP interfaces

**Installation Commands:**

```bash
pip install mcp>=0.9.0
pip install httpx>=0.25.0
pip install python-jsonrpc-server>=0.4.0
```

**Dependency Location:**

- Add to `requirements.txt` under AI/LLM section
- Add to `pyproject.toml` in `[project.optional-dependencies]` under `mcp` extra
- Document in `docs/requirements.md`

### Step 1.3: Design MCP Client Architecture

**Technical Approach:**

- Create `MCPClientManager` class to manage multiple MCP server connections
- Implement server discovery mechanism (read from config file)
- Design connection pooling for efficient server reuse
- Add health checking for server availability
- Implement automatic reconnection on failure

**Architecture Components:**

1. **MCPClientManager** (`src/proxima/agent/mcp/client_manager.py`):
   - Manages lifecycle of MCP connections
   - Registers MCP servers from configuration
   - Provides interface for tool discovery and execution
   - Handles connection pooling and cleanup

2. **MCPServerConfig** (dataclass):
   - Server name and description
   - Transport type (stdio or SSE)
   - Command to start server (for stdio)
   - URL for remote servers (for SSE)
   - Environment variables
   - Timeout configuration

3. **MCPConnection** (wrapper class):
   - Single connection to one MCP server
   - Manages JSON-RPC communication
   - Implements retry logic
   - Tracks connection state

**Configuration File** (`configs/mcp_servers.yaml`):

```yaml
mcp_servers:
  # Quantum backend servers
  ibm_quantum:
    name: "IBM Quantum Cloud"
    transport: "sse"
    url: "https://quantum-computing.ibm.com/mcp"
    requires_auth: true
    env:
      IBM_QUANTUM_TOKEN: "${IBM_QUANTUM_TOKEN}"

  aws_braket:
    name: "AWS Braket"
    transport: "sse"
    url: "https://braket.amazonaws.com/mcp"
    requires_auth: true
    env:
      AWS_ACCESS_KEY_ID: "${AWS_ACCESS_KEY_ID}"
      AWS_SECRET_ACCESS_KEY: "${AWS_SECRET_ACCESS_KEY}"

  # File system server (local)
  filesystem:
    name: "File System Access"
    transport: "stdio"
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "${WORKSPACE_DIR}"]
    auto_start: true

  # Database server (local)
  sqlite:
    name: "SQLite Database"
    transport: "stdio"
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-sqlite", "${DATABASE_PATH}"]
    auto_start: true

  # Git server (local)
  git:
    name: "Git Operations"
    transport: "stdio"
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-git"]
    auto_start: true
```

**Key Design Decisions:**

- Use factory pattern for creating different transport types
- Implement circuit breaker pattern for failed connections
- Cache server capabilities to reduce roundtrips
- Use weak references to prevent memory leaks
- Support both sync and async operation modes

### Step 1.4: Implement MCP Client Manager

**Technical Approach:**

- Load MCP server configurations from YAML file
- Initialize connections to enabled servers on startup
- Expose unified API for tool discovery and execution
- Handle transport-specific details transparently
- Implement connection lifecycle management

**Core Implementation** (`src/proxima/agent/mcp/client_manager.py`):

**Class Structure:**

```python
class MCPClientManager:
    def __init__(self, config_path: Path):
        # Initialize with server configuration

    async def connect_all(self) -> None:
        # Connect to all enabled servers

    async def disconnect_all(self) -> None:
        # Gracefully disconnect from servers

    async def list_available_tools(self) -> List[MCPTool]:
        # Aggregate tools from all connected servers

    async def list_available_resources(self) -> List[MCPResource]:
        # Aggregate resources from all servers

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> MCPToolResult:
        # Execute a tool on specific server

    async def read_resource(
        self,
        server_name: str,
        resource_uri: str
    ) -> MCPResourceContent:
        # Read resource content

    async def get_prompt(
        self,
        server_name: str,
        prompt_name: str,
        arguments: Dict[str, str]
    ) -> MCPPromptMessages:
        # Get prompt template with substituted variables

    def get_server_status(self, server_name: str) -> ServerStatus:
        # Check if server is connected and healthy
```

**Connection Lifecycle:**

1. **Initialization**: Parse configuration, validate settings
2. **Connection**: Establish transport (stdio process or SSE connection)
3. **Capability Exchange**: Call `initialize` to get server capabilities
4. **Active Use**: Handle tool calls, resource reads
5. **Disconnection**: Send shutdown message, close transport

**Error Handling:**

- Timeout on unresponsive servers (30 second default)
- Retry failed connections with exponential backoff
- Fallback to cached capabilities if server temporarily unavailable
- Log all errors to `logs/mcp_errors.log`

**Concurrency:**

- Use `asyncio.Lock` per server to prevent concurrent modifications
- Allow parallel tool calls to different servers
- Queue requests during server reconnection

### Step 1.5: Create Stdio Transport Implementation

**Technical Approach:**

- Spawn MCP server as subprocess using `asyncio.create_subprocess_exec`
- Use stdin/stdout for JSON-RPC communication
- Implement message framing (newline-delimited JSON)
- Handle process lifecycle (start, monitor, terminate)
- Capture stderr for error diagnostics

**Implementation** (`src/proxima/agent/mcp/transports/stdio.py`):

**Key Functions:**

```python
class StdioTransport:
    async def start_server(
        self,
        command: str,
        args: List[str],
        env: Dict[str, str]
    ) -> None:
        # Start MCP server process using asyncio
        # Set environment variables
        # Configure stdin/stdout/stderr pipes

    async def send_request(
        self,
        method: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Format JSON-RPC request
        # Write to stdin with newline
        # Read response from stdout
        # Parse and validate JSON-RPC response

    async def close(self) -> None:
        # Send graceful shutdown
        # Wait for process termination (max 5 seconds)
        # Force kill if necessary
```

**Message Format:**

- Each message is single line JSON
- Request: `{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}\n`
- Response: `{"jsonrpc": "2.0", "id": 1, "result": {...}}\n`
- Notification: `{"jsonrpc": "2.0", "method": "notification", "params": {...}}\n`

**Process Management:**

- Track process ID (PID) for monitoring
- Check process.returncode to detect crashes
- Restart crashed servers automatically (max 3 attempts)
- Kill subprocess tree on cleanup (parent + children)

**Stderr Handling:**

- Create async task to read stderr continuously
- Parse for error messages and warnings
- Log to separate file `logs/mcp_servers/<server_name>.log`
- Display critical errors in UI

### Step 1.6: Create SSE Transport Implementation

**Technical Approach:**

- Use SSE (Server-Sent Events) for remote MCP servers
- Establish long-lived HTTP connection for server→client events
- Use regular HTTP POST for client→server requests
- Implement reconnection logic for dropped connections
- Handle authentication headers

**Implementation** (`src/proxima/agent/mcp/transports/sse.py`):

**Key Functions:**

```python
class SSETransport:
    async def connect(
        self,
        url: str,
        headers: Dict[str, str]
    ) -> None:
        # Open SSE connection using httpx
        # Start background task to read events
        # Handle connection establishment

    async def send_request(
        self,
        method: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Send HTTP POST to request endpoint
        # Wait for response via SSE event or direct response
        # Match request ID with response

    async def _read_events(self) -> None:
        # Background task to read SSE stream
        # Parse event data
        # Route to appropriate handler

    async def close(self) -> None:
        # Close SSE connection
        # Cancel background tasks
```

**SSE Event Types:**

- `message`: Server→client messages (responses, notifications)
- `error`: Error events from server
- `ping`: Keep-alive to prevent timeout

**Authentication:**

- Support Bearer token in Authorization header
- Support API key in custom headers (X-API-Key)
- Support OAuth 2.0 token refresh
- Store credentials securely (OS keyring via `keyring` library)

**Reconnection Strategy:**

- Detect connection loss via timeout or error
- Wait exponential backoff: 1s, 2s, 4s, 8s, 16s (max)
- Preserve request queue during reconnection
- Fail after 5 reconnection attempts

### Step 1.7: Integrate MCP Tools with Agent Tool Registry

**Technical Approach:**

- Discover MCP tools from all connected servers
- Convert MCP tool schema to Proxima ToolDefinition format
- Register MCP tools in existing tool registry
- Route tool execution requests to appropriate MCP server
- Handle tool results and errors uniformly

**Integration Point** (`src/proxima/agent/tools.py`):

**Add MCP Tool Discovery:**

```python
class ToolRegistry:
    def __init__(self):
        self._mcp_manager: Optional[MCPClientManager] = None

    async def register_mcp_servers(
        self,
        mcp_manager: MCPClientManager
    ) -> None:
        # Store reference to MCP manager
        # Discover all available MCP tools
        # Convert to ToolDefinition objects
        # Register in tool registry with "mcp_" prefix

    async def _convert_mcp_tool(
        self,
        mcp_tool: MCPTool,
        server_name: str
    ) -> ToolDefinition:
        # Map MCP tool schema to ToolDefinition
        # Set category based on tool type
        # Mark as requiring network permission
        # Store server reference in metadata
```

**Tool Naming Convention:**

- Prefix with server name: `ibm_quantum:submit_job`
- Keep original tool name for disambiguation
- Add server description to tool description

**Schema Conversion:**

- MCP `inputSchema` (JSON Schema) → ToolDefinition `parameters`
- MCP string/number/boolean types map directly
- MCP object types become nested parameter groups
- MCP array types supported via array ToolParameter

**Execution Routing:**

```python
async def execute_tool(
    self,
    tool_name: str,
    arguments: Dict[str, Any]
) -> ToolResult:
    if tool_name.startswith("mcp_"):
        # Parse server name from tool name
        # Route to MCPClientManager
        # Convert MCP result to ToolResult
    else:
        # Execute built-in tool
```

### Step 1.8: Add MCP Resource Access

**Technical Approach:**

- List available resources from MCP servers
- Allow agents to read resource content
- Cache resource data to reduce network calls
- Invalidate cache based on resource metadata
- Present resources as tool results

**Implementation** (`src/proxima/agent/mcp/resources.py`):

**Resource Discovery:**

```python
class MCPResourceManager:
    async def list_resources(
        self,
        server_name: Optional[str] = None
    ) -> List[MCPResource]:
        # Get resources from specific server or all servers
        # Filter by resource type (file, database, api)
        # Sort by relevance or name

    async def read_resource(
        self,
        resource_uri: str
    ) -> ResourceContent:
        # Check cache first
        # Read from MCP server if not cached
        # Store in cache with TTL
        # Return content with metadata
```

**Resource Types:**

- `file://...` - File system resources
- `db://...` - Database records
- `api://...` - API responses
- `quantum://...` - Quantum backend information

**Caching Strategy:**

- Use LRU cache with max 100 resources
- TTL based on resource metadata (default 5 minutes)
- Invalidate on explicit refresh request
- Skip cache for large resources (> 1 MB)

**Integration with Agent:**

- Agents can query "what resources are available?"
- Agents can read specific resources by URI
- Resources appear as context in LLM prompts
- Resources can be attached to tool calls

### Step 1.9: Implement MCP Prompt Templates

**Technical Approach:**

- Discover prompt templates from MCP servers
- Allow agents to retrieve prompts with variable substitution
- Use prompts to guide agent behavior
- Store commonly used prompts for quick access

**Implementation** (`src/proxima/agent/mcp/prompts.py`):

**Prompt Discovery:**

```python
class MCPPromptManager:
    async def list_prompts(
        self,
        server_name: Optional[str] = None
    ) -> List[MCPPrompt]:
        # Get prompt templates from servers
        # Include prompt arguments/variables
        # Provide prompt descriptions

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: Dict[str, str]
    ) -> List[PromptMessage]:
        # Retrieve prompt from server
        # Substitute variables in template
        # Return formatted messages
```

**Prompt Usage:**

- Predefined workflows: "analyze_quantum_circuit", "optimize_backend_selection"
- Dynamic system prompts based on context
- User-created prompt templates
- Prompt chaining for multi-step tasks

**Variable Substitution:**

- Replace `{{variable_name}}` with provided values
- Validate required variables are provided
- Support default values for optional variables
- Escape special characters in substituted values

### Step 1.10: Add MCP UI Integration

**Technical Approach:**

- Display MCP server status in settings screen
- Show available tools and resources in agent UI
- Allow users to enable/disable MCP servers
- Provide MCP server configuration editor
- Show MCP operation logs in real-time

**UI Locations:**

1. **Settings Screen** (`src/proxima/tui/screens/settings.py`):
   - New "MCP Servers" section
   - List of configured servers with status indicators
   - Connect/Disconnect buttons per server
   - Configuration editor (YAML or form-based)

2. **Agent UI** (`src/proxima/tui/screens/agent_ai_assistant.py`):
   - Show MCP tools in tool list
   - Display "via MCP: <server>" badge on MCP tools
   - Show MCP operation progress
   - Log MCP calls in agent conversation

3. **Status Bar** (bottom of TUI):
   - MCP indicator: `MCP: 3/5 servers connected`
   - Warning icon if servers disconnected
   - Click to view MCP status details

**Settings UI Elements:**

```
┌─ MCP Servers ────────────────────────────────────┐
│ ✅ filesystem       (File System Access)  [●]    │
│ ✅ git              (Git Operations)      [●]    │
│ ⚠️  ibm_quantum     (IBM Quantum Cloud)   [○]    │
│ ❌ aws_braket       (AWS Braket)          [○]    │
│ ➕ Add New Server...                              │
│                                                   │
│ [Connect All] [Disconnect All] [Reload Config]   │
└───────────────────────────────────────────────────┘

Legend: ✅ Connected  ⚠️ Error  ❌ Disconnected  [●] Enabled  [○] Disabled
```

**Real-time Updates:**

- Use Textual reactive properties for server status
- Update UI when servers connect/disconnect
- Show progress bar during tool execution
- Display errors in notification popups

### Step 1.11: Create Built-in MCP Servers for Proxima

**Technical Approach:**

- Create custom MCP servers specifically for Proxima capabilities
- Expose quantum backends as MCP tools
- Expose Proxima database as MCP resources
- Enable external tools to integrate with Proxima

**Custom MCP Servers:**

1. **Proxima Backends Server** (`src/proxima/agent/mcp/servers/backends_server.py`):
   - Tools: `list_backends`, `execute_circuit`, `compare_backends`
   - Resources: Backend configurations, capability information
   - Prompts: "Select optimal backend", "Compare execution results"

2. **Proxima Results Server** (`src/proxima/agent/mcp/servers/results_server.py`):
   - Tools: `query_results`, `export_results`, `analyze_trends`
   - Resources: Historical execution results, benchmark data
   - Prompts: "Analyze performance trends", "Generate report"

3. **Proxima Workspace Server** (`src/proxima/agent/mcp/servers/workspace_server.py`):
   - Tools: `create_circuit_file`, `load_algorithm`, `save_configuration`
   - Resources: Circuit definitions, algorithm implementations
   - Prompts: "Create new quantum algorithm", "Load existing circuit"

**Server Implementation:**

- Use official MCP Python SDK to build servers
- Implement stdio transport for local use
- Export servers as standalone executables
- Document server capabilities in README

**Server Registration:**

- Add to `configs/mcp_servers.yaml` with `auto_start: true`
- Start automatically when Proxima TUI launches
- Stop gracefully when TUI exits

### Step 1.12: Add MCP Tool Composition

**Technical Approach:**

- Allow agents to chain multiple MCP tools together
- Implement workflow engine for multi-step operations
- Support conditional execution based on tool results
- Provide data transformation between tool steps

**Workflow Definition** (`src/proxima/agent/mcp/workflow.py`):

**Workflow Structure:**

```python
@dataclass
class WorkflowStep:
    tool_name: str  # MCP tool to execute
    arguments: Dict[str, Any]  # Input arguments
    output_mapping: Dict[str, str]  # Map outputs to next step inputs
    condition: Optional[str]  # Execute only if condition true

@dataclass
class Workflow:
    name: str
    description: str
    steps: List[WorkflowStep]

class WorkflowEngine:
    async def execute_workflow(
        self,
        workflow: Workflow,
        initial_context: Dict[str, Any]
    ) -> WorkflowResult:
        # Execute steps in sequence
        # Pass outputs to next step inputs
        # Evaluate conditions
        # Handle errors gracefully
```

**Example Workflow:**

```yaml
workflows:
  quantum_cloud_execution:
    name: "Execute on Quantum Cloud"
    description: "Submit circuit to IBM Quantum and retrieve results"
    steps:
      - tool: "filesystem:read_file"
        arguments:
          path: "{{circuit_file}}"
        output: "circuit_qasm"

      - tool: "ibm_quantum:submit_job"
        arguments:
          qasm: "{{circuit_qasm}}"
          backend: "ibmq_qasm_simulator"
          shots: 1024
        output: "job_id"

      - tool: "ibm_quantum:wait_for_job"
        arguments:
          job_id: "{{job_id}}"
          timeout: 300
        output: "job_result"

      - tool: "proxima:save_results"
        arguments:
          data: "{{job_result}}"
          label: "IBM Quantum Execution"
```

**Workflow Execution:**

- Parse workflow from YAML or JSON
- Validate all tools exist before execution
- Execute steps sequentially (no parallelism initially)
- Store intermediate results in context
- Support variable substitution using `{{variable_name}}`

**Error Handling:**

- Stop on first error (fail-fast)
- Optionally continue on errors (ignore-errors flag)
- Rollback mechanism for reversible operations
- Detailed error reporting with step identification

### Step 1.13: Add MCP Configuration Management

**Technical Approach:**

- Provide UI for editing MCP server configurations
- Validate configuration before saving
- Support environment variable expansion
- Allow importing/exporting configurations
- Store sensitive data securely

**Configuration Editor** (`src/proxima/tui/dialogs/mcp_config.py`):

**UI Features:**

- Form-based editor for adding new servers
- YAML editor for advanced users
- Validation with error highlighting
- Test connection button
- Import/export configurations

**Security:**

- Store API keys in OS keyring (Windows Credential Manager, macOS Keychain, Linux Secret Service)
- Never display full API keys (show `sk-...****...xyz`)
- Prompt for credentials when needed
- Support credential rotation

**Environment Variables:**

- Expand `${VAR_NAME}` in configuration
- Support `.env` file for local development
- Document required environment variables
- Warn if variables undefined

### Step 1.14: Implement MCP Monitoring and Logging

**Technical Approach:**

- Log all MCP operations for debugging
- Track MCP tool execution metrics
- Monitor server health continuously
- Alert on server failures
- Provide MCP-specific troubleshooting

**Logging** (`logs/mcp_operations.log`):

**Log Entries:**

```
[2026-02-02 10:15:23] INFO: Connected to MCP server 'filesystem'
[2026-02-02 10:15:24] INFO: Discovered 5 tools from 'filesystem'
[2026-02-02 10:15:30] DEBUG: Calling tool 'filesystem:read_file' with args {'path': '/workspace/circuit.qasm'}
[2026-02-02 10:15:30] DEBUG: Tool result: 200 bytes, success
[2026-02-02 10:15:45] WARNING: MCP server 'ibm_quantum' not responding (timeout)
[2026-02-02 10:15:50] ERROR: Failed to connect to 'aws_braket': Authentication failed
```

**Metrics to Track:**

- Tool call count per server
- Average tool execution time
- Success/failure rates
- Network latency per server
- Cache hit rates for resources

**Health Monitoring:**

- Periodic ping to check server responsiveness (every 60 seconds)
- Detect hung servers (no response for 30 seconds)
- Auto-restart crashed servers (max 3 attempts)
- Report health status in UI

**Troubleshooting Mode:**

- Enable verbose logging with `--mcp-debug` flag
- Capture full JSON-RPC messages
- Display messages in dedicated log viewer
- Export logs for bug reports

### Step 1.15: Document MCP Integration

**Technical Approach:**

- Create comprehensive MCP documentation
- Provide examples of creating custom MCP servers
- Document built-in Proxima MCP servers
- Add troubleshooting guide
- Create video tutorials

**Documentation Files:**

1. **`docs/agent-guide/mcp-integration.md`**: Complete MCP guide
2. **`docs/agent-guide/mcp-server-development.md`**: Creating custom MCP servers
3. **`docs/agent-guide/mcp-workflows.md`**: Workflow creation guide
4. **`docs/agent-guide/mcp-troubleshooting.md`**: Common issues and solutions

**Example Documentation Sections:**

- What is MCP and why use it?
- Installing and configuring MCP servers
- Using MCP tools in Proxima agents
- Creating custom MCP servers for Proxima
- Security best practices
- Performance optimization
- FAQ and troubleshooting

**Code Examples:**

- Python script to create simple MCP server
- JavaScript/TypeScript examples using MCP SDK
- Configuration examples for popular services
- Workflow YAML templates

---

## Feature 2: Agent Workspace Memory System

**Objective:** Implement a comprehensive long-term memory system that enables Proxima agents to remember past conversations, learn from interactions, build workspace knowledge, and maintain context across sessions.

**Value Proposition:**

- **Session Continuity**: Seamlessly resume work from previous sessions
- **Learning Capability**: Improve recommendations based on historical data
- **Workspace Intelligence**: Understand project structure, patterns, and conventions
- **Personalization**: Adapt to user preferences and working style
- **Efficiency**: Eliminate redundant questions and repeated explanations
- **Knowledge Base**: Build a searchable database of project-specific information

### Step 2.1: Design Memory Architecture

**Technical Background:**

- Multi-tiered memory system inspired by human cognition
- Short-term memory: Current conversation context (in LLM context window)
- Working memory: Recent session data (last 24 hours, in memory)
- Long-term memory: Historical data (persistent storage, indexed database)
- Episodic memory: Specific events and interactions
- Semantic memory: Facts, concepts, and learned patterns

**Memory Hierarchy:**

```
┌─────────────────────────────────────────────┐
│  Short-Term Memory (STM)                    │
│  Current conversation context               │
│  Size: ~8K tokens, Duration: Session       │
└────────────────┬────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│  Working Memory (WM)                        │
│  Recent session data, active tasks          │
│  Size: ~50 items, Duration: 24 hours       │
└────────────────┬────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│  Long-Term Memory (LTM)                     │
│  Persistent storage, full history           │
│  Size: Unlimited, Duration: Forever        │
└─────────────────────────────────────────────┘
```

**Storage Strategy:**

- Use SQLite for structured memory data (conversations, facts, patterns)
- Use vector database (Chroma or FAISS) for semantic search
- Use JSON files for workspace snapshots
- Use file system for large artifacts (circuit files, logs)

**Key Design Principles:**

- Privacy-first: All data stored locally, no cloud sync by default
- Efficient retrieval: Fast semantic search (<100ms)
- Automatic cleanup: Remove irrelevant old memories
- User control: Clear memory, export/import, selective deletion

### Step 2.2: Implement Memory Database Schema

**Technical Approach:**

- Design SQLite schema for memory storage
- Create tables for conversations, facts, workspace state, user preferences
- Add indexes for fast queries
- Implement database migrations for schema updates

**Database Schema** (`src/proxima/agent/memory/schema.sql`):

**Tables:**

1. **conversations** - Chat history

```sql
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    role TEXT NOT NULL,  -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    tokens_used INTEGER DEFAULT 0,
    model TEXT,
    metadata JSON,  -- Tool calls, attachments, etc.
    embedding BLOB,  -- Vector embedding for semantic search
    INDEX idx_session (session_id),
    INDEX idx_timestamp (timestamp)
);
```

2. **workspace_facts** - Learned facts about workspace

```sql
CREATE TABLE workspace_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_type TEXT NOT NULL,  -- 'file_structure', 'pattern', 'preference', 'config'
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,  -- 0.0 to 1.0
    source TEXT,  -- 'user_told', 'inferred', 'observed'
    first_seen REAL NOT NULL,
    last_accessed REAL NOT NULL,
    access_count INTEGER DEFAULT 1,
    embedding BLOB,
    UNIQUE(fact_type, key)
);
```

3. **user_preferences** - User-specific settings and preferences

```sql
CREATE TABLE user_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,  -- 'backend', 'ui', 'agent_behavior'
    preference TEXT NOT NULL,
    value TEXT NOT NULL,
    learned_from TEXT,  -- How preference was learned
    updated_at REAL NOT NULL,
    UNIQUE(category, preference)
);
```

4. **task_history** - Record of completed tasks

```sql
CREATE TABLE task_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_description TEXT NOT NULL,
    task_type TEXT,  -- 'execute', 'build', 'analyze', 'file_operation'
    started_at REAL NOT NULL,
    completed_at REAL,
    success BOOLEAN NOT NULL,
    tool_calls JSON,  -- Array of tool calls made
    error_message TEXT,
    outcome_summary TEXT,
    embedding BLOB
);
```

5. **workspace_snapshots** - Periodic workspace state captures

```sql
CREATE TABLE workspace_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    file_count INTEGER,
    backend_configs JSON,
    recent_activity JSON,
    git_status JSON,
    checksum TEXT  -- For detecting changes
);
```

**Indexes for Performance:**

```sql
CREATE INDEX idx_facts_type ON workspace_facts(fact_type);
CREATE INDEX idx_facts_accessed ON workspace_facts(last_accessed DESC);
CREATE INDEX idx_tasks_type ON task_history(task_type);
CREATE INDEX idx_tasks_success ON task_history(success);
CREATE INDEX idx_prefs_category ON user_preferences(category);
```

**Database Location:**

- Path: `{USER_DATA_DIR}/proxima/memory/agent_memory.db`
- Windows: `%APPDATA%\Proxima\memory\agent_memory.db`
- macOS: `~/Library/Application Support/Proxima/memory/agent_memory.db`
- Linux: `~/.local/share/proxima/memory/agent_memory.db`

### Step 2.3: Implement Vector Embeddings for Semantic Search

**Technical Approach:**

- Generate embeddings for conversation messages and facts
- Use OpenAI embeddings API or local sentence transformers
- Store embeddings in database for similarity search
- Build FAISS index for fast vector search

**Embedding Generation** (`src/proxima/agent/memory/embeddings.py`):

**Embedding Providers:**

```python
class EmbeddingProvider(ABC):
    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        pass

class OpenAIEmbeddingProvider(EmbeddingProvider):
    # Use OpenAI text-embedding-3-small (1536 dimensions)
    # Cost: $0.02 per 1M tokens
    # API endpoint: https://api.openai.com/v1/embeddings

class LocalEmbeddingProvider(EmbeddingProvider):
    # Use sentence-transformers library
    # Model: all-MiniLM-L6-v2 (384 dimensions)
    # Fast, runs locally, no API calls
    # Library: sentence-transformers
```

**Embedding Strategy:**

- Embed user messages for finding similar past questions
- Embed assistant responses for finding relevant knowledge
- Embed workspace facts for contextual retrieval
- Embed task descriptions for finding similar completed tasks

**Vector Search Implementation:**

```python
class VectorSearchEngine:
    def __init__(self, embedding_provider: EmbeddingProvider):
        # Initialize FAISS index (IndexFlatL2 or IndexIVFFlat)
        # Load existing embeddings from database

    async def search_similar(
        self,
        query_text: str,
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[SearchResult]:
        # Generate embedding for query
        # Search FAISS index for nearest neighbors
        # Filter by cosine similarity threshold
        # Return results sorted by relevance
```

**Libraries Required:**

- `sentence-transformers` - Local embedding models
- `faiss-cpu` or `faiss-gpu` - Fast vector search
- `openai` - OpenAI embeddings API (optional)
- `numpy` - Vector operations

**Performance Optimization:**

- Cache embeddings in memory for recent items
- Batch embedding generation (process 100 items at once)
- Use quantized FAISS index for memory efficiency
- Lazy load embeddings only when search requested

### Step 2.4: Build Memory Manager Core

**Technical Approach:**

- Create `MemoryManager` class to coordinate all memory operations
- Implement methods for storing and retrieving memories
- Handle memory consolidation (moving working memory to long-term)
- Provide unified API for agents to access memory

**Core Implementation** (`src/proxima/agent/memory/manager.py`):

**Class Structure:**

```python
class MemoryManager:
    def __init__(
        self,
        db_path: Path,
        embedding_provider: EmbeddingProvider
    ):
        # Initialize database connection
        # Initialize vector search engine
        # Load working memory into RAM

    async def store_conversation_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> None:
        # Store message in database
        # Generate and store embedding
        # Update working memory cache

    async def store_fact(
        self,
        fact_type: str,
        key: str,
        value: str,
        confidence: float = 1.0,
        source: str = "inferred"
    ) -> None:
        # Store or update workspace fact
        # Generate embedding for semantic search
        # Update confidence if fact already exists

    async def recall_similar_conversations(
        self,
        query: str,
        limit: int = 5
    ) -> List[ConversationMemory]:
        # Search for similar past conversations
        # Return relevant messages with context

    async def recall_relevant_facts(
        self,
        context: str,
        limit: int = 10
    ) -> List[WorkspaceFact]:
        # Search for relevant workspace facts
        # Filter by confidence and recency

    async def learn_preference(
        self,
        category: str,
        preference: str,
        value: str,
        source: str = "observed"
    ) -> None:
        # Store user preference
        # Update if already exists

    async def get_workspace_context(self) -> WorkspaceContext:
        # Aggregate recent facts, preferences, and activities
        # Build context summary for agent

    async def consolidate_memory(self) -> None:
        # Move working memory to long-term storage
        # Update embeddings
        # Archive old conversations
```

**Memory Consolidation:**

- Run consolidation every 24 hours or on shutdown
- Summarize long conversations to save tokens
- Identify important facts to keep in working memory
- Archive completed tasks older than 30 days

**Cache Management:**

- Keep last 50 conversation turns in RAM
- Keep top 100 most-accessed facts in RAM
- Keep current session preferences in RAM
- Evict least recently used items when cache full

### Step 2.5: Implement Automatic Fact Learning

**Technical Approach:**

- Monitor agent interactions to extract learnable facts
- Use LLM to identify important information
- Store facts with confidence scores
- Update facts when new information contradicts old

**Fact Extraction** (`src/proxima/agent/memory/fact_extractor.py`):

**Types of Facts to Learn:**

1. **File Structure**: "The quantum algorithms are in src/algorithms/"
2. **Naming Conventions**: "Backend config files use snake_case"
3. **Common Patterns**: "User always tests with 1024 shots first"
4. **Tool Preferences**: "User prefers Cirq over Qiskit for small circuits"
5. **Error Solutions**: "Import error fixed by activating conda env"
6. **Performance Notes**: "LRET backend is fastest for < 10 qubits"

**Extraction Method:**

```python
class FactExtractor:
    async def extract_facts_from_conversation(
        self,
        messages: List[Message]
    ) -> List[Fact]:
        # Send conversation to LLM with fact extraction prompt
        # Parse LLM response into structured facts
        # Assign confidence scores based on source
        # Validate facts before storing

    async def extract_facts_from_observation(
        self,
        action: str,
        outcome: str
    ) -> List[Fact]:
        # Learn from action outcomes
        # Example: "Building LRET takes 5 minutes" from build time
        # Example: "GPU backend requires CUDA" from build failure
```

**Fact Extraction Prompt:**

```
Analyze this conversation and extract factual information about the workspace,
user preferences, and technical details. For each fact, provide:
- Type: file_structure, pattern, preference, config, or technical
- Key: Short identifier
- Value: The actual fact
- Confidence: 0.0 to 1.0 based on how certain the fact is

Example output:
{
  "facts": [
    {"type": "preference", "key": "default_backend", "value": "cirq", "confidence": 0.9},
    {"type": "file_structure", "key": "algorithm_dir", "value": "src/algorithms/", "confidence": 1.0}
  ]
}
```

**Fact Validation:**

- Check if fact contradicts existing facts
- If contradiction, keep fact with higher confidence
- Merge complementary facts (e.g., multiple backend preferences)
- Flag uncertain facts for user confirmation

### Step 2.6: Create Memory-Aware Agent Prompting

**Technical Approach:**

- Augment agent system prompts with relevant memories
- Retrieve similar past conversations for context
- Include learned facts in agent instructions
- Adapt agent behavior based on user preferences

**Context Building** (`src/proxima/agent/memory/context_builder.py`):

**Context Assembly:**

```python
class MemoryContextBuilder:
    async def build_context_for_query(
        self,
        query: str,
        session_history: List[Message]
    ) -> AgentContext:
        # Search for similar past conversations
        # Retrieve relevant workspace facts
        # Get user preferences
        # Get recent task history
        # Build context summary

    def format_context_for_prompt(
        self,
        context: AgentContext
    ) -> str:
        # Format memories as natural language
        # Include in system prompt or user message
        # Keep under token budget
```

**Enhanced System Prompt:**

```
You are Proxima's AI agent with long-term memory capabilities.

WORKSPACE CONTEXT:
{{relevant_facts}}

USER PREFERENCES:
{{user_preferences}}

RECENT ACTIVITIES:
{{recent_tasks}}

RELEVANT PAST CONVERSATIONS:
{{similar_conversations}}

Use this context to provide more personalized and informed assistance.
```

**Token Budget Management:**

- Allocate 20% of context window for memories
- Prioritize most relevant memories by similarity score
- Summarize very similar conversations to save tokens
- Truncate old messages if budget exceeded

### Step 2.7: Add Memory UI Components

**Technical Approach:**

- Create "Memory" tab in agent UI
- Display conversation history with search
- Show learned facts and allow editing
- Visualize memory statistics
- Allow memory export/import/clear

**UI Implementation** (`src/proxima/tui/screens/agent_memory.py`):

**Memory Screen Layout:**

```
┌─ Agent Memory ────────────────────────────────────┐
│ 🔍 Search: [_______________________] [Search]     │
├───────────────────────────────────────────────────┤
│ 📊 Statistics                                     │
│ Total Conversations: 247                          │
│ Workspace Facts: 83                               │
│ User Preferences: 12                              │
│ Task History: 156                                 │
│                                                   │
│ 📚 Conversation History          [Clear All]      │
│ ┌───────────────────────────────────────┐        │
│ │ 2026-02-01 10:30 - Build LRET backend │        │
│ │ 2026-02-01 11:45 - Execute bell state  │        │
│ │ 2026-01-31 14:20 - Compare backends    │        │
│ │ ... (scroll for more)                  │        │
│ └───────────────────────────────────────┘        │
│                                                   │
│ 🧠 Learned Facts                 [Edit] [Delete]  │
│ ┌───────────────────────────────────────┐        │
│ │ • Default backend: cirq (conf: 0.95)  │        │
│ │ • Algorithms in: src/algorithms/       │        │
│ │ • Prefers 1024 shots for testing       │        │
│ │ ... (scroll for more)                  │        │
│ └───────────────────────────────────────┘        │
│                                                   │
│ [Export Memory] [Import Memory] [Settings]        │
└───────────────────────────────────────────────────┘
```

**Search Functionality:**

- Semantic search across all conversations
- Filter by date range
- Filter by success/failure
- Filter by task type

**Fact Editing:**

- Click fact to edit
- Adjust confidence slider
- Delete incorrect facts
- Add manual facts

### Step 2.8: Implement Memory-Based Suggestions

**Technical Approach:**

- Analyze memory to proactively suggest actions
- Detect patterns in user behavior
- Recommend based on past successes
- Warn about past failures

**Suggestion Engine** (`src/proxima/agent/memory/suggestions.py`):

**Suggestion Types:**

1. **Action Suggestions**: "You usually build the backend after pulling from git"
2. **Parameter Suggestions**: "Last time you used 2048 shots for this circuit"
3. **Optimization Suggestions**: "Cirq was faster than Qiskit for similar circuits"
4. **Warning Suggestions**: "Building this backend failed last week due to missing CUDA"

**Implementation:**

```python
class MemorySuggestionEngine:
    async def generate_suggestions(
        self,
        current_context: str
    ) -> List[Suggestion]:
        # Search for similar past situations
        # Analyze outcomes of past actions
        # Generate actionable suggestions
        # Rank by relevance and confidence

    async def detect_patterns(self) -> List[Pattern]:
        # Analyze task history for recurring patterns
        # Identify common sequences of actions
        # Learn typical parameter values
```

**Pattern Detection Examples:**

- "User always builds backend before running experiments"
- "User runs tests after code changes"
- "User exports results as JSON after comparing backends"

**Display in UI:**

- Show suggestions in agent chat as proactive messages
- Display as notification badges
- Allow dismissing or accepting suggestions
- Learn from user's response to suggestions

### Step 2.9: Add Privacy and Control Features

**Technical Approach:**

- Implement memory privacy controls
- Allow selective memory deletion
- Provide memory encryption option
- Enable memory export for backup

**Privacy Controls** (`src/proxima/agent/memory/privacy.py`):

**Features:**

1. **Selective Memory Deletion**:
   - Delete by date range
   - Delete by keyword search
   - Delete specific conversations
   - Delete all memories for fresh start

2. **Memory Encryption** (optional):
   - Encrypt database with password
   - Use AES-256-GCM encryption
   - Decrypt on demand when accessing
   - Library: `cryptography` (Fernet)

3. **Memory Export/Import**:
   - Export as JSON for backup
   - Import from previous export
   - Merge imported memories with existing
   - Validate import format

4. **Automatic Cleanup**:
   - Delete conversations older than N days (configurable)
   - Delete low-confidence facts not accessed in N days
   - Archive completed tasks after N days
   - User can configure retention policies

**Settings UI:**

```
Memory & Privacy Settings

Data Retention:
○ Keep forever
● Keep for 90 days
○ Keep for 30 days
○ Custom: [___] days

Memory Encryption:
☐ Encrypt memory database
   Password: [___________]

Automatic Cleanup:
☑ Delete old conversations
☑ Archive completed tasks
☐ Remove low-confidence facts

[Export Memory] [Import Memory] [Clear All Memory]
```

### Step 2.10: Implement Memory Analytics

**Technical Approach:**

- Provide insights into agent learning progress
- Visualize memory growth over time
- Show most useful facts
- Track memory system performance

**Analytics Dashboard** (`src/proxima/agent/memory/analytics.py`):

**Metrics to Track:**

1. **Memory Growth**: Number of facts/conversations over time
2. **Fact Confidence Distribution**: How many high/low confidence facts
3. **Most Accessed Facts**: Which facts are queried most often
4. **Search Performance**: Average search latency
5. **Memory Effectiveness**: Correlation between memory use and task success

**Visualization:**

```
Memory Analytics

Memory Growth (Last 30 Days)
  Conversations  ███████████████ 247 (+34 this week)
  Facts          █████████ 83 (+12 this week)
  Preferences    ████ 12 (no change)

Top Accessed Facts:
1. Default backend: cirq (accessed 127 times)
2. Algorithm directory: src/algorithms/ (accessed 89 times)
3. Preferred shots: 1024 (accessed 76 times)

Search Performance:
  Average latency: 45ms
  Cache hit rate: 78%

Memory Impact on Success Rate:
  Tasks with memory context: 89% success
  Tasks without memory context: 72% success
  Improvement: +17%
```

**Export Analytics:**

- Generate reports as PDF or HTML
- Export metrics as CSV for analysis
- Share insights with development team

### Step 2.11: Create Memory Migration System

**Technical Approach:**

- Handle database schema changes gracefully
- Migrate old memory format to new format
- Preserve all existing memories during updates
- Validate migrated data

**Migration System** (`src/proxima/agent/memory/migrations.py`):

**Migration Framework:**

```python
class MigrationManager:
    def get_current_version(self) -> int:
        # Read version from database metadata

    def get_latest_version(self) -> int:
        # Return latest schema version

    async def run_migrations(self) -> None:
        # Run all pending migrations in order
        # Each migration is a separate SQL file or Python function

    async def rollback_migration(self, target_version: int) -> None:
        # Rollback to specific version (for development)
```

**Migration Files** (`src/proxima/agent/memory/migrations/`):

- `001_initial_schema.sql` - Create initial tables
- `002_add_embeddings.sql` - Add embedding columns
- `003_add_preferences.sql` - Add preferences table
- `004_add_analytics.sql` - Add analytics tables

**Backward Compatibility:**

- Maintain compatibility with previous version
- Provide upgrade path from v1.x to v2.x
- Test migrations on sample data
- Backup database before migration

### Step 2.12: Integrate Memory with Existing Agent System

**Technical Approach:**

- Modify agent initialization to load memory manager
- Inject memory context into agent prompts
- Store agent actions in memory automatically
- Update memory after each tool execution

**Integration Points:**

1. **Agent Initialization** (`src/proxima/agent/controller.py`):

```python
class AgentController:
    def __init__(self):
        self.memory_manager = MemoryManager(
            db_path=get_memory_db_path(),
            embedding_provider=get_embedding_provider()
        )
        await self.memory_manager.initialize()
```

2. **Message Handling** (`src/proxima/tui/screens/agent_ai_assistant.py`):

```python
async def _handle_user_message(self, message: str):
    # Store user message in memory
    await self.memory_manager.store_conversation_turn(
        session_id=self.session_id,
        role="user",
        content=message
    )

    # Build context from memory
    context = await self.memory_manager.build_context_for_query(message)

    # Generate response with memory context
    response = await self.llm.generate(message, context=context)

    # Store assistant response in memory
    await self.memory_manager.store_conversation_turn(
        session_id=self.session_id,
        role="assistant",
        content=response
    )
```

3. **Tool Execution** (`src/proxima/agent/tools.py`):

```python
async def execute_tool(self, tool_name: str, args: Dict) -> ToolResult:
    # Execute tool
    result = await self._execute(tool_name, args)

    # Store in task history
    await self.memory_manager.store_task(
        task_description=f"Execute {tool_name}",
        tool_calls=[{" name": tool_name, "args": args}],
        success=result.success,
        outcome=result.output
    )

    # Extract learnable facts
    facts = await self.fact_extractor.extract_from_outcome(
        action=tool_name,
        outcome=result
    )
    for fact in facts:
        await self.memory_manager.store_fact(**fact)

    return result
```

### Step 2.13: Test Memory System

**Technical Approach:**

- Create comprehensive test suite for memory system
- Test memory storage and retrieval
- Test semantic search accuracy
- Test memory consolidation
- Test migration system

**Test Categories:**

1. **Unit Tests** (`tests/agent/memory/test_memory_manager.py`):
   - Test fact storage and retrieval
   - Test conversation storage
   - Test preference management
   - Test cache behavior

2. **Integration Tests** (`tests/agent/memory/test_memory_integration.py`):
   - Test memory with real agent interactions
   - Test memory consolidation process
   - Test memory migration
   - Test memory export/import

3. **Performance Tests** (`tests/agent/memory/test_memory_performance.py`):
   - Benchmark search latency
   - Test with large datasets (10K+ conversations)
   - Memory usage profiling
   - Cache effectiveness

**Test Data:**

- Generate synthetic conversation history
- Create diverse workspace facts
- Simulate user preferences
- Test edge cases (empty memory, corrupted database)

### Step 2.14: Document Memory System

**Technical Approach:**

- Create user-facing documentation
- Create developer documentation
- Provide examples and best practices
- Document privacy and security aspects

**Documentation Files:**

1. **`docs/agent-guide/memory-system.md`**: User guide
   - What is agent memory?
   - How memory improves agent performance
   - Privacy and security
   - Managing your memory
   - FAQ

2. **`docs/developer-guide/memory-architecture.md`**: Technical docs
   - Memory system architecture
   - Database schema
   - Embedding system
   - API reference
   - Extending memory system

3. **`docs/agent-guide/memory-examples.md`**: Usage examples
   - Teaching agent about workspace
   - Using memory for personalization
   - Memory-based automation
   - Troubleshooting memory issues

**User Guide Topics:**

- Understanding agent memory
- What information is stored?
- How to view and manage memories
- Exporting and backing up memory
- Privacy best practices
- Clearing or resetting memory

---

## Feature 3: Multi-Agent Collaboration Framework

**Objective:** Create a framework that enables multiple specialized AI agents to collaborate on complex tasks with coordination, communication, and task distribution capabilities.

**Value Proposition:**

- **Specialized Expertise**: Each agent focuses on specific domain (quantum, coding, analysis)
- **Parallel Processing**: Multiple agents work simultaneously for faster completion
- **Quality Assurance**: Peer review and validation between agents
- **Complex Task Handling**: Break large projects into manageable agent-specific subtasks
- **Improved Accuracy**: Different agents check each other's work
- **Scalability**: Add new specialist agents without modifying core system

### Step 3.1: Design Multi-Agent Architecture

**Technical Background:**

- Multi-agent systems enable autonomous agents to work cooperatively
- Agents communicate via message passing (actor model)
- Central coordinator distributes tasks and aggregates results
- Each agent has specific capabilities and constraints

**Architecture Pattern: Hierarchical Multi-Agent System**

```
                    ┌─────────────────────┐
                    │ Coordinator Agent   │
                    │ (Task Distribution) │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   ┌────▼────┐           ┌────▼────┐          ┌─────▼────┐
   │ Quantum │           │  Code   │          │ Analysis │
   │ Expert  │           │ Expert  │          │  Expert  │
   └────┬────┘           └────┬────┘          └─────┬────┘
        │                     │                      │
        └─────────────────────┴──────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Shared Resources  │
                    │  (Memory, Tools)   │
                    └────────────────────┘
```

**Agent Roles:**

1. **Coordinator Agent**: Manages workflow, distributes tasks, aggregates results
2. **Quantum Expert Agent**: Handles quantum circuit design, optimization
3. **Code Expert Agent**: Generates code, refactors, fixes bugs
4. **Analysis Expert Agent**: Analyzes results, generates insights, creates visualizations
5. **Documentation Agent**: Writes documentation, creates tutorials
6. **Testing Agent**: Creates and runs tests, validates implementations

**Key Design Principles:**

- Agents are independent processes/tasks (true parallelism)
- Message-based communication (no shared mutable state)
- Each agent has clear responsibilities (single responsibility principle)
- Coordinator handles conflicts and priorities
- Agents can request help from other agents

### Step 3.2: Implement Agent Communication Protocol

**Technical Approach:**

- Use message queues for async communication
- Define structured message format (JSON)
- Implement request/response and pub/sub patterns
- Handle message routing and delivery

**Communication Layer** (`src/proxima/agent/multi/communication.py`):

**Message Format:**

```python
@dataclass
class AgentMessage:
    id: str  # Unique message ID
    from_agent: str  # Sender agent ID
    to_agent: str  # Recipient agent ID (or "broadcast")
    message_type: MessageType  # REQUEST, RESPONSE, NOTIFICATION, ERROR
    timestamp: float
    content: Dict[str, Any]  # Message payload
    reply_to: Optional[str]  # ID of message being replied to
    priority: int = 0  # Higher = more urgent
```

**Message Types:**

```python
class MessageType(Enum):
    REQUEST = "request"  # Request action from another agent
    RESPONSE = "response"  # Response to a request
    NOTIFICATION = "notification"  # Broadcast information
    QUERY = "query"  # Query for information
    RESULT = "result"  # Task completion result
    ERROR = "error"  # Error notification
    HEARTBEAT = "heartbeat"  # Health check
```

**Message Bus Implementation:**

```python
class MessageBus:
    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}  # Per-agent queues
        self._subscriptions: Dict[str, List[str]] = {}  # Topic subscriptions

    async def send_message(self, message: AgentMessage) -> None:
        # Route message to recipient's queue
        # Handle broadcast messages
        # Log message for debugging

    async def receive_message(
        self,
        agent_id: str,
        timeout: float = None
    ) -> Optional[AgentMessage]:
        # Get message from agent's queue
        # Block until message or timeout

    async def subscribe(
        self,
        agent_id: str,
        topic: str
    ) -> None:
        # Subscribe agent to topic for notifications

    async def publish(
        self,
        topic: str,
        message: AgentMessage
    ) -> None:
        # Publish message to all subscribed agents
```

**Communication Patterns:**

1. **Request-Response**:

```python
# Agent A requests help from Agent B
request_msg = AgentMessage(
    id=generate_id(),
    from_agent="quantum_expert",
    to_agent="code_expert",
    message_type=MessageType.REQUEST,
    content={"action": "generate_circuit_code", "circuit": "..."}
)
await message_bus.send_message(request_msg)

# Wait for response
response = await message_bus.receive_message("quantum_expert", timeout=30.0)
```

2. **Pub-Sub**:

```python
# Agent subscribes to topic
await message_bus.subscribe("analysis_expert", topic="results_updated")

# Coordinator publishes notification
notification = AgentMessage(
    from_agent="coordinator",
    message_type=MessageType.NOTIFICATION,
    content={"event": "new_results", "data": {...}}
)
await message_bus.publish("results_updated", notification)
```

**Libraries:**

- `asyncio.Queue` - Message queues
- `msgpack` or `json` - Message serialization
- Optional: `aio-pika` (RabbitMQ) for distributed systems

### Step 3.3: Create Base Agent Class

**Technical Approach:**

- Define abstract base class for all agents
- Implement common agent functionality
- Provide lifecycle management (start, stop, restart)
- Handle message processing loop

**Base Agent** (`src/proxima/agent/multi/base_agent.py`):

**Class Structure:**

```python
class BaseAgent(ABC):
    def __init__(
        self,
        agent_id: str,
        message_bus: MessageBus,
        config: AgentConfig
    ):
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.config = config
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._capabilities: List[str] = []

    async def start(self) -> None:
        # Start agent's message processing loop
        # Initialize agent-specific resources
        # Send READY notification to coordinator

    async def stop(self) -> None:
        # Graceful shutdown
        # Complete current tasks
        # Send STOPPING notification
        # Clean up resources

    async def _message_loop(self) -> None:
        # Main loop: receive and process messages
        while self._running:
            msg = await self.message_bus.receive_message(self.agent_id, timeout=1.0)
            if msg:
                await self._handle_message(msg)

    async def _handle_message(self, message: AgentMessage) -> None:
        # Route message to appropriate handler
        if message.message_type == MessageType.REQUEST:
            await self._handle_request(message)
        elif message.message_type == MessageType.QUERY:
            await self._handle_query(message)
        # ... other message types

    @abstractmethod
    async def _handle_request(self, message: AgentMessage) -> None:
        # Override in subclasses to handle requests
        pass

    async def send_request(
        self,
        to_agent: str,
        action: str,
        params: Dict
    ) -> Dict:
        # Send request and wait for response
        # Implement timeout and retry logic

    async def send_notification(self, event: str, data: Dict) -> None:
        # Broadcast notification to interested agents

    def get_capabilities(self) -> List[str]:
        # Return list of actions this agent can perform
        return self._capabilities
```

**Agent Lifecycle States:**

```python
class AgentState(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    WAITING = "waiting"
    ERROR = "error"
    STOPPING = "stopping"
    STOPPED = "stopped"
```

**Agent Monitoring:**

- Each agent sends heartbeat every 10 seconds
- Coordinator detects unresponsive agents
- Auto-restart crashed agents (configurable)
- Track agent performance metrics

### Step 3.4: Implement Coordinator Agent

**Technical Approach:**

- Create coordinator agent to orchestrate multi-agent tasks
- Implement task decomposition algorithm
- Handle task assignment to appropriate agents
- Aggregate results from multiple agents

**Coordinator Implementation** (`src/proxima/agent/multi/coordinator.py`):

**Class Structure:**

```python
class CoordinatorAgent(BaseAgent):
    def __init__(
        self,
        message_bus: MessageBus,
        config: AgentConfig
    ):
        super().__init__("coordinator", message_bus, config)
        self._agents: Dict[str, AgentInfo] = {}  # Registered agents
        self._tasks: Dict[str, Task] = {}  # Active tasks
        self._task_queue: asyncio.Queue = asyncio.Queue()

    async def register_agent(
        self,
        agent_id: str,
        capabilities: List[str]
    ) -> None:
        # Register agent and its capabilities
        # Store agent metadata
        # Send welcome message

    async def decompose_task(
        self,
        task: ComplexTask
    ) -> List[SubTask]:
        # Break complex task into subtasks
        # Identify required capabilities for each subtask
        # Determine task dependencies
        # Order subtasks for execution

    async def assign_task(
        self,
        subtask: SubTask
    ) -> str:
        # Find agent with required capabilities
        # Check agent availability
        # Assign task to best-fit agent
        # Return assigned agent ID

    async def execute_complex_task(
        self,
        task: ComplexTask
    ) -> TaskResult:
        # Decompose into subtasks
        # Assign subtasks to agents
        # Monitor progress
        # Handle failures and retries
        # Aggregate results
        # Return final result
```

**Task Decomposition Strategy:**

```python
# Example: "Create and test quantum algorithm"
complex_task = ComplexTask(description="Create and test QAOA algorithm")

subtasks = [
    SubTask(
        id="1",
        description="Design QAOA circuit",
        assigned_to="quantum_expert",
        capabilities_required=["circuit_design", "optimization"],
        dependencies=[]
    ),
    SubTask(
        id="2",
        description="Generate Python code",
        assigned_to="code_expert",
        capabilities_required=["code_generation", "python"],
        dependencies=["1"]  # Depends on circuit design
    ),
    SubTask(
        id="3",
        description="Create unit tests",
        assigned_to="testing_agent",
        capabilities_required=["test_generation", "pytest"],
        dependencies=["2"]  # Depends on code
    ),
    SubTask(
        id="4",
        description="Execute and analyze results",
        assigned_to="analysis_expert",
        capabilities_required=["execution", "data_analysis"],
        dependencies=["2", "3"]  # Depends on code and tests
    ),
    SubTask(
        id="5",
        description="Generate documentation",
        assigned_to="documentation_agent",
        capabilities_required=["documentation", "markdown"],
        dependencies=["1", "2"]  # Depends on design and code
    ),
]
```

**Parallel Execution:**

- Subtasks without dependencies run in parallel
- Use `asyncio.gather()` for concurrent execution
- Track progress of parallel tasks
- Handle partial failures gracefully

**Result Aggregation:**

- Collect results from all subtasks
- Combine into cohesive final result
- Identify conflicts or inconsistencies
- Request clarification if needed

### Step 3.5: Create Specialized Agent Implementations

**Technical Approach:**

- Implement specific agent types
- Give each agent domain expertise
- Configure agent with appropriate tools and LLM
- Define agent capabilities

**Quantum Expert Agent** (`src/proxima/agent/multi/agents/quantum_expert.py`):

**Responsibilities:**

- Design quantum circuits
- Optimize circuits for specific backends
- Explain quantum concepts
- Recommend quantum algorithms

**Implementation:**

```python
class QuantumExpertAgent(BaseAgent):
    def __init__(self, message_bus: MessageBus, config: AgentConfig):
        super().__init__("quantum_expert", message_bus, config)
        self._capabilities = [
            "circuit_design",
            "circuit_optimization",
            "algorithm_recommendation",
            "quantum_explanation"
        ]
        self._llm = create_llm_client(
            provider=config.llm_provider,
            model=config.llm_model,
            system_prompt=QUANTUM_EXPERT_SYSTEM_PROMPT
        )

    async def _handle_request(self, message: AgentMessage) -> None:
        action = message.content.get("action")

        if action == "design_circuit":
            result = await self._design_circuit(message.content["specs"])
        elif action == "optimize_circuit":
            result = await self._optimize_circuit(message.content["circuit"])
        elif action == "recommend_algorithm":
            result = await self._recommend_algorithm(message.content["problem"])
        else:
            result = {"error": f"Unknown action: {action}"}

        # Send response
        response = AgentMessage(
            id=generate_id(),
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            message_type=MessageType.RESPONSE,
            content=result,
            reply_to=message.id
        )
        await self.message_bus.send_message(response)
```

**System Prompt:**

```
You are a quantum computing expert specializing in quantum circuit design and optimization.
Your expertise includes:
- Designing quantum circuits for various algorithms (VQE, QAOA, Grover, Shor, etc.)
- Optimizing circuits for specific quantum backends
- Explaining quantum concepts in accessible terms
- Recommending appropriate algorithms for given problems

You work as part of a multi-agent system. Provide concise, technically accurate responses.
```

**Code Expert Agent** (`src/proxima/agent/multi/agents/code_expert.py`):

**Responsibilities:**

- Generate Python code for quantum algorithms
- Refactor existing code
- Fix bugs and errors
- Implement backend adapters

**Capabilities:**

```python
self._capabilities = [
    "code_generation",
    "code_refactoring",
    "bug_fixing",
    "code_review",
    "backend_adapter_creation"
]
```

**Analysis Expert Agent** (`src/proxima/agent/multi/agents/analysis_expert.py`):

**Responsibilities:**

- Analyze quantum simulation results
- Generate visualizations
- Compare backend performance
- Provide insights and recommendations

**Capabilities:**

```python
self._capabilities = [
    "result_analysis",
    "data_visualization",
    "performance_comparison",
    "statistical_analysis",
    "recommendation_generation"
]
```

### Step 3.6: Implement Agent Discovery and Registration

**Technical Approach:**

- Agents self-register with coordinator on startup
- Advertise capabilities during registration
- Coordinator maintains agent registry
- Support dynamic agent addition/removal

**Registration Protocol:**

```python
# Agent startup sequence
async def start_agent(agent: BaseAgent):
    # 1. Start message processing loop
    await agent.start()

    # 2. Register with coordinator
    registration_msg = AgentMessage(
        from_agent=agent.agent_id,
        to_agent="coordinator",
        message_type=MessageType.REQUEST,
        content={
            "action": "register",
            "capabilities": agent.get_capabilities(),
            "metadata": {
                "version": agent.get_version(),
                "max_concurrent_tasks": agent.config.max_concurrent_tasks
            }
        }
    )
    await agent.message_bus.send_message(registration_msg)

    # 3. Wait for registration confirmation
    response = await agent.wait_for_response(registration_msg.id, timeout=10.0)
    if response.content.get("status") == "registered":
        logger.info(f"Agent {agent.agent_id} registered successfully")
```

**Agent Registry** (`src/proxima/agent/multi/registry.py`):

```python
@dataclass
class AgentInfo:
    agent_id: str
    capabilities: List[str]
    state: AgentState
    last_heartbeat: float
    tasks_completed: int
    tasks_failed: int
    average_response_time: float

class AgentRegistry:
    def register(self, agent_info: AgentInfo) -> None:
        # Add agent to registry
        # Index by agent_id and capabilities

    def find_agents_with_capability(
        self,
        capability: str
    ) -> List[AgentInfo]:
        # Find all agents capable of performing action
        # Filter by availability
        # Sort by performance metrics

    def update_heartbeat(self, agent_id: str) -> None:
        # Update last seen timestamp
        # Mark agent as healthy

    def get_agent_status(self, agent_id: str) -> AgentInfo:
        # Get current agent information
```

### Step 3.7: Create Task Distribution Algorithm

**Technical Approach:**

- Implement intelligent task assignment
- Consider agent load, capabilities, and performance
- Support priority-based scheduling
- Handle agent failures gracefully

**Task Scheduler** (`src/proxima/agent/multi/scheduler.py`):

**Scheduling Strategies:**

1. **Capability-Based**: Assign to agent with required capabilities
2. **Load-Balanced**: Distribute tasks evenly across agents
3. **Performance-Based**: Assign to fastest agent
4. **Priority-Based**: High-priority tasks first
5. **Affinity-Based**: Keep related tasks on same agent (cache locality)

**Implementation:**

```python
class TaskScheduler:
    def __init__(self, agent_registry: AgentRegistry):
        self.registry = agent_registry
        self.strategy = SchedulingStrategy.CAPABILITY_LOAD_BALANCED

    async def assign_task(
        self,
        task: SubTask
    ) -> str:
        # Get candidate agents
        candidates = self.registry.find_agents_with_capability(
            task.capabilities_required[0]
        )

        if not candidates:
            raise NoCapableAgentError(f"No agent can handle: {task.capabilities_required}")

        # Apply scheduling strategy
        if self.strategy == SchedulingStrategy.LOAD_BALANCED:
            # Choose agent with fewest active tasks
            agent = min(candidates, key=lambda a: a.active_tasks)
        elif self.strategy == SchedulingStrategy.PERFORMANCE_BASED:
            # Choose fastest agent
            agent = min(candidates, key=lambda a: a.average_response_time)
        else:
            # Default: first available
            agent = candidates[0]

        return agent.agent_id
```

**Failure Handling:**

- If agent doesn't respond within timeout, reassign task
- Track agent failure rate, disable if too high
- Implement circuit breaker pattern for unreliable agents
- Log failures for debugging

### Step 3.8: Add Inter-Agent Peer Review

**Technical Approach:**

- Implement peer review protocol
- Allow agents to validate each other's work
- Improve output quality through multiple perspectives
- Handle disagreements with voting or escalation

**Peer Review System** (`src/proxima/agent/multi/peer_review.py`):

**Review Protocol:**

```python
class PeerReviewManager:
    async def request_review(
        self,
        work: AgentWork,
        reviewers: List[str]
    ) -> ReviewResult:
        # Send work to multiple agents for review
        # Collect feedback
        # Aggregate reviews
        # Determine if work passes review

    async def resolve_conflicts(
        self,
        reviews: List[Review]
    ) -> Resolution:
        # Handle disagreements between reviewers
        # Use voting (majority wins)
        # Or escalate to human user
```

**Example Workflow:**

```python
# Code Expert generates code
code_result = await code_expert.generate_code(spec)

# Send to Quantum Expert for review
review_request = {
    "action": "review_code",
    "code": code_result.code,
    "criteria": ["correctness", "optimization", "quantum_principles"]
}
review = await quantum_expert.review(review_request)

# If issues found, send back for revision
if not review.approved:
    revised_code = await code_expert.revise_code(
        code=code_result.code,
        feedback=review.comments
    )
```

**Review Criteria:**

- **Correctness**: Does it work as intended?
- **Performance**: Is it efficient?
- **Best Practices**: Follows conventions?
- **Completeness**: All requirements met?

### Step 3.9: Implement Shared Knowledge Base

**Technical Approach:**

- Create shared memory accessible to all agents
- Store intermediate results, learned patterns
- Enable agents to learn from each other
- Avoid duplicate work across agents

**Shared Knowledge** (`src/proxima/agent/multi/shared_knowledge.py`):

**Knowledge Store:**

```python
class SharedKnowledgeBase:
    def __init__(self):
        self._facts: Dict[str, Any] = {}  # Shared facts
        self._cache: Dict[str, Any] = {}  # Cached results
        self._learnings: List[Learning] = []  # Insights from agents

    async def store_fact(
        self,
        key: str,
        value: Any,
        source_agent: str
    ) -> None:
        # Store fact from agent
        # Make available to other agents

    async def query_fact(
        self,
        key: str
    ) -> Optional[Any]:
        # Retrieve fact if exists

    async def cache_result(
        self,
        operation: str,
        params: Dict,
        result: Any,
        ttl: int = 3600
    ) -> None:
        # Cache operation result
        # Other agents can reuse without recomputing

    async def share_learning(
        self,
        agent_id: str,
        learning: Learning
    ) -> None:
        # Share insight with other agents
        # Example: "LRET is fastest for < 15 qubits"
```

**Benefits:**

- Avoid redundant computations
- Faster task completion
- Collective intelligence
- Consistent knowledge across agents

### Step 3.10: Create Multi-Agent UI

**Technical Approach:**

- Visualize agent activities in TUI
- Show task distribution and progress
- Display agent communication
- Allow user to intervene or provide guidance

**UI Implementation** (`src/proxima/tui/screens/multi_agent.py`):

**Screen Layout:**

```
┌─ Multi-Agent System ──────────────────────────────┐
│ Active Agents: 5/5                                │
│ ●quantum_expert  ●code_expert  ●analysis_expert   │
│ ●testing_agent   ●documentation_agent             │
├───────────────────────────────────────────────────┤
│ Current Task: Create QAOA algorithm               │
│ Progress: [████████░░] 75% (3/4 subtasks done)    │
│                                                   │
│ Subtasks:                                         │
│ ✓ Design circuit (quantum_expert) - 2.3s         │
│ ✓ Generate code (code_expert) - 4.1s             │
│ ⏳ Run tests (testing_agent) - in progress...     │
│ ⏸ Create docs (documentation_agent) - waiting     │
├───────────────────────────────────────────────────┤
│ Agent Communication (recent):                     │
│ [10:30:15] coordinator → quantum_expert: design   │
│ [10:30:18] quantum_expert → coordinator: done     │
│ [10:30:19] coordinator → code_expert: generate    │
│ [10:30:23] code_expert → quantum_expert: review?  │
│ [10:30:24] quantum_expert → code_expert: approved │
├───────────────────────────────────────────────────┤
│ [Pause] [Stop] [View Logs] [Agent Details]       │
└───────────────────────────────────────────────────┘
```

**Real-time Updates:**

- Use Textual reactive properties
- Update agent status indicators
- Animate progress bars
- Stream agent messages
- Highlight active agents

**Agent Details View:**

- Click agent to see details
- View agent capabilities
- See task history
- Check performance metrics

### Step 3.11: Add Human-in-the-Loop Controls

**Technical Approach:**

- Allow user to intervene in multi-agent workflow
- Provide approval gates for critical decisions
- Enable manual task assignment override
- Support user feedback to agents

**User Controls:**

1. **Approval Gates**: Pause before executing destructive operations
2. **Manual Override**: User can reassign tasks or change priorities
3. **Feedback**: User can correct agent outputs
4. **Emergency Stop**: Halt all agents immediately

**Implementation:**

```python
class HumanInTheLoop:
    async def request_approval(
        self,
        action: str,
        details: Dict,
        timeout: float = 300.0
    ) -> bool:
        # Show approval dialog in UI
        # Wait for user response
        # Return approval status

    async def request_feedback(
        self,
        agent_id: str,
        output: Any
    ) -> Optional[str]:
        # Show output to user
        # Collect feedback
        # Return feedback to agent
```

### Step 3.12: Implement Multi-Agent Monitoring

**Technical Approach:**

- Track metrics for all agents
- Monitor system health
- Detect and alert on issues
- Generate performance reports

**Monitoring System** (`src/proxima/agent/multi/monitoring.py`):

**Metrics to Track:**

```python
@dataclass
class AgentMetrics:
    agent_id: str
    uptime: float
    tasks_completed: int
    tasks_failed: int
    average_response_time: float
    messages_sent: int
    messages_received: int
    cpu_usage: float
    memory_usage: float
```

**System Metrics:**

```python
@dataclass
class SystemMetrics:
    total_agents: int
    healthy_agents: int
    total_tasks_completed: int
    average_task_duration: float
    system_throughput: float  # Tasks per minute
    inter_agent_latency: float  # Message latency
```

**Alerting:**

- Alert if agent unresponsive for > 30 seconds
- Alert if agent error rate > 20%
- Alert if system throughput drops significantly
- Display alerts in UI and log files

### Step 3.13: Test Multi-Agent System

**Technical Approach:**

- Create comprehensive test scenarios
- Test agent communication
- Test failure recovery
- Test complex multi-step workflows

**Test Scenarios:**

1. **Basic Communication**: Agents can send/receive messages
2. **Task Distribution**: Tasks assigned correctly
3. **Parallel Execution**: Multiple tasks run concurrently
4. **Failure Recovery**: System handles agent crashes
5. **Complex Workflow**: End-to-end multi-agent task completion

**Test Implementation** (`tests/agent/multi/test_multi_agent.py`):

```python
@pytest.mark.asyncio
async def test_multi_agent_collaboration():
    # Setup
    message_bus = MessageBus()
    coordinator = CoordinatorAgent(message_bus, config)
    quantum_expert = QuantumExpertAgent(message_bus, config)
    code_expert = CodeExpertAgent(message_bus, config)

    # Start agents
    await coordinator.start()
    await quantum_expert.start()
    await code_expert.start()

    # Execute complex task
    task = ComplexTask(description="Create Bell state circuit")
    result = await coordinator.execute_complex_task(task)

    # Verify
    assert result.success
    assert "circuit" in result.data
    assert "code" in result.data
```

### Step 3.14: Document Multi-Agent System

**Technical Approach:**

- Create user and developer documentation
- Provide examples and tutorials
- Document agent protocols
- Explain customization

**Documentation Files:**

1. **`docs/agent-guide/multi-agent-system.md`**: User guide
2. **`docs/developer-guide/creating-custom-agents.md`**: Developer guide
3. **`docs/agent-guide/multi-agent-examples.md`**: Usage examples
4. **`docs/architecture/multi-agent-design.md`**: Architecture docs

**Topics to Cover:**

- What is multi-agent collaboration?
- How agents work together
- Available specialized agents
- Creating custom agents
- Troubleshooting agent issues
- Performance tuning
- Best practices

---

## Feature 4: Visual Circuit Builder Integration

**Objective:** Integrate a drag-and-drop visual circuit builder that allows users to design quantum circuits graphically, with AI-assisted suggestions and automatic code generation.

**Value Proposition:**

- **Accessibility**: No coding required for circuit creation
- **Visual Learning**: See circuit structure at a glance
- **Rapid Prototyping**: Quickly experiment with different gate combinations
- **AI Assistance**: Get suggestions for optimal gate placement
- **Multi-Format Export**: Generate code for Qiskit, Cirq, PennyLane, etc.
- **Educational**: Perfect for learning quantum computing concepts

### Step 4.1: Design Visual Circuit Architecture

**Technical Background:**

- Visual circuit builders use canvas-based rendering
- Gates represented as graphical elements with input/output ports
- Wires connect gates to show qubit flow
- Support for classical bits and measurements
- Real-time validation and simulation preview

**Architecture Components:**

```
┌────────────────────────────────────────────────┐
│          Visual Circuit Builder                │
├────────────────────────────────────────────────┤
│  Canvas Layer (rendering gates and wires)      │
│  Gate Palette (drag source for gates)          │
│  Circuit Validator (check circuit validity)    │
│  Code Generator (convert visual to code)       │
│  AI Suggester (recommend next gates)           │
│  Export Manager (save in multiple formats)     │
└────────────────────────────────────────────────┘
```

**Visual Representation:**

```
Qubit 0: ──H────●────X────M──
                │    │
Qubit 1: ───────X────●────M──
                     │
Qubit 2: ────────────X────M──
```

**Key Design Principles:**

- Use web-based canvas (HTML5 Canvas or SVG) for rendering
- Support touch and mouse interactions
- Undo/redo for every action
- Real-time circuit updates
- Responsive layout (works on different screen sizes)

### Step 4.2: Choose Rendering Technology

**Technical Approach:**

- Evaluate rendering options for circuit visualization
- Choose technology that balances performance and features
- Ensure cross-platform compatibility
- Support export to common image formats

**Rendering Options:**

1. **Web-Based (Recommended)**:
   - **Technology**: HTML5 Canvas + JavaScript/TypeScript
   - **Frameworks**: Fabric.js, Konva.js, or custom canvas rendering
   - **Integration**: Embed in TUI using webview or separate web UI
   - **Pros**: Rich ecosystem, easy styling, good performance
   - **Cons**: Requires web integration

2. **Native Python (Alternative)**:
   - **Technology**: PyQt5/PyQt6 with QGraphicsScene
   - **Libraries**: PyQt5, PyQt6, or PySide6
   - **Integration**: Native desktop widget
   - **Pros**: No web dependency, tight Python integration
   - **Cons**: More complex GUI programming

3. **Terminal-Based (For TUI)**:
   - **Technology**: Unicode box drawing characters
   - **Libraries**: Rich, Textual
   - **Integration**: Direct TUI integration
   - **Pros**: Works in terminal, no external dependencies
   - **Cons**: Limited visual fidelity

**Recommended Choice: Web-Based with Fabric.js**

- Fabric.js provides object-oriented canvas API
- Built-in drag-and-drop support
- Easy event handling
- Excellent documentation
- Active community

**Integration Strategy:**

- Create separate web interface for circuit builder
- Communicate with Proxima backend via REST API or WebSocket
- Optionally embed in TUI using terminal web browser (browsh) or external browser

### Step 4.3: Implement Canvas and Gate Rendering

**Technical Approach:**

- Create canvas element for circuit drawing
- Define gate visual representations
- Implement rendering pipeline
- Support zoom and pan

**Canvas Setup** (`src/proxima/circuit_builder/web/canvas.ts`):

**Gate Visual Definitions:**

```typescript
interface GateVisual {
  id: string;
  type: GateType; // 'H', 'X', 'Y', 'Z', 'CNOT', 'CZ', etc.
  x: number; // Position on canvas
  y: number;
  width: number;
  height: number;
  style: GateStyle;
  inputs: Port[]; // Input connection points
  outputs: Port[]; // Output connection points
}

interface Port {
  id: string;
  x: number; // Relative to gate
  y: number;
  type: "input" | "output";
  connected: boolean;
}
```

**Gate Rendering:**

- Single-qubit gates: Square boxes with gate label (H, X, Y, Z, etc.)
- Two-qubit gates: Connected circles/squares (CNOT, CZ, SWAP)
- Multi-qubit gates: Larger boxes spanning multiple qubits
- Measurement: Meter icon on target qubit
- Barriers: Dashed vertical line across all qubits

**Gate Styles:**

```typescript
const gateStyles = {
  H: { color: "#4A90E2", label: "H", description: "Hadamard" },
  X: { color: "#E24A4A", label: "X", description: "Pauli X (NOT)" },
  Y: { color: "#E2A04A", label: "Y", description: "Pauli Y" },
  Z: { color: "#9B4AE2", label: "Z", description: "Pauli Z" },
  CNOT: { color: "#4AE290", controlStyle: "dot", targetStyle: "X" },
  CZ: { color: "#4AE2E2", controlStyle: "dot", targetStyle: "dot" },
  // ... more gates
};
```

**Rendering Pipeline:**

1. Clear canvas
2. Draw qubit wires (horizontal lines)
3. Draw gates in order (left to right)
4. Draw connections between gates
5. Draw measurement symbols
6. Render selection indicators
7. Apply zoom/pan transformation

**Zoom and Pan:**

- Mouse wheel for zoom (1x to 5x)
- Click and drag to pan canvas
- Fit-to-screen button
- Reset view button

### Step 4.4: Create Gate Palette

**Technical Approach:**

- Display available gates in sidebar
- Support drag-and-drop from palette to canvas
- Organize gates by category
- Include search functionality

**Palette Layout:**

```
┌─ Gate Palette ────────┐
│ 🔍 Search gates...     │
├───────────────────────┤
│ 📁 Single-Qubit       │
│   [H] Hadamard        │
│   [X] Pauli X         │
│   [Y] Pauli Y         │
│   [Z] Pauli Z         │
│   [S] S Gate          │
│   [T] T Gate          │
│   [Rx] Rotation X     │
│   [Ry] Rotation Y     │
│   [Rz] Rotation Z     │
│                       │
│ 📁 Two-Qubit          │
│   [CNOT] Control-NOT  │
│   [CZ] Control-Z      │
│   [SWAP] Swap         │
│   [CRx] Control-Rx    │
│                       │
│ 📁 Multi-Qubit        │
│   [CCX] Toffoli       │
│   [CSWAP] Fredkin     │
│                       │
│ 📁 Measurement        │
│   [M] Measure         │
│   [Reset] Reset       │
│                       │
│ 📁 Utilities          │
│   [Barrier] Barrier   │
│   [Comment] Comment   │
└───────────────────────┘
```

**Drag-and-Drop Implementation:**

```typescript
class GatePalette {
  setupDragHandlers(): void {
    // For each gate in palette
    this.gates.forEach((gate) => {
      gate.element.addEventListener("dragstart", (e) => {
        e.dataTransfer.setData("gate-type", gate.type);
        e.dataTransfer.effectAllowed = "copy";
      });
    });
  }
}

class CircuitCanvas {
  setupDropHandlers(): void {
    this.canvas.addEventListener("dragover", (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
    });

    this.canvas.addEventListener("drop", (e) => {
      e.preventDefault();
      const gateType = e.dataTransfer.getData("gate-type");
      const position = this.getCanvasPosition(e);
      const qubit = this.getQubitAtPosition(position.y);
      this.addGate(gateType, qubit, position.x);
    });
  }
}
```

**Search Functionality:**

- Filter gates by name or description
- Highlight matching gates
- Show gate usage hints
- Recently used gates section

### Step 4.5: Implement Gate Placement and Connection

**Technical Approach:**

- Snap gates to qubit wire positions
- Validate gate placement (no overlaps)
- Auto-connect gates in sequence
- Support multi-qubit gate placement

**Gate Placement Logic:**

```typescript
class CircuitManager {
  addGate(
    gateType: string,
    targetQubits: number[],
    timeSlot: number,
  ): Gate | null {
    // Validate placement
    if (!this.canPlaceGate(targetQubits, timeSlot)) {
      this.showError("Cannot place gate here");
      return null;
    }

    // Create gate object
    const gate = new Gate({
      type: gateType,
      qubits: targetQubits,
      position: timeSlot,
      params: this.getDefaultParams(gateType),
    });

    // Add to circuit
    this.circuit.addGate(gate);

    // Render
    this.renderGate(gate);

    // Add to undo stack
    this.undoStack.push(new AddGateAction(gate));

    return gate;
  }

  canPlaceGate(qubits: number[], timeSlot: number): boolean {
    // Check if qubits are valid
    if (qubits.some((q) => q >= this.numQubits)) {
      return false;
    }

    // Check for gate collisions
    const existingGates = this.getGatesAt(timeSlot);
    for (const gate of existingGates) {
      if (this.gatesOverlap(qubits, gate.qubits)) {
        return false;
      }
    }

    return true;
  }
}
```

**Multi-Qubit Gate Placement:**

- First click: Select control qubit
- Second click: Select target qubit
- Draw visual guide line during selection
- Cancel with Escape key

**Gate Parameters:**

- For parameterized gates (Rx, Ry, Rz), show parameter input dialog
- Support angle input in degrees or radians
- Validate parameter ranges
- Show parameter value on gate label

### Step 4.6: Add Circuit Editing Features

**Technical Approach:**

- Support gate selection, moving, and deletion
- Implement undo/redo stack
- Add copy/paste functionality
- Support circuit templates

**Editing Operations:**

1. **Selection**: Click gate to select, Ctrl+Click for multi-select
2. **Moving**: Drag selected gates to new position
3. **Deletion**: Delete key or right-click > Delete
4. **Copy/Paste**: Ctrl+C / Ctrl+V
5. **Duplicate**: Ctrl+D to duplicate selected gates

**Undo/Redo Implementation:**

```typescript
interface Action {
  execute(): void;
  undo(): void;
  redo(): void;
}

class AddGateAction implements Action {
  constructor(private gate: Gate) {}

  execute(): void {
    this.circuit.addGate(this.gate);
  }

  undo(): void {
    this.circuit.removeGate(this.gate.id);
  }

  redo(): void {
    this.execute();
  }
}

class UndoManager {
  private undoStack: Action[] = [];
  private redoStack: Action[] = [];

  executeAction(action: Action): void {
    action.execute();
    this.undoStack.push(action);
    this.redoStack = []; // Clear redo on new action
  }

  undo(): void {
    const action = this.undoStack.pop();
    if (action) {
      action.undo();
      this.redoStack.push(action);
    }
  }

  redo(): void {
    const action = this.redoStack.pop();
    if (action) {
      action.redo();
      this.undoStack.push(action);
    }
  }
}
```

**Circuit Templates:**

- Provide common circuit patterns (Bell state, GHZ, QFT, etc.)
- User can save custom templates
- Load template and customize
- Template library browser

### Step 4.7: Integrate AI-Assisted Gate Suggestions

**Technical Approach:**

- Analyze current circuit state
- Use LLM to suggest next optimal gate
- Show suggestions as ghost gates
- User can accept or dismiss suggestions

**AI Suggestion Engine** (`src/proxima/circuit_builder/ai_suggester.py`):

**Suggestion Logic:**

```python
class AIGateSuggester:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.suggestion_history = []

    async def suggest_next_gates(
        self,
        current_circuit: Circuit,
        goal: str = None
    ) -> List[GateSuggestion]:
        # Analyze circuit structure
        analysis = self.analyze_circuit(current_circuit)

        # Build prompt
        prompt = f"""
        Current quantum circuit:
        {current_circuit.to_qasm()}

        Circuit analysis:
        - Qubits: {analysis['num_qubits']}
        - Gates: {analysis['gate_count']}
        - Entanglement: {analysis['entanglement_pattern']}

        Goal: {goal or 'Continue building this circuit'}

        Suggest 3 most appropriate next gates to add, considering:
        1. Quantum algorithm patterns
        2. Circuit optimization
        3. Common quantum operations

        Provide suggestions with reasoning.
        """

        # Get LLM response
        response = await self.llm.generate(prompt)

        # Parse suggestions
        suggestions = self.parse_suggestions(response)

        return suggestions
```

**Displaying Suggestions:**

```
┌─ AI Suggestions ──────────────────────┐
│ 💡 Based on your Bell state circuit:  │
│                                       │
│ 1. Add CNOT(0→1)  [90% confidence]   │
│    Reason: Creates entanglement       │
│    [Apply]  [Dismiss]                 │
│                                       │
│ 2. Add H gate on q2  [75% confidence]│
│    Reason: Extends to 3-qubit GHZ     │
│    [Apply]  [Dismiss]                 │
│                                       │
│ 3. Add Measurement  [85% confidence]  │
│    Reason: Complete circuit           │
│    [Apply]  [Dismiss]                 │
└───────────────────────────────────────┘
```

**Suggestion Display:**

- Show ghost gates on canvas (semi-transparent)
- Highlight suggested positions
- Provide reasoning tooltips
- Track suggestion acceptance rate

### Step 4.8: Implement Circuit Validation

**Technical Approach:**

- Real-time validation during circuit construction
- Check for common errors
- Provide helpful error messages
- Suggest fixes for issues

**Validation Rules:**

```typescript
class CircuitValidator {
  validate(circuit: Circuit): ValidationResult {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    // Check qubit indices
    if (!this.checkQubitIndices(circuit)) {
      errors.push({
        type: "invalid_qubit",
        message: "Gate targets invalid qubit",
        gates: this.findInvalidQubitGates(circuit),
      });
    }

    // Check measurement placement
    if (!this.checkMeasurements(circuit)) {
      warnings.push({
        type: "measurement_not_final",
        message: "Measurements should be at circuit end",
        suggestion: "Move measurements to final time slot",
      });
    }

    // Check gate ordering
    if (!this.checkGateOrdering(circuit)) {
      warnings.push({
        type: "suboptimal_ordering",
        message: "Gate ordering could be optimized",
        suggestion: "Reorder gates to reduce circuit depth",
      });
    }

    // Check for disconnected qubits
    const disconnected = this.findDisconnectedQubits(circuit);
    if (disconnected.length > 0) {
      warnings.push({
        type: "unused_qubits",
        message: `Qubits ${disconnected.join(", ")} are not used`,
        suggestion: "Remove unused qubits or add gates",
      });
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings,
    };
  }
}
```

**Error Display:**

- Red border around invalid gates
- Error icon with tooltip
- Validation panel showing all issues
- Quick-fix buttons for common problems

### Step 4.9: Create Code Generator

**Technical Approach:**

- Convert visual circuit to multiple code formats
- Support Qiskit, Cirq, PennyLane, Braket, OpenQASM
- Generate clean, documented code
- Include execution boilerplate

**Code Generator** (`src/proxima/circuit_builder/code_generator.py`):

**Multi-Format Support:**

```python
class CircuitCodeGenerator:
    def generate_code(
        self,
        circuit: VisualCircuit,
        target_framework: str,
        include_execution: bool = True
    ) -> str:
        if target_framework == 'qiskit':
            return self._generate_qiskit(circuit, include_execution)
        elif target_framework == 'cirq':
            return self._generate_cirq(circuit, include_execution)
        elif target_framework == 'pennylane':
            return self._generate_pennylane(circuit, include_execution)
        elif target_framework == 'braket':
            return self._generate_braket(circuit, include_execution)
        elif target_framework == 'qasm':
            return self._generate_qasm(circuit)
        else:
            raise ValueError(f'Unsupported framework: {target_framework}')

    def _generate_qiskit(
        self,
        circuit: VisualCircuit,
        include_execution: bool
    ) -> str:
        # Generate imports
        code = 'from qiskit import QuantumCircuit\n'
        code += 'from qiskit import Aer, execute\n\n'

        # Create circuit
        code += f'# Create quantum circuit with {circuit.num_qubits} qubits\n'
        code += f'qc = QuantumCircuit({circuit.num_qubits})\n\n'

        # Add gates
        for gate in circuit.gates:
            code += self._qiskit_gate_code(gate) + '\n'

        # Add measurements
        if circuit.has_measurements:
            code += f'\n# Measure all qubits\n'
            code += f'qc.measure_all()\n'

        # Add execution code
        if include_execution:
            code += '\n# Execute circuit\n'
            code += 'simulator = Aer.get_backend("qasm_simulator")\n'
            code += 'job = execute(qc, simulator, shots=1024)\n'
            code += 'result = job.result()\n'
            code += 'counts = result.get_counts()\n'
            code += 'print(counts)\n'

        return code
```

**Generated Code Example:**

```python
# Qiskit output
from qiskit import QuantumCircuit
from qiskit import Aer, execute

# Create quantum circuit with 2 qubits
qc = QuantumCircuit(2)

# Create Bell state
qc.h(0)  # Hadamard on qubit 0
qc.cx(0, 1)  # CNOT from qubit 0 to 1

# Measure all qubits
qc.measure_all()

# Execute circuit
simulator = Aer.get_backend("qasm_simulator")
job = execute(qc, simulator, shots=1024)
result = job.result()
counts = result.get_counts()
print(counts)
```

**Code Export Options:**

- Copy to clipboard
- Download as .py file
- Send directly to Proxima backend for execution
- Open in IDE (VS Code integration)

### Step 4.10: Add Circuit Simulation Preview

**Technical Approach:**

- Simulate circuit in real-time as user builds
- Show state vector evolution
- Display measurement probabilities
- Animate qubit state changes

**Simulation Engine** (`src/proxima/circuit_builder/simulator.py`):

**Real-time Simulation:**

```python
class CircuitSimulator:
    def __init__(self, backend: str = 'statevector'):
        self.backend = backend
        self.cache = {}  # Cache simulation results

    async def simulate(
        self,
        circuit: VisualCircuit
    ) -> SimulationResult:
        # Check cache
        cache_key = circuit.get_hash()
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Convert to backend format
        backend_circuit = circuit.to_backend_format(self.backend)

        # Simulate
        if self.backend == 'statevector':
            result = self._simulate_statevector(backend_circuit)
        elif self.backend == 'density_matrix':
            result = self._simulate_density_matrix(backend_circuit)
        else:
            result = self._simulate_sampling(backend_circuit)

        # Cache result
        self.cache[cache_key] = result

        return result

    def _simulate_statevector(self, circuit) -> SimulationResult:
        # Use numpy for small circuits (<10 qubits)
        # Use backend for larger circuits
        # Return state vector and probabilities
        pass
```

**Preview Display:**

```
┌─ Simulation Preview ──────────────────────┐
│ State Vector (2 qubits):                  │
│ |00⟩: 0.707 (50.0%)  ████████████         │
│ |01⟩: 0.000 (0.0%)                        │
│ |10⟩: 0.000 (0.0%)                        │
│ |11⟩: 0.707 (50.0%)  ████████████         │
│                                           │
│ Measurement Probabilities:                │
│ [Pie Chart showing 50% |00⟩, 50% |11⟩]   │
│                                           │
│ Entanglement: Yes                         │
│ Circuit Depth: 2                          │
│ Gate Count: 2                             │
│                                           │
│ [⚡ Re-simulate] [📊 Detailed View]       │
└───────────────────────────────────────────┘
```

**Performance Optimization:**

- Only simulate when circuit changes
- Use incremental simulation for small changes
- Limit simulation to small circuits (< 15 qubits)
- Offer full simulation via Proxima backend

### Step 4.11: Implement Import from Code

**Technical Approach:**

- Parse existing circuit code
- Convert to visual representation
- Support multiple input formats
- Handle complex circuits gracefully

**Circuit Parser** (`src/proxima/circuit_builder/parser.py`):

**Parsing Strategy:**

```python
class CircuitParser:
    def parse(
        self,
        code: str,
        source_format: str
    ) -> VisualCircuit:
        if source_format == 'qiskit':
            return self._parse_qiskit(code)
        elif source_format == 'cirq':
            return self._parse_cirq(code)
        elif source_format == 'qasm':
            return self._parse_qasm(code)
        else:
            raise ValueError(f'Unsupported format: {source_format}')

    def _parse_qasm(self, qasm_code: str) -> VisualCircuit:
        # Parse QASM using regex or parser library
        # Extract qubit declarations
        # Extract gate operations
        # Build VisualCircuit object

        lines = qasm_code.split('\n')
        circuit = VisualCircuit()

        for line in lines:
            line = line.strip()

            # Parse qubit declaration
            if line.startswith('qreg'):
                match = re.match(r'qreg q\[(\d+)\]', line)
                if match:
                    num_qubits = int(match.group(1))
                    circuit.num_qubits = num_qubits

            # Parse gates
            elif line.startswith('h'):
                # Parse Hadamard: h q[0];
                qubit = self._extract_qubit(line)
                circuit.add_gate('H', [qubit])

            elif line.startswith('cx'):
                # Parse CNOT: cx q[0], q[1];
                control, target = self._extract_two_qubits(line)
                circuit.add_gate('CNOT', [control, target])

            # ... more gate types

        return circuit
```

**Import UI:**

```
┌─ Import Circuit ──────────────────────────┐
│ Paste code or upload file:               │
│ ┌───────────────────────────────────────┐ │
│ │ OPENQASM 2.0;                         │ │
│ │ include "qelib1.inc";                 │ │
│ │ qreg q[2];                            │ │
│ │ h q[0];                               │ │
│ │ cx q[0], q[1];                        │ │
│ └───────────────────────────────────────┘ │
│                                           │
│ Format: ○ Auto-detect ● Qiskit ○ Cirq   │
│         ○ QASM ○ PennyLane               │
│                                           │
│ [Import] [Upload File] [Cancel]          │
└───────────────────────────────────────────┘
```

### Step 4.12: Add Collaboration Features

**Technical Approach:**

- Share circuits with other users
- Real-time collaborative editing
- Version control for circuits
- Comment and annotation system

**Sharing Implementation:**

```python
class CircuitSharing:
    async def share_circuit(
        self,
        circuit: VisualCircuit,
        share_settings: ShareSettings
    ) -> ShareLink:
        # Generate unique circuit ID
        circuit_id = generate_unique_id()

        # Store circuit in database
        await self.db.store_circuit(circuit_id, circuit)

        # Create share link
        share_link = f'proxima://circuit/{circuit_id}'

        # Set permissions
        if share_settings.allow_edit:
            permissions = ['view', 'edit']
        else:
            permissions = ['view']

        # Store sharing metadata
        await self.db.store_share_metadata(
            circuit_id,
            permissions,
            share_settings.expiration
        )

        return ShareLink(url=share_link, permissions=permissions)

    async def load_shared_circuit(
        self,
        share_link: str
    ) -> VisualCircuit:
        # Extract circuit ID from link
        circuit_id = self.extract_circuit_id(share_link)

        # Load circuit from database
        circuit = await self.db.load_circuit(circuit_id)

        return circuit
```

**Real-time Collaboration:**

- Use WebSocket for live updates
- Show cursors of other users
- Conflict resolution for simultaneous edits
- Chat sidebar for communication

### Step 4.13: Create Mobile-Friendly Interface

**Technical Approach:**

- Responsive design for tablets and phones
- Touch-optimized controls
- Simplified interface for small screens
- Progressive Web App (PWA) support

**Responsive Layout:**

```
Desktop:
┌────────────────────────────────────────┐
│ Toolbar          │  Canvas            │
├──────────────────┼────────────────────┤
│ Gate Palette     │  Circuit           │
│                  │                    │
│ [Gates...]       │  q0: ──H──●──      │
│                  │           │        │
│                  │  q1: ─────X──      │
│                  │                    │
├──────────────────┼────────────────────┤
│ Properties       │  Simulation        │
└────────────────────────────────────────┘

Mobile:
┌──────────────────┐
│ ☰ Menu      [+]  │
├──────────────────┤
│   Circuit        │
│                  │
│  q0: ──H──●──    │
│           │      │
│  q1: ─────X──    │
│                  │
├──────────────────┤
│ [Gates ▼]        │
└──────────────────┘
```

**Touch Gestures:**

- Tap to select gate
- Long-press for context menu
- Pinch to zoom
- Two-finger pan
- Swipe between tabs

### Step 4.14: Integrate with Proxima Backend

**Technical Approach:**

- Connect circuit builder to Proxima execution engine
- Send circuits for simulation
- Retrieve and display results
- Save circuits to workspace

**Backend Integration** (`src/proxima/circuit_builder/backend_connector.py`):

**API Communication:**

```python
class ProximaBackendConnector:
    def __init__(self, api_url: str, auth_token: str = None):
        self.api_url = api_url
        self.auth_token = auth_token
        self.http_client = httpx.AsyncClient()

    async def execute_circuit(
        self,
        circuit: VisualCircuit,
        backend: str = 'auto',
        shots: int = 1024
    ) -> ExecutionResult:
        # Convert circuit to QASM
        qasm = circuit.to_qasm()

        # Prepare request
        request_data = {
            'circuit': qasm,
            'backend': backend,
            'shots': shots,
            'format': 'qasm'
        }

        # Send to Proxima API
        response = await self.http_client.post(
            f'{self.api_url}/api/v1/circuits/submit',
            json=request_data,
            headers={'Authorization': f'Bearer {self.auth_token}'}
        )

        result = response.json()
        return ExecutionResult.from_json(result)

    async def save_circuit(
        self,
        circuit: VisualCircuit,
        name: str,
        description: str = ''
    ) -> str:
        # Save circuit to Proxima workspace
        circuit_data = {
            'name': name,
            'description': description,
            'circuit': circuit.to_json(),
            'metadata': {
                'created_by': 'circuit_builder',
                'num_qubits': circuit.num_qubits,
                'gate_count': circuit.gate_count
            }
        }

        response = await self.http_client.post(
            f'{self.api_url}/api/v1/circuits',
            json=circuit_data,
            headers={'Authorization': f'Bearer {self.auth_token}'}
        )

        return response.json()['circuit_id']
```

**Execute Button Integration:**

```
[▶ Execute]  Backend: [Cirq ▼]  Shots: [1024]

After execution:
┌─ Execution Results ────────────────────┐
│ Backend: cirq                          │
│ Shots: 1024                            │
│ Duration: 0.234s                       │
│                                        │
│ Results:                               │
│ |00⟩: 512 (50.0%) ████████████         │
│ |11⟩: 512 (50.0%) ████████████         │
│                                        │
│ [Export] [Visualize] [Save]            │
└────────────────────────────────────────┘
```

### Step 4.15: Document Visual Circuit Builder

**Technical Approach:**

- Create user guide with screenshots
- Video tutorials for common tasks
- Developer docs for extending builder
- Keyboard shortcuts reference

**Documentation Files:**

1. **`docs/user-guide/circuit-builder.md`**: Complete user guide
2. **`docs/user-guide/circuit-builder-tutorial.md`**: Step-by-step tutorial
3. **`docs/developer-guide/circuit-builder-api.md`**: API reference
4. **`docs/user-guide/circuit-builder-shortcuts.md`**: Keyboard shortcuts

**Tutorial Topics:**

- Creating your first circuit
- Using gate palette
- AI-assisted circuit design
- Exporting to code
- Executing circuits
- Sharing circuits
- Advanced tips and tricks

**Keyboard Shortcuts:**

```
General:
  Ctrl+N   New circuit
  Ctrl+O   Open circuit
  Ctrl+S   Save circuit
  Ctrl+Z   Undo
  Ctrl+Y   Redo

Editing:
  Delete   Remove selected gates
  Ctrl+C   Copy
  Ctrl+V   Paste
  Ctrl+D   Duplicate
  H        Add Hadamard gate
  X        Add X gate
  C        Add CNOT gate

Navigation:
  +/-      Zoom in/out
  Space    Pan mode (drag canvas)
  F        Fit to screen
  Home     Reset view
```

---

## Feature 5: Real-Time Quantum State Visualization

**Objective:** Create advanced real-time visualization system for quantum states, including Bloch sphere animations, state vector evolution, density matrices, and measurement dynamics.

**Value Proposition:**

- **Visual Understanding**: See quantum states evolve in real-time
- **Educational Tool**: Perfect for learning quantum mechanics concepts
- **Debugging Aid**: Identify issues in quantum circuits visually
- **Interactive Exploration**: Manipulate states and see immediate effects
- **Multiple Representations**: Bloch sphere, state vector, density matrix, Q-sphere
- **Animation Support**: Watch state evolution through circuit execution

### Step 5.1: Design Visualization Architecture

**Technical Background:**

- Quantum states can be visualized in multiple ways
- Bloch sphere: Single-qubit states as points on sphere
- State vector: Complex amplitudes as bar charts
- Density matrix: Heatmap representation
- Q-sphere: Multi-qubit generalization of Bloch sphere
- Wigner function: Phase-space representation

**Visualization Types:**

```
┌─────────────────────────────────────────┐
│  Visualization Modes                    │
├─────────────────────────────────────────┤
│  1. Bloch Sphere (1 qubit)             │
│  2. State Vector (amplitude/phase)      │
│  3. Density Matrix (heatmap)            │
│  4. Q-Sphere (multi-qubit)              │
│  5. Probability Distribution (bars)     │
│  6. Wigner Function (contour plot)      │
│  7. Pauli Expectation Values            │
│  8. Entanglement Visualization          │
└─────────────────────────────────────────┘
```

**Architecture Components:**

- **Rendering Engine**: 3D graphics for Bloch sphere, 2D for others
- **Animation System**: Smooth transitions between states
- **Interaction Handler**: Mouse/touch controls for rotation
- **Data Pipeline**: Convert quantum states to visual coordinates
- **Export System**: Save visualizations as images/videos

**Key Design Principles:**

- Real-time updates (< 16ms per frame)
- Smooth animations with interpolation
- Multiple simultaneous views
- Responsive to window resize
- Hardware-accelerated rendering

### Step 5.2: Choose Visualization Technology Stack

**Technical Approach:**

- Evaluate 3D rendering libraries
- Choose web-based or native solution
- Ensure cross-platform compatibility
- Support export to images and animations

**Technology Options:**

1. **Web-Based (Recommended)**:
   - **3D Library**: Three.js (WebGL)
   - **2D Charts**: D3.js, Chart.js, or Plotly.js
   - **Animation**: GSAP (GreenSock)
   - **Integration**: Embed in web UI or separate window
   - **Pros**: Rich ecosystem, excellent documentation, portable
   - **Cons**: Requires web browser

2. **Native Python (Alternative)**:
   - **3D Library**: PyQt3D, VTK, or Mayavi
   - **2D Charts**: Matplotlib, Plotly
   - **Animation**: Matplotlib animation API
   - **Integration**: Native desktop widgets
   - **Pros**: No web dependency, Python-native
   - **Cons**: More complex setup

3. **Hybrid Approach**:
   - **Backend**: Python for state calculations
   - **Frontend**: Three.js for rendering
   - **Communication**: WebSocket for real-time updates
   - **Benefits**: Best of both worlds

**Recommended Choice: Three.js + D3.js**

- Three.js for 3D Bloch sphere rendering
- D3.js for 2D plots (state vector, density matrix)
- WebSocket for real-time data streaming
- Export via canvas.toBlob()

### Step 5.3: Implement Bloch Sphere Renderer

**Technical Approach:**

- Create 3D Bloch sphere using Three.js
- Render state as arrow/point on sphere
- Support rotation and zoom
- Animate state transitions

**Bloch Sphere Implementation** (`src/proxima/visualization/web/bloch_sphere.ts`):

**Sphere Geometry:**

```typescript
class BlochSphere {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private sphere: THREE.Mesh;
  private stateVector: THREE.ArrowHelper;

  constructor(container: HTMLElement) {
    this.initScene();
    this.createSphere();
    this.createAxes();
    this.setupControls();
    this.animate();
  }

  private createSphere(): void {
    // Create transparent sphere
    const geometry = new THREE.SphereGeometry(1, 32, 32);
    const material = new THREE.MeshPhongMaterial({
      color: 0x3498db,
      transparent: true,
      opacity: 0.3,
      wireframe: false,
    });
    this.sphere = new THREE.Mesh(geometry, material);
    this.scene.add(this.sphere);

    // Add wireframe overlay
    const wireframe = new THREE.WireframeGeometry(geometry);
    const line = new THREE.LineSegments(wireframe);
    line.material.opacity = 0.5;
    this.scene.add(line);
  }

  private createAxes(): void {
    // X axis (red)
    this.addArrow(new THREE.Vector3(1, 0, 0), 0xff0000, "X");

    // Y axis (green)
    this.addArrow(new THREE.Vector3(0, 1, 0), 0x00ff00, "Y");

    // Z axis (blue) - |0⟩ at top, |1⟩ at bottom
    this.addArrow(new THREE.Vector3(0, 0, 1), 0x0000ff, "Z");

    // Add labels
    this.addLabel(new THREE.Vector3(0, 0, 1.2), "|0⟩");
    this.addLabel(new THREE.Vector3(0, 0, -1.2), "|1⟩");
    this.addLabel(new THREE.Vector3(1.2, 0, 0), "|+⟩");
    this.addLabel(new THREE.Vector3(-1.2, 0, 0), "|-⟩");
  }

  updateState(theta: number, phi: number): void {
    // Convert Bloch angles to Cartesian
    const x = Math.sin(theta) * Math.cos(phi);
    const y = Math.sin(theta) * Math.sin(phi);
    const z = Math.cos(theta);

    // Update state vector arrow
    const direction = new THREE.Vector3(x, y, z);
    this.stateVector.setDirection(direction);

    // Animate transition
    this.animateTransition(direction);
  }

  private animateTransition(newDirection: THREE.Vector3): void {
    // Smooth interpolation from current to new direction
    const currentDirection = this.stateVector.position.clone();

    gsap.to(currentDirection, {
      x: newDirection.x,
      y: newDirection.y,
      z: newDirection.z,
      duration: 0.3,
      ease: "power2.inOut",
      onUpdate: () => {
        this.stateVector.setDirection(currentDirection);
      },
    });
  }
}
```

**State Conversion:**

```python
def state_to_bloch(state_vector: np.ndarray) -> Tuple[float, float]:
    """Convert single-qubit state to Bloch sphere angles."""
    # Normalize
    state = state_vector / np.linalg.norm(state_vector)

    # Extract amplitudes
    alpha = state[0]  # Amplitude of |0⟩
    beta = state[1]   # Amplitude of |1⟩

    # Calculate Bloch sphere angles
    theta = 2 * np.arccos(np.abs(alpha))
    phi = np.angle(beta) - np.angle(alpha)

    return theta, phi
```

**Interactive Controls:**

- Mouse drag to rotate sphere
- Scroll to zoom
- Click state to see details
- Reset view button

### Step 5.4: Create State Vector Visualizer

**Technical Approach:**

- Display amplitude and phase for each basis state
- Use bar charts for amplitudes
- Use color wheels or angle indicators for phase
- Support multi-qubit states

**State Vector Visualization** (`src/proxima/visualization/web/state_vector.ts`):

**Layout Design:**

```
┌─ State Vector ────────────────────────────┐
│ Basis State │ Amplitude │ Phase │ Prob   │
├─────────────┼───────────┼───────┼────────┤
│ |00⟩       │ ████ 0.71 │ 0°    │ 50.0%  │
│ |01⟩       │           │ -     │ 0.0%   │
│ |10⟩       │           │ -     │ 0.0%   │
│ |11⟩       │ ████ 0.71 │ 180°  │ 50.0%  │
└───────────────────────────────────────────┘

Visual representation:
|00⟩: █████████████ (0.707, phase: 0°)
|11⟩: █████████████ (0.707, phase: 180°)
```

**Implementation:**

```typescript
class StateVectorVisualizer {
  renderStateVector(state: ComplexArray): void {
    const numQubits = Math.log2(state.length);
    const container = document.getElementById("state-vector");

    // Create table rows for each basis state
    state.forEach((amplitude, index) => {
      const basisState = this.indexToBasisState(index, numQubits);
      const magnitude = amplitude.abs();
      const phase = (amplitude.arg() * 180) / Math.PI;
      const probability = magnitude * magnitude;

      const row = this.createStateRow(
        basisState,
        magnitude,
        phase,
        probability,
      );

      container.appendChild(row);
    });
  }

  private createStateRow(
    basis: string,
    magnitude: number,
    phase: number,
    probability: number,
  ): HTMLElement {
    const row = document.createElement("div");
    row.className = "state-row";

    // Basis state label
    const label = document.createElement("span");
    label.textContent = `|${basis}⟩`;
    label.className = "basis-label";

    // Amplitude bar
    const amplitudeBar = document.createElement("div");
    amplitudeBar.className = "amplitude-bar";
    amplitudeBar.style.width = `${magnitude * 100}%`;
    amplitudeBar.textContent = magnitude.toFixed(3);

    // Phase indicator (color wheel)
    const phaseIndicator = this.createPhaseWheel(phase);

    // Probability
    const probText = document.createElement("span");
    probText.textContent = `${(probability * 100).toFixed(1)}%`;
    probText.className = "probability";

    row.appendChild(label);
    row.appendChild(amplitudeBar);
    row.appendChild(phaseIndicator);
    row.appendChild(probText);

    return row;
  }

  private createPhaseWheel(phaseDegrees: number): HTMLElement {
    // Create circular phase indicator
    const wheel = document.createElement("div");
    wheel.className = "phase-wheel";

    // Color based on phase
    const hue = (phaseDegrees + 180) % 360; // Map -180..180 to 0..360
    wheel.style.background = `hsl(${hue}, 70%, 60%)`;

    // Add angle text
    wheel.textContent = `${phaseDegrees.toFixed(0)}°`;

    return wheel;
  }
}
```

**Phase Visualization:**

- Use color wheel (hue represents phase angle)
- Arrow pointing in phase direction
- Numeric display of angle
- Tooltip with complex notation (a + bi)

### Step 5.5: Implement Density Matrix Heatmap

**Technical Approach:**

- Render density matrix as 2D heatmap
- Use color to represent magnitude
- Display both real and imaginary parts
- Support interactive tooltips

**Density Matrix Visualization** (`src/proxima/visualization/web/density_matrix.ts`):

**Heatmap Implementation:**

```typescript
class DensityMatrixVisualizer {
  renderDensityMatrix(rho: ComplexMatrix): void {
    const size = rho.shape[0];

    // Create two heatmaps: real and imaginary parts
    this.renderRealPart(rho.real);
    this.renderImaginaryPart(rho.imag);
  }

  private renderRealPart(matrix: number[][]): void {
    const canvas = document.getElementById("density-real") as HTMLCanvasElement;
    const ctx = canvas.getContext("2d");
    const size = matrix.length;
    const cellSize = canvas.width / size;

    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const value = matrix[i][j];

        // Map value to color (-1 to 1 → blue to red)
        const color = this.valueToColor(value);

        ctx.fillStyle = color;
        ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);

        // Add grid lines
        ctx.strokeStyle = "#ccc";
        ctx.strokeRect(j * cellSize, i * cellSize, cellSize, cellSize);

        // Add value text for small matrices
        if (size <= 8) {
          ctx.fillStyle = "#000";
          ctx.font = "10px monospace";
          ctx.fillText(value.toFixed(2), j * cellSize + 5, i * cellSize + 15);
        }
      }
    }
  }

  private valueToColor(value: number): string {
    // Blue-white-red colormap
    if (value < 0) {
      const intensity = Math.abs(value);
      const blue = Math.floor(255 * intensity);
      return `rgb(${255 - blue}, ${255 - blue}, 255)`;
    } else {
      const intensity = value;
      const red = Math.floor(255 * intensity);
      return `rgb(255, ${255 - red}, ${255 - red})`;
    }
  }
}
```

**Display Layout:**

```
┌─ Density Matrix ──────────────────────────┐
│ Real Part:                Imaginary Part: │
│ ┌──────────────┐         ┌──────────────┐ │
│ │ [Heatmap]    │         │ [Heatmap]    │ │
│ │              │         │              │ │
│ │ Color scale: │         │ Color scale: │ │
│ │ Blue(-) Red(+)│         │ Blue(-) Red(+)│ │
│ └──────────────┘         └──────────────┘ │
│                                           │
│ Properties:                               │
│ • Purity: 1.000 (pure state)             │
│ • Trace: 1.000                           │
│ • Rank: 1                                │
│ • Entanglement Entropy: 0.693            │
└───────────────────────────────────────────┘
```

**Tooltip on Hover:**

```
Element (i, j): ρ[0][1]
Real: 0.500
Imaginary: 0.000
Magnitude: 0.500
Phase: 0.0°
```

### Step 5.6: Create Q-Sphere Visualization

**Technical Approach:**

- Extend Bloch sphere concept to multi-qubit states
- Plot basis states on sphere surface
- Size/color represents probability
- Support interactive exploration

**Q-Sphere Implementation** (`src/proxima/visualization/web/qsphere.ts`):

**Q-Sphere Algorithm:**

```typescript
class QSphere {
  plotState(state: ComplexArray): void {
    const numQubits = Math.log2(state.length);

    // Map each basis state to point on sphere
    state.forEach((amplitude, index) => {
      const probability = Math.pow(amplitude.abs(), 2);

      if (probability > 0.001) {
        // Skip negligible amplitudes
        // Map index to sphere coordinates
        const [theta, phi] = this.basisToSphere(index, numQubits);

        // Create sphere marker
        const marker = this.createMarker(
          theta,
          phi,
          probability,
          amplitude.arg(),
        );

        this.scene.add(marker);
      }
    });
  }

  private basisToSphere(index: number, numQubits: number): [number, number] {
    // Fibonacci sphere mapping for uniform distribution
    const goldenAngle = Math.PI * (3 - Math.sqrt(5));

    const theta = Math.acos(1 - (2 * (index + 0.5)) / 2 ** numQubits);
    const phi = index * goldenAngle;

    return [theta, phi];
  }

  private createMarker(
    theta: number,
    phi: number,
    probability: number,
    phase: number,
  ): THREE.Mesh {
    // Convert spherical to Cartesian
    const x = Math.sin(theta) * Math.cos(phi);
    const y = Math.sin(theta) * Math.sin(phi);
    const z = Math.cos(theta);

    // Create sphere marker
    const radius = 0.05 + probability * 0.15; // Size based on probability
    const geometry = new THREE.SphereGeometry(radius, 16, 16);

    // Color based on phase
    const hue = ((phase * 180) / Math.PI + 180) % 360;
    const material = new THREE.MeshPhongMaterial({
      color: `hsl(${hue}, 70%, 60%)`,
    });

    const marker = new THREE.Mesh(geometry, material);
    marker.position.set(x * 1.5, y * 1.5, z * 1.5);

    return marker;
  }
}
```

**Visual Representation:**

- Larger spheres = higher probability
- Color = phase angle
- Position = basis state
- Hover shows basis state label

### Step 5.7: Add Animation System

**Technical Approach:**

- Animate state evolution through circuit
- Smooth transitions between gate applications
- Support playback controls (play, pause, step)
- Adjust animation speed

**Animation Controller** (`src/proxima/visualization/animator.py`):

**Animation Pipeline:**

```python
class StateAnimator:
    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
        self.states = []  # State at each step
        self.current_step = 0
        self.playing = False
        self.speed = 1.0  # 1x speed

    def compute_state_evolution(self) -> List[np.ndarray]:
        """Compute state at each gate application."""
        num_qubits = self.circuit.num_qubits
        state = np.zeros(2 ** num_qubits, dtype=complex)
        state[0] = 1.0  # Initialize to |0...0⟩

        self.states = [state.copy()]

        # Apply gates one by one
        for gate in self.circuit.gates:
            state = self.apply_gate(state, gate)
            self.states.append(state.copy())

        return self.states

    def animate_to_step(self, target_step: int) -> None:
        """Animate transition to specific step."""
        if target_step >= len(self.states):
            return

        current_state = self.states[self.current_step]
        target_state = self.states[target_step]

        # Interpolate between states
        num_frames = int(30 * self.speed)  # 30 frames at 1x speed

        for frame in range(num_frames):
            t = frame / num_frames
            interpolated = self.lerp_states(current_state, target_state, t)
            self.update_visualization(interpolated)
            time.sleep(1 / 60)  # 60 FPS

        self.current_step = target_step

    def lerp_states(
        self,
        state1: np.ndarray,
        state2: np.ndarray,
        t: float
    ) -> np.ndarray:
        """Linear interpolation between quantum states."""
        # Normalize interpolated state
        interpolated = (1 - t) * state1 + t * state2
        return interpolated / np.linalg.norm(interpolated)

    def play(self) -> None:
        """Play animation from current step."""
        self.playing = True
        while self.playing and self.current_step < len(self.states) - 1:
            self.step_forward()
            time.sleep(1.0 / self.speed)

    def pause(self) -> None:
        """Pause animation."""
        self.playing = False

    def step_forward(self) -> None:
        """Move to next step."""
        if self.current_step < len(self.states) - 1:
            self.animate_to_step(self.current_step + 1)

    def step_backward(self) -> None:
        """Move to previous step."""
        if self.current_step > 0:
            self.animate_to_step(self.current_step - 1)
```

**Animation Controls:**

```
┌─ Animation Controls ──────────────────────┐
│ [◀◀] [◀] [▶] [▶▶]  Step: 3/10           │
│                                           │
│ Progress: ████████░░░░░░ 30%              │
│                                           │
│ Speed: [0.5x] [1x] [2x] [5x]             │
│                                           │
│ Current Gate: H q[0]                     │
│ Next Gate: CNOT q[0]→q[1]                │
└───────────────────────────────────────────┘
```

### Step 5.8: Implement Measurement Visualization

**Technical Approach:**

- Animate measurement collapse
- Show probability distribution before measurement
- Highlight measured outcome
- Support repeated measurements visualization

**Measurement Animation:**

```typescript
class MeasurementVisualizer {
  async animateMeasurement(
    preState: ComplexArray,
    measuredQubit: number,
    outcome: number,
  ): Promise<void> {
    // Show pre-measurement probabilities
    this.showProbabilities(preState, measuredQubit);
    await this.delay(1000);

    // Animate collapse
    await this.animateCollapse(preState, measuredQubit, outcome);

    // Show post-measurement state
    const postState = this.computePostMeasurementState(
      preState,
      measuredQubit,
      outcome,
    );
    this.updateVisualization(postState);
  }

  private async animateCollapse(
    state: ComplexArray,
    qubit: number,
    outcome: number,
  ): Promise<void> {
    // Highlight measured qubit
    this.highlightQubit(qubit);

    // Show "collapsing" animation
    for (let frame = 0; frame < 30; frame++) {
      const t = frame / 30;

      // Gradually reduce non-outcome amplitudes
      const partialState = this.partialCollapse(state, qubit, outcome, t);
      this.updateVisualization(partialState);

      await this.delay(33); // 30 FPS
    }
  }

  private showProbabilities(state: ComplexArray, qubit: number): void {
    // Calculate P(|0⟩) and P(|1⟩) for measured qubit
    const prob0 = this.calculateMeasurementProb(state, qubit, 0);
    const prob1 = this.calculateMeasurementProb(state, qubit, 1);

    // Display probability bars
    this.displayProbBar("|0⟩", prob0);
    this.displayProbBar("|1⟩", prob1);
  }
}
```

**Measurement Display:**

```
Before Measurement:
┌────────────────────────────────┐
│ Measuring qubit 0:             │
│ P(|0⟩) = 50.0%  ████████       │
│ P(|1⟩) = 50.0%  ████████       │
└────────────────────────────────┘

[Collapse Animation]

After Measurement:
┌────────────────────────────────┐
│ Result: |0⟩                    │
│ State collapsed to |0⟩ branch  │
└────────────────────────────────┘
```

### Step 5.9: Create Entanglement Visualizer

**Technical Approach:**

- Visualize entanglement structure between qubits
- Use network graph to show connections
- Color/width indicates entanglement strength
- Calculate entanglement measures (concurrence, negativity)

**Entanglement Network** (`src/proxima/visualization/entanglement.py`):

**Network Visualization:**

```python
class EntanglementVisualizer:
    def visualize_entanglement(self, state: np.ndarray) -> None:
        """Create network graph of entangled qubits."""
        num_qubits = int(np.log2(len(state)))

        # Calculate pairwise entanglement
        entanglement_matrix = np.zeros((num_qubits, num_qubits))

        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                # Calculate entanglement measure
                entanglement = self.calculate_mutual_information(
                    state, i, j
                )
                entanglement_matrix[i, j] = entanglement
                entanglement_matrix[j, i] = entanglement

        # Create network graph
        self.draw_entanglement_network(entanglement_matrix)

    def calculate_mutual_information(
        self,
        state: np.ndarray,
        qubit1: int,
        qubit2: int
    ) -> float:
        """Calculate quantum mutual information."""
        # Compute reduced density matrices
        rho1 = self.partial_trace(state, keep=[qubit1])
        rho2 = self.partial_trace(state, keep=[qubit2])
        rho12 = self.partial_trace(state, keep=[qubit1, qubit2])

        # Calculate von Neumann entropies
        S1 = self.von_neumann_entropy(rho1)
        S2 = self.von_neumann_entropy(rho2)
        S12 = self.von_neumann_entropy(rho12)

        # Mutual information: I(1:2) = S1 + S2 - S12
        return S1 + S2 - S12

    def draw_entanglement_network(
        self,
        matrix: np.ndarray
    ) -> None:
        """Draw network using D3.js or networkx."""
        num_qubits = len(matrix)

        # Create nodes (qubits)
        nodes = [{'id': i, 'label': f'q{i}'} for i in range(num_qubits)]

        # Create edges (entanglement connections)
        edges = []
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                strength = matrix[i, j]
                if strength > 0.01:  # Threshold
                    edges.append({
                        'source': i,
                        'target': j,
                        'strength': strength
                    })

        # Render network
        self.render_network(nodes, edges)
```

**Network Display:**

```
┌─ Entanglement Network ────────────────────┐
│         q0                                 │
│        ╱  ╲                               │
│   0.8 ╱    ╲ 0.3                          │
│      ╱      ╲                             │
│     q1────q2                              │
│       0.6                                 │
│                                           │
│ Legend:                                   │
│ Line thickness = entanglement strength    │
│ 0.0 (none) ─── 1.0 (maximum)  ═══        │
│                                           │
│ Metrics:                                  │
│ • Total Entanglement: 1.7                │
│ • Max Entanglement: q0↔q1 (0.8)          │
│ • Entangled Pairs: 3                     │
└───────────────────────────────────────────┘
```

### Step 5.10: Add Interactive Manipulation

**Technical Approach:**

- Allow users to manipulate quantum states directly
- Apply single gates interactively
- See immediate visualization update
- Useful for exploration and learning

**Interactive Controls:**

```typescript
class InteractiveStateManipulator {
  setupControls(): void {
    // Add gate buttons for each qubit
    this.createGateButtons();

    // Add parameter sliders for rotation gates
    this.createParameterSliders();

    // Add reset button
    this.createResetButton();
  }

  private createGateButtons(): void {
    const gates = ["H", "X", "Y", "Z", "S", "T"];

    gates.forEach((gate) => {
      const button = document.createElement("button");
      button.textContent = gate;
      button.onclick = () => this.applyGate(gate);
      document.getElementById("gate-controls").appendChild(button);
    });
  }

  private async applyGate(gateName: string): Promise<void> {
    // Get current state
    const currentState = this.getCurrentState();

    // Apply gate
    const newState = this.applyGateToState(currentState, gateName);

    // Animate transition
    await this.animateStateTransition(currentState, newState);

    // Update visualization
    this.updateAllVisualizations(newState);
  }

  private createParameterSliders(): void {
    // Rotation angle slider
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = "360";
    slider.value = "0";
    slider.oninput = (e) => {
      const angle = parseFloat((e.target as HTMLInputElement).value);
      this.updateRotationPreview(angle);
    };

    document.getElementById("param-controls").appendChild(slider);
  }
}
```

**Interactive UI:**

```
┌─ Interactive State Manipulation ──────────┐
│ Select Qubit: ○ q0  ● q1  ○ q2           │
│                                           │
│ Single-Qubit Gates:                       │
│ [H] [X] [Y] [Z] [S] [T]                  │
│                                           │
│ Rotation Gates:                           │
│ Rx: [slider: 0°──────────────360°] 45°   │
│ Ry: [slider: 0°──────────────360°] 90°   │
│ Rz: [slider: 0°──────────────360°] 180°  │
│                                           │
│ [Apply Rx] [Apply Ry] [Apply Rz]         │
│                                           │
│ [Reset to |0⟩] [Reset to |+⟩]            │
└───────────────────────────────────────────┘
```

### Step 5.11: Implement Multi-View Layout

**Technical Approach:**

- Display multiple visualizations simultaneously
- Synchronized updates across all views
- Customizable layout
- Save/load layout preferences

**Multi-View Manager:**

```typescript
class MultiViewManager {
  private views: Map<string, VisualizationView> = new Map();
  private layout: LayoutConfig;

  setupLayout(config: LayoutConfig): void {
    this.layout = config;

    // Create view containers
    config.views.forEach((viewConfig) => {
      const view = this.createView(viewConfig);
      this.views.set(viewConfig.id, view);
    });

    // Setup synchronization
    this.setupSynchronization();
  }

  updateAllViews(state: QuantumState): void {
    // Update all active views with new state
    this.views.forEach((view, id) => {
      if (view.isActive) {
        view.update(state);
      }
    });
  }

  private setupSynchronization(): void {
    // When any view changes (e.g., user interaction),
    // update all other views
    this.views.forEach((view) => {
      view.on("state-changed", (newState) => {
        this.updateAllViews(newState);
      });
    });
  }
}
```

**Layout Examples:**

```
2x2 Grid:
┌─────────────────────────────────┐
│ Bloch Sphere  │ State Vector    │
├───────────────┼─────────────────┤
│ Density Matrix│ Probabilities   │
└─────────────────────────────────┘

Side-by-Side:
┌─────────────────────────────────┐
│ Q-Sphere      │ Entanglement    │
│               │ Network         │
│               │                 │
└─────────────────────────────────┘

Focus Mode (Single View):
┌─────────────────────────────────┐
│                                 │
│    Bloch Sphere (Full Screen)  │
│                                 │
└─────────────────────────────────┘
```

### Step 5.12: Add Export Functionality

**Technical Approach:**

- Export visualizations as images (PNG, SVG)
- Export animations as GIF or MP4
- Export data as JSON or CSV
- Support high-resolution exports

**Export System:**

```python
class VisualizationExporter:
    def export_image(
        self,
        visualization: str,
        format: str = 'png',
        resolution: Tuple[int, int] = (1920, 1080)
    ) -> bytes:
        """Export current visualization as image."""
        if format == 'png':
            return self._export_png(visualization, resolution)
        elif format == 'svg':
            return self._export_svg(visualization)
        else:
            raise ValueError(f'Unsupported format: {format}')

    def export_animation(
        self,
        states: List[np.ndarray],
        format: str = 'gif',
        fps: int = 30,
        duration: float = 5.0
    ) -> bytes:
        """Export state evolution as animation."""
        frames = []

        for state in states:
            # Render each state
            frame = self.render_state_to_frame(state)
            frames.append(frame)

        if format == 'gif':
            return self._create_gif(frames, fps)
        elif format == 'mp4':
            return self._create_mp4(frames, fps)

    def export_data(
        self,
        state: np.ndarray,
        format: str = 'json'
    ) -> str:
        """Export state data."""
        if format == 'json':
            return json.dumps({
                'state_vector': state.tolist(),
                'probabilities': np.abs(state) ** 2).tolist(),
                'num_qubits': int(np.log2(len(state)))
            })
        elif format == 'csv':
            return self._export_csv(state)
```

**Export UI:**

```
┌─ Export Visualization ────────────────────┐
│ Export Type:                              │
│ ○ Current View  ● All Views  ○ Animation │
│                                           │
│ Format:                                   │
│ ● PNG  ○ SVG  ○ GIF  ○ MP4               │
│                                           │
│ Resolution:                               │
│ ○ 1920x1080  ● 3840x2160 (4K)  ○ Custom │
│                                           │
│ Quality: ───────●─────── High             │
│                                           │
│ [Export] [Cancel]                         │
└───────────────────────────────────────────┘
```

### Step 5.13: Integrate with Proxima Backend

**Technical Approach:**

- Stream state data from backend to visualization
- Support real-time updates during execution
- Handle large state vectors efficiently
- Provide fallback for slow connections

**Backend Connection:**

```python
class VisualizationBackendConnector:
    def __init__(self, websocket_url: str):
        self.ws_url = websocket_url
        self.websocket = None

    async def connect(self) -> None:
        """Establish WebSocket connection."""
        self.websocket = await websockets.connect(self.ws_url)

        # Start listening for state updates
        asyncio.create_task(self.listen_for_updates())

    async def listen_for_updates(self) -> None:
        """Listen for state updates from backend."""
        async for message in self.websocket:
            data = json.loads(message)

            if data['type'] == 'state_update':
                state = np.array(data['state'], dtype=complex)
                await self.on_state_update(state)

            elif data['type'] == 'measurement':
                await self.on_measurement(
                    data['qubit'],
                    data['outcome']
                )

    async def request_state(self, circuit_id: str) -> None:
        """Request current state for a circuit."""
        await self.websocket.send(json.dumps({
            'type': 'get_state',
            'circuit_id': circuit_id
        }))

    async def on_state_update(self, state: np.ndarray) -> None:
        """Handle state update from backend."""
        # Update all visualizations
        await self.visualization_manager.update_all(state)
```

**Message Protocol:**

```json
// State update message
{
  "type": "state_update",
  "circuit_id": "abc123",
  "step": 5,
  "gate": "CNOT q[0]→q[1]",
  "state": [0.707, 0, 0, 0.707],
  "timestamp": 1738454400.123
}

// Measurement message
{
  "type": "measurement",
  "circuit_id": "abc123",
  "qubit": 0,
  "outcome": 1,
  "probability": 0.5,
  "timestamp": 1738454401.456
}
```

### Step 5.14: Optimize Performance

**Technical Approach:**

- Implement WebGL rendering for large state spaces
- Use web workers for state calculations
- Lazy loading for complex visualizations
- Progressive rendering for large circuits

**Performance Optimizations:**

1. **WebGL Acceleration**: Use Three.js WebGL renderer
2. **State Caching**: Cache rendered frames
3. **Level-of-Detail**: Reduce complexity for distant views
4. **Culling**: Don't render invisible elements
5. **Throttling**: Limit update frequency

**Performance Monitor:**

```typescript
class PerformanceMonitor {
  private metrics = {
    fps: 0,
    renderTime: 0,
    stateUpdateTime: 0,
    memory: 0,
  };

  measurePerformance(): void {
    // Measure FPS
    this.metrics.fps = this.calculateFPS();

    // Measure render time
    const renderStart = performance.now();
    this.render();
    this.metrics.renderTime = performance.now() - renderStart;

    // Display metrics
    this.displayMetrics();
  }

  optimizeIfNeeded(): void {
    if (this.metrics.fps < 30) {
      // Reduce quality
      this.reduceQuality();
    }

    if (this.metrics.memory > 500 * 1024 * 1024) {
      // 500 MB
      // Clear caches
      this.clearCaches();
    }
  }
}
```

### Step 5.15: Document Visualization System

**Technical Approach:**

- Create user guide with examples
- Document all visualization types
- Provide API reference for developers
- Include educational materials

**Documentation Files:**

1. **`docs/user-guide/visualization.md`**: User guide
2. **`docs/user-guide/visualization-tutorial.md`**: Interactive tutorial
3. **`docs/developer-guide/visualization-api.md`**: API docs
4. **`docs/education/quantum-visualization.md`**: Educational content

**Tutorial Topics:**

- Understanding the Bloch sphere
- Reading state vectors
- Interpreting density matrices
- Visualizing entanglement
- Using animation controls
- Exporting visualizations
- Interactive state manipulation

---

## Feature 6: Plugin Marketplace & Extension System

**Objective:** Create a comprehensive plugin marketplace and extension system that allows developers to extend Proxima's functionality through custom backends, algorithms, visualizations, and tools.

**Value Proposition:**

- **Community Extensions**: Users can share and download custom backends, algorithms
- **Easy Installation**: One-click plugin installation from marketplace
- **Developer Friendly**: Simple API for creating plugins
- **Monetization**: Optional paid plugins for professional features
- **Quality Control**: Review and rating system for plugins
- **Auto-Updates**: Plugins automatically update to latest versions

### Step 6.1: Design Plugin Architecture

**Technical Background:**

- Plugin system allows runtime extension of Proxima functionality
- Plugins are isolated Python packages with defined entry points
- Plugins can add backends, algorithms, visualizations, AI tools
- Sandboxing ensures plugins can't harm core system

**Plugin Types:**

```
┌─────────────────────────────────────────┐
│  Plugin Categories                      │
├─────────────────────────────────────────┤
│  1. Backend Plugins                     │
│     - New quantum simulators            │
│     - Cloud service integrations        │
│     - Hardware backends                 │
│                                         │
│  2. Algorithm Plugins                   │
│     - VQE implementations               │
│     - Custom optimization               │
│     - Error mitigation                  │
│                                         │
│  3. Visualization Plugins               │
│     - Custom plots                      │
│     - 3D renderers                      │
│     - Animation tools                   │
│                                         │
│  4. Tool Plugins                        │
│     - Code generators                   │
│     - Debugging tools                   │
│     - Profilers                         │
│                                         │
│  5. Agent Plugins                       │
│     - Custom AI agents                  │
│     - LLM integrations                  │
│     - Specialized assistants            │
│                                         │
│  6. UI Plugins                          │
│     - Custom screens                    │
│     - Widgets                           │
│     - Themes                            │
└─────────────────────────────────────────┘
```

**Plugin Structure:**

```
my-plugin/
├── plugin.yaml          # Plugin manifest
├── __init__.py         # Entry point
├── backend.py          # Backend implementation (if applicable)
├── algorithm.py        # Algorithm implementation
├── requirements.txt    # Dependencies
├── README.md           # Documentation
├── LICENSE             # License file
├── tests/              # Unit tests
└── examples/           # Usage examples
```

**Key Design Principles:**

- Plugins are Python packages installed via pip
- Isolated from core Proxima code
- Clear API boundaries
- Version compatibility checks
- Secure execution environment

### Step 6.2: Define Plugin Manifest Format

**Technical Approach:**

- Create standardized manifest file (YAML)
- Define plugin metadata, dependencies, entry points
- Specify required Proxima version
- Declare permissions needed

**Manifest Schema** (`plugin.yaml`):

**Example Manifest:**

```yaml
# Plugin Manifest
name: "advanced-vqe"
version: "1.2.0"
display_name: "Advanced VQE Solver"
description: "Enhanced Variational Quantum Eigensolver with advanced optimizers"
author: "Dr. Jane Quantum"
email: "jane@quantum-research.org"
website: "https://github.com/janequantum/advanced-vqe"
license: "MIT"

# Compatibility
proxima_version: ">=2.0.0,<3.0.0"
python_version: ">=3.9"

# Plugin type and capabilities
type: "algorithm"
category: "optimization"
tags:
  - "vqe"
  - "variational"
  - "optimization"
  - "quantum-chemistry"

# Entry points
entry_points:
  algorithms:
    - name: "AdvancedVQE"
      class: "advanced_vqe.solver.AdvancedVQE"
      description: "VQE with COBYLA, SPSA, and custom optimizers"

  backends:
    - name: "vqe_simulator"
      class: "advanced_vqe.backend.VQESimulator"
      description: "Optimized simulator for VQE circuits"

# Dependencies
dependencies:
  python:
    - "numpy>=1.21.0"
    - "scipy>=1.7.0"
    - "qiskit>=0.39.0"

  system:
    - "fortran-compiler" # Optional system dependencies

# Permissions
permissions:
  - "execute_circuits"
  - "file_read"
  - "file_write:workspace" # Limited to workspace directory
  - "network:github.com" # Specific domain only

# Configuration
config_schema:
  optimizer:
    type: "string"
    enum: ["COBYLA", "SPSA", "ADAM", "LBFGS"]
    default: "COBYLA"

  max_iterations:
    type: "integer"
    minimum: 1
    maximum: 10000
    default: 1000

  convergence_threshold:
    type: "number"
    default: 1e-6

# Marketplace information
marketplace:
  price: "free" # or "paid" with amount
  featured: false
  mature_rating: false
  support_url: "https://github.com/janequantum/advanced-vqe/issues"
  documentation_url: "https://advanced-vqe.readthedocs.io"
```

**Validation:**

- Validate manifest on plugin submission
- Check required fields present
- Verify version formats
- Validate entry point paths

### Step 6.3: Create Plugin Manager

**Technical Approach:**

- Implement plugin discovery and loading
- Handle plugin lifecycle (install, enable, disable, uninstall)
- Manage plugin dependencies
- Provide API for accessing plugin functionality

**Plugin Manager** (`src/proxima/plugins/manager.py`):

**Core Implementation:**

```python
class PluginManager:
    def __init__(self, plugin_dir: Path):
        self.plugin_dir = plugin_dir
        self.plugins: Dict[str, Plugin] = {}
        self.enabled_plugins: Set[str] = set()

    def discover_plugins(self) -> List[PluginInfo]:
        """Discover all installed plugins."""
        plugins = []

        for plugin_path in self.plugin_dir.iterdir():
            if plugin_path.is_dir():
                manifest_path = plugin_path / 'plugin.yaml'
                if manifest_path.exists():
                    plugin_info = self.load_plugin_manifest(manifest_path)
                    plugins.append(plugin_info)

        return plugins

    def install_plugin(
        self,
        plugin_source: str,
        version: str = None
    ) -> InstallResult:
        """Install plugin from source."""
        # Parse source (PyPI, GitHub, local path)
        if plugin_source.startswith('github:'):
            return self._install_from_github(plugin_source, version)
        elif plugin_source.startswith('pypi:'):
            return self._install_from_pypi(plugin_source, version)
        else:
            return self._install_from_path(Path(plugin_source))

    def enable_plugin(self, plugin_name: str) -> None:
        """Enable a plugin."""
        plugin = self.plugins.get(plugin_name)
        if not plugin:
            raise PluginNotFoundError(f'Plugin not found: {plugin_name}')

        # Check dependencies
        if not self._check_dependencies(plugin):
            raise DependencyError(f'Missing dependencies for {plugin_name}')

        # Load plugin
        plugin.load()

        # Register entry points
        self._register_entry_points(plugin)

        self.enabled_plugins.add(plugin_name)

        logger.info(f'Enabled plugin: {plugin_name}')

    def disable_plugin(self, plugin_name: str) -> None:
        """Disable a plugin."""
        if plugin_name not in self.enabled_plugins:
            return

        plugin = self.plugins[plugin_name]

        # Unregister entry points
        self._unregister_entry_points(plugin)

        # Unload plugin
        plugin.unload()

        self.enabled_plugins.remove(plugin_name)

        logger.info(f'Disabled plugin: {plugin_name}')

    def uninstall_plugin(self, plugin_name: str) -> None:
        """Uninstall a plugin."""
        # Disable first
        self.disable_plugin(plugin_name)

        # Remove plugin files
        plugin_path = self.plugin_dir / plugin_name
        shutil.rmtree(plugin_path)

        # Remove from registry
        del self.plugins[plugin_name]

        logger.info(f'Uninstalled plugin: {plugin_name}')

    def get_plugin_info(self, plugin_name: str) -> PluginInfo:
        """Get information about a plugin."""
        return self.plugins[plugin_name].info

    def list_plugins(
        self,
        enabled_only: bool = False
    ) -> List[PluginInfo]:
        """List all plugins."""
        if enabled_only:
            return [
                self.plugins[name].info
                for name in self.enabled_plugins
            ]
        else:
            return [p.info for p in self.plugins.values()]
```

**Plugin Lifecycle:**

```
Discovered → Installed → Enabled → Running → Disabled → Uninstalled
     ↓          ↓          ↓          ↓          ↓          ↓
   Listed   Downloaded  Loaded   Active   Stopped   Removed
```

### Step 6.4: Implement Plugin API

**Technical Approach:**

- Define clear API for plugin developers
- Provide base classes for different plugin types
- Document all hooks and extension points
- Ensure API stability across versions

**Base Plugin Classes** (`src/proxima/plugins/base.py`):

**Algorithm Plugin:**

```python
class AlgorithmPlugin(ABC):
    """Base class for algorithm plugins."""

    @abstractmethod
    def execute(
        self,
        circuit: QuantumCircuit,
        **params
    ) -> AlgorithmResult:
        """Execute the algorithm."""
        pass

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for the algorithm."""
        pass

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate algorithm parameters."""
        return True

    def get_description(self) -> str:
        """Get algorithm description."""
        return self.__doc__ or "No description"
```

**Backend Plugin:**

```python
class BackendPlugin(BaseBackendAdapter):
    """Base class for backend plugins."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    @abstractmethod
    async def execute(
        self,
        circuit: Any,
        shots: int,
        **options
    ) -> ExecutionResult:
        """Execute quantum circuit."""
        pass

    def get_capabilities(self) -> Capabilities:
        """Return backend capabilities."""
        return Capabilities(
            max_qubits=self.config.get('max_qubits', 20),
            supports_noise=False,
            simulator_types=[SimulatorType.STATEVECTOR]
        )
```

**Visualization Plugin:**

```python
class VisualizationPlugin(ABC):
    """Base class for visualization plugins."""

    @abstractmethod
    def render(
        self,
        data: Any,
        **options
    ) -> bytes:
        """Render visualization."""
        pass

    def get_supported_formats(self) -> List[str]:
        """Get supported output formats."""
        return ['png', 'svg']

    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema."""
        return {}
```

**Agent Tool Plugin:**

```python
class AgentToolPlugin(ABC):
    """Base class for agent tool plugins."""

    @abstractmethod
    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition for LLM."""
        pass

    @abstractmethod
    async def execute_tool(
        self,
        **params
    ) -> ToolResult:
        """Execute the tool."""
        pass
```

### Step 6.5: Create Plugin Marketplace Backend

**Technical Approach:**

- Build web service for plugin marketplace
- Store plugin metadata in database
- Implement search and filtering
- Handle plugin versioning

**Marketplace API** (`src/proxima/marketplace/api.py`):

**API Endpoints:**

```python
class MarketplaceAPI:
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.client = httpx.AsyncClient()

    async def search_plugins(
        self,
        query: str = None,
        category: str = None,
        tags: List[str] = None,
        sort_by: str = 'relevance'
    ) -> List[PluginListing]:
        """Search for plugins in marketplace."""
        params = {
            'query': query,
            'category': category,
            'tags': ','.join(tags) if tags else None,
            'sort': sort_by
        }

        response = await self.client.get(
            f'{self.api_url}/api/plugins/search',
            params=params
        )

        return [PluginListing(**p) for p in response.json()['plugins']]

    async def get_plugin_details(
        self,
        plugin_id: str
    ) -> PluginDetails:
        """Get detailed information about a plugin."""
        response = await self.client.get(
            f'{self.api_url}/api/plugins/{plugin_id}'
        )

        return PluginDetails(**response.json())

    async def download_plugin(
        self,
        plugin_id: str,
        version: str = 'latest'
    ) -> bytes:
        """Download plugin package."""
        response = await self.client.get(
            f'{self.api_url}/api/plugins/{plugin_id}/download',
            params={'version': version}
        )

        return response.content

    async def submit_plugin(
        self,
        plugin_package: bytes,
        metadata: PluginMetadata
    ) -> SubmissionResult:
        """Submit a new plugin to marketplace."""
        files = {'package': plugin_package}
        data = {'metadata': metadata.json()}

        response = await self.client.post(
            f'{self.api_url}/api/plugins/submit',
            files=files,
            data=data,
            headers={'Authorization': f'Bearer {self.api_token}'}
        )

        return SubmissionResult(**response.json())

    async def rate_plugin(
        self,
        plugin_id: str,
        rating: int,
        review: str = None
    ) -> None:
        """Rate and review a plugin."""
        await self.client.post(
            f'{self.api_url}/api/plugins/{plugin_id}/rate',
            json={
                'rating': rating,
                'review': review
            },
            headers={'Authorization': f'Bearer {self.api_token}'}
        )
```

**Database Schema:**

```sql
CREATE TABLE plugins (
    id UUID PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    display_name VARCHAR(255),
    description TEXT,
    author_id UUID REFERENCES users(id),
    category VARCHAR(50),
    type VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    downloads INTEGER DEFAULT 0,
    average_rating DECIMAL(3, 2)
);

CREATE TABLE plugin_versions (
    id UUID PRIMARY KEY,
    plugin_id UUID REFERENCES plugins(id),
    version VARCHAR(50) NOT NULL,
    changelog TEXT,
    package_url TEXT,
    package_hash VARCHAR(64),
    proxima_version_min VARCHAR(50),
    proxima_version_max VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    downloads INTEGER DEFAULT 0,
    UNIQUE(plugin_id, version)
);

CREATE TABLE plugin_ratings (
    id UUID PRIMARY KEY,
    plugin_id UUID REFERENCES plugins(id),
    user_id UUID REFERENCES users(id),
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    review TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(plugin_id, user_id)
);

CREATE TABLE plugin_tags (
    plugin_id UUID REFERENCES plugins(id),
    tag VARCHAR(50),
    PRIMARY KEY(plugin_id, tag)
);
```

### Step 6.6: Build Marketplace UI

**Technical Approach:**

- Create web or TUI interface for browsing plugins
- Implement search and filtering
- Display plugin details and ratings
- Provide installation buttons

**TUI Marketplace Screen** (`src/proxima/tui/screens/marketplace.py`):

**Screen Layout:**

```
┌─ Plugin Marketplace ──────────────────────────────┐
│ 🔍 Search: [quantum chemistry___________] [Search]│
│ Category: [All ▼]  Sort: [Popular ▼]             │
├───────────────────────────────────────────────────┤
│ ┌─ Advanced VQE Solver ───────────────┐ ⭐⭐⭐⭐⭐│
│ │ Enhanced VQE with advanced optimizers│ 4.8/5.0│
│ │ by Dr. Jane Quantum                  │ 2.5K ⬇ │
│ │ [Install] [Details]                  │        │
│ └──────────────────────────────────────┘        │
│                                                   │
│ ┌─ QuTiP Backend ─────────────────────┐ ⭐⭐⭐⭐ │
│ │ Quantum Toolbox in Python backend    │ 4.2/5.0│
│ │ by QuTiP Team                        │ 1.8K ⬇ │
│ │ [Install] [Details]                  │        │
│ └──────────────────────────────────────┘        │
│                                                   │
│ ┌─ 3D State Visualizer ──────────────┐ ⭐⭐⭐⭐⭐│
│ │ Advanced 3D quantum state rendering  │ 4.9/5.0│
│ │ by VizQuantum                        │ 3.2K ⬇ │
│ │ [Installed ✓] [Details]              │        │
│ └──────────────────────────────────────┘        │
│                                                   │
│ [1-3 of 127 plugins] [◀ Prev] [Next ▶]         │
└───────────────────────────────────────────────────┘
```

**Plugin Details View:**

```
┌─ Advanced VQE Solver ─────────────────────────────┐
│ by Dr. Jane Quantum  |  Version: 1.2.0            │
│ ⭐⭐⭐⭐⭐ 4.8/5.0 (247 reviews) | 2,543 downloads  │
├───────────────────────────────────────────────────┤
│ Description:                                      │
│ Enhanced Variational Quantum Eigensolver with     │
│ advanced optimization algorithms including COBYLA,│
│ SPSA, and custom gradient-based methods.          │
│                                                   │
│ Features:                                         │
│ • Multiple optimizer support                      │
│ • Adaptive convergence                            │
│ • Parallel execution                              │
│ • Custom ansatz support                           │
│                                                   │
│ Requirements:                                     │
│ • Proxima >= 2.0.0                                │
│ • Python >= 3.9                                   │
│ • numpy, scipy, qiskit                            │
│                                                   │
│ Permissions Needed:                               │
│ • Execute circuits                                │
│ • File read/write (workspace only)                │
│                                                   │
│ [Install Plugin] [View Source] [Report Issue]    │
│                                                   │
│ Reviews (showing 3 of 247):                       │
│ ⭐⭐⭐⭐⭐ by user123: "Excellent VQE implementation"│
│ ⭐⭐⭐⭐ by quantumdev: "Works great, minor bugs"   │
│ ⭐⭐⭐⭐⭐ by researcher: "Best VQE solver available"│
│                                                   │
│ [Write Review] [View All Reviews]                 │
└───────────────────────────────────────────────────┘
```

### Step 6.7: Implement Plugin Sandboxing

**Technical Approach:**

- Execute plugins in restricted environment
- Limit file system access to specific directories
- Control network access
- Monitor resource usage

**Sandbox Implementation** (`src/proxima/plugins/sandbox.py`):

**Sandboxing Strategy:**

```python
class PluginSandbox:
    def __init__(self, plugin_name: str, permissions: List[str]):
        self.plugin_name = plugin_name
        self.permissions = permissions
        self.allowed_paths = self._compute_allowed_paths()
        self.allowed_hosts = self._compute_allowed_hosts()

    def execute_in_sandbox(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function in sandboxed environment."""
        # Create restricted global namespace
        sandbox_globals = self._create_sandbox_globals()

        # Wrap file operations
        sandbox_globals['open'] = self._sandboxed_open

        # Wrap network operations
        sandbox_globals['socket'] = self._sandboxed_socket

        # Execute with restricted permissions
        with self._resource_limits():
            result = func(*args, **kwargs)

        return result

    def _sandboxed_open(
        self,
        path: str,
        mode: str = 'r',
        **kwargs
    ):
        """Sandboxed file open."""
        abs_path = Path(path).resolve()

        # Check if path is allowed
        if not self._is_path_allowed(abs_path):
            raise PermissionError(
                f'Plugin {self.plugin_name} cannot access {path}'
            )

        # Check mode permissions
        if 'w' in mode or 'a' in mode:
            if not self._has_permission('file_write'):
                raise PermissionError(
                    f'Plugin {self.plugin_name} does not have write permission'
                )

        return open(path, mode, **kwargs)

    def _sandboxed_socket(self, *args, **kwargs):
        """Sandboxed socket creation."""
        if not self._has_permission('network'):
            raise PermissionError(
                f'Plugin {self.plugin_name} does not have network permission'
            )

        # Return wrapped socket that checks allowed hosts
        return SandboxedSocket(self.allowed_hosts, *args, **kwargs)

    @contextmanager
    def _resource_limits(self):
        """Apply resource limits."""
        import resource

        # Limit CPU time (10 minutes)
        resource.setrlimit(resource.RLIMIT_CPU, (600, 600))

        # Limit memory (2 GB)
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 ** 3, 2 * 1024 ** 3))

        yield

        # Reset limits
        resource.setrlimit(resource.RLIMIT_CPU, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
```

**Permission System:**

```python
PERMISSION_DESCRIPTIONS = {
    'execute_circuits': 'Execute quantum circuits',
    'file_read': 'Read files from workspace',
    'file_write': 'Write files to workspace',
    'file_write:all': 'Write files anywhere (dangerous)',
    'network': 'Access network (all hosts)',
    'network:domain.com': 'Access specific domain only',
    'subprocess': 'Run system commands',
    'database': 'Access Proxima database'
}
```

### Step 6.8: Add Plugin Dependency Management

**Technical Approach:**

- Resolve plugin dependencies automatically
- Handle version conflicts
- Support dependency trees
- Use pip/conda for Python packages

**Dependency Resolver** (`src/proxima/plugins/dependencies.py`):

**Implementation:**

```python
class DependencyResolver:
    def resolve_dependencies(
        self,
        plugin: PluginInfo
    ) -> List[Dependency]:
        """Resolve all dependencies for a plugin."""
        dependencies = []

        # Python package dependencies
        for dep in plugin.dependencies.get('python', []):
            resolved = self._resolve_python_dep(dep)
            dependencies.append(resolved)

        # Plugin dependencies
        for dep in plugin.dependencies.get('plugins', []):
            resolved = self._resolve_plugin_dep(dep)
            dependencies.append(resolved)

        # System dependencies
        for dep in plugin.dependencies.get('system', []):
            resolved = self._resolve_system_dep(dep)
            dependencies.append(resolved)

        return dependencies

    def install_dependencies(
        self,
        dependencies: List[Dependency]
    ) -> None:
        """Install all dependencies."""
        for dep in dependencies:
            if dep.type == 'python':
                self._install_python_package(dep)
            elif dep.type == 'plugin':
                self._install_plugin_dependency(dep)
            elif dep.type == 'system':
                self._install_system_package(dep)

    def check_conflicts(
        self,
        plugin: PluginInfo,
        installed_plugins: List[PluginInfo]
    ) -> List[Conflict]:
        """Check for version conflicts."""
        conflicts = []

        for installed in installed_plugins:
            # Check if plugins have conflicting dependencies
            conflict = self._find_conflict(plugin, installed)
            if conflict:
                conflicts.append(conflict)

        return conflicts
```

**Conflict Resolution:**

- Warn user about conflicts
- Suggest compatible versions
- Optionally disable conflicting plugins
- Create virtual environment per plugin (advanced)

### Step 6.9: Implement Plugin Auto-Updates

**Technical Approach:**

- Check for plugin updates periodically
- Notify users of available updates
- Support automatic or manual updates
- Maintain changelog history

**Update Manager** (`src/proxima/plugins/updater.py`):

**Auto-Update Implementation:**

```python
class PluginUpdater:
    async def check_for_updates(
        self,
        plugins: List[Plugin]
    ) -> List[UpdateInfo]:
        """Check for available updates."""
        updates = []

        for plugin in plugins:
            latest_version = await self.marketplace.get_latest_version(
                plugin.name
            )

            if self._is_newer_version(latest_version, plugin.version):
                update_info = UpdateInfo(
                    plugin_name=plugin.name,
                    current_version=plugin.version,
                    latest_version=latest_version,
                    changelog=await self.marketplace.get_changelog(
                        plugin.name,
                        plugin.version,
                        latest_version
                    )
                )
                updates.append(update_info)

        return updates

    async def update_plugin(
        self,
        plugin_name: str,
        version: str = 'latest'
    ) -> UpdateResult:
        """Update a plugin to specified version."""
        # Download new version
        package = await self.marketplace.download_plugin(
            plugin_name,
            version
        )

        # Disable current version
        await self.plugin_manager.disable_plugin(plugin_name)

        # Backup current version
        self._backup_plugin(plugin_name)

        # Install new version
        try:
            await self.plugin_manager.install_plugin(
                package,
                overwrite=True
            )

            # Enable new version
            await self.plugin_manager.enable_plugin(plugin_name)

            return UpdateResult(success=True, version=version)

        except Exception as e:
            # Rollback on failure
            self._restore_backup(plugin_name)
            await self.plugin_manager.enable_plugin(plugin_name)

            return UpdateResult(success=False, error=str(e))
```

**Update Notification:**

```
┌─ Plugin Updates Available ────────────────┐
│ 3 plugins have updates:                   │
│                                           │
│ 1. Advanced VQE: 1.2.0 → 1.3.0           │
│    • Added ADAM optimizer                 │
│    • Fixed convergence bug                │
│    [Update] [View Changes]                │
│                                           │
│ 2. QuTiP Backend: 2.1.0 → 2.2.0          │
│    • Performance improvements             │
│    • Support for 30 qubits                │
│    [Update] [View Changes]                │
│                                           │
│ 3. 3D Visualizer: 1.5.0 → 1.6.0          │
│    • New animation features               │
│    [Update] [View Changes]                │
│                                           │
│ [Update All] [Skip] [Auto-Update: ON ▼]  │
└───────────────────────────────────────────┘
```

### Step 6.10: Create Plugin Developer Tools

**Technical Approach:**

- Provide CLI tools for plugin development
- Generate plugin template/boilerplate
- Validate plugin structure
- Test plugin locally before submission

**CLI Tools** (`src/proxima/plugins/cli.py`):

**Commands:**

```bash
# Create new plugin from template
proxima plugin new my-algorithm --type=algorithm

# Validate plugin structure
proxima plugin validate ./my-algorithm

# Test plugin locally
proxima plugin test ./my-algorithm

# Package plugin for distribution
proxima plugin build ./my-algorithm

# Submit to marketplace
proxima plugin publish ./my-algorithm-1.0.0.tar.gz

# Generate documentation
proxima plugin docs ./my-algorithm
```

**Plugin Template Generator:**

```python
class PluginTemplateGenerator:
    def generate_template(
        self,
        plugin_name: str,
        plugin_type: str,
        author: str,
        description: str
    ) -> Path:
        """Generate plugin template."""
        plugin_dir = Path(plugin_name)
        plugin_dir.mkdir(exist_ok=True)

        # Generate manifest
        self._generate_manifest(
            plugin_dir,
            plugin_name,
            plugin_type,
            author,
            description
        )

        # Generate __init__.py
        self._generate_init(plugin_dir, plugin_type)

        # Generate implementation file
        self._generate_implementation(plugin_dir, plugin_type)

        # Generate tests
        self._generate_tests(plugin_dir)

        # Generate README
        self._generate_readme(plugin_dir, plugin_name, description)

        # Generate requirements.txt
        self._generate_requirements(plugin_dir)

        return plugin_dir
```

**Generated Plugin Structure:**

```
my-algorithm/
├── plugin.yaml           # Generated manifest
├── __init__.py          # Entry point
├── algorithm.py         # Implementation template
├── requirements.txt     # Dependencies
├── README.md           # Documentation template
├── LICENSE             # MIT license template
├── tests/
│   ├── __init__.py
│   └── test_algorithm.py  # Test template
└── examples/
    └── example_usage.py   # Usage example
```

### Step 6.11: Implement Plugin Review System

**Technical Approach:**

- Manual review for submitted plugins
- Automated security scanning
- Code quality checks
- Community moderation

**Review Process:**

```
Submit → Automated Checks → Manual Review → Approve/Reject
   ↓           ↓                  ↓              ↓
  Upload   Security Scan    Human Review    Published
           Code Analysis    Test Execution
           Manifest Check
```

**Automated Checks:**

```python
class PluginReviewer:
    async def review_plugin(
        self,
        plugin_package: bytes
    ) -> ReviewResult:
        """Perform automated review."""
        results = []

        # Extract and validate manifest
        manifest = self._extract_manifest(plugin_package)
        manifest_check = self._validate_manifest(manifest)
        results.append(manifest_check)

        # Security scan
        security_check = await self._security_scan(plugin_package)
        results.append(security_check)

        # Code quality analysis
        quality_check = await self._analyze_code_quality(plugin_package)
        results.append(quality_check)

        # Run tests
        test_check = await self._run_tests(plugin_package)
        results.append(test_check)

        # Check license
        license_check = self._check_license(plugin_package)
        results.append(license_check)

        return ReviewResult(
            passed=all(r.passed for r in results),
            checks=results
        )
```

**Security Scanning:**

- Check for known vulnerabilities (bandit, safety)
- Scan for malicious code patterns
- Verify no hardcoded credentials
- Check imported packages for security issues

### Step 6.12: Add Plugin Analytics

**Technical Approach:**

- Track plugin usage statistics
- Monitor performance impact
- Collect crash reports
- Provide analytics dashboard for developers

**Analytics System:**

```python
class PluginAnalytics:
    async def track_event(
        self,
        plugin_name: str,
        event_type: str,
        data: Dict[str, Any]
    ) -> None:
        """Track plugin event."""
        event = AnalyticsEvent(
            plugin_name=plugin_name,
            event_type=event_type,
            data=data,
            timestamp=time.time(),
            user_id=self._get_anonymous_user_id()
        )

        await self._send_to_analytics_service(event)

    async def track_error(
        self,
        plugin_name: str,
        error: Exception,
        context: Dict[str, Any]
    ) -> None:
        """Track plugin error."""
        error_report = ErrorReport(
            plugin_name=plugin_name,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            context=context,
            timestamp=time.time()
        )

        await self._send_error_report(error_report)
```

**Analytics Dashboard:**

```
┌─ Plugin Analytics: Advanced VQE ──────────┐
│ Period: Last 30 days                      │
├───────────────────────────────────────────┤
│ Usage:                                    │
│ • Total Executions: 1,247                 │
│ • Unique Users: 89                        │
│ • Avg Execution Time: 2.3s                │
│                                           │
│ Performance:                              │
│ • Success Rate: 98.4%                     │
│ • Error Rate: 1.6% (20 errors)            │
│ • Crashes: 0                              │
│                                           │
│ Popular Features:                         │
│ • COBYLA optimizer: 67%                   │
│ • SPSA optimizer: 28%                     │
│ • Custom optimizer: 5%                    │
│                                           │
│ User Satisfaction:                        │
│ • Average Rating: 4.8/5.0                 │
│ • Recent Reviews: 12 (10 positive)        │
│                                           │
│ [View Detailed Report] [Export Data]      │
└───────────────────────────────────────────┘
```

### Step 6.13: Create Plugin Documentation System

**Technical Approach:**

- Auto-generate API documentation from code
- Support Markdown documentation files
- Integrate with marketplace
- Version documentation alongside code

**Documentation Generator:**

```python
class PluginDocGenerator:
    def generate_docs(
        self,
        plugin_path: Path
    ) -> Path:
        """Generate documentation for plugin."""
        docs_dir = plugin_path / 'docs'
        docs_dir.mkdir(exist_ok=True)

        # Generate API reference from docstrings
        api_docs = self._generate_api_docs(plugin_path)
        (docs_dir / 'api.md').write_text(api_docs)

        # Extract usage examples
        examples = self._extract_examples(plugin_path)
        (docs_dir / 'examples.md').write_text(examples)

        # Generate index page
        index = self._generate_index(plugin_path)
        (docs_dir / 'index.md').write_text(index)

        return docs_dir
```

### Step 6.14: Test Plugin System

**Technical Approach:**

- Test plugin loading and unloading
- Test sandboxing and permissions
- Test dependency resolution
- Test marketplace integration

**Test Suite:**

```python
class TestPluginSystem:
    def test_plugin_installation(self):
        # Test installing plugin from marketplace
        pass

    def test_plugin_sandboxing(self):
        # Test that plugins cannot access restricted resources
        pass

    def test_dependency_resolution(self):
        # Test resolving plugin dependencies
        pass

    def test_plugin_updates(self):
        # Test auto-update mechanism
        pass

    def test_plugin_conflicts(self):
        # Test handling version conflicts
        pass
```

### Step 6.15: Document Plugin System

**Technical Approach:**

- Create comprehensive plugin developer guide
- Document API and best practices
- Provide example plugins
- Create video tutorials

**Documentation Files:**

1. **`docs/developer-guide/plugin-development.md`**: Complete guide
2. **`docs/developer-guide/plugin-api-reference.md`**: API docs
3. **`docs/developer-guide/plugin-examples.md`**: Example plugins
4. **`docs/marketplace/publishing-guide.md`**: Publishing guide

**Topics:**

- Getting started with plugin development
- Plugin types and capabilities
- API reference
- Sandboxing and permissions
- Testing and debugging plugins
- Submitting to marketplace
- Best practices and guidelines

---

## PART B: EXISTING IMPLEMENTATION PHASES

---

## Architecture Overview

### System Components

1. **Agent Controller Layer**
   - Centralized orchestration of all agent operations
   - Tool registry and execution dispatcher
   - Session management with persistent state
   - Consent and safety management

2. **Terminal Executor Layer**
   - Cross-platform shell abstraction (PowerShell, bash, zsh)
   - Asynchronous process spawning with PTY
   - Real-time output streaming with line buffering
   - Session pooling for resource efficiency

3. **UI Presentation Layer**
   - Textual-based TUI with reactive widgets
   - Split-pane layouts with resizable panels
   - Real-time log viewers with auto-scroll
   - Progress indicators and status displays

4. **Safety & Rollback Layer**
   - File system checkpoints before modifications
   - Operation approval workflows
   - Undo/redo stack management
   - Audit trail logging

### Data Flow Architecture

```
User Input → LLM Provider → Tool Selection → Agent Controller →
Terminal Executor → Real-time Output Stream → UI Update →
Safety Layer (if needed) → Result Storage → Stats Update
```

---

## Phase 1: Real-Time Execution & Results Monitoring

**Objective:** Transform the Execution and Results tabs into hardcore real-time monitoring systems that display live output from all running processes.

### Step 1.1: Implement Real-Time Event Bus System

**Technical Approach:**

- Create an event bus using Python's `asyncio.Queue` for non-blocking message passing
- Design event types: `PROCESS_STARTED`, `OUTPUT_LINE`, `ERROR_LINE`, `PROCESS_COMPLETED`, `PROGRESS_UPDATE`, `STAGE_CHANGED`
- Implement event listeners that can subscribe to specific event types
- Use weak references to prevent memory leaks from orphaned listeners

**Libraries:**

- `asyncio` - Core async primitives
- `typing` - Type hints for event contracts
- `dataclasses` - Event data structure definitions
- `weakref` - Weak reference management

**Key Design Decisions:**

- Events should be immutable dataclass instances
- Each event carries a timestamp, source ID, and payload
- Event bus maintains a circular buffer (max 10,000 events) for replay
- Events are dispatched on the asyncio event loop

### Step 1.2: Refactor Execution Screen for Real-Time Updates

**Technical Approach:**

- Replace polling-based updates with reactive updates triggered by events
- Implement a `RichLog` widget from Textual that auto-scrolls with new content
- Use Textual's `reactive` decorators for automatic UI updates when data changes
- Create a connection between the event bus and the UI update cycle
- Implement debouncing for high-frequency updates (max 60 FPS)

**Textual Components:**

- `RichLog` - Rich text log viewer with syntax highlighting
- `ProgressBar` - Indeterminate and determinate progress bars
- `DataTable` - Tabular data for process lists
- `Static` with `reactive` - For dynamic text updates

**Update Strategy:**

- Batch UI updates every 16ms (60 FPS) to prevent UI blocking
- Use Textual's `call_later` for deferred updates
- Implement virtual scrolling for logs exceeding 10,000 lines
- Store full log in memory but only render visible window

### Step 1.3: Implement Process Output Streaming

**Technical Approach:**

- Use `asyncio.create_subprocess_exec` for non-blocking process spawning
- Set `stdout=asyncio.subprocess.PIPE` and `stderr=asyncio.subprocess.PIPE`
- Create separate async tasks for reading stdout and stderr streams
- Implement line-based buffering using `StreamReader.readline()`
- Detect ANSI escape codes and strip or interpret them for rendering

**Stream Handling:**

- Use `asyncio.StreamReader` for non-blocking line reading
- Implement timeout on readline to prevent hung processes
- Detect process termination via returncode checking
- Capture both stdout and stderr separately but merge in UI

**ANSI Code Handling:**

- Use `rich.console.Console.render_str()` for ANSI interpretation
- Strip codes if terminal doesn't support colors
- Map ANSI colors to Textual theme colors

### Step 1.4: Create Multi-Process Monitor Widget

**Technical Approach:**

- Design a split-pane widget showing multiple terminal outputs simultaneously
- Implement a `MultiTerminalView` container with dynamic columns
- Allow up to 4 concurrent terminal views in 2x2 grid layout
- Each terminal view has independent scrolling and search
- Highlight active terminal with colored border

**Textual Layout:**

- Use `Horizontal` and `Vertical` containers for grid layout
- Implement `ScrollableContainer` for each terminal pane
- Use `TabbedContent` as alternative to grid for many terminals
- Implement terminal switcher with keyboard shortcuts (Alt+1-9)

**State Management:**

- Each terminal has unique ID and associated metadata
- Store terminal state: command, start time, status, output buffer
- Implement terminal history (last 20 terminals) for quick access
- Auto-close completed terminals after configurable timeout

### Step 1.5: Enhance Results Tab with Real-Time Data

**Technical Approach:**

- Refactor results tab to subscribe to execution completion events
- Display results as they arrive, not after all processes complete
- Implement streaming results viewer for long-running benchmarks
- Show partial results with "in-progress" indicators

**Data Structures:**

- Use `list` with `append()` for incremental result accumulation
- Store results in SQLite database for persistence and queries
- Index by execution ID, timestamp, backend name
- Support JSON export of results at any time

**UI Components:**

- `DataTable` for tabular results with sortable columns
- `PlotextPlot` from textual-plotext for real-time graphs
- `ProgressBar` for per-result completion status
- Collapsible sections for detailed result inspection

---

## Phase 2: Agent UI/UX Enhancements

**Objective:** Improve the agent UI to be more professional, user-friendly, and efficient with word wrapping, resizable panels, clean layouts, and toggle-able real-time stats.

### Step 2.1: Implement Word Wrapping for Agent Responses

**Technical Approach:**

- Configure `RichLog` widget to use word wrapping by default
- Set `wrap=True` parameter when creating log widgets
- Use Rich's `Text` object with automatic wrapping
- Implement soft wrapping that respects word boundaries

**Rich Text Configuration:**

- Use `rich.console.Console` with `soft_wrap=True`
- Set maximum width to container width minus padding
- Use `overflow="fold"` to handle long unbreakable strings
- Remove horizontal scrollbar from wrapped containers

**Code-Specific Handling:**

- Use `Syntax` widget from Rich for code blocks with language detection
- Enable line numbers for code snippets
- Use horizontal scrolling ONLY for code blocks (user expects this)
- Wrap natural language responses but not code

### Step 2.2: Add Resizable Panel Sliders

**Technical Approach:**

- Implement custom `ResizableContainer` widget extending Textual's `Container`
- Add drag handles (vertical bars) between panels
- Detect mouse drag events on handles
- Recalculate panel widths dynamically during drag
- Store resize preferences in user settings JSON

**Textual Mouse Handling:**

- Override `on_mouse_down`, `on_mouse_move`, `on_mouse_up` methods
- Detect if click occurred on resize handle (5px width region)
- Update width percentages as mouse moves
- Implement minimum width constraints (20% each panel)

**Keyboard Alternative:**

- Bind `Ctrl+[` and `Ctrl+]` to decrease/increase agent panel width
- Step size: 5% width change per keypress
- Show temporary tooltip with current width percentage

**Settings Persistence:**

- Store panel width ratios in `~/.proxima/tui_settings.json`
- Load saved widths on screen initialization
- Provide "Reset Layout" button to restore defaults

### Step 2.3: Redesign Agent UI for Professional Look

**Technical Approach:**

- Create custom CSS theme specifically for agent panels
- Use consistent spacing: 1-unit padding, 2-unit margins
- Implement color hierarchy: primary (headings), accent (active), muted (metadata)
- Add subtle borders with rounded corners (if terminal supports)
- Use icons from Unicode or Nerd Fonts for visual clarity

**Theme Elements:**

- **Header:** Bold, large text with icon, solid background color
- **Message Bubbles:** Distinct user vs AI styling (border vs background)
- **Tool Execution:** Collapsible sections with status icons
- **Statistics:** Card-based layout with icons and colored numbers
- **Buttons:** Consistent variants (primary, success, warning, error)

**Layout Structure:**

```
┌─ Agent Panel ──────────────────────────────────┐
│ [Icon] AI Assistant          [≡ Menu] [✕ Hide] │ ← Header
├────────────────────────────────────────────────┤
│ ┌─ Stats (collapsible) ────────────────────┐  │
│ │ Provider: OpenAI | Model: gpt-4         │  │
│ │ Messages: 42 | Tokens: 15.2K            │  │
│ └──────────────────────────────────────────┘  │
├────────────────────────────────────────────────┤
│                                                │
│ 👤 User: Build LRET backend                   │ ← Message
│ ─────────────────────────────────────────────  │
│ 🤖 AI: I'll build the LRET Cirq backend...    │
│                                                │
│   ┌─ Tool: build_backend ───────────────┐     │ ← Tool
│   │ ✓ Completed in 45s                  │     │
│   │ Backend: lret_cirq                  │     │
│   └─────────────────────────────────────┘     │
│                                                │
│ [Scroll for more messages...]                 │
├────────────────────────────────────────────────┤
│ [Input field: Ask AI for help...]        [⮕]  │ ← Input
└────────────────────────────────────────────────┘
```

**Textual CSS Best Practices:**

- Use CSS variables for colors defined in theme
- Employ relative units (fr, auto) over fixed pixels
- Implement hover states for interactive elements
- Add focus indicators for keyboard navigation

### Step 2.4: Implement Toggle-able Real-Time Stats Panel

**Technical Approach:**

- Create collapsible stats widget with expand/collapse animation
- Bind keyboard shortcut (Ctrl+T) to toggle visibility
- Add small button in agent header to toggle stats
- Animate collapse using Textual's height transition
- When visible, update stats every 500ms

**Stats Layout:**

- Use grid layout with label-value pairs
- Position stats at top of agent panel, below header
- Use different panel for stats vs messages to prevent overlap
- Stats panel should push message content down, not overlay

**Stats to Display:**

- **LLM Stats:** Provider, model, temperature, max tokens
- **Session Stats:** Messages sent, tokens used, requests made
- **Performance:** Avg response time, uptime, errors
- **Agent Stats:** Tools executed, files modified, commands run
- **Terminal Stats:** Active terminals, completed processes

**Real-Time Update Mechanism:**

- Create reactive properties for all stat values
- Update properties on every event (tool execution, message, etc.)
- Textual automatically re-renders when reactive properties change
- Use `set_interval()` to refresh time-based stats (uptime, elapsed time)

### Step 2.5: Ensure Stats Don't Overlap Other Content

**Technical Approach:**

- Use proper Textual layout constraints (no absolute positioning)
- Stats panel is part of normal document flow
- When stats expand, increase panel height
- When stats collapse, decrease panel height
- Use `Vertical` container to stack stats → messages → input

**Layout Strategy:**

```
Vertical Container (height: 100%)
  ├─ Header (height: 3)
  ├─ Stats Panel (height: auto, max: 10)
  │   └─ Collapsible(visible=True/False)
  ├─ Messages Area (height: 1fr)  ← Takes remaining space
  │   └─ ScrollableContainer
  └─ Input Section (height: auto, min: 5)
```

**Z-Index Management:**

- Don't use overlays for stats
- Use proper container nesting instead
- Only modal dialogs should overlay content
- Stats are part of the normal widget tree

---

## Phase 3: Terminal Integration & Multi-Process Management

**Objective:** Implement robust terminal execution with support for multiple concurrent processes, real-time output streaming, and cross-platform compatibility.

### Step 3.1: Design Terminal Executor Core

**Technical Approach:**

- Create `TerminalExecutor` class that abstracts shell differences
- Detect OS using `platform.system()` (Windows, Linux, Darwin)
- On Windows: Use PowerShell Core (pwsh) or fallback to PowerShell 5.1
- On Unix: Use bash, zsh, or sh based on availability
- Store default shell and working directory per executor instance

**Shell Detection Logic:**

- Windows: Check for `pwsh.exe` in PATH, else use `powershell.exe`
- macOS: Use `zsh` (default since Catalina), fallback to `bash`
- Linux: Check $SHELL environment variable, fallback to `bash`
- Verify shell exists using `shutil.which()` before use

**Working Directory Management:**

- Maintain stack of working directories (pushd/popd semantics)
- Every command execution records current working directory
- Support `cd` command to change working directory for session
- Validate directory exists before changing

**Environment Variables:**

- Inherit parent process environment
- Allow per-session environment variable overrides
- Support setting environment variables in command string
- Persist environment across commands in same session

### Step 3.2: Implement Asynchronous Process Execution

**Technical Approach:**

- Use `asyncio.create_subprocess_exec()` for process spawning
- Pass command through shell with `-c` flag for proper parsing
- Create async tasks for stdout and stderr reading
- Use `asyncio.gather()` to wait for both streams
- Capture return code when process completes

**Subprocess Configuration:**

```python
# Pseudocode structure
process = await asyncio.create_subprocess_exec(
    shell_executable,
    shell_flag,  # '-c' for bash/zsh, '-Command' for PowerShell
    command_string,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
    cwd=working_directory,
    env=environment_dict,
    start_new_session=True  # For process group management
)
```

**Stream Reading Strategy:**

- Read line-by-line using `stream.readline()`
- Decode bytes to UTF-8 (handle errors with 'replace' mode)
- Call callback function for each line received
- Continue reading until stream reaches EOF
- Handle partial lines at end of stream

**Timeout Handling:**

- Implement configurable timeout per command (default: 300s)
- Use `asyncio.wait_for()` to enforce timeout
- Send SIGTERM on timeout, then SIGKILL after grace period
- Return timeout error in execution result

### Step 3.3: Create Session Management System

**Technical Approach:**

- Design `AgentSession` dataclass to store session state
- Include fields: id, name, created_at, working_dir, environment, command_history
- Maintain session pool (max 10 concurrent sessions)
- Implement session factory for creating/destroying sessions
- Support session persistence to disk for restoration

**Session Lifecycle:**

1. **Creation:** Generate unique ID, set initial working directory
2. **Active:** Execute commands, update state, append to history
3. **Idle:** Session unused for 5 minutes, can be reclaimed
4. **Destroyed:** Clean up resources, save history to disk

**Session Storage:**

- Store session data in `~/.proxima/agent_sessions/{session_id}.json`
- Include: all commands executed, working directory changes, environment
- Limit stored history to last 100 commands per session
- Allow session import to continue previous work

**Concurrency Control:**

- Use `asyncio.Semaphore` to limit concurrent executions (max 10)
- Queue commands when all slots occupied
- Priority queue: user commands > background tasks > cleanup tasks

### Step 3.4: Implement Multi-Terminal Monitoring

**Technical Approach:**

- Create `MultiTerminalMonitor` class to track all active processes
- Maintain dictionary mapping terminal ID to terminal state
- Each state includes: command, status, output buffer, timestamps
- Implement event emitter to notify UI of changes
- Support filtering and searching across all terminals

**Terminal State Machine:**

- **PENDING:** Command queued but not started
- **STARTING:** Process spawning in progress
- **RUNNING:** Process actively executing
- **COMPLETED:** Process finished successfully (return code 0)
- **FAILED:** Process exited with non-zero return code
- **TIMEOUT:** Process killed due to timeout
- **CANCELLED:** User manually stopped process

**Output Buffering Strategy:**

- Maintain circular buffer per terminal (max 10,000 lines)
- Store each line with metadata: timestamp, stream (stdout/stderr), line number
- Implement efficient line retrieval for UI rendering
- Support streaming buffer to file for very long outputs

**Event System:**

- Emit events for all state changes: `STARTED`, `OUTPUT_RECEIVED`, `COMPLETED`
- Include terminal ID and relevant data in each event
- UI components subscribe to events and update accordingly
- Debounce rapid events (multiple lines in quick succession)

### Step 3.5: Add Cross-Platform Command Normalization

**Technical Approach:**

- Create command translation layer for common operations
- Map generic commands to platform-specific equivalents
- Handle path separators (forward slash vs backslash)
- Convert environment variable syntax ($VAR vs %VAR% vs $env:VAR)

**Command Mappings:**

- List directory: `ls` → `dir` (CMD), `Get-ChildItem` (PowerShell), `ls` (bash/zsh)
- Copy file: `cp` → `copy` (CMD), `Copy-Item` (PowerShell), `cp` (bash/zsh)
- Remove file: `rm` → `del` (CMD), `Remove-Item` (PowerShell), `rm` (bash/zsh)
- Create directory: `mkdir` → `mkdir` (all), but use `-Force` flag in PowerShell

**Path Handling:**

- Normalize all paths using `pathlib.Path`
- Convert to OS-specific format before passing to shell
- Handle relative paths relative to current working directory
- Support home directory expansion (~/ → C:\Users\username\ on Windows)

**Environment Variables:**

- Detect variable references in command string
- Convert to platform-specific syntax before execution
- On Windows PowerShell: Use `$env:VAR_NAME`
- On Unix shells: Use `$VAR_NAME` or `${VAR_NAME}`

---

## Phase 4: Backend Building & Compilation System

**Objective:** Enable AI agents to build and compile quantum simulator backends through terminal commands with proper error handling and progress reporting.

### Step 4.1: Define Backend Build Profiles

**Technical Approach:**

- Create configuration file (YAML or JSON) listing all supported backends
- Each backend profile includes: name, build commands, dependencies, working directory
- Store profiles in `configs/backend_build_profiles.yaml`
- Load profiles on agent initialization

**Backend Profile Structure:**

```yaml
backends:
  lret_cirq:
    name: "LRET Cirq Scalability"
    directory: "src/proxima/backends/lret_cirq"
    build_steps:
      - command: "pip install -r requirements.txt"
        description: "Install dependencies"
      - command: "python setup.py build_ext --inplace"
        description: "Compile Cython extensions"
      - command: "python -m pytest tests/"
        description: "Run tests"
    dependencies: ["python>=3.9", "cirq>=1.0", "cython"]
    platform: "all"
    gpu_required: false
```

**Supported Backends:**

1. **LRET Cirq Scalability:** CPU-based Cirq backend with scaling optimizations
2. **LRET PennyLane Hybrid:** Hybrid quantum-classical backend using PennyLane
3. **LRET Phase 7 Unified:** Unified multi-backend interface
4. **Cirq:** Google's quantum computing framework
5. **Qiskit Aer:** IBM's high-performance simulator
6. **QuEST:** Quantum Exact Simulation Toolkit
7. **qsim:** Google's quantum circuit simulator
8. **cuQuantum:** NVIDIA's GPU-accelerated quantum simulator

### Step 4.2: Implement Build Command Executor

**Technical Approach:**

- Create `BackendBuilder` class that uses `TerminalExecutor`
- Validate backend name against known profiles
- Load build profile and resolve paths
- Execute each build step sequentially
- Stream output to UI in real-time

**Build Process Flow:**

1. **Pre-build Validation:**
   - Check if backend directory exists
   - Verify required dependencies are available
   - Ensure sufficient disk space for build artifacts
2. **Dependency Installation:**
   - Parse requirements file or dependencies list
   - Install using pip, conda, or system package manager
   - Handle virtual environment activation if needed

3. **Compilation:**
   - Execute build commands in correct order
   - Set appropriate environment variables (CC, CFLAGS, etc.)
   - Handle platform-specific compilation flags
   - Compile Cython, C++, CUDA code as needed

4. **Post-build Verification:**
   - Run test suite to verify build
   - Check for expected output files (shared libraries, Python modules)
   - Report build statistics (time, artifact size)

**Error Handling:**

- Detect compilation errors from stderr
- Parse error messages to extract file:line information
- Provide helpful suggestions based on common error patterns
- Support build retry with clean workspace

### Step 4.3: Add GPU Backend Support

**Technical Approach:**

- Detect GPU availability using appropriate APIs
- For CUDA: Use `nvidia-smi` command to check GPU presence
- For ROCm: Use `rocm-smi` command for AMD GPUs
- Only offer GPU backends if hardware detected
- Set CUDA/ROCm environment variables before compilation

**GPU Detection:**

- **NVIDIA:** Run `nvidia-smi --query-gpu=name,memory.total --format=csv`
- **AMD:** Run `rocm-smi --showproductname`
- Parse output to determine GPU model and memory
- Store GPU info in agent context for reference

**CUDA Compilation:**

- Set `CUDA_HOME` environment variable to CUDA toolkit path
- Add CUDA bin and lib directories to PATH/LD_LIBRARY_PATH
- Specify compute capability flags for nvcc compiler
- Link against cuBLAS, cuSPARSE for numerical operations

**cuQuantum Integration:**

- Check for cuQuantum library installation
- Set `CUQUANTUM_ROOT` environment variable
- Use cuStateVec for state vector simulation
- Use cuTensorNet for tensor network contraction

### Step 4.4: Implement Build Progress Tracking

**Technical Approach:**

- Parse build output for progress indicators
- Detect compilation steps (compiling X.cpp, linking Y.so)
- Estimate total build time based on previous builds
- Update progress bar in UI as build progresses

**Progress Estimation:**

- Count total build steps from profile
- Assign weight to each step (dependency install: 20%, compile: 60%, test: 20%)
- Update progress percentage as each step completes
- Show indeterminate progress during long-running steps

**Output Parsing:**

- Use regex to detect compilation messages: `Compiling (\w+\.cpp)`
- Detect link messages: `Linking (\w+\.so)`
- Detect test execution: `Running test: (\w+)`
- Extract warnings and errors for summary report

**Build Caching:**

- Check if backend already built (check for output files)
- Offer to skip rebuild if no source changes detected
- Use file modification times to determine if rebuild needed
- Implement `--force-rebuild` flag to override cache

### Step 4.5: Add Build Artifact Management

**Technical Approach:**

- Store build artifacts in dedicated directory (`build/`)
- Organize by backend name and timestamp
- Keep last 3 successful builds for rollback
- Clean old builds automatically to save space

**Artifact Storage Structure:**

```
build/
├── lret_cirq/
│   ├── 20260201_143022/  # Latest build
│   │   ├── lib/
│   │   ├── logs/
│   │   └── manifest.json
│   ├── 20260131_092011/  # Previous build
│   └── 20260130_151405/
├── cirq/
└── qiskit/
```

**Build Manifest:**

- JSON file with build metadata: timestamp, duration, success status
- List of all generated files with checksums
- Compilation flags used
- Test results summary

---

## Phase 5: File System Operations & Administrative Access

**Objective:** Grant AI agents secure access to the local file system with proper permissions, safety checks, and administrative privilege handling.

### Step 5.1: Implement Safe File Operations

**Technical Approach:**

- Create `FileSystemOperations` class with methods for read, write, list, delete
- Implement path validation to prevent directory traversal attacks
- Use `pathlib.Path.resolve()` to normalize and validate paths
- Check file permissions before operations
- Implement operation whitelist and blacklist

**Path Validation:**

- Resolve all paths to absolute paths
- Ensure path is within allowed directories (project root, temp directory)
- Reject paths containing `..` after normalization
- Block access to system directories (/etc, /sys, C:\Windows\System32)
- Create allowed path list in configuration

**File Reading:**

- Implement configurable max file size (default: 10MB)
- Use encoding detection for text files (chardet library)
- Handle binary files gracefully (show hex preview or metadata)
- Support reading file ranges (lines X-Y)
- Stream large files instead of loading entirely in memory

**File Writing:**

- Create backup before overwriting existing files
- Use atomic write operations (write to temp, then rename)
- Set appropriate file permissions after creation
- Verify write success by reading back content checksum

**Directory Operations:**

- List directory with metadata: size, modification time, permissions
- Support recursive directory listing with max depth
- Implement file search with glob patterns
- Exclude hidden files by default (show with flag)

### Step 5.2: Add Administrative Privilege Handling

**Technical Approach:**

- Detect current privilege level on startup
- On Windows: Use `ctypes.windll.shell32.IsUserAnAdmin()`
- On Unix: Check if effective UID is 0 (root)
- Prompt user for elevation when needed
- Provide non-privileged alternatives where possible

**Elevation Mechanisms:**

- **Windows:** Use `ShellExecute` with `runas` verb to launch elevated process
- **macOS/Linux:** Use `sudo` command with password prompt
- **UAC Bypass:** Store credentials temporarily in secure memory (not on disk)
- **Alternative:** Request one-time admin approval for sensitive operations

**Privilege Checking:**

- Before privileged operations, check if elevation needed
- Show clear warning to user about privileged operation
- Require explicit consent (not just "allow all")
- Log all privileged operations to audit log

**Privileged Operations:**

- Installing system packages (apt, yum, brew)
- Writing to system directories
- Modifying system services
- Changing file ownership or permissions
- Binding to privileged ports (<1024)

### Step 5.3: Implement File System Monitoring

**Technical Approach:**

- Use `watchdog` library for cross-platform file system monitoring
- Monitor project directories for changes
- Detect file creation, modification, deletion, move events
- Notify agent when relevant files change
- Trigger automatic actions based on events (e.g., rebuild on source change)

**Watchdog Configuration:**

- Create `FileSystemWatcher` class wrapping watchdog's Observer
- Set up recursive monitoring on project root
- Filter events by file patterns (ignore .git, **pycache**, etc.)
- Implement rate limiting to handle bursts of events

**Event Handling:**

- On file created: Update internal file index
- On file modified: Invalidate cached file contents
- On file deleted: Remove from index, notify if important file
- On file moved: Update references in project database

**Smart Rebuild Triggers:**

- Detect changes to source files (.py, .cpp, .cu)
- Automatically suggest rebuild when backend sources change
- Show diff of what changed since last build
- Offer to run tests after automatic rebuild

### Step 5.4: Add File Content Search and Analysis

**Technical Approach:**

- Implement code search using `ripgrep` (rg) for speed
- Support regex and literal string search
- Search across multiple files simultaneously
- Show context around matches (3 lines before/after)
- Highlight matches in UI with syntax highlighting

**Search Features:**

- **Full-text search:** Search all files in project
- **Code-aware search:** Respect language syntax (skip comments, strings)
- **Semantic search:** Use embeddings to find similar code (optional)
- **File type filtering:** Search only .py, .cpp, etc.
- **Case sensitivity:** Toggle case-sensitive/insensitive search

**Code Analysis:**

- Parse Python files with `ast` module for structure
- Extract function/class definitions
- Identify imports and dependencies
- Calculate code metrics (lines of code, complexity)
- Detect code smells and anti-patterns

### Step 5.5: Implement Secure Temporary File Management

**Technical Approach:**

- Use Python's `tempfile` module for secure temp file creation
- Create agent-specific temp directory on startup
- Clean up temp files on agent shutdown
- Implement auto-cleanup for temp files older than 24 hours

**Temp File Usage:**

- Store intermediate build artifacts
- Cache downloaded files
- Store large command outputs
- Create scratch space for file modifications

**Security Considerations:**

- Set restrictive permissions on temp files (owner read/write only)
- Use unpredictable filenames to prevent race conditions
- Ensure temp directory is on encrypted volume if available
- Sanitize filenames to prevent injection attacks

---

## Phase 6: Natural Language Planning & Execution

**Objective:** Enable agents to understand natural language requests, create execution plans, and autonomously execute multi-step tasks.

### Step 6.1: Design Task Planner System

**Technical Approach:**

- Create `TaskPlanner` class that uses LLM for plan generation
- Given user request, generate structured execution plan
- Plan consists of sequential or parallel steps
- Each step specifies tool to use and arguments
- Validate plan feasibility before execution

**Plan Generation Process:**

1. **Intent Recognition:** Classify user request into categories (build, analyze, modify)
2. **Dependency Analysis:** Identify prerequisites and order of operations
3. **Tool Selection:** Choose appropriate tools for each step
4. **Argument Extraction:** Extract values for tool parameters from request
5. **Validation:** Ensure all required parameters present and valid

**Plan Structure:**

```json
{
  "plan_id": "plan_20260201_143022",
  "description": "Build LRET Cirq backend and run tests",
  "steps": [
    {
      "step_id": 1,
      "tool": "execute_command",
      "arguments": { "command": "pip install cirq" },
      "description": "Install Cirq dependency",
      "depends_on": [],
      "estimated_duration": 30
    },
    {
      "step_id": 2,
      "tool": "build_backend",
      "arguments": { "backend_name": "lret_cirq" },
      "description": "Build LRET Cirq backend",
      "depends_on": [1],
      "estimated_duration": 120
    }
  ],
  "parallel_groups": [[1], [2]],
  "estimated_total_duration": 150
}
```

**LLM Prompt Engineering:**

- Provide LLM with tool descriptions and examples
- Use few-shot learning with example plans
- Request structured JSON output
- Validate JSON against schema
- Re-prompt if plan is invalid or incomplete

### Step 6.2: Implement Plan Execution Engine

**Technical Approach:**

- Create `PlanExecutor` class to run generated plans
- Execute steps in dependency order
- Support parallel execution of independent steps
- Monitor progress and update UI
- Handle step failures gracefully

**Execution Strategy:**

- Build dependency graph from `depends_on` relationships
- Use topological sort to determine execution order
- Execute steps without dependencies first
- Once step completes, trigger dependent steps
- Use `asyncio.gather()` for parallel execution

**Progress Tracking:**

- Calculate overall progress as percentage of completed steps
- Weight steps by estimated duration for accurate progress
- Update UI with current step and status
- Show step-by-step breakdown in collapsible section

**Error Handling:**

- On step failure, mark step as failed
- Determine if subsequent steps can still execute
- Offer to skip failed step and continue
- Provide option to retry failed step with modifications

**Plan Adaptation:**

- If step fails, ask LLM to revise plan
- Allow user to modify plan mid-execution
- Support inserting new steps between existing steps
- Enable plan pause, resume, and abort

### Step 6.3: Add Natural Language Command Parser

**Technical Approach:**

- Create command parser that extracts intent and parameters
- Use LLM to interpret ambiguous requests
- Support common command patterns (build X, run Y, show Z)
- Fall back to LLM if pattern matching fails

**Pattern Matching:**

- Regex patterns for common commands: `^build (.+)$`, `^run (.+)$`
- Extract backend name, script name, file path from patterns
- Validate extracted values against known entities
- Provide suggestions if entity not recognized

**LLM Fallback:**

- If no pattern matches, send request to LLM
- Ask LLM to categorize as: tool execution, question, clarification needed
- Extract tool name and arguments from LLM response
- Confirm with user before executing ambiguous commands

**Example Mappings:**

```
"build lret cirq" → build_backend(backend_name="lret_cirq")
"show git status" → git_status(path=".")
"run tests for cirq" → execute_command(command="python -m pytest tests/test_cirq.py")
"what files changed?" → git_status(path=".") + interpret_results
```

**Clarification Handling:**

- If request ambiguous, ask follow-up questions
- Provide multiple choice options when applicable
- Remember context from previous messages
- Support multi-turn dialogue for complex requests

### Step 6.4: Implement Script Execution Framework

**Technical Approach:**

- Support executing scripts in multiple languages: Python, Bash, PowerShell, JavaScript
- Auto-detect script language from file extension or shebang
- Set up appropriate interpreter and environment
- Pass arguments to scripts
- Capture and parse script output

**Language Support:**

- **Python:** Use same Python interpreter as Proxima (`sys.executable`)
- **Bash:** Use bash shell on Unix, Git Bash or WSL on Windows
- **PowerShell:** Use pwsh or powershell
- **JavaScript:** Use Node.js if available
- **Lua:** Use Lua interpreter if available (for embedded scripts)

**Script Execution:**

1. **Validation:** Check if script file exists and is readable
2. **Permission Check:** Ensure script has execute permission (chmod +x on Unix)
3. **Interpreter Selection:** Choose interpreter based on extension/shebang
4. **Environment Setup:** Set working directory, environment variables
5. **Execution:** Run script with `TerminalExecutor`
6. **Result Capture:** Collect stdout, stderr, return code

**Argument Passing:**

- Pass arguments as command-line parameters
- Support both positional and named arguments
- Handle spaces and special characters in arguments
- Support reading arguments from files (for large argument lists)

**Script Output Handling:**

- Parse JSON output for structured results
- Detect and extract specific patterns (e.g., test results)
- Stream output to UI in real-time
- Store full output in execution history

### Step 6.5: Add Context-Aware Command Suggestions

**Technical Approach:**

- Track user's recent commands and intents
- Suggest relevant follow-up commands
- Use LLM to generate contextual suggestions
- Learn from user's command patterns over time

**Suggestion Generation:**

- After executing command, analyze result
- Suggest natural next steps (e.g., after build, suggest run tests)
- Show suggestions as clickable buttons or autocomplete
- Rank suggestions by relevance and frequency

**Contextual Examples:**

```
After "build lret_cirq":
  - Suggest: "Run tests for LRET Cirq"
  - Suggest: "Show build logs"
  - Suggest: "Check compilation warnings"

After "git status" shows changes:
  - Suggest: "Commit changes"
  - Suggest: "Show diff"
  - Suggest: "Discard changes"
```

**Learning System:**

- Store command sequences in database
- Identify common patterns (A → B → C)
- Increase suggestion rank for frequently used sequences
- Personalize suggestions per user

---

## Phase 7: Git Operations Integration

**Objective:** Enable agents to perform full git operations including clone, pull, push, commit, branch management, and conflict resolution.

### Step 7.1: Implement Core Git Commands

**Technical Approach:**

- Create `GitOperations` class wrapping git CLI commands
- Execute git commands using `TerminalExecutor`
- Parse git output to structured format
- Detect and handle git errors gracefully

**Git Command Execution:**

- Use `git` command-line tool (ensure it's in PATH)
- Pass `--porcelain` flag for machine-readable output when available
- Set GIT_TERMINAL_PROMPT=0 to disable interactive prompts
- Use `--no-pager` to avoid less/more invocation

**Supported Commands:**

1. **clone:** `git clone <url> [destination]`
2. **pull:** `git pull [remote] [branch]`
3. **push:** `git push [remote] [branch]`
4. **commit:** `git commit -m "message"`
5. **status:** `git status --porcelain`
6. **add:** `git add <files>`
7. **branch:** `git branch`, `git branch <name>`
8. **checkout:** `git checkout <branch>`, `git checkout -b <new_branch>`
9. **merge:** `git merge <branch>`
10. **diff:** `git diff [commit1] [commit2]`
11. **log:** `git log --oneline --graph`
12. **remote:** `git remote add/remove/show`
13. **stash:** `git stash push/pop/list`
14. **reset:** `git reset [--hard/--soft] [commit]`
15. **rebase:** `git rebase <branch>`

### Step 7.2: Add Git Status Parser

**Technical Approach:**

- Parse `git status --porcelain` output
- Extract file statuses: modified, added, deleted, untracked
- Detect merge conflicts
- Show human-readable status summary in UI

**Porcelain Format Parsing:**

```
Porcelain output format:
XY PATH
X  PATH -> NEWPATH

Where X is status in index, Y is status in working tree:
' ' = unmodified
M = modified
A = added
D = deleted
R = renamed
C = copied
U = updated but unmerged
? = untracked
! = ignored
```

**Status Object:**

```python
@dataclass
class GitFileStatus:
    path: str
    status: str  # "modified", "added", "deleted", "untracked", "conflict"
    staged: bool
    index_status: str  # Raw X from XY
    worktree_status: str  # Raw Y from XY
```

**UI Presentation:**

- Group files by status (modified, added, deleted, untracked)
- Color-code each status: red (deleted), green (added), yellow (modified)
- Show summary: "5 files changed, 12 insertions(+), 3 deletions(-)"
- Provide quick action buttons: Stage All, Commit, Discard

### Step 7.3: Implement Git Diff Visualization

**Technical Approach:**

- Execute `git diff` command with `--no-color` or `--color=never`
- Parse unified diff format
- Display side-by-side diff in UI using custom widget
- Highlight added (green) and removed (red) lines

**Diff Parsing:**

- Use `difflib` or regex to parse unified diff output
- Extract hunks (sections of changes)
- Parse line numbers and change types (+/-)
- Store as structured diff object

**Diff Viewer Widget:**

- Create `DiffViewer` Textual widget
- Support three modes: unified, split, inline
- Use syntax highlighting for code files
- Allow navigation between hunks (next/prev buttons)

**Advanced Diff Features:**

- Show word-level diffs within lines (not just line-level)
- Fold unchanged context (show only changed sections)
- Support viewing diffs for specific commits
- Compare branches or tags

### Step 7.4: Add Commit & Push Workflow

**Technical Approach:**

- Implement staged commit workflow
- Support amending previous commits
- Push to remote with progress indication
- Handle authentication (SSH keys, HTTPS credentials)

**Commit Process:**

1. **Stage Files:** Use `git add` for selected files or `git add -A` for all
2. **Verify Staged:** Show list of staged files for confirmation
3. **Compose Message:** Prompt for commit message (with template)
4. **Validate Message:** Ensure message follows conventions (e.g., length limits)
5. **Commit:** Execute `git commit -m "message"`
6. **Confirm:** Show commit hash and summary

**Commit Message Assistance:**

- Provide template based on changes (e.g., "Fix bug in X", "Add feature Y")
- Use LLM to generate commit message from diff
- Enforce commit message conventions (Conventional Commits)
- Store commit message history for reuse

**Push Operation:**

- Detect current branch and tracking remote
- Show push preview: commits to be pushed
- Execute `git push origin <branch>`
- Handle errors: non-fast-forward, rejected push
- Suggest solutions: pull first, force push (with warning)

**Authentication Handling:**

- For SSH: Ensure SSH agent has key loaded
- For HTTPS: Use git credential helper
- Support personal access tokens (PAT)
- Store credentials securely in OS keychain (Windows Credential Manager, macOS Keychain, libsecret on Linux)

### Step 7.5: Implement Branch Management

**Technical Approach:**

- List all branches (local and remote)
- Create new branches
- Switch between branches
- Delete branches safely
- Visualize branch history

**Branch Listing:**

- Execute `git branch -a --format="%(refname:short)"`
- Parse output to extract branch names
- Mark current branch with indicator
- Show remote-tracking branches separately

**Branch Switching:**

- Use `git checkout <branch>` or `git switch <branch>` (Git 2.23+)
- Check for uncommitted changes before switch
- Offer to stash changes if present
- Restore working directory after switch

**Branch Creation:**

- Create from current HEAD: `git branch <name>`
- Create from specific commit: `git branch <name> <commit>`
- Create and switch: `git checkout -b <name>`
- Push new branch to remote: `git push -u origin <name>`

**Branch Visualization:**

- Use `git log --graph --oneline --all --decorate`
- Parse output to tree structure
- Display as ASCII graph in TUI
- Use Unicode box-drawing characters for clean appearance

### Step 7.6: Add Merge Conflict Resolution

**Technical Approach:**

- Detect merge conflicts from git status
- Extract conflict markers from files
- Provide UI for manual resolution
- Support automatic resolution strategies

**Conflict Detection:**

- Parse `git status --porcelain` for "U" status
- List files with conflicts
- Extract conflict sections from files

**Conflict Marker Parsing:**

```
<<<<<<< HEAD
Current branch content
=======
Incoming branch content
>>>>>>> incoming_branch
```

**Resolution UI:**

- Show three-way merge view: base, ours, theirs
- Provide buttons: Accept Ours, Accept Theirs, Edit Manually
- Highlight differences between versions
- Save resolved file and mark as resolved: `git add <file>`

**Automatic Resolution:**

- For simple conflicts (one side only changed), auto-resolve
- Use merge strategies: ours, theirs, union
- Implement smart resolution for common patterns (imports, config)

---

## Phase 8: Backend Code Modification with Safety

**Objective:** Enable agents to modify quantum backend code with full safety mechanisms including consent, backup, undo/redo, and rollback.

### Step 8.1: Implement File Backup System

**Technical Approach:**

- Before any modification, create backup of original file
- Store backups in `.proxima/backups/` directory
- Include timestamp and operation ID in backup filename
- Implement automatic cleanup of old backups (keep last 50)

**Backup Structure:**

```
.proxima/backups/
├── 20260201_143022_abc123/
│   ├── manifest.json
│   ├── lret_cirq/
│   │   └── simulator.py
│   └── lret_cirq/
│       └── optimizer.py
```

**Manifest File:**

```json
{
  "operation_id": "abc123",
  "timestamp": "2026-02-01T14:30:22Z",
  "operation_type": "modify_backend_code",
  "files": [
    {
      "path": "src/proxima/backends/lret_cirq/simulator.py",
      "checksum": "sha256:abcdef...",
      "size": 12345
    }
  ],
  "description": "Optimize state vector allocation"
}
```

**Backup Creation:**

- Use `shutil.copy2()` to preserve metadata (mtime, permissions)
- Calculate SHA256 checksum of original file
- Store relative path from project root
- Create manifest with operation metadata

### Step 8.2: Design Modification Preview System

**Technical Approach:**

- Generate diff preview before applying changes
- Show side-by-side comparison of old vs new code
- Require user approval before actual modification
- Support "dry run" mode that only shows changes

**Preview Generation:**

1. **Parse Request:** Extract file path and modification type (replace, insert, delete)
2. **Read Current Content:** Load file into memory
3. **Apply Modification (in memory):** Generate new content
4. **Generate Diff:** Use `difflib.unified_diff()` to create diff
5. **Display Preview:** Show diff in UI with syntax highlighting

**Modification Types:**

- **Replace:** Find specific text and replace with new text
- **Insert:** Add lines at specific line number
- **Delete:** Remove specific lines or text blocks
- **Append:** Add content to end of file
- **Prepend:** Add content to beginning of file

**Preview Display:**

- Use `DiffViewer` widget from earlier phase
- Show file path and modification summary
- Display lines added (green) and removed (red)
- Show context (5 lines before/after changes)
- Calculate impact: lines changed, new/old file size

### Step 8.3: Implement Consent Management System

**Technical Approach:**

- Create `ConsentManager` class to handle approval requests
- Support multiple consent scopes: once, session, always
- Track approval history in database
- Implement consent revocation

**Consent Types:**

1. **Once:** Approve this specific operation only
2. **Session:** Approve all similar operations in current session
3. **Always:** Approve automatically in future (stored in settings)
4. **Deny:** Reject operation

**Consent Request Structure:**

```python
@dataclass
class ConsentRequest:
    request_id: str
    operation: str  # "modify_backend_code", "execute_command", etc.
    description: str  # Human-readable description
    risk_level: str  # "low", "medium", "high", "critical"
    details: Dict[str, Any]  # Operation-specific details
    timestamp: datetime
```

**Consent Dialog:**

- Show modal dialog blocking UI
- Display operation description and details
- Show risk level with appropriate color coding
- Provide buttons: Approve, Approve Always, Deny
- Include timeout (auto-deny after 5 minutes)

**Consent Storage:**

- Store "always" approvals in `~/.proxima/consent_rules.json`
- Include: operation pattern, granted_at timestamp, granted_by user
- Support wildcard patterns (e.g., "execute_command:git \*")
- Provide UI to review and revoke stored consents

### Step 8.4: Add Checkpoint-Based Rollback

**Technical Approach:**

- Create checkpoint before each operation
- Checkpoint includes: all affected files, operation metadata, timestamp
- Maintain checkpoint stack (last 20 operations)
- Support rolling back to any checkpoint

**Checkpoint Creation:**

```python
@dataclass
class Checkpoint:
    id: str
    timestamp: datetime
    operation: str
    description: str
    files: List[FileSnapshot]  # Path, content, checksum
    metadata: Dict[str, Any]
```

**FileSnapshot:**

- Store full file content (for small files < 1MB)
- Store diff only (for large files)
- Include: path, content/diff, checksum, size, mtime

**Rollback Process:**

1. **Select Checkpoint:** Choose checkpoint to restore
2. **Validate Files:** Ensure target files haven't changed since backup
3. **Prompt Confirmation:** Show what will be restored
4. **Restore Files:** Write backup content back to original locations
5. **Verify Restoration:** Check checksums match backup
6. **Update Checkpoint Stack:** Mark newer checkpoints as invalidated

**Rollback Safety:**

- Detect if files modified after checkpoint creation
- Warn user if rollback will lose recent changes
- Offer to create snapshot of current state before rollback
- Implement "rollback of rollback" to undo mistaken rollbacks

### Step 8.5: Implement Undo/Redo Stack

**Technical Approach:**

- Maintain separate undo and redo stacks
- Push operation to undo stack on each modification
- Support sequential undo/redo
- Integrate with checkpoint system

**Undo Stack Structure:**

```python
undo_stack: List[Checkpoint] = []
redo_stack: List[Checkpoint] = []
```

**Undo Operation:**

1. **Pop from Undo Stack:** Get most recent operation
2. **Create Redo Checkpoint:** Save current state to redo stack
3. **Restore Files:** Apply undo checkpoint
4. **Update UI:** Show undo confirmation

**Redo Operation:**

1. **Pop from Redo Stack:** Get last undone operation
2. **Create Undo Checkpoint:** Save current state to undo stack
3. **Restore Files:** Apply redo checkpoint
4. **Update UI:** Show redo confirmation

**Stack Management:**

- Limit stack size to 20 operations each
- Clear redo stack when new modification made after undo
- Provide "Clear History" option to free memory
- Support jumping to specific point in history

**Keyboard Shortcuts:**

- Ctrl+Z: Undo
- Ctrl+Y or Ctrl+Shift+Z: Redo
- Alt+Z: Show undo history
- Bind to undo/redo buttons in UI

### Step 8.6: Add Code Modification Intelligence

**Technical Approach:**

- Use LLM to understand code structure
- Parse code with AST (Abstract Syntax Tree) for Python
- Use Tree-sitter for multi-language parsing
- Intelligent suggestion for modification location

**AST-Based Modification:**

- Parse Python file with `ast` module
- Locate specific function/class by name
- Insert, modify, or delete code at precise AST node location
- Regenerate source code from modified AST

**Tree-sitter Integration:**

- Use Tree-sitter for C++, CUDA, JavaScript parsing
- Build language-specific grammars
- Query syntax nodes with Tree-sitter queries
- Support incremental parsing for large files

**Smart Code Insertion:**

- Detect appropriate indentation level
- Match surrounding code style (spaces vs tabs)
- Auto-import required modules
- Suggest variable names matching conventions

**Modification Templates:**

- Provide templates for common modifications
- "Add new method to class"
- "Add parameter to function"
- "Insert logging statement"
- "Add type annotations"

---

## Phase 9: Agent Statistics & Telemetry System

**Objective:** Implement comprehensive real-time statistics tracking for all agent operations with toggle-able display and minimal performance overhead.

### Step 9.1: Design Telemetry Data Model

**Technical Approach:**

- Create `AgentTelemetry` class to collect all metrics
- Use reactive properties for real-time updates
- Store metrics in time-series structure for historical analysis
- Implement efficient in-memory storage with periodic persistence

**Metrics Categories:**

**1. LLM Metrics:**

- Provider name (OpenAI, Anthropic, etc.)
- Model name (gpt-4, claude-3, etc.)
- Temperature, max tokens settings
- Prompt tokens used (cumulative)
- Completion tokens used (cumulative)
- Total tokens used
- Requests sent
- Requests failed
- Average response time (ms)
- Token rate (tokens/second)
- Cost estimate (if pricing available)

**2. Agent Performance:**

- Uptime (seconds since start)
- Messages processed
- Tools executed (by type)
- Commands executed
- Files modified
- Errors encountered
- Success rate (%)
- Average tool execution time

**3. Terminal Statistics:**

- Active terminals
- Completed processes
- Failed processes
- Total output lines captured
- Current terminal count
- Peak terminal count

**4. Git Operations:**

- Commits made
- Branches created
- Files staged
- Pushes/pulls executed
- Merge conflicts resolved

**5. File Operations:**

- Files read
- Files written
- Files deleted
- Bytes read
- Bytes written
- Search operations

**6. Backend Building:**

- Builds initiated
- Builds succeeded
- Builds failed
- Total build time (seconds)
- Average build time

### Step 9.2: Implement Real-Time Metric Collection

**Technical Approach:**

- Instrument all agent operations to emit metric events
- Use decorator pattern to wrap functions with telemetry
- Collect metrics asynchronously to avoid blocking
- Aggregate metrics in background thread

**Instrumentation Decorator:**

```python
# Pseudocode
@collect_metrics(category="tool_execution")
async def execute_tool(tool_name: str, args: Dict):
    start = time.time()
    try:
        result = await _execute_tool_impl(tool_name, args)
        metrics.record_success(tool_name, time.time() - start)
        return result
    except Exception as e:
        metrics.record_failure(tool_name, str(e))
        raise
```

**Metric Emission:**

- Every operation emits start and end events
- Include: operation type, timestamp, duration, result
- Send events to centralized metric collector
- Collector aggregates and stores metrics

**Aggregation Strategy:**

- Update cumulative counters (total tokens, total requests)
- Maintain moving averages for rates (messages/minute)
- Store recent values in circular buffer (last 100 operations)
- Calculate percentiles (p50, p90, p99) for latencies

### Step 9.3: Create Statistics Display Widget

**Technical Approach:**

- Design `StatsPanel` Textual widget
- Use card-based layout with categories
- Update display every 500ms
- Implement smooth transitions for changing values

**UI Layout:**

```
┌─ Agent Statistics ──────────────────────────┐
│ 🤖 LLM                                      │
│   Provider: OpenAI          Model: gpt-4    │
│   Tokens: 15,234           Requests: 42     │
│   Avg Response: 1.2s       Cost: $0.23      │
├─────────────────────────────────────────────┤
│ ⚙️ Performance                               │
│   Uptime: 2h 15m           Messages: 156    │
│   Tools Run: 89            Success: 97.8%   │
│   Errors: 2                Avg Time: 850ms  │
├─────────────────────────────────────────────┤
│ 🖥️ Terminal                                 │
│   Active: 3                Completed: 47    │
│   Output Lines: 12,405     Processes: 50    │
└─────────────────────────────────────────────┘
```

**Value Formatting:**

- Large numbers: Use K/M/B suffixes (15234 → 15.2K)
- Time durations: Show in appropriate units (seconds, minutes, hours)
- Percentages: Show to 1 decimal place
- Currencies: Format with currency symbol and 2 decimals
- Rates: Show per-minute or per-hour as appropriate

**Real-Time Updates:**

- Subscribe to metric update events
- Update reactive properties when metrics change
- Textual automatically re-renders changed properties
- Use color changes to indicate significant events (spike in errors → red)

### Step 9.4: Implement Toggle Functionality

**Technical Approach:**

- Add keyboard shortcut (Ctrl+I) to toggle stats visibility
- Add button in agent header to toggle
- Animate expand/collapse with CSS transitions
- Store preference in user settings

**Toggle Implementation:**

- Use Textual's `display` property to show/hide
- When hidden: `stats_panel.display = False`
- When shown: `stats_panel.display = True`
- Bind to reactive property for automatic UI update

**Animation:**

- Use CSS transition for height change
- Animate from 0 to full height over 200ms
- Use easing function for smooth motion
- Fade in/out opacity simultaneously

**Persistence:**

- Store visibility preference in settings JSON
- Restore state on agent restart
- Per-agent setting (AI Assistant, AI Analysis, AI Thinking)

**Compact Mode:**

- Offer compact mode showing only key metrics
- Single row with most important stats
- Full mode shows all categories
- Toggle between compact and full with separate shortcut

### Step 9.5: Add Historical Metrics & Graphing

**Technical Approach:**

- Store time-series data in SQLite database
- Implement rolling window (keep last 7 days)
- Create graphs for key metrics over time
- Use Plotext or textual-plotext for terminal-based graphs

**Time-Series Storage:**

```sql
CREATE TABLE metrics (
    timestamp INTEGER PRIMARY KEY,
    metric_name TEXT,
    value REAL,
    metadata TEXT  -- JSON for additional context
);

CREATE INDEX idx_metric_time ON metrics(metric_name, timestamp);
```

**Graph Types:**

- **Line Graph:** Tokens used over time
- **Bar Graph:** Tools executed by type
- **Scatter Plot:** Response time vs token count
- **Histogram:** Distribution of command execution times

**Plotext Integration:**

- Use `plotext.plot()` for line graphs
- Render in Textual `Static` widget
- Update graph every 5 seconds
- Support zoom and pan with keyboard

**Export Features:**

- Export metrics to CSV for external analysis
- Generate HTML report with graphs
- Support Prometheus format for monitoring tools
- Create JSON API endpoint for dashboard integration

### Step 9.6: Implement Performance Monitoring & Alerts

**Technical Approach:**

- Monitor agent performance metrics
- Detect anomalies (response time spike, error rate increase)
- Show alerts in UI when issues detected
- Log alerts for later review

**Anomaly Detection:**

- Calculate baseline metrics (average, stddev)
- Flag values exceeding threshold (mean + 3\*stddev)
- Use sliding window for dynamic thresholds
- Implement cooldown to prevent alert spam

**Alert Types:**

- **High Latency:** Response time > 5s
- **Error Spike:** Error rate > 5% in last 10 requests
- **Token Limit:** Approaching token limit (> 80% of max)
- **Memory Usage:** Process memory > 1GB
- **Disk Space:** Available disk < 1GB

**Alert Display:**

- Show notification banner in UI
- Play sound alert (if enabled)
- Log to alert history
- Provide "Dismiss" and "View Details" buttons

**Performance Optimization:**

- If high latency detected, suggest reducing token limit
- If many errors, suggest checking LLM connectivity
- If memory high, offer to clear caches
- Auto-clear old logs to free disk space

---

## Phase 10: Integration & Testing

**Objective:** Integrate all components, ensure seamless interaction, and comprehensively test the entire agent system.

### Step 10.1: Component Integration

**Technical Approach:**

- Wire all modules together through `AgentController`
- Ensure proper dependency injection
- Test inter-component communication
- Validate event flow from source to sink

**Integration Points:**

1. **UI → Agent Controller:** User actions trigger agent operations
2. **Agent Controller → LLM Router:** Route requests to appropriate LLM provider
3. **Agent Controller → Terminal Executor:** Execute commands and capture output
4. **Terminal Executor → Multi-Terminal Monitor:** Stream output to monitoring system
5. **Agent Controller → Safety Manager:** Request consent for dangerous operations
6. **Safety Manager → UI:** Display consent dialogs
7. **Agent Controller → Telemetry System:** Record all operations for statistics
8. **Telemetry System → Stats Widget:** Display real-time metrics

**Dependency Management:**

- Use constructor injection for all dependencies
- Avoid circular dependencies (use interfaces/protocols)
- Implement factory pattern for complex object creation
- Use singleton pattern for shared resources (LLM router, terminal pool)

**Event Flow Validation:**

- Trace event path from generation to consumption
- Ensure no events lost or duplicated
- Verify event ordering (causality)
- Test event bus under high load

### Step 10.2: Unit Testing

**Technical Approach:**

- Write unit tests for each module using pytest
- Mock external dependencies (LLM APIs, file system, subprocess)
- Achieve >80% code coverage
- Use parameterized tests for multiple scenarios

**Test Structure:**

```
tests/
├── agent/
│   ├── test_terminal_executor.py
│   ├── test_session_manager.py
│   ├── test_tools.py
│   ├── test_safety.py
│   ├── test_git_operations.py
│   ├── test_backend_modifier.py
│   ├── test_multi_terminal_monitor.py
│   └── test_agent_controller.py
├── tui/
│   ├── test_agent_widgets.py
│   └── test_agent_ai_assistant.py
└── fixtures/
    ├── mock_llm.py
    ├── mock_subprocess.py
    └── sample_files/
```

**Mocking Strategy:**

- Mock `asyncio.create_subprocess_exec` for terminal tests
- Mock LLM responses with predefined outputs
- Use `pytest-mock` for flexible mocking
- Create fixtures for common test data

**Coverage Tools:**

- Use `pytest-cov` for coverage reports
- Generate HTML coverage report for review
- Identify untested code paths
- Add tests for edge cases

### Step 10.3: Integration Testing

**Technical Approach:**

- Test complete workflows end-to-end
- Use test fixtures for realistic scenarios
- Validate UI updates in response to operations
- Test error handling and recovery

**Test Scenarios:**

**1. Backend Build Workflow:**

- User requests "Build LRET Cirq backend"
- Agent parses request → creates plan → requests consent
- User approves → build starts → progress updates in UI
- Build completes → results shown → stats updated
- Verify: Output captured, progress accurate, stats correct

**2. Git Clone & Modify Workflow:**

- User requests "Clone repo from GitHub and modify X"
- Agent clones repo → checks out branch → modifies file
- Shows diff preview → requests consent → applies changes
- Commits changes → pushes to remote
- Verify: Repo cloned, file modified correctly, commit created

**3. Multi-Terminal Monitoring:**

- Start 4 concurrent backend builds
- Verify all terminals show output
- Verify output correctly attributed to each terminal
- Check performance under load
- Verify stats show 4 active terminals

**4. Error Recovery:**

- Trigger operation that fails
- Verify error captured and displayed
- Check rollback functionality
- Ensure agent remains operational after error

**Test Automation:**

- Use `pytest-bdd` for behavior-driven tests
- Write Gherkin scenarios for user stories
- Automate UI testing with Textual's `pilot` API
- Run integration tests in CI/CD pipeline

### Step 10.4: Performance Testing

**Technical Approach:**

- Test agent under high load
- Measure response times and throughput
- Identify bottlenecks
- Optimize hot paths

**Performance Scenarios:**

**1. High-Frequency Operations:**

- Execute 100 commands in rapid succession
- Measure average execution time
- Check for memory leaks
- Verify UI remains responsive

**2. Large Output Handling:**

- Execute command generating 100K lines of output
- Measure memory usage
- Verify scrolling performance
- Check log truncation works

**3. Concurrent Operations:**

- Run 10 tools simultaneously
- Measure resource utilization (CPU, memory)
- Verify no deadlocks or race conditions
- Check event bus handles concurrent events

**4. Long-Running Operations:**

- Start operation that runs for 10 minutes
- Verify progress updates continue
- Check UI doesn't freeze
- Test cancellation works

**Profiling:**

- Use `cProfile` for Python profiling
- Identify slow functions
- Use `line_profiler` for line-level analysis
- Optimize critical paths (event dispatch, UI updates)

**Benchmarks:**

- Set performance targets: < 100ms tool execution overhead
- Measure against targets
- Track performance over time
- Prevent performance regressions in CI

### Step 10.5: User Acceptance Testing

**Technical Approach:**

- Create test scenarios based on user stories
- Conduct usability testing with real users
- Gather feedback on UI/UX
- Iterate based on findings

**Test Users:**

- Quantum computing researchers (primary users)
- Backend developers
- Novice users (for accessibility)
- Power users (for advanced features)

**Usability Metrics:**

- Task completion rate
- Time to complete tasks
- Error rate
- User satisfaction (survey)
- Net Promoter Score (NPS)

**Feedback Collection:**

- In-app feedback button
- User surveys after testing sessions
- Analytics on feature usage
- Bug reports via GitHub issues

**Iteration Process:**

1. Identify pain points from feedback
2. Prioritize issues by impact and frequency
3. Implement fixes and improvements
4. Re-test with users
5. Repeat until satisfaction threshold met

### Step 10.6: Documentation & Deployment

**Technical Approach:**

- Write comprehensive documentation
- Create video tutorials
- Set up automated deployment
- Monitor production usage

**Documentation:**

- **User Guide:** How to use agent features
- **Developer Guide:** Architecture and extension points
- **API Reference:** Tool descriptions and parameters
- **Troubleshooting:** Common issues and solutions
- **FAQ:** Frequently asked questions

**Tutorials:**

- Getting started with AI agent
- Building backends with agent
- Using git operations
- Modifying backend code safely
- Advanced scripting and automation

**Deployment:**

- Package agent module as part of Proxima
- Distribute via pip: `pip install proxima-quantum`
- Create standalone TUI executable for non-Python users
- Set up auto-update mechanism

**Monitoring:**

- Implement telemetry opt-in for usage analytics
- Track feature adoption rates
- Monitor error rates in production
- Collect performance metrics

**Support:**

- Set up community forum
- Create support ticket system
- Provide email support for critical issues
- Maintain changelog for updates

---

## Implementation Timeline

**Total Estimated Duration:** 12-16 weeks

### Phase Breakdown:

- **Phase 1 (Real-Time Execution):** 2 weeks
- **Phase 2 (UI Enhancements):** 2 weeks
- **Phase 3 (Terminal Integration):** 2 weeks
- **Phase 4 (Backend Building):** 1.5 weeks
- **Phase 5 (File System Operations):** 1.5 weeks
- **Phase 6 (Natural Language Planning):** 2 weeks
- **Phase 7 (Git Operations):** 1.5 weeks
- **Phase 8 (Code Modification):** 2 weeks
- **Phase 9 (Statistics System):** 1.5 weeks
- **Phase 10 (Integration & Testing):** 3 weeks

### Milestones:

1. **M1 (Week 4):** Real-time execution and UI enhancements complete
2. **M2 (Week 8):** Terminal integration and backend building working
3. **M3 (Week 12):** All core features implemented
4. **M4 (Week 16):** Testing complete, ready for deployment

---

## Key Technologies & Libraries

### Python Libraries:

- **Textual** (v0.47+): TUI framework
- **Rich** (v13.7+): Terminal text formatting
- **asyncio**: Asynchronous I/O
- **aiohttp**: Async HTTP client
- **pytest** (v7.4+): Testing framework
- **SQLite**: Embedded database
- **watchdog**: File system monitoring
- **gitpython** (optional): Git operations (alternative to CLI)
- **tree-sitter**: Code parsing
- **chardet**: Encoding detection

### External Tools:

- **git**: Version control
- **ripgrep (rg)**: Fast code search
- **PowerShell Core (pwsh)**: Windows shell
- **bash/zsh**: Unix shells

### LLM Providers:

- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude 3)
- Google AI (Gemini)
- xAI (Grok)
- Ollama (local models)
- LM Studio (local models)

### Backend Build Tools:

- **pip**: Python package manager
- **cmake**: C++ build system
- **nvcc**: CUDA compiler
- **g++/clang++**: C++ compilers

---

## Success Criteria

The implementation is considered successful when:

1. ✅ All three AI agents (Assistant, Analysis, Thinking) functional
2. ✅ Real-time execution and results monitoring working
3. ✅ Word wrapping and resizable panels implemented
4. ✅ Clean, professional UI with toggle-able stats
5. ✅ Backend building supports all target simulators
6. ✅ File system operations work securely
7. ✅ Natural language command parsing functional
8. ✅ Admin access handling works on all platforms
9. ✅ Script execution supports multiple languages
10. ✅ Multi-terminal monitoring shows real-time output
11. ✅ Git operations complete and reliable
12. ✅ Backend code modification with safety features working
13. ✅ All statistics display in real-time
14. ✅ Test coverage >80%
15. ✅ Performance meets targets (<100ms overhead)
16. ✅ User acceptance testing passed
17. ✅ Documentation complete

---

## Risk Mitigation

### Technical Risks:

**Risk:** LLM API rate limits or failures  
**Mitigation:** Implement retry logic, caching, fallback providers

**Risk:** Terminal command incompatibility across platforms  
**Mitigation:** Extensive testing on Windows, macOS, Linux; command normalization

**Risk:** File system race conditions  
**Mitigation:** Use file locking, atomic operations, careful ordering

**Risk:** Security vulnerabilities in code modification  
**Mitigation:** Sandbox execution, consent system, audit logging

**Risk:** Performance degradation with many concurrent operations  
**Mitigation:** Resource pooling, rate limiting, async design

### User Experience Risks:

**Risk:** UI complexity overwhelming users  
**Mitigation:** Progressive disclosure, good defaults, tutorials

**Risk:** Consent fatigue (too many approval requests)  
**Mitigation:** Smart defaults, "always allow" option, operation grouping

**Risk:** Confusing error messages  
**Mitigation:** Clear, actionable error messages; suggested fixes

---

## Future Enhancements

Beyond initial implementation:

1. **AI Model Fine-Tuning:** Train custom models on Proxima domain
2. **Collaborative Editing:** Multiple users on same agent session
3. **Cloud Sync:** Sync sessions and history across devices
4. **Mobile Companion App:** Monitor executions from mobile
5. **Voice Commands:** Speak commands to agent
6. **IDE Integration:** VS Code extension for Proxima agent
7. **Jupyter Integration:** Use agent in Jupyter notebooks
8. **CI/CD Integration:** Agent as part of build pipeline
9. **Kubernetes Support:** Deploy backends to K8s clusters
10. **Advanced Analytics:** ML-based performance analysis

---

## Conclusion

This comprehensive guide provides all necessary technical details for implementing a production-grade AI agent system for Proxima. The system will transform the quantum computing platform with real-time monitoring, intelligent automation, and a professional user experience.

The phase-by-phase approach ensures steady progress with clear milestones and deliverables. Each phase builds upon previous work, culminating in a fully integrated, tested, and documented system.

Implementation by GPT-5.1 Codex Max, GPT-5.2 Codex Max, or Claude Opus 4.5 following this guide should result in a robust, feature-complete agent system that meets all specified requirements.

---

**Document Version:** 1.0  
**Last Updated:** February 1, 2026  
**Authors:** Proxima Development Team  
**Status:** Ready for Implementation
