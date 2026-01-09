# Competitive Analysis: Proxima vs Crush (OpenCode AI)

**Date:** January 10, 2026  
**Status:** Strategic Assessment  
**Purpose:** Compare Proxima with industry-leading AI coding agents and identify gaps to achieve competitive advantage

---

## Executive Summary

**Crush** (formerly OpenCode AI) is the current market leader in terminal-based AI coding agents with:
- ⭐ **17.2k GitHub stars** (vs Proxima: new/unreleased)
- 👥 **65 contributors** (vs Proxima: small team)
- 📦 **95 releases** with active development (commits within hours)
- 🔧 **Mature ecosystem** with extensive LLM, MCP, LSP, and tool integrations

### Current Assessment: **Proxima is NOT yet competitive with Crush**

However, Proxima has a **unique value proposition** that Crush lacks: **Quantum Simulation Orchestration**

---

## Feature Comparison Matrix

| Feature Category | Crush | Proxima (Current) | Proxima (Planned) | Gap Status |
|-----------------|-------|-------------------|-------------------|------------|
| **Core AI Capabilities** |
| Multi-LLM Support | ✅ Extensive (15+ providers) | 🔶 Foundation only | ✅ Phase 3 | 🔴 Major Gap |
| Session Management | ✅ Full persistence | 🔶 Basic | ✅ Phase 1 | 🟡 Minor Gap |
| Auto-Compact (Context Window) | ✅ Advanced | ❌ None | ⚪ Not planned | 🔴 Major Gap |
| Model Switching Mid-Session | ✅ Yes | ❌ No | ⚪ Not planned | 🟡 Consider Adding |
| **TUI/CLI Experience** |
| Terminal UI | ✅ Bubble Tea (excellent) | 🔶 Textual (planned) | ✅ Phase 6 | 🟡 In Progress |
| CLI Commands | ✅ Comprehensive | ✅ Complete foundation | ✅ Done | ✅ Competitive |
| Keyboard Shortcuts | ✅ Vim-like, extensive | ❌ Basic | ✅ Phase 6 | 🔴 Major Gap |
| Non-interactive Mode | ✅ Yes (-p flag) | ❌ No | ⚪ Should add | 🟡 Minor Gap |
| **Developer Tools Integration** |
| LSP Integration | ✅ Multi-language | ❌ None | ⚪ Not planned | 🔴 Critical Gap |
| MCP Support | ✅ stdio, http, sse | ❌ None | ⚪ Not planned | 🔴 Critical Gap |
| Git Integration | ✅ Attribution, commits | ❌ None | ⚪ Not planned | 🟡 Consider Adding |
| File Operations | ✅ view, edit, patch, write | ❌ None | ⚪ Limited | 🔴 Major Gap |
| Sourcegraph Search | ✅ Yes | ❌ No | ⚪ Not planned | 🟡 Nice to Have |
| **Agent Skills & Extensibility** |
| Custom Commands | ✅ Markdown-based | ❌ None | ⚪ Not planned | 🟡 Minor Gap |
| Agent Skills (AgentSkills.io) | ✅ Yes | ❌ No | ⚪ Not planned | 🟡 Consider Adding |
| Plugin System | ✅ MCP-based | 🔶 Foundation | ✅ Phase 2 | 🟡 In Progress |
| Custom Providers | ✅ OpenAI/Anthropic compat | 🔶 Planned | ✅ Phase 3 | 🟡 In Progress |
| **Configuration & Settings** |
| Config Hierarchy | ✅ Project > Global | ✅ Complete | ✅ Done | ✅ Competitive |
| Environment Variables | ✅ Extensive | ✅ Complete | ✅ Done | ✅ Competitive |
| JSON Schema Validation | ✅ Yes | ✅ Pydantic | ✅ Done | ✅ Competitive |
| .ignore Files | ✅ .crushignore | ❌ None | ⚪ Should add | 🟡 Minor Gap |
| **Unique to Proxima (Differentiation)** |
| Quantum Backend Support | ❌ N/A | ✅ LRET, Cirq, Qiskit | ✅ Phase 2 | 🎯 Advantage |
| Multi-Backend Comparison | ❌ N/A | 🔶 Foundation | ✅ Phase 5 | 🎯 Advantage |
| Backend Auto-Selection | ❌ N/A | 🔶 Foundation | ✅ Phase 3 | 🎯 Advantage |
| Quantum Result Insights | ❌ N/A | 🔶 Foundation | ✅ Phase 5 | 🎯 Advantage |
| Resource-Aware Execution | 🔶 Basic | ✅ Complete | ✅ Phase 4 | 🎯 Advantage |
| Execution Control (Pause/Resume/Rollback) | ❌ Cancel only | ✅ Full FSM | ✅ Phase 4 | 🎯 Advantage |
| Explicit Consent System | 🔶 Permission prompts | ✅ Comprehensive | ✅ Phase 4 | 🎯 Advantage |
| proxima_agent.md Support | ❌ N/A | 🔶 Planned | ✅ Phase 5 | 🎯 Unique |
| **Observability & Safety** |
| Logging System | ✅ File-based + CLI viewer | ✅ Structlog + Rich | ✅ Done | ✅ Competitive |
| Execution Timer | 🔶 Basic | ✅ Full transparency | ✅ Phase 1/4 | 🎯 Advantage |
| Memory Monitoring | ❌ None | ✅ psutil-based | ✅ Phase 4 | 🎯 Advantage |
| Fail-Safe Mechanisms | ❌ None | ✅ Planned | ✅ Phase 4 | 🎯 Advantage |
| **Distribution & Packaging** |
| PyPI Package | ❌ Go-based | ✅ Ready | ✅ Done | ✅ Competitive |
| Docker Support | ❌ Not provided | ✅ Complete | ✅ Done | ✅ Competitive |
| Homebrew | ✅ Yes | ✅ Ready | ✅ Done | ✅ Competitive |
| npm | ✅ Yes | ❌ No | ⚪ Consider | 🟡 Minor Gap |
| Standalone Binaries | ✅ Multi-platform | ✅ Planned | ✅ Phase 6 | 🟡 In Progress |

**Legend:** ✅ Complete | 🔶 Partial/Foundation | ❌ None | ⚪ Not Planned | 🎯 Proxima Advantage

---

## Critical Gaps to Address

### 🔴 HIGH PRIORITY (Must Have to Compete)

#### 1. LSP (Language Server Protocol) Integration
**Why Critical:** Crush uses LSPs for code intelligence (diagnostics, completions, definitions). Without this, Proxima cannot provide meaningful code assistance.

**Recommendation:**
```python
# Add to Phase 6 or create new Phase 3.5
# src/proxima/lsp/
#   ├── __init__.py
#   ├── client.py      # LSP client implementation
#   ├── languages.py   # Per-language configurations
#   └── diagnostics.py # Diagnostic collection
```

**Effort:** 3-4 weeks

#### 2. MCP (Model Context Protocol) Support
**Why Critical:** Industry standard for extending AI agent capabilities. Crush supports stdio, http, and sse transports.

**Recommendation:**
```python
# Add to Phase 5 or create dedicated phase
# src/proxima/mcp/
#   ├── __init__.py
#   ├── transport.py   # stdio, http, sse
#   ├── tools.py       # Tool discovery and execution
#   └── registry.py    # MCP server registry
```

**Effort:** 4-5 weeks

#### 3. File Operations Tools
**Why Critical:** AI coding agents need to view, edit, search, and manipulate files.

**Recommendation:**
```python
# Extend src/proxima/utils/ or create src/proxima/tools/
# Tools needed:
#   - view: Read file contents with line ranges
#   - edit: Modify files (diff-based)
#   - patch: Apply patches
#   - grep: Search file contents
#   - glob: Find files by pattern
#   - ls: List directory contents
```

**Effort:** 2-3 weeks

#### 4. Auto-Compact / Context Window Management
**Why Critical:** Long conversations exceed model context limits. Crush auto-summarizes at 95% capacity.

**Recommendation:**
```python
# Add to src/proxima/intelligence/
#   ├── context_manager.py  # Track token usage
#   └── summarizer.py       # Auto-summarize when needed
```

**Effort:** 2 weeks

### 🟡 MEDIUM PRIORITY (Should Have)

#### 5. Git Integration
- Commit attribution (`Assisted-by: Model via Proxima`)
- File change tracking during sessions
- PR description generation

#### 6. Non-Interactive Mode
```bash
proxima -p "Explain quantum entanglement" -f json
```

#### 7. Custom Commands System
- Markdown-based command definitions
- Named argument support
- User and project-level commands

#### 8. .proximaignore Support
- Respect .gitignore by default
- Additional ignore patterns for Proxima

### 🟢 LOW PRIORITY (Nice to Have)

- Sourcegraph integration
- npm distribution
- Agent Skills (AgentSkills.io) support
- Model switching mid-session

---

## Strategic Recommendations

### 1. **Double Down on Quantum Differentiation**

Proxima's unique value is **quantum simulation orchestration**. No competitor offers:
- Multi-backend quantum execution (LRET, Cirq, Qiskit Aer)
- Automatic backend selection with explanation
- Quantum-specific result interpretation and insights
- Resource-aware quantum simulation

**Action:** Make quantum capabilities exceptional and market-leading.

### 2. **Add "Coding Agent" Capabilities as Secondary Feature**

To compete with Crush, Proxima needs basic coding agent features:
- File operations (view, edit, grep)
- LSP integration for code intelligence
- MCP support for extensibility

**However:** Position these as "AI-assisted quantum workflow development" rather than competing directly with Crush.

### 3. **Focus on Enterprise/Research Use Cases**

Crush targets individual developers. Proxima should target:
- **Research Labs:** Quantum computing research workflows
- **Enterprise:** Production quantum simulation with consent and audit trails
- **Education:** Learning quantum computing with guided insights

### 4. **Leverage Python Ecosystem**

Crush is Go-based. Proxima being Python-based provides:
- Native integration with quantum libraries (Cirq, Qiskit, NumPy)
- Jupyter notebook compatibility
- Familiar ecosystem for data scientists and researchers

---

## Implementation Roadmap to Achieve Parity

### Phase 3.5: Developer Tools Integration (NEW - 4 weeks)
**Goal:** Add essential coding agent capabilities

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | File Operations | view, edit, patch, grep, glob, ls tools |
| 2 | LSP Client | Basic LSP integration (diagnostics focus) |
| 3 | MCP Foundation | stdio transport support |
| 4 | Integration Testing | E2E tests for new capabilities |

### Updated Phase 5: Advanced Features (5 weeks → 6 weeks)
Add:
- Context window management / auto-compact
- Non-interactive mode (-p flag)
- .proximaignore support

### Updated Phase 6: Production (4 weeks → 5 weeks)
Add:
- Custom commands system
- Git integration with attribution
- Agent Skills support (optional)

---

## Competitive Positioning Statement

> **Proxima** is the first intelligent quantum simulation orchestration framework that combines 
> multi-backend quantum execution with AI-powered insights. Unlike general-purpose coding agents,
> Proxima specializes in quantum computing workflows with resource-aware execution, explicit 
> consent management, and comprehensive result interpretation.

---

## Success Metrics

### To Match Crush (General Capabilities)
- [ ] LSP integration with 3+ languages
- [ ] MCP support (stdio transport minimum)
- [ ] File operations (view, edit, grep)
- [ ] Auto-compact / context management
- [ ] Non-interactive mode

### To Beat Crush (Unique Differentiation)
- [x] Multi-backend quantum support ✅
- [x] Execution control (pause/resume/rollback) ✅
- [x] Resource-aware execution ✅
- [x] Explicit consent system ✅
- [ ] Quantum-specific insights
- [ ] proxima_agent.md compatibility
- [ ] Multi-backend comparison with visualizations

---

## Conclusion

**Current State:** Proxima is not competitive with Crush for general AI coding tasks.

**Path to Success:** 
1. **Accept this limitation** and focus on quantum simulation as the primary use case
2. **Add essential coding features** (file ops, LSP, MCP) to support quantum workflow development
3. **Excel in quantum differentiation** where no competitor exists
4. **Target specialized audience** (researchers, enterprise, education) rather than general developers

**Timeline to Competitive Position:** 10-12 weeks of focused development

---

## Appendix: Crush Architecture Reference

```
Crush (Go-based)
├── cmd/                    # CLI entry point (Cobra)
├── internal/
│   ├── app/               # Core application services
│   ├── config/            # Configuration management
│   ├── db/                # SQLite database + migrations
│   ├── llm/               # LLM providers and tools
│   ├── tui/               # Terminal UI (Bubble Tea)
│   ├── logging/           # Logging infrastructure
│   ├── message/           # Message handling
│   ├── session/           # Session management
│   └── lsp/               # Language Server Protocol
```

**Key Technologies:**
- **Language:** Go 1.24+
- **TUI Framework:** Bubble Tea (Charmbracelet)
- **Database:** SQLite
- **CLI:** Cobra
- **LLM Protocol:** MCP (Model Context Protocol)
- **Code Intelligence:** LSP
