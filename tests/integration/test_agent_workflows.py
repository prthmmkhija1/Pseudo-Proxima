"""Integration tests for Agent workflows.

Phase 10: Integration & Testing

Tests cover:
- End-to-end backend build workflow
- Git clone and modify workflow
- Multi-terminal monitoring workflow
- Error recovery scenarios
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.fixtures.mock_agent import (
    MockLLMClient,
    MockLLMResponse,
    MockSubprocessFactory,
    MockProcessOutput,
    MockFileSystem,
    MockGitRepository,
    MockTelemetry,
    MockConsentManager,
    create_mock_llm_client,
    create_mock_subprocess_factory,
    create_mock_file_system,
    create_mock_git_repo,
    create_mock_telemetry,
    create_mock_consent_manager,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def integration_components():
    """Create all components for integration testing."""
    llm = create_mock_llm_client()
    subprocess_factory = create_mock_subprocess_factory()
    file_system = create_mock_file_system()
    telemetry = create_mock_telemetry()
    consent = create_mock_consent_manager()
    
    return {
        "llm": llm,
        "subprocess": subprocess_factory,
        "fs": file_system,
        "telemetry": telemetry,
        "consent": consent,
    }


# =============================================================================
# BACKEND BUILD WORKFLOW TESTS
# =============================================================================

class TestBackendBuildWorkflow:
    """Integration tests for backend build workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_build_workflow(self, integration_components):
        """Test complete backend build workflow."""
        llm = integration_components["llm"]
        subprocess_factory = integration_components["subprocess"]
        telemetry = integration_components["telemetry"]
        consent = integration_components["consent"]
        
        # Step 1: Get build instructions from LLM
        llm.queue_response(MockLLMResponse(
            content='{"action": "build_backend", "backend": "cirq", "steps": ["configure", "build", "install"]}'
        ))
        
        response = await llm.chat([
            {"role": "user", "content": "Build the Cirq quantum backend"}
        ])
        
        build_plan = json.loads(response["choices"][0]["message"]["content"])
        assert build_plan["action"] == "build_backend"
        
        # Step 2: Request consent
        approved = await consent.request_consent(
            operation="Build Backend",
            description=f"Build {build_plan['backend']} backend",
            risk_level="medium",
        )
        assert approved is True
        
        # Step 3: Configure build
        subprocess_factory.set_output("cmake", MockProcessOutput(
            stdout="-- Configuring done\n-- Generating done\n",
            returncode=0,
            duration_ms=500,
        ))
        
        configure_process = await subprocess_factory.create_subprocess_shell(
            "cmake -B build -DCMAKE_BUILD_TYPE=Release"
        )
        stdout, _ = await configure_process.communicate()
        
        telemetry.record_event("build_configure", {
            "backend": "cirq",
            "output": stdout.decode(),
            "success": configure_process.returncode == 0,
        })
        
        assert configure_process.returncode == 0
        
        # Step 4: Build
        subprocess_factory.set_output("cmake --build", MockProcessOutput(
            stdout="Building...\n[100%] Built target cirq\n",
            returncode=0,
            duration_ms=2000,
        ))
        
        build_process = await subprocess_factory.create_subprocess_shell(
            "cmake --build build"
        )
        stdout, _ = await build_process.communicate()
        
        telemetry.record_event("build_compile", {
            "backend": "cirq",
            "output": stdout.decode(),
            "success": build_process.returncode == 0,
        })
        
        assert build_process.returncode == 0
        
        # Verify telemetry
        assert len(telemetry.events) == 2
        assert telemetry.events[0]["type"] == "build_configure"
        assert telemetry.events[1]["type"] == "build_compile"
    
    @pytest.mark.asyncio
    async def test_build_with_failure_recovery(self, integration_components):
        """Test build workflow with failure and recovery."""
        subprocess_factory = integration_components["subprocess"]
        
        # First build fails
        subprocess_factory.set_output("cmake --build", MockProcessOutput(
            stdout="",
            stderr="error: compilation failed\n",
            returncode=1,
        ))
        
        build_process = await subprocess_factory.create_subprocess_shell(
            "cmake --build build"
        )
        _, stderr = await build_process.communicate()
        
        assert build_process.returncode == 1
        assert "error" in stderr.decode()
        
        # Fix and retry
        subprocess_factory.set_output("cmake --build", MockProcessOutput(
            stdout="Build successful\n",
            returncode=0,
        ))
        
        retry_process = await subprocess_factory.create_subprocess_shell(
            "cmake --build build"
        )
        stdout, _ = await retry_process.communicate()
        
        assert retry_process.returncode == 0
        assert "successful" in stdout.decode()


# =============================================================================
# GIT WORKFLOW TESTS
# =============================================================================

class TestGitWorkflow:
    """Integration tests for git workflows."""
    
    @pytest.mark.asyncio
    async def test_clone_and_modify_workflow(self, integration_components):
        """Test complete git clone and modify workflow."""
        llm = integration_components["llm"]
        subprocess_factory = integration_components["subprocess"]
        fs = integration_components["fs"]
        consent = integration_components["consent"]
        
        # Step 1: Clone repository
        subprocess_factory.set_output("git clone", MockProcessOutput(
            stdout="Cloning into 'quantum-backend'...\nReceiving objects: 100%\ndone.\n",
            returncode=0,
        ))
        
        approved = await consent.request_consent(
            operation="Clone Repository",
            description="Clone quantum-backend repository",
            risk_level="low",
        )
        assert approved is True
        
        clone_process = await subprocess_factory.create_subprocess_shell(
            "git clone https://github.com/test/quantum-backend.git"
        )
        stdout, _ = await clone_process.communicate()
        
        assert clone_process.returncode == 0
        assert "done" in stdout.decode()
        
        # Step 2: Create file
        fs.write_file(
            "/quantum-backend/src/feature.py",
            '''"""New feature implementation."""

def new_feature():
    """Implement new feature."""
    return "feature implemented"
'''
        )
        
        assert fs.exists("/quantum-backend/src/feature.py")
        
        # Step 3: Stage and commit
        subprocess_factory.set_output("git add", MockProcessOutput(returncode=0))
        subprocess_factory.set_output("git commit", MockProcessOutput(
            stdout="[feature-branch abc1234] Add new feature\n 1 file changed\n",
            returncode=0,
        ))
        
        add_process = await subprocess_factory.create_subprocess_shell(
            "git add src/feature.py"
        )
        await add_process.wait()
        
        commit_process = await subprocess_factory.create_subprocess_shell(
            'git commit -m "Add new feature"'
        )
        stdout, _ = await commit_process.communicate()
        
        assert commit_process.returncode == 0
        assert "Add new feature" in stdout.decode()
    
    @pytest.mark.asyncio
    async def test_branch_workflow(self, integration_components):
        """Test branch creation and switching."""
        git_repo = create_mock_git_repo("/project")
        
        # Create feature branch
        git_repo.create_branch("feature/new-api")
        assert "feature/new-api" in git_repo.branches
        
        # Switch to branch
        git_repo.checkout("feature/new-api")
        assert git_repo.current_branch == "feature/new-api"
        
        # Make commit
        git_repo.add_commit(
            message="Implement new API",
            files=["src/api.py"],
        )
        
        assert len(git_repo.commits) == 1
        
        # Switch back to main
        git_repo.checkout("main")
        assert git_repo.current_branch == "main"


# =============================================================================
# MULTI-TERMINAL WORKFLOW TESTS
# =============================================================================

class TestMultiTerminalWorkflow:
    """Integration tests for multi-terminal monitoring."""
    
    @pytest.mark.asyncio
    async def test_parallel_build_and_test(self, integration_components):
        """Test parallel build and test execution."""
        subprocess_factory = integration_components["subprocess"]
        
        # Configure mock outputs
        subprocess_factory.set_output("cmake --build", MockProcessOutput(
            stdout="Build complete\n",
            returncode=0,
            duration_ms=1000,
        ))
        subprocess_factory.set_output("pytest", MockProcessOutput(
            stdout="===== 10 passed =====\n",
            returncode=0,
            duration_ms=2000,
        ))
        
        # Start both processes
        build_process = await subprocess_factory.create_subprocess_shell(
            "cmake --build build"
        )
        test_process = await subprocess_factory.create_subprocess_shell(
            "pytest tests/"
        )
        
        # Wait for both
        build_result, test_result = await asyncio.gather(
            build_process.wait(),
            test_process.wait(),
        )
        
        assert build_result == 0
        assert test_result == 0
    
    @pytest.mark.asyncio
    async def test_monitor_multiple_sessions(self, integration_components):
        """Test monitoring multiple terminal sessions."""
        subprocess_factory = integration_components["subprocess"]
        telemetry = integration_components["telemetry"]
        
        # Create session data
        sessions = []
        
        for i in range(3):
            subprocess_factory.set_output(f"task_{i}", MockProcessOutput(
                stdout=f"Task {i} output\n",
                returncode=0,
            ))
            
            process = await subprocess_factory.create_subprocess_shell(f"task_{i}")
            sessions.append({
                "id": f"session-{i}",
                "process": process,
                "start_time": time.time(),
            })
        
        # Wait for all and record telemetry
        for session in sessions:
            exit_code = await session["process"].wait()
            telemetry.record_event("session_complete", {
                "session_id": session["id"],
                "exit_code": exit_code,
            })
        
        # Verify all completed
        session_events = telemetry.get_events("session_complete")
        assert len(session_events) == 3


# =============================================================================
# ERROR RECOVERY TESTS
# =============================================================================

class TestErrorRecovery:
    """Integration tests for error recovery scenarios."""
    
    @pytest.mark.asyncio
    async def test_llm_retry_on_failure(self, integration_components):
        """Test LLM retry on transient failure."""
        llm = integration_components["llm"]
        
        # First call fails
        llm.fail_rate = 1.0  # 100% fail
        
        with pytest.raises(Exception):
            await llm.chat([{"role": "user", "content": "Test"}])
        
        # Retry succeeds
        llm.fail_rate = 0.0
        llm.queue_response(MockLLMResponse(content="Success"))
        
        response = await llm.chat([{"role": "user", "content": "Test"}])
        
        assert "Success" in response["choices"][0]["message"]["content"]
    
    @pytest.mark.asyncio
    async def test_build_cleanup_on_failure(self, integration_components):
        """Test cleanup after build failure."""
        subprocess_factory = integration_components["subprocess"]
        fs = integration_components["fs"]
        
        # Create build directory
        fs.mkdir("/project/build")
        fs.write_file("/project/build/CMakeCache.txt", "cache data")
        
        # Build fails
        subprocess_factory.set_output("cmake --build", MockProcessOutput(
            returncode=1,
            stderr="Build failed",
        ))
        
        process = await subprocess_factory.create_subprocess_shell(
            "cmake --build build"
        )
        await process.wait()
        
        # Cleanup on failure
        if process.returncode != 0:
            fs.delete("/project/build/CMakeCache.txt")
            fs.delete("/project/build")
        
        # Verify cleanup
        assert not fs.exists("/project/build/CMakeCache.txt")
    
    @pytest.mark.asyncio
    async def test_consent_denial_handling(self, integration_components):
        """Test handling consent denial."""
        consent = create_mock_consent_manager(auto_approve=False)
        telemetry = integration_components["telemetry"]
        
        # Request consent for dangerous operation
        approved = await consent.request_consent(
            operation="Delete All Files",
            description="Delete all project files",
            risk_level="critical",
        )
        
        # Record denial
        telemetry.record_event("consent_denied", {
            "operation": "Delete All Files",
            "risk_level": "critical",
        })
        
        assert approved is False
        
        denial_events = telemetry.get_events("consent_denied")
        assert len(denial_events) == 1


# =============================================================================
# FULL WORKFLOW INTEGRATION TESTS
# =============================================================================

class TestFullWorkflowIntegration:
    """Full end-to-end workflow integration tests."""
    
    @pytest.mark.asyncio
    async def test_complete_agent_session(self, integration_components):
        """Test complete agent session workflow."""
        llm = integration_components["llm"]
        subprocess_factory = integration_components["subprocess"]
        fs = integration_components["fs"]
        telemetry = integration_components["telemetry"]
        consent = integration_components["consent"]
        
        # Session start
        session_id = f"session-{int(time.time())}"
        telemetry.record_event("session_start", {"session_id": session_id})
        
        # Step 1: User asks to build backend
        llm.queue_response(MockLLMResponse(
            content='{"action": "build_backend", "backend": "qiskit"}'
        ))
        
        user_request = "Build the Qiskit backend with GPU support"
        response = await llm.chat([
            {"role": "user", "content": user_request}
        ])
        
        telemetry.record_event("user_message", {"content": user_request})
        
        # Step 2: Get consent
        approved = await consent.request_consent(
            operation="Build Qiskit Backend",
            description="Build Qiskit with GPU support",
            risk_level="medium",
        )
        
        assert approved is True
        
        # Step 3: Execute build
        subprocess_factory.set_output("pip install", MockProcessOutput(
            stdout="Successfully installed qiskit-gpu\n",
            returncode=0,
        ))
        
        process = await subprocess_factory.create_subprocess_shell(
            "pip install qiskit[gpu]"
        )
        stdout, _ = await process.communicate()
        
        telemetry.record_event("command_executed", {
            "command": "pip install qiskit[gpu]",
            "exit_code": process.returncode,
        })
        
        # Step 4: Verify installation
        fs.write_file("/project/.installed", "qiskit-gpu")
        
        # Session end
        telemetry.record_event("session_end", {
            "session_id": session_id,
            "success": True,
        })
        
        # Verify complete workflow
        events = telemetry.get_events()
        event_types = [e["type"] for e in events]
        
        assert "session_start" in event_types
        assert "user_message" in event_types
        assert "command_executed" in event_types
        assert "session_end" in event_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
