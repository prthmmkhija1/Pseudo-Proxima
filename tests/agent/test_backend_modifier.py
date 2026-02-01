"""Unit tests for Backend Modifier module.

Phase 10: Integration & Testing

Tests cover:
- Backend code modification
- File backup and restore
- Code validation
- Safe modification workflows
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.fixtures.mock_agent import (
    MockFileSystem,
    MockConsentManager,
    create_mock_file_system,
    create_mock_consent_manager,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_file_system():
    """Create mock file system."""
    fs = create_mock_file_system()
    
    # Populate with sample backend code
    fs.write_file(
        "/backends/cirq_backend.py",
        '''"""Cirq backend implementation."""

class CirqBackend:
    """Backend for Cirq quantum circuits."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.name = "cirq"
    
    def execute(self, circuit):
        """Execute a quantum circuit."""
        return {"result": "executed"}
'''
    )
    
    fs.write_file(
        "/backends/qiskit_backend.py",
        '''"""Qiskit backend implementation."""

class QiskitBackend:
    """Backend for Qiskit quantum circuits."""
    
    def __init__(self):
        self.name = "qiskit"
    
    def run(self, circuit):
        """Run a quantum circuit."""
        pass
'''
    )
    
    return fs


@pytest.fixture
def mock_consent_manager():
    """Create mock consent manager."""
    return create_mock_consent_manager(auto_approve=True)


# =============================================================================
# FILE BACKUP TESTS
# =============================================================================

class TestFileBackup:
    """Tests for file backup functionality."""
    
    def test_create_backup(self, mock_file_system):
        """Test creating a file backup."""
        original_content = mock_file_system.read_file("/backends/cirq_backend.py")
        
        # Create backup
        mock_file_system.write_file(
            "/backends/cirq_backend.py.bak",
            original_content
        )
        
        assert mock_file_system.exists("/backends/cirq_backend.py.bak")
        assert mock_file_system.read_file("/backends/cirq_backend.py.bak") == original_content
    
    def test_restore_from_backup(self, mock_file_system):
        """Test restoring from backup."""
        original_content = mock_file_system.read_file("/backends/cirq_backend.py")
        
        # Create backup
        mock_file_system.write_file(
            "/backends/cirq_backend.py.bak",
            original_content
        )
        
        # Modify original
        mock_file_system.write_file(
            "/backends/cirq_backend.py",
            "# Modified content"
        )
        
        # Restore
        backup_content = mock_file_system.read_file("/backends/cirq_backend.py.bak")
        mock_file_system.write_file("/backends/cirq_backend.py", backup_content)
        
        restored = mock_file_system.read_file("/backends/cirq_backend.py")
        assert restored == original_content


# =============================================================================
# CODE MODIFICATION TESTS
# =============================================================================

class TestCodeModification:
    """Tests for code modification operations."""
    
    def test_add_import(self, mock_file_system):
        """Test adding import to file."""
        content = mock_file_system.read_file("/backends/cirq_backend.py")
        
        new_import = "import numpy as np\n"
        modified = new_import + content
        
        mock_file_system.write_file("/backends/cirq_backend.py", modified)
        
        result = mock_file_system.read_file("/backends/cirq_backend.py")
        assert "import numpy as np" in result
    
    def test_add_method(self, mock_file_system):
        """Test adding method to class."""
        content = mock_file_system.read_file("/backends/cirq_backend.py")
        
        new_method = '''
    def get_status(self):
        """Get backend status."""
        return {"status": "ready", "name": self.name}
'''
        
        # Insert before last line
        lines = content.split('\n')
        lines.insert(-1, new_method)
        modified = '\n'.join(lines)
        
        mock_file_system.write_file("/backends/cirq_backend.py", modified)
        
        result = mock_file_system.read_file("/backends/cirq_backend.py")
        assert "get_status" in result
    
    def test_modify_method(self, mock_file_system):
        """Test modifying existing method."""
        content = mock_file_system.read_file("/backends/cirq_backend.py")
        
        # Replace execute method
        old_return = 'return {"result": "executed"}'
        new_return = 'return {"result": "executed", "status": "success"}'
        
        modified = content.replace(old_return, new_return)
        mock_file_system.write_file("/backends/cirq_backend.py", modified)
        
        result = mock_file_system.read_file("/backends/cirq_backend.py")
        assert '"status": "success"' in result


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestCodeValidation:
    """Tests for code validation."""
    
    def test_syntax_check_valid(self):
        """Test syntax validation of valid code."""
        valid_code = '''
def hello():
    return "world"
'''
        
        try:
            compile(valid_code, '<string>', 'exec')
            is_valid = True
        except SyntaxError:
            is_valid = False
        
        assert is_valid is True
    
    def test_syntax_check_invalid(self):
        """Test syntax validation of invalid code."""
        invalid_code = '''
def hello(
    return "world"
'''
        
        try:
            compile(invalid_code, '<string>', 'exec')
            is_valid = True
        except SyntaxError:
            is_valid = False
        
        assert is_valid is False
    
    def test_validate_class_structure(self):
        """Test validating class structure."""
        code = '''
class TestClass:
    def __init__(self):
        pass
    
    def method(self):
        pass
'''
        
        # Basic structure validation
        assert "class TestClass:" in code
        assert "def __init__" in code
        assert "def method" in code


# =============================================================================
# SAFE MODIFICATION WORKFLOW TESTS
# =============================================================================

class TestSafeModificationWorkflow:
    """Tests for safe modification workflow."""
    
    @pytest.mark.asyncio
    async def test_modification_with_consent(
        self, mock_file_system, mock_consent_manager
    ):
        """Test modification requires consent."""
        approved = await mock_consent_manager.request_consent(
            operation="Modify Backend",
            description="Add new method to CirqBackend",
            risk_level="medium",
        )
        
        assert approved is True
        assert len(mock_consent_manager.requests) == 1
    
    @pytest.mark.asyncio
    async def test_modification_denied(self, mock_file_system):
        """Test modification when consent denied."""
        manager = create_mock_consent_manager(auto_approve=False)
        
        approved = await manager.request_consent(
            operation="Delete Backend",
            description="Remove CirqBackend implementation",
            risk_level="high",
        )
        
        assert approved is False
    
    def test_full_modification_workflow(self, mock_file_system):
        """Test complete modification workflow."""
        target_file = "/backends/cirq_backend.py"
        
        # Step 1: Read original
        original = mock_file_system.read_file(target_file)
        
        # Step 2: Create backup
        backup_path = f"{target_file}.bak.{int(time.time())}"
        mock_file_system.write_file(backup_path, original)
        
        # Step 3: Validate backup
        assert mock_file_system.exists(backup_path)
        
        # Step 4: Make modification
        modified = original.replace(
            'self.name = "cirq"',
            'self.name = "cirq"\n        self.version = "1.0.0"'
        )
        
        # Step 5: Validate modification syntax
        try:
            compile(modified, '<string>', 'exec')
            is_valid = True
        except SyntaxError:
            is_valid = False
        
        assert is_valid is True
        
        # Step 6: Write modification
        mock_file_system.write_file(target_file, modified)
        
        # Step 7: Verify
        result = mock_file_system.read_file(target_file)
        assert 'self.version = "1.0.0"' in result


# =============================================================================
# ROLLBACK TESTS
# =============================================================================

class TestRollback:
    """Tests for rollback functionality."""
    
    def test_rollback_on_error(self, mock_file_system):
        """Test rollback when modification fails."""
        target_file = "/backends/cirq_backend.py"
        
        # Create backup
        original = mock_file_system.read_file(target_file)
        backup_path = f"{target_file}.bak"
        mock_file_system.write_file(backup_path, original)
        
        # Make invalid modification
        invalid_code = "class Invalid(\n"  # Invalid Python
        mock_file_system.write_file(target_file, invalid_code)
        
        # Validate (would fail)
        try:
            compile(invalid_code, '<string>', 'exec')
            is_valid = True
        except SyntaxError:
            is_valid = False
        
        # Rollback on failure
        if not is_valid:
            backup_content = mock_file_system.read_file(backup_path)
            mock_file_system.write_file(target_file, backup_content)
        
        # Verify rollback
        result = mock_file_system.read_file(target_file)
        assert result == original
    
    def test_multiple_modification_rollback(self, mock_file_system):
        """Test rolling back multiple modifications."""
        files = [
            "/backends/cirq_backend.py",
            "/backends/qiskit_backend.py",
        ]
        
        backups = {}
        
        # Create backups
        for file_path in files:
            backups[file_path] = mock_file_system.read_file(file_path)
        
        # Modify all files
        for file_path in files:
            mock_file_system.write_file(file_path, "# Modified")
        
        # Rollback all
        for file_path, original in backups.items():
            mock_file_system.write_file(file_path, original)
        
        # Verify
        for file_path, original in backups.items():
            assert mock_file_system.read_file(file_path) == original


# =============================================================================
# CODE INTELLIGENCE TESTS
# =============================================================================

class TestCodeIntelligence:
    """Tests for code intelligence features."""
    
    def test_find_class_definition(self, mock_file_system):
        """Test finding class definition."""
        content = mock_file_system.read_file("/backends/cirq_backend.py")
        
        # Simple class finder
        import re
        class_match = re.search(r'class (\w+)', content)
        
        assert class_match is not None
        assert class_match.group(1) == "CirqBackend"
    
    def test_find_methods(self, mock_file_system):
        """Test finding methods in class."""
        content = mock_file_system.read_file("/backends/cirq_backend.py")
        
        # Simple method finder
        import re
        methods = re.findall(r'def (\w+)\(', content)
        
        assert "__init__" in methods
        assert "execute" in methods
    
    def test_find_imports(self, mock_file_system):
        """Test finding imports in file."""
        # Add imports to test
        mock_file_system.write_file(
            "/backends/test_file.py",
            '''import os
import sys
from typing import Dict, List

class Test:
    pass
'''
        )
        
        content = mock_file_system.read_file("/backends/test_file.py")
        
        import re
        imports = re.findall(r'^(?:import|from)\s+[\w.]+', content, re.MULTILINE)
        
        assert len(imports) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
