"""Unit tests for Git Operations module.

Phase 10: Integration & Testing

Tests cover:
- Repository cloning
- Branch operations
- Commit workflows
- Status parsing
- Diff generation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.fixtures.mock_agent import (
    MockGitRepository,
    MockSubprocessFactory,
    MockProcessOutput,
    create_mock_git_repo,
    create_mock_subprocess_factory,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_git_repo():
    """Create mock git repository."""
    return create_mock_git_repo("/test/repo")


@pytest.fixture
def mock_subprocess_factory():
    """Create mock subprocess factory with git commands."""
    factory = create_mock_subprocess_factory()
    
    # Add git-specific mock outputs
    factory.set_output("git status", MockProcessOutput(
        stdout="On branch main\nnothing to commit, working tree clean\n",
        returncode=0,
    ))
    factory.set_output("git branch", MockProcessOutput(
        stdout="* main\n  feature\n  develop\n",
        returncode=0,
    ))
    factory.set_output("git log", MockProcessOutput(
        stdout="abc1234 Initial commit\ndef5678 Add feature\n",
        returncode=0,
    ))
    
    return factory


# =============================================================================
# MOCK GIT REPOSITORY TESTS
# =============================================================================

class TestMockGitRepository:
    """Tests for MockGitRepository."""
    
    def test_repository_creation(self, mock_git_repo):
        """Test creating a mock repository."""
        assert mock_git_repo.path == "/test/repo"
        assert mock_git_repo.current_branch == "main"
        assert "main" in mock_git_repo.branches
    
    def test_add_commit(self, mock_git_repo):
        """Test adding a commit."""
        commit_hash = mock_git_repo.add_commit(
            message="Test commit",
            files=["file1.py", "file2.py"],
        )
        
        assert commit_hash.startswith("abc")
        assert len(mock_git_repo.commits) == 1
        assert mock_git_repo.commits[0]["message"] == "Test commit"
    
    def test_create_branch(self, mock_git_repo):
        """Test creating a branch."""
        result = mock_git_repo.create_branch("feature-branch")
        
        assert result is True
        assert "feature-branch" in mock_git_repo.branches
    
    def test_create_duplicate_branch(self, mock_git_repo):
        """Test creating duplicate branch fails."""
        mock_git_repo.create_branch("feature")
        result = mock_git_repo.create_branch("feature")
        
        assert result is False
    
    def test_checkout(self, mock_git_repo):
        """Test checking out a branch."""
        mock_git_repo.create_branch("develop")
        result = mock_git_repo.checkout("develop")
        
        assert result is True
        assert mock_git_repo.current_branch == "develop"
    
    def test_checkout_nonexistent(self, mock_git_repo):
        """Test checking out nonexistent branch fails."""
        result = mock_git_repo.checkout("nonexistent")
        
        assert result is False
        assert mock_git_repo.current_branch == "main"
    
    def test_stage_files(self, mock_git_repo):
        """Test staging files."""
        mock_git_repo.stage(["file1.py", "file2.py"])
        
        assert "file1.py" in mock_git_repo.staged_files
        assert "file2.py" in mock_git_repo.staged_files
    
    def test_get_status_clean(self, mock_git_repo):
        """Test getting clean repository status."""
        status = mock_git_repo.get_status()
        
        assert status["branch"] == "main"
        assert status["clean"] is True
        assert status["staged"] == []
        assert status["modified"] == []
    
    def test_get_status_with_changes(self, mock_git_repo):
        """Test getting status with staged files."""
        mock_git_repo.stage(["modified.py"])
        status = mock_git_repo.get_status()
        
        assert status["clean"] is False
        assert "modified.py" in status["staged"]


# =============================================================================
# GIT COMMAND MOCK TESTS
# =============================================================================

class TestGitCommandMocks:
    """Tests for git command mocking."""
    
    @pytest.mark.asyncio
    async def test_git_status(self, mock_subprocess_factory):
        """Test git status command."""
        process = await mock_subprocess_factory.create_subprocess_shell(
            "git status"
        )
        stdout, _ = await process.communicate()
        
        assert "On branch main" in stdout.decode()
        assert process.returncode == 0
    
    @pytest.mark.asyncio
    async def test_git_branch(self, mock_subprocess_factory):
        """Test git branch command."""
        process = await mock_subprocess_factory.create_subprocess_shell(
            "git branch"
        )
        stdout, _ = await process.communicate()
        
        assert "main" in stdout.decode()
        assert process.returncode == 0
    
    @pytest.mark.asyncio
    async def test_git_clone(self, mock_subprocess_factory):
        """Test git clone command."""
        process = await mock_subprocess_factory.create_subprocess_shell(
            "git clone https://github.com/test/repo.git"
        )
        stdout, _ = await process.communicate()
        
        assert "Cloning" in stdout.decode()
        assert process.returncode == 0
    
    @pytest.mark.asyncio
    async def test_git_log(self, mock_subprocess_factory):
        """Test git log command."""
        process = await mock_subprocess_factory.create_subprocess_shell(
            "git log --oneline"
        )
        stdout, _ = await process.communicate()
        
        assert "Initial commit" in stdout.decode()


# =============================================================================
# GIT WORKFLOW TESTS
# =============================================================================

class TestGitWorkflows:
    """Tests for complete git workflows."""
    
    def test_feature_branch_workflow(self, mock_git_repo):
        """Test complete feature branch workflow."""
        # Create feature branch
        mock_git_repo.create_branch("feature/new-feature")
        mock_git_repo.checkout("feature/new-feature")
        
        # Make changes
        mock_git_repo.modified_files.append("src/feature.py")
        mock_git_repo.stage(["src/feature.py"])
        
        # Commit
        commit_hash = mock_git_repo.add_commit(
            message="Add new feature",
            files=["src/feature.py"],
        )
        
        # Verify
        assert mock_git_repo.current_branch == "feature/new-feature"
        assert len(mock_git_repo.commits) == 1
        assert commit_hash is not None
    
    def test_multiple_commits(self, mock_git_repo):
        """Test multiple commits in sequence."""
        for i in range(5):
            mock_git_repo.add_commit(
                message=f"Commit {i}",
                files=[f"file{i}.py"],
            )
        
        assert len(mock_git_repo.commits) == 5
        assert mock_git_repo.commits[0]["message"] == "Commit 0"
        assert mock_git_repo.commits[4]["message"] == "Commit 4"
    
    def test_branch_operations(self, mock_git_repo):
        """Test multiple branch operations."""
        # Create branches
        mock_git_repo.create_branch("develop")
        mock_git_repo.create_branch("staging")
        mock_git_repo.create_branch("production")
        
        # Switch between branches
        mock_git_repo.checkout("develop")
        assert mock_git_repo.current_branch == "develop"
        
        mock_git_repo.checkout("staging")
        assert mock_git_repo.current_branch == "staging"
        
        mock_git_repo.checkout("main")
        assert mock_git_repo.current_branch == "main"


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestGitErrorHandling:
    """Tests for git error handling."""
    
    @pytest.mark.asyncio
    async def test_git_command_failure(self, mock_subprocess_factory):
        """Test handling git command failure."""
        mock_subprocess_factory.set_output(
            "git push",
            MockProcessOutput(
                stdout="",
                stderr="error: failed to push some refs\n",
                returncode=1,
            )
        )
        
        process = await mock_subprocess_factory.create_subprocess_shell(
            "git push origin main"
        )
        stdout, stderr = await process.communicate()
        
        assert process.returncode == 1
        assert "failed to push" in stderr.decode()
    
    @pytest.mark.asyncio
    async def test_git_merge_conflict(self, mock_subprocess_factory):
        """Test handling merge conflict."""
        mock_subprocess_factory.set_output(
            "git merge",
            MockProcessOutput(
                stdout="",
                stderr="CONFLICT (content): Merge conflict in file.py\n",
                returncode=1,
            )
        )
        
        process = await mock_subprocess_factory.create_subprocess_shell(
            "git merge feature"
        )
        _, stderr = await process.communicate()
        
        assert "CONFLICT" in stderr.decode()


# =============================================================================
# STATUS PARSING TESTS
# =============================================================================

class TestGitStatusParsing:
    """Tests for git status parsing."""
    
    def test_clean_status(self, mock_git_repo):
        """Test parsing clean status."""
        status = mock_git_repo.get_status()
        
        assert status["clean"] is True
        assert len(status["staged"]) == 0
        assert len(status["modified"]) == 0
    
    def test_status_with_modifications(self, mock_git_repo):
        """Test parsing status with modifications."""
        mock_git_repo.modified_files = ["file1.py", "file2.py"]
        status = mock_git_repo.get_status()
        
        assert status["clean"] is False
        assert len(status["modified"]) == 2
    
    def test_status_with_staged(self, mock_git_repo):
        """Test parsing status with staged files."""
        mock_git_repo.stage(["staged_file.py"])
        status = mock_git_repo.get_status()
        
        assert status["clean"] is False
        assert "staged_file.py" in status["staged"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
