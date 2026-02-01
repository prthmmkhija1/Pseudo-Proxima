"""Build Artifact Manager for Proxima Agent.

Phase 4: Backend Building & Compilation System

Manages build artifacts including:
- Storing compiled backends
- Version management (keeping last N builds)
- Build manifests with metadata
- Rollback support
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

from proxima.utils.logging import get_logger

logger = get_logger("agent.build_artifact_manager")


class ArtifactType(Enum):
    """Types of build artifacts."""
    BINARY = "binary"           # Compiled binaries
    LIBRARY = "library"         # Shared/static libraries
    CONFIG = "config"           # Configuration files
    LOG = "log"                 # Build logs
    METADATA = "metadata"       # Build metadata
    SOURCE = "source"           # Source code backup


@dataclass
class ArtifactInfo:
    """Information about a single artifact."""
    name: str
    path: str
    artifact_type: ArtifactType
    size_bytes: int
    checksum: str
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "artifact_type": self.artifact_type.value,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactInfo":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            path=data["path"],
            artifact_type=ArtifactType(data["artifact_type"]),
            size_bytes=data["size_bytes"],
            checksum=data["checksum"],
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class BuildManifest:
    """Build manifest containing all artifact info."""
    build_id: str
    backend_name: str
    version: str
    timestamp: datetime
    duration_seconds: float
    success: bool
    artifacts: List[ArtifactInfo] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    build_config: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "build_id": self.build_id,
            "backend_name": self.backend_name,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "environment": self.environment,
            "build_config": self.build_config,
            "error_message": self.error_message,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuildManifest":
        """Create from dictionary."""
        return cls(
            build_id=data["build_id"],
            backend_name=data["backend_name"],
            version=data["version"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            duration_seconds=data["duration_seconds"],
            success=data["success"],
            artifacts=[ArtifactInfo.from_dict(a) for a in data.get("artifacts", [])],
            environment=data.get("environment", {}),
            build_config=data.get("build_config", {}),
            error_message=data.get("error_message"),
        )
    
    @property
    def total_size_bytes(self) -> int:
        """Get total size of all artifacts."""
        return sum(a.size_bytes for a in self.artifacts)


@dataclass
class BuildVersion:
    """A specific build version."""
    build_id: str
    timestamp: datetime
    path: Path
    manifest: BuildManifest
    
    @property
    def age_days(self) -> float:
        """Get age in days."""
        delta = datetime.now() - self.timestamp
        return delta.total_seconds() / 86400


class BuildArtifactManager:
    """Manage build artifacts for backends.
    
    Handles:
    - Artifact storage in build/{backend}/{timestamp}/
    - Build manifests with checksums
    - Version management (keep last N builds)
    - Cleanup of old artifacts
    - Rollback support
    
    Example:
        >>> manager = BuildArtifactManager()
        >>> 
        >>> # Start a new build
        >>> build_dir = manager.create_build_directory("qsim")
        >>> 
        >>> # After build completes, register artifacts
        >>> manager.register_artifact(build_dir, "qsim.so", ArtifactType.LIBRARY)
        >>> 
        >>> # Save the manifest
        >>> manager.save_manifest(build_dir, manifest)
        >>> 
        >>> # Get previous builds
        >>> versions = manager.get_build_versions("qsim")
    """
    
    DEFAULT_ARTIFACTS_DIR = "build"
    DEFAULT_MAX_VERSIONS = 3
    MANIFEST_FILENAME = "manifest.json"
    
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        max_versions: int = DEFAULT_MAX_VERSIONS,
    ):
        """Initialize the artifact manager.
        
        Args:
            base_dir: Base directory for artifacts (default: ./build)
            max_versions: Maximum number of versions to keep per backend
        """
        self.base_dir = Path(base_dir) if base_dir else Path(self.DEFAULT_ARTIFACTS_DIR)
        self.max_versions = max_versions
        
        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"BuildArtifactManager initialized at {self.base_dir}")
    
    def create_build_directory(
        self,
        backend_name: str,
        timestamp: Optional[datetime] = None,
    ) -> Path:
        """Create a new build directory for a backend.
        
        Args:
            backend_name: Name of the backend
            timestamp: Build timestamp (default: now)
            
        Returns:
            Path to the new build directory
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Format: build/{backend}/{YYYYMMDD_HHMMSS}/
        time_str = timestamp.strftime("%Y%m%d_%H%M%S")
        build_dir = self.base_dir / backend_name / time_str
        
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (build_dir / "bin").mkdir(exist_ok=True)
        (build_dir / "lib").mkdir(exist_ok=True)
        (build_dir / "logs").mkdir(exist_ok=True)
        
        logger.info(f"Created build directory: {build_dir}")
        return build_dir
    
    def compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex-encoded checksum
        """
        sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def register_artifact(
        self,
        build_dir: Path,
        file_path: Path,
        artifact_type: ArtifactType,
    ) -> ArtifactInfo:
        """Register an artifact from a build.
        
        Args:
            build_dir: Build directory
            file_path: Path to artifact file
            artifact_type: Type of artifact
            
        Returns:
            ArtifactInfo for the registered artifact
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Artifact not found: {file_path}")
        
        # Get relative path within build dir
        try:
            rel_path = file_path.relative_to(build_dir)
        except ValueError:
            # File is outside build dir, copy it
            dest_dir = build_dir / "artifacts"
            dest_dir.mkdir(exist_ok=True)
            dest_path = dest_dir / file_path.name
            shutil.copy2(file_path, dest_path)
            rel_path = dest_path.relative_to(build_dir)
            file_path = dest_path
        
        artifact = ArtifactInfo(
            name=file_path.name,
            path=str(rel_path),
            artifact_type=artifact_type,
            size_bytes=file_path.stat().st_size,
            checksum=self.compute_checksum(file_path),
            created_at=datetime.now(),
        )
        
        logger.debug(f"Registered artifact: {artifact.name} ({artifact_type.value})")
        return artifact
    
    def save_manifest(
        self,
        build_dir: Path,
        manifest: BuildManifest,
    ) -> Path:
        """Save a build manifest to disk.
        
        Args:
            build_dir: Build directory
            manifest: Build manifest to save
            
        Returns:
            Path to saved manifest file
        """
        manifest_path = build_dir / self.MANIFEST_FILENAME
        
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        
        logger.info(f"Saved manifest: {manifest_path}")
        return manifest_path
    
    def load_manifest(self, build_dir: Path) -> Optional[BuildManifest]:
        """Load a build manifest from disk.
        
        Args:
            build_dir: Build directory
            
        Returns:
            BuildManifest or None if not found
        """
        manifest_path = build_dir / self.MANIFEST_FILENAME
        
        if not manifest_path.exists():
            return None
        
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return BuildManifest.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load manifest: {e}")
            return None
    
    def get_build_versions(
        self,
        backend_name: str,
        successful_only: bool = False,
    ) -> List[BuildVersion]:
        """Get all build versions for a backend.
        
        Args:
            backend_name: Name of the backend
            successful_only: Only return successful builds
            
        Returns:
            List of BuildVersion, sorted newest first
        """
        backend_dir = self.base_dir / backend_name
        
        if not backend_dir.exists():
            return []
        
        versions = []
        
        for build_dir in backend_dir.iterdir():
            if not build_dir.is_dir():
                continue
            
            manifest = self.load_manifest(build_dir)
            if manifest is None:
                continue
            
            if successful_only and not manifest.success:
                continue
            
            version = BuildVersion(
                build_id=manifest.build_id,
                timestamp=manifest.timestamp,
                path=build_dir,
                manifest=manifest,
            )
            versions.append(version)
        
        # Sort newest first
        versions.sort(key=lambda v: v.timestamp, reverse=True)
        
        return versions
    
    def get_latest_build(
        self,
        backend_name: str,
        successful_only: bool = True,
    ) -> Optional[BuildVersion]:
        """Get the latest build for a backend.
        
        Args:
            backend_name: Name of the backend
            successful_only: Only return successful builds
            
        Returns:
            Latest BuildVersion or None
        """
        versions = self.get_build_versions(backend_name, successful_only)
        return versions[0] if versions else None
    
    def cleanup_old_builds(
        self,
        backend_name: str,
        keep_count: Optional[int] = None,
    ) -> int:
        """Remove old builds, keeping only the most recent N.
        
        Args:
            backend_name: Name of the backend
            keep_count: Number of builds to keep (default: max_versions)
            
        Returns:
            Number of builds removed
        """
        if keep_count is None:
            keep_count = self.max_versions
        
        versions = self.get_build_versions(backend_name)
        
        if len(versions) <= keep_count:
            return 0
        
        # Remove old versions
        removed = 0
        for version in versions[keep_count:]:
            try:
                shutil.rmtree(version.path)
                logger.info(f"Removed old build: {version.path}")
                removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove {version.path}: {e}")
        
        return removed
    
    def cleanup_all_old_builds(self) -> Dict[str, int]:
        """Clean up old builds for all backends.
        
        Returns:
            Dict mapping backend name to number of builds removed
        """
        results = {}
        
        for backend_dir in self.base_dir.iterdir():
            if backend_dir.is_dir():
                removed = self.cleanup_old_builds(backend_dir.name)
                if removed > 0:
                    results[backend_dir.name] = removed
        
        return results
    
    def rollback(
        self,
        backend_name: str,
        target_build_id: Optional[str] = None,
    ) -> Optional[BuildVersion]:
        """Rollback to a previous build.
        
        Args:
            backend_name: Name of the backend
            target_build_id: Specific build ID, or None for previous successful
            
        Returns:
            The build version rolled back to, or None
        """
        versions = self.get_build_versions(backend_name, successful_only=True)
        
        if len(versions) < 2:
            logger.warning(f"No previous build to rollback to for {backend_name}")
            return None
        
        if target_build_id:
            # Find specific build
            target = next(
                (v for v in versions if v.build_id == target_build_id),
                None
            )
            if target is None:
                logger.warning(f"Build {target_build_id} not found")
                return None
        else:
            # Use previous (second newest)
            target = versions[1]
        
        logger.info(f"Rolling back {backend_name} to build {target.build_id}")
        
        # Mark as current (implementation depends on deployment mechanism)
        # For now, we just return the target version
        return target
    
    def get_total_size(self, backend_name: Optional[str] = None) -> int:
        """Get total size of artifacts.
        
        Args:
            backend_name: Specific backend, or None for all
            
        Returns:
            Total size in bytes
        """
        if backend_name:
            target_dir = self.base_dir / backend_name
        else:
            target_dir = self.base_dir
        
        if not target_dir.exists():
            return 0
        
        total = 0
        for path in target_dir.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        
        return total
    
    def verify_build(self, build_dir: Path) -> Dict[str, Any]:
        """Verify build integrity using manifest checksums.
        
        Args:
            build_dir: Build directory to verify
            
        Returns:
            Verification results
        """
        manifest = self.load_manifest(build_dir)
        
        if manifest is None:
            return {
                "success": False,
                "error": "Manifest not found",
                "verified": 0,
                "failed": 0,
            }
        
        results = {
            "success": True,
            "verified": 0,
            "failed": 0,
            "missing": [],
            "checksum_mismatch": [],
        }
        
        for artifact in manifest.artifacts:
            artifact_path = build_dir / artifact.path
            
            if not artifact_path.exists():
                results["missing"].append(artifact.name)
                results["failed"] += 1
                continue
            
            checksum = self.compute_checksum(artifact_path)
            
            if checksum != artifact.checksum:
                results["checksum_mismatch"].append({
                    "name": artifact.name,
                    "expected": artifact.checksum,
                    "actual": checksum,
                })
                results["failed"] += 1
            else:
                results["verified"] += 1
        
        results["success"] = results["failed"] == 0
        
        return results
    
    def export_build(
        self,
        build_dir: Path,
        output_path: Path,
    ) -> Path:
        """Export a build as a compressed archive.
        
        Args:
            build_dir: Build directory to export
            output_path: Output archive path (without extension)
            
        Returns:
            Path to created archive
        """
        output_path = Path(output_path)
        
        # Create tar.gz archive
        archive_path = shutil.make_archive(
            str(output_path),
            "gztar",
            root_dir=build_dir.parent,
            base_dir=build_dir.name,
        )
        
        logger.info(f"Exported build to: {archive_path}")
        return Path(archive_path)


def generate_build_id(backend_name: str, timestamp: datetime) -> str:
    """Generate a unique build ID.
    
    Args:
        backend_name: Name of the backend
        timestamp: Build timestamp
        
    Returns:
        Unique build ID
    """
    time_str = timestamp.strftime("%Y%m%d%H%M%S")
    hash_input = f"{backend_name}-{time_str}".encode()
    short_hash = hashlib.sha256(hash_input).hexdigest()[:8]
    return f"{backend_name}-{time_str}-{short_hash}"
