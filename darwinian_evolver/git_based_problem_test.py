"""Tests for GitBasedOrganism class."""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from darwinian_evolver.git_based_problem import GitBasedOrganism
from darwinian_evolver.learning_log import LearningLogEntry  # noqa: F401
from darwinian_evolver.problem import EvaluationFailureCase  # noqa: F401

# Rebuild the model to resolve forward references
GitBasedOrganism.model_rebuild()


@pytest.fixture
def test_repo():
    """Create a temporary git repository for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir) / "test_repo"
        repo_path.mkdir()

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

        # Create initial files
        (repo_path / "file1.txt").write_text("Initial content of file1\n")
        (repo_path / "file2.txt").write_text("Initial content of file2\n")
        (repo_path / "subdir").mkdir()
        (repo_path / "subdir" / "file3.txt").write_text("Initial content of file3\n")

        # Make initial commit
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

        # Get the commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        initial_hash = result.stdout.strip()

        yield {
            "path": str(repo_path),
            "initial_hash": initial_hash,
        }


# ============================================================================
# make_initial_organism_from_repo Tests
# ============================================================================


def test_make_initial_organism_from_repo(test_repo) -> None:
    """Test creating initial organism from a git repository."""
    organism = GitBasedOrganism.make_initial_organism_from_repo(
        repo_root=test_repo["path"],
        files_to_capture=["file1.txt", "file2.txt"],
    )

    assert organism.repo_root == test_repo["path"]
    assert organism.git_hash == test_repo["initial_hash"]
    assert "file1.txt" in organism.file_contents
    assert "file2.txt" in organism.file_contents
    assert organism.file_contents["file1.txt"] == "Initial content of file1\n"
    assert organism.file_contents["file2.txt"] == "Initial content of file2\n"
    assert organism.parent is None


def test_make_initial_organism_with_explicit_hash(test_repo) -> None:
    """Test creating organism with explicitly provided git hash."""
    organism = GitBasedOrganism.make_initial_organism_from_repo(
        repo_root=test_repo["path"],
        files_to_capture=["file1.txt"],
        git_hash=test_repo["initial_hash"],
    )

    assert organism.git_hash == test_repo["initial_hash"]
    assert organism.file_contents["file1.txt"] == "Initial content of file1\n"


def test_make_initial_organism_with_subdirectory_file(test_repo) -> None:
    """Test capturing files in subdirectories."""
    organism = GitBasedOrganism.make_initial_organism_from_repo(
        repo_root=test_repo["path"],
        files_to_capture=["subdir/file3.txt"],
    )

    assert "subdir/file3.txt" in organism.file_contents
    assert organism.file_contents["subdir/file3.txt"] == "Initial content of file3\n"


def test_make_initial_organism_with_multiple_files(test_repo) -> None:
    """Test capturing multiple files at once."""
    organism = GitBasedOrganism.make_initial_organism_from_repo(
        repo_root=test_repo["path"],
        files_to_capture=["file1.txt", "file2.txt", "subdir/file3.txt"],
    )

    assert len(organism.file_contents) == 3
    assert "file1.txt" in organism.file_contents
    assert "file2.txt" in organism.file_contents
    assert "subdir/file3.txt" in organism.file_contents


# ============================================================================
# diff_from_parent Tests
# ============================================================================


def test_diff_from_parent_no_parent() -> None:
    """Test diff_from_parent returns empty string when there's no parent."""
    organism = GitBasedOrganism(
        repo_root="/fake/path",
        git_hash="abc123",
        file_contents={"file.txt": "content"},
    )

    diff = organism.diff_from_parent

    assert diff == ""


def test_diff_from_parent_no_changes(test_repo) -> None:
    """Test diff when files are identical to parent."""
    parent = GitBasedOrganism.make_initial_organism_from_repo(
        repo_root=test_repo["path"],
        files_to_capture=["file1.txt", "file2.txt"],
    )

    child = GitBasedOrganism(
        repo_root=test_repo["path"],
        git_hash=test_repo["initial_hash"],
        file_contents=parent.file_contents.copy(),
        parent=parent,
    )

    diff = child.diff_from_parent

    assert diff == ""


def test_diff_from_parent_with_changes(test_repo) -> None:
    """Test diff generation when files are modified."""
    parent = GitBasedOrganism.make_initial_organism_from_repo(
        repo_root=test_repo["path"],
        files_to_capture=["file1.txt", "file2.txt"],
    )

    # Create child with modified content
    modified_contents = parent.file_contents.copy()
    modified_contents["file1.txt"] = "Modified content of file1\n"

    child = GitBasedOrganism(
        repo_root=test_repo["path"],
        git_hash=test_repo["initial_hash"],
        file_contents=modified_contents,
        parent=parent,
    )

    diff = child.diff_from_parent

    # Verify diff contains expected elements
    assert "file1.txt" in diff
    assert "Initial content of file1" in diff
    assert "Modified content of file1" in diff
    assert "---" in diff or "+++" in diff  # Unified diff markers


def test_diff_from_parent_multiple_files_changed(test_repo) -> None:
    """Test diff when multiple files are modified."""
    parent = GitBasedOrganism.make_initial_organism_from_repo(
        repo_root=test_repo["path"],
        files_to_capture=["file1.txt", "file2.txt"],
    )

    # Modify both files
    modified_contents = parent.file_contents.copy()
    modified_contents["file1.txt"] = "New content for file1\n"
    modified_contents["file2.txt"] = "New content for file2\n"

    child = GitBasedOrganism(
        repo_root=test_repo["path"],
        git_hash=test_repo["initial_hash"],
        file_contents=modified_contents,
        parent=parent,
    )

    diff = child.diff_from_parent

    # Both files should appear in diff
    assert "file1.txt" in diff
    assert "file2.txt" in diff


def test_diff_from_parent_only_one_file_changed(test_repo) -> None:
    """Test diff when only one file out of many is modified."""
    parent = GitBasedOrganism.make_initial_organism_from_repo(
        repo_root=test_repo["path"],
        files_to_capture=["file1.txt", "file2.txt"],
    )

    # Only modify file1
    modified_contents = parent.file_contents.copy()
    modified_contents["file1.txt"] = "Changed\n"

    child = GitBasedOrganism(
        repo_root=test_repo["path"],
        git_hash=test_repo["initial_hash"],
        file_contents=modified_contents,
        parent=parent,
    )

    diff = child.diff_from_parent

    # Only file1 should appear in diff
    assert "file1.txt" in diff
    # file2 should not appear since it's unchanged
    assert diff.count("file2.txt") == 0 or "file2.txt" not in diff


def test_diff_from_parent_multiline_changes(test_repo) -> None:
    """Test diff with multiline content changes."""
    parent = GitBasedOrganism.make_initial_organism_from_repo(
        repo_root=test_repo["path"],
        files_to_capture=["file1.txt"],
    )

    # Create multiline content
    modified_contents = parent.file_contents.copy()
    modified_contents["file1.txt"] = "Line 1\nLine 2\nLine 3\nLine 4\n"

    child = GitBasedOrganism(
        repo_root=test_repo["path"],
        git_hash=test_repo["initial_hash"],
        file_contents=modified_contents,
        parent=parent,
    )

    diff = child.diff_from_parent

    assert len(diff) > 0
    assert [text in diff for text in ["Line 1", "Line 2", "Line 3", "Line 4"]]


# ============================================================================
# build_repo Tests
# ============================================================================


def test_build_repo_creates_temp_directory(test_repo) -> None:
    """Test that build_repo creates a temporary directory."""
    organism = GitBasedOrganism.make_initial_organism_from_repo(
        repo_root=test_repo["path"],
        files_to_capture=["file1.txt"],
    )

    with organism.build_repo() as temp_dir:
        # Verify the directory exists
        assert os.path.exists(temp_dir)
        assert os.path.isdir(temp_dir)

    # Verify the directory is cleaned up after exiting context
    assert not os.path.exists(temp_dir)


def test_build_repo_clones_repository(test_repo) -> None:
    """Test that build_repo clones the repository."""
    organism = GitBasedOrganism.make_initial_organism_from_repo(
        repo_root=test_repo["path"],
        files_to_capture=["file1.txt"],
    )

    with organism.build_repo() as temp_dir:
        # Verify it's a git repository
        assert os.path.exists(Path(temp_dir) / ".git")

        # Verify files exist
        assert os.path.exists(Path(temp_dir) / "file1.txt")


def test_build_repo_checks_out_correct_hash(test_repo) -> None:
    """Test that build_repo checks out the correct git hash."""
    organism = GitBasedOrganism.make_initial_organism_from_repo(
        repo_root=test_repo["path"],
        files_to_capture=["file1.txt"],
    )

    with organism.build_repo() as temp_dir:
        # Get the current HEAD in the temp repo
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=temp_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        current_hash = result.stdout.strip()

        assert current_hash == organism.git_hash


def test_build_repo_writes_modified_files(test_repo) -> None:
    """Test that build_repo writes modified file contents."""
    parent = GitBasedOrganism.make_initial_organism_from_repo(
        repo_root=test_repo["path"],
        files_to_capture=["file1.txt", "file2.txt"],
    )

    # Create organism with modified content
    modified_contents = parent.file_contents.copy()
    modified_contents["file1.txt"] = "Modified content\n"

    organism = GitBasedOrganism(
        repo_root=test_repo["path"],
        git_hash=test_repo["initial_hash"],
        file_contents=modified_contents,
        parent=parent,
    )

    with organism.build_repo() as temp_dir:
        # Verify file1 has the modified content
        with open(Path(temp_dir) / "file1.txt", "r") as f:
            content = f.read()
        assert content == "Modified content\n"

        # Verify file2 has the original content (unchanged)
        with open(Path(temp_dir) / "file2.txt", "r") as f:
            content = f.read()
        assert content == "Initial content of file2\n"


def test_build_repo_writes_all_tracked_files(test_repo) -> None:
    """Test that all files in file_contents are written."""
    parent = GitBasedOrganism.make_initial_organism_from_repo(
        repo_root=test_repo["path"],
        files_to_capture=["file1.txt", "file2.txt", "subdir/file3.txt"],
    )

    # Modify all files
    modified_contents = {
        "file1.txt": "Modified 1\n",
        "file2.txt": "Modified 2\n",
        "subdir/file3.txt": "Modified 3\n",
    }

    organism = GitBasedOrganism(
        repo_root=test_repo["path"],
        git_hash=test_repo["initial_hash"],
        file_contents=modified_contents,
        parent=parent,
    )

    with organism.build_repo() as temp_dir:
        # Verify all files have new content
        with open(Path(temp_dir) / "file1.txt", "r") as f:
            assert f.read() == "Modified 1\n"
        with open(Path(temp_dir) / "file2.txt", "r") as f:
            assert f.read() == "Modified 2\n"
        with open(Path(temp_dir) / "subdir" / "file3.txt", "r") as f:
            assert f.read() == "Modified 3\n"


def test_build_repo_preserves_other_files(test_repo) -> None:
    """Test that files not in file_contents are preserved from git."""
    organism = GitBasedOrganism.make_initial_organism_from_repo(
        repo_root=test_repo["path"],
        files_to_capture=["file1.txt"],  # Only tracking file1
    )

    with organism.build_repo() as temp_dir:
        # file2.txt should still exist from the git checkout
        assert (Path(temp_dir) / "file2.txt").exists()

        # And it should have the original content from git
        with open(Path(temp_dir) / "file2.txt", "r") as f:
            content = f.read()
        assert content == "Initial content of file2\n"


def test_build_repo_multiple_uses(test_repo) -> None:
    """Test that build_repo can be called multiple times."""
    organism = GitBasedOrganism.make_initial_organism_from_repo(
        repo_root=test_repo["path"],
        files_to_capture=["file1.txt"],
    )

    # First use
    with organism.build_repo() as temp_dir1:
        path1 = temp_dir1
        assert os.path.exists(temp_dir1)

    # Directory should be cleaned up
    assert not os.path.exists(path1)

    # Second use should work fine
    with organism.build_repo() as temp_dir2:
        path2 = temp_dir2
        assert os.path.exists(temp_dir2)

    # Second directory should also be cleaned up
    assert not os.path.exists(path2)


# ============================================================================
# _get_file_content Tests
# ============================================================================


def test_get_file_content(test_repo) -> None:
    """Test retrieving file content from git."""
    # _get_file_content runs git commands in current directory
    original_cwd = os.getcwd()
    try:
        os.chdir(test_repo["path"])
        content = GitBasedOrganism._get_file_content(
            test_repo["initial_hash"],
            "file1.txt",
        )
        assert content == "Initial content of file1\n"
    finally:
        os.chdir(original_cwd)


def test_get_file_content_subdirectory(test_repo) -> None:
    """Test retrieving file content from subdirectory."""
    original_cwd = os.getcwd()
    try:
        os.chdir(test_repo["path"])
        content = GitBasedOrganism._get_file_content(
            test_repo["initial_hash"],
            "subdir/file3.txt",
        )
        assert content == "Initial content of file3\n"
    finally:
        os.chdir(original_cwd)


def test_get_file_content_multiple_files(test_repo) -> None:
    """Test retrieving content from multiple files."""
    original_cwd = os.getcwd()
    try:
        os.chdir(test_repo["path"])
        content1 = GitBasedOrganism._get_file_content(
            test_repo["initial_hash"],
            "file1.txt",
        )
        content2 = GitBasedOrganism._get_file_content(
            test_repo["initial_hash"],
            "file2.txt",
        )

        assert content1 == "Initial content of file1\n"
        assert content2 == "Initial content of file2\n"
    finally:
        os.chdir(original_cwd)


def test_get_file_content_nonexistent_file(test_repo) -> None:
    """Test that getting nonexistent file raises an error."""
    with pytest.raises(subprocess.CalledProcessError):
        GitBasedOrganism._get_file_content(
            test_repo["initial_hash"],
            "nonexistent.txt",
        )


def test_get_file_content_invalid_hash(test_repo) -> None:
    """Test that invalid git hash raises an error."""
    with pytest.raises(subprocess.CalledProcessError):
        GitBasedOrganism._get_file_content(
            "invalid_hash_12345",
            "file1.txt",
        )
