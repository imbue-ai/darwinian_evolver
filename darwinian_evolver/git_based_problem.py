"""This file contains an extended Organism base class that can be used for problems that operate on a git repository."""

from __future__ import annotations

import contextlib
import difflib
import os
import subprocess
import tempfile
from contextlib import contextmanager
from typing import Generator
from typing import Iterable
from typing import Self

from pydantic import computed_field

from darwinian_evolver.problem import Organism


class GitBasedOrganism(Organism):
    """
    Organism base class that represents a modified git repository.

    The state of the organism is represented by a git commit hash and a set of file contents that might have been modified on top of the commit.
    """

    repo_root: str
    git_hash: str
    file_contents: dict[str, str]

    @classmethod
    def make_initial_organism_from_repo(
        cls,
        repo_root: str,
        files_to_capture: Iterable[str],
        git_hash: str | None = None,
        **kwargs,
    ) -> Self:
        """
        Initialize the organism with the current state of the git repository.

        This method captures the current git commit hash and the contents of specified files.
        """
        # Change working directory to the repository root
        with contextlib.chdir(repo_root):
            if git_hash is None:
                # Populate the git hash from the current repository
                git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()

            file_contents = {file_path: cls._get_file_content(git_hash, file_path) for file_path in files_to_capture}

        return cls(repo_root=repo_root, git_hash=git_hash, file_contents=file_contents, **kwargs)

    @computed_field
    @property
    def diff_from_parent(self) -> str:
        """
        Generate a diff between this organism and its parent.

        The diff is primarily intended for visual inspection to better understand what changes occurred.
        """
        if self.parent is None:
            return ""

        assert isinstance(self.parent, GitBasedOrganism), "Parent must be a GitBasedOrganism as well"
        assert self.file_contents.keys() == self.parent.file_contents.keys(), (
            "The set of captured files must match between parent and current organism"
        )

        diffs_per_file = []
        for file_path, content in self.file_contents.items():
            if self.parent.file_contents[file_path] != content:
                file_diff = difflib.unified_diff(
                    self.parent.file_contents[file_path].splitlines(keepends=True),
                    content.splitlines(keepends=True),
                    fromfile=file_path,
                    tofile=file_path,
                )
                file_diff_str = "\n".join(file_diff)
                diffs_per_file.append(file_diff_str)

        return "\n".join(diffs_per_file)

    @contextmanager
    def build_repo(self) -> Generator[str, None, None]:
        """
        Build a temporary git repository with the contents of this organism.

        Intended to be used as a context manager: `with organism.build_repo() as temp_dir: ...`

        This is useful for running evaluations or mutations within the context of this organism.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(["git", "clone", self.repo_root, temp_dir], check=True, capture_output=True)
            subprocess.run(["git", "checkout", self.git_hash], cwd=temp_dir, check=True, capture_output=True)

            # Replace the relevant files with the contents from this organism
            for file_path, content in self.file_contents.items():
                temp_file_path = f"{temp_dir}/{file_path}"
                # Currently, we do not support adding new files. Though this could be added in the future.
                assert os.path.exists(temp_file_path), f"File {temp_file_path} does not exist in the repository."
                with open(temp_file_path, "w") as f:
                    f.write(content)

            yield temp_dir

    @staticmethod
    def _get_file_content(git_hash: str, file_path: str) -> str:
        """Get the contents of a file in the current git repository."""
        return subprocess.check_output(["git", "show", f"{git_hash}:{file_path}"]).decode("utf-8")
