#!/usr/bin/env python3
# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Graphify: Codebase Knowledge Graph Builder and Query Engine.

Parses all modules, imports, classes, functions, docstrings, and historical
changelogs in the repository to build a unified codebase knowledge graph.
Provides dependency tracing, circular import detection, change history extraction,
and Mermaid visualization capability.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Global output file path for the knowledge graph
GRAPH_FILE = Path("repo_knowledge_graph.json")


class CodebaseGraphBuilder:
    """Parses codebase and documents to construct a knowledge graph."""

    def __init__(self, root_dir: str | Path = ".") -> None:
        self.root_dir = Path(root_dir).resolve()
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, str]] = []

    def add_node(self, node_id: str, node_type: str, properties: dict[str, Any]) -> None:
        """Add or update a node in the graph."""
        self.nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "properties": properties,
        }

    def add_edge(self, source: str, target: str, edge_type: str) -> None:
        """Add a directed edge in the graph."""
        self.edges.append({
            "source": source,
            "target": target,
            "type": edge_type,
        })

    def run(self) -> dict[str, Any]:
        """Run full extraction pipeline."""
        print("1. Parsing Python files and AST symbols...")
        self._parse_python_files()

        print("2. Parsing CHANGELOG.md version history...")
        self._parse_changelog()

        print("3. Parsing project master specifications...")
        self._parse_master_spec()

        print("4. Resolving dependency and import edges...")
        self._resolve_imports()

        graph = {
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
            "stats": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
            }
        }
        return graph

    def _parse_python_files(self) -> None:
        """Scan directory and parse all python files using AST."""
        for root, _, files in os.walk(self.root_dir):
            # Skip virtual environments, caches, git directories
            if any(p in root for p in [".venv", ".venv312", ".git", ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache", "build", "dist"]):
                continue

            for file in files:
                if not file.endswith(".py"):
                    continue

                file_path = Path(root) / file
                rel_path = file_path.relative_to(self.root_dir).as_posix()
                
                try:
                    code = file_path.read_text(encoding="utf-8", errors="ignore")
                    tree = ast.parse(code)
                except Exception as e:
                    print(f"  Warning: Failed to parse AST for {rel_path} ({e})")
                    continue

                lines = code.splitlines()
                loc = len(lines)

                # Add file node
                self.add_node(
                    node_id=rel_path,
                    node_type="file",
                    properties={
                        "path": rel_path,
                        "loc": loc,
                        "docstring": ast.get_docstring(tree) or "",
                    }
                )

                # Analyze contents
                self._analyze_ast_tree(tree, rel_path)

    def _analyze_ast_tree(self, tree: ast.AST, file_id: str) -> None:
        """Analyze symbols and imports inside a file's AST."""
        for node in ast.walk(tree):
            # Track classes
            if isinstance(node, ast.ClassDef):
                class_id = f"{file_id}::{node.name}"
                doc = ast.get_docstring(node) or ""
                bases = [ast.unparse(b) for b in node.bases]
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                
                self.add_node(
                    node_id=class_id,
                    node_type="class",
                    properties={
                        "name": node.name,
                        "docstring": doc,
                        "bases": bases,
                        "methods": methods,
                    }
                )
                # Link file -> class
                self.add_edge(file_id, class_id, "defines_class")

            # Track functions/methods (only top-level script functions)
            elif isinstance(node, ast.FunctionDef):
                # Ensure it's not nested inside a class (we look at parent during linear check, but keep it simple)
                # If we just do simple check:
                func_id = f"{file_id}::{node.name}"
                doc = ast.get_docstring(node) or ""
                self.add_node(
                    node_id=func_id,
                    node_type="function",
                    properties={
                        "name": node.name,
                        "docstring": doc,
                        "args": [arg.arg for arg in node.args.args],
                    }
                )
                self.add_edge(file_id, func_id, "defines_function")

            # Track imports
            elif isinstance(node, ast.Import):
                for name in node.names:
                    # Generic import
                    self.add_node(
                        node_id=f"import::{name.name}",
                        node_type="import_spec",
                        properties={"module": name.name}
                    )
                    self.add_edge(file_id, f"import::{name.name}", "imports_module")

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for name in node.names:
                    full_import = f"{module}.{name.name}" if module else name.name
                    self.add_node(
                        node_id=f"import::{full_import}",
                        node_type="import_spec",
                        properties={"module": module, "name": name.name}
                    )
                    self.add_edge(file_id, f"import::{full_import}", "imports_symbol")

    def _parse_changelog(self) -> None:
        """Parse CHANGELOG.md to extract version nodes and history."""
        changelog_path = self.root_dir / "CHANGELOG.md"
        if not changelog_path.exists():
            return

        content = changelog_path.read_text(encoding="utf-8", errors="ignore")
        # Find sections like: ## [1.6.0] - 2026-07-05 or similar
        version_headers = re.finditer(r"##\s+\[?([0-9a-zA-Z\.\-]+)\]?\s*-\s*([0-9\-]+)", content)
        
        versions = []
        for match in version_headers:
            ver = match.group(1)
            date = match.group(2)
            versions.append((ver, date, match.start()))

        for i, (ver, date, pos) in enumerate(versions):
            # Extract content until next version pos
            end_pos = versions[i+1][2] if i + 1 < len(versions) else len(content)
            body = content[pos:end_pos].strip()
            
            # Simple change category parsing (Added, Fixed, Changed...)
            changes = []
            for line in body.splitlines():
                if line.strip().startswith("-") or line.strip().startswith("*"):
                    changes.append(line.strip().lstrip("-* ").strip())

            version_id = f"version::{ver}"
            self.add_node(
                node_id=version_id,
                node_type="version_milestone",
                properties={
                    "version": ver,
                    "date": date,
                    "changes": changes,
                }
            )
            # Link version sequence
            if i + 1 < len(versions):
                next_ver = versions[i+1][0]
                self.add_edge(version_id, f"version::{next_ver}", "succeeds")

    def _parse_master_spec(self) -> None:
        """Parse IVERI_PROJECT_MASTER.md to extract architecture and objectives."""
        master_path = self.root_dir / "IVERI_PROJECT_MASTER.md"
        if not master_path.exists():
            return

        content = master_path.read_text(encoding="utf-8", errors="ignore")
        
        # Look for Hypothesis H1 - H10
        hypotheses = re.finditer(r"\b(H\d+)\b[:\-]?\s*([^\n]+)", content)
        for match in hypotheses:
            h_id = match.group(1)
            h_desc = match.group(2).strip()
            
            self.add_node(
                node_id=f"hypothesis::{h_id}",
                node_type="hypothesis",
                properties={
                    "label": h_id,
                    "description": h_desc,
                }
            )

        # Look for objectives or requirements like OBJ1, OBJ2 or similar
        objs = re.finditer(r"\b(OBJ-?\d+)\b[:\-]?\s*([^\n]+)", content)
        for match in objs:
            o_id = match.group(1).replace("-", "")
            o_desc = match.group(2).strip()
            
            self.add_node(
                node_id=f"requirement::{o_id}",
                node_type="spec_requirement",
                properties={
                    "label": o_id,
                    "description": o_desc,
                }
            )

    def _resolve_imports(self) -> None:
        """Resolve import edges to point to actual internal file nodes when possible."""
        # Map internal package paths to filenames
        # E.g. "model.backbone" -> "model/backbone.py"
        # E.g. "core.constants" -> "core/constants.py"
        internal_modules: dict[str, str] = {}
        for node_id, node in self.nodes.items():
            if node["type"] == "file":
                filepath = node["properties"]["path"]
                # Convert "model/backbone.py" -> "model.backbone"
                mod_name = filepath.replace(".py", "").replace("/", ".")
                # Also handle __init__.py modules
                if mod_name.endswith(".__init__"):
                    mod_name = mod_name[:-9]
                internal_modules[mod_name] = filepath

        # Reroute import edges
        resolved_edges = []
        for edge in self.edges:
            target = edge["target"]
            if target.startswith("import::"):
                import_path = target[8:]
                
                # Check if this maps to a known local file
                matched_file = None
                # Try exact match
                if import_path in internal_modules:
                    matched_file = internal_modules[import_path]
                else:
                    # Try prefixes (e.g. from model.moe.router import SparseMoERouter)
                    parts = import_path.split(".")
                    for i in range(len(parts), 0, -1):
                        sub_prefix = ".".join(parts[:i])
                        if sub_prefix in internal_modules:
                            matched_file = internal_modules[sub_prefix]
                            break

                if matched_file:
                    edge["target"] = matched_file
                    edge["type"] = "depends_on_file"
                    
            resolved_edges.append(edge)
        self.edges = resolved_edges


class CodebaseGraphQuery:
    """Query and analyze the codebase knowledge graph."""

    def __init__(self, graph: dict[str, Any]) -> None:
        self.graph = graph
        self.nodes = {n["id"]: n for n in graph["nodes"]}
        
        # Build adjacency list
        self.adj: dict[str, list[dict[str, str]]] = {n_id: [] for n_id in self.nodes}
        self.rev_adj: dict[str, list[dict[str, str]]] = {n_id: [] for n_id in self.nodes}
        for edge in graph["edges"]:
            src = edge["source"]
            tgt = edge["target"]
            if src in self.adj and tgt in self.adj:
                self.adj[src].append(edge)
                self.rev_adj[tgt].append(edge)

    def stats(self) -> None:
        """Print high level graph stats."""
        print("=== Graph Statistics ===")
        types: dict[str, int] = {}
        for node in self.nodes.values():
            types[node["type"]] = types.get(node["type"], 0) + 1
        for k, v in types.items():
            print(f"  {k}: {v}")
        print(f"  Total Nodes: {len(self.nodes)}")
        print(f"  Total Edges: {len(self.graph['edges'])}")

    def trace(self, symbol: str) -> None:
        """Trace dependencies and usage of a file or symbol."""
        print(f"=== Dependency Trace for: '{symbol}' ===")
        
        # Find matching node(s)
        matches = [n_id for n_id in self.nodes if symbol in n_id]
        if not matches:
            print(f"  No nodes found matching '{symbol}'")
            return

        for node_id in matches:
            node = self.nodes[node_id]
            print(f"\nNode: {node_id} ({node['type']})")
            if "docstring" in node["properties"] and node["properties"]["docstring"]:
                doc = node["properties"]["docstring"].splitlines()[0]
                print(f"  Doc: {doc}")

            # Outgoing dependencies
            deps = [e["target"] for e in self.adj[node_id] if e["type"] == "depends_on_file"]
            if deps:
                print("  Depends on files:")
                for d in sorted(set(deps)):
                    print(f"    -> {d}")

            # Incoming usages
            usages = [e["source"] for e in self.rev_adj[node_id] if e["type"] == "depends_on_file"]
            if usages:
                print("  Used by files:")
                for u in sorted(set(usages)):
                    print(f"    <- {u}")

    def check_circular(self) -> None:
        """Detect circular dependencies among file nodes."""
        print("=== Circular Dependency Audit ===")
        
        # Filter nodes to files only
        file_nodes = [n_id for n_id, n in self.nodes.items() if n["type"] == "file"]
        
        visited: dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=visited
        for n_id in file_nodes:
            visited[n_id] = 0

        cycles: list[list[str]] = []

        def dfs(node_id: str, path: list[str]) -> None:
            visited[node_id] = 1  # visiting
            path.append(node_id)

            for edge in self.adj[node_id]:
                if edge["type"] == "depends_on_file":
                    tgt = edge["target"]
                    if tgt in visited:
                        if visited[tgt] == 1:  # Cycle detected
                            cycle_start = path.index(tgt)
                            cycles.append(path[cycle_start:] + [tgt])
                        elif visited[tgt] == 0:
                            dfs(tgt, path)

            path.pop()
            visited[node_id] = 2  # visited

        for n_id in file_nodes:
            if visited[n_id] == 0:
                dfs(n_id, [])

        if not cycles:
            print("  PASS: No circular dependencies detected among codebase files.")
        else:
            print(f"  FAIL: Found {len(cycles)} circular dependencies:")
            for idx, cycle in enumerate(cycles[:10]):
                path_str = " -> ".join(cycle)
                print(f"    Cycle {idx+1}: {path_str}")

    def mermaid(self) -> str:
        """Generate a Mermaid dependency diagram of core modules."""
        lines = ["graph TD", "    subgraph Core System"]
        
        # Categorize files into components
        components = {
            "Model": ["model/iveri_core.py", "model/backbone.py", "model/attention.py", "model/norms.py", "model/rope.py", "model/swiglu.py"],
            "BLT": ["model/blt/entropy_model.py", "model/blt/patcher.py", "model/blt/encoder.py", "model/blt/decoder.py"],
            "Titans": ["model/titans/memory.py", "model/titans/updater.py", "model/titans/lr_gen.py"],
            "MoR": ["model/mor/router.py", "model/mor/recursion.py", "model/mor/kv_cache.py"],
            "Mamba2": ["model/mamba2/block.py", "model/mamba2/scan.py", "model/mamba2/math.py"],
            "MoE": ["model/moe/router.py", "model/moe/experts.py"],
            "Training": ["training/trainer.py", "training/optimizer.py", "training/scheduler.py", "training/checkpointing.py"],
            "Evaluation": ["evaluation/evaluator.py", "evaluation/sft_evaluator.py", "evaluation/coding_evaluator.py"],
        }

        # Add node styles and nicknames
        nicknames: dict[str, str] = {}
        counter = 0
        for comp_name, files in components.items():
            lines.append(f"        subgraph {comp_name}")
            for f in files:
                if f in self.nodes:
                    nick = f"node_{counter}"
                    nicknames[f] = nick
                    basename = f.split("/")[-1]
                    lines.append(f"            {nick}[\"{basename}\"]")
                    counter += 1
            lines.append("        end")
        lines.append("    end")

        # Add connection edges
        drawn_edges = set()
        for edge in self.graph["edges"]:
            if edge["type"] == "depends_on_file":
                src = edge["source"]
                tgt = edge["target"]
                if src in nicknames and tgt in nicknames:
                    edge_key = (nicknames[src], nicknames[tgt])
                    if edge_key not in drawn_edges:
                        lines.append(f"    {nicknames[src]} --> {nicknames[tgt]}")
                        drawn_edges.add(edge_key)

        return "\n".join(lines)

    def version_history(self) -> None:
        """Display version history timeline and changes."""
        print("=== Version Timeline ===")
        ver_nodes = [n for n in self.nodes.values() if n["type"] == "version_milestone"]
        # Sort by version/date (latest first)
        ver_nodes.sort(key=lambda x: x["properties"].get("date", ""), reverse=True)
        
        for v in ver_nodes:
            props = v["properties"]
            sys.stdout.buffer.write(f"\nVersion: {props['version']} ({props.get('date', 'N/A')})\n".encode("utf-8", errors="replace"))
            for c in props.get("changes", [])[:5]:
                sys.stdout.buffer.write(f"  - {c}\n".encode("utf-8", errors="replace"))
            if len(props.get("changes", [])) > 5:
                sys.stdout.buffer.write(f"  ... and {len(props['changes']) - 5} more changes.\n".encode("utf-8", errors="replace"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Codebase Knowledge Graph Engine")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    subparsers.add_parser("build", help="Scan codebase and compile JSON knowledge graph")
    subparsers.add_parser("stats", help="Show summary statistics from the compiled graph")
    subparsers.add_parser("check-circular", help="Audit the codebase for circular imports")
    subparsers.add_parser("history", help="Show version timeline and milestones")
    subparsers.add_parser("mermaid", help="Generate a Mermaid dependency diagram")

    trace_parser = subparsers.add_parser("trace", help="Trace dependencies of a file or symbol")
    trace_parser.add_argument("symbol", type=str, help="Module, file, or class name to trace")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "build":
        builder = CodebaseGraphBuilder()
        graph = builder.run()
        GRAPH_FILE.write_text(json.dumps(graph, indent=2), encoding="utf-8")
        print(f"Success: Knowledge graph successfully written to {GRAPH_FILE}")
        sys.exit(0)

    # All other commands require the graph to be built first
    if not GRAPH_FILE.exists():
        print(f"Error: {GRAPH_FILE} does not exist. Please run 'python graphify.py build' first.")
        sys.exit(1)

    try:
        graph = json.loads(GRAPH_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error: Failed to load graph file: {e}")
        sys.exit(1)

    query = CodebaseGraphQuery(graph)

    if args.command == "stats":
        query.stats()
    elif args.command == "trace":
        query.trace(args.symbol)
    elif args.command == "check-circular":
        query.check_circular()
    elif args.command == "history":
        query.version_history()
    elif args.command == "mermaid":
        print(query.mermaid())


if __name__ == "__main__":
    main()
