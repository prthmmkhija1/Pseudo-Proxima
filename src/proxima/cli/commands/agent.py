"""
Agent CLI commands.

Execute proxima_agent.md files.
"""

from pathlib import Path

import typer

app = typer.Typer(help="Execute proxima_agent.md files.")


@app.callback(invoke_without_command=True)
def agent_callback(ctx: typer.Context) -> None:
    """Show agent command help."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("run")
def run_agent(
    agent_file: Path = typer.Argument(..., help="Path to proxima_agent.md file"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show plan without executing"
    ),
    step: bool = typer.Option(False, "--step", help="Execute one task at a time"),
    resume_from: int | None = typer.Option(
        None, "--resume", "-r", help="Resume from task N"
    ),
) -> None:
    """Execute a proxima_agent.md file."""
    from proxima.core.agent_interpreter import AgentFileParser, AgentInterpreter

    if not agent_file.exists():
        typer.echo(f"Agent file not found: {agent_file}", err=True)
        raise typer.Exit(1)

    if not agent_file.suffix == ".md":
        typer.echo("Warning: Agent file should be a .md file", err=True)

    typer.echo(f"Loading agent file: {agent_file}")

    try:
        parser = AgentFileParser()
        agent_config = parser.parse_file(agent_file)
    except Exception as e:
        typer.echo(f"Failed to parse agent file: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Agent: {agent_config.metadata.name} v{agent_config.metadata.version}")
    typer.echo(f"Tasks: {len(agent_config.tasks)}")

    if dry_run:
        typer.echo("\n--- Execution Plan (Dry Run) ---")
        for i, task in enumerate(agent_config.tasks, 1):
            typer.echo(f"\n[Task {i}] {task.name}")
            typer.echo(f"  Type: {task.type.value}")
            typer.echo(
                f"  Description: {task.description[:80]}..."
                if len(task.description) > 80
                else f"  Description: {task.description}"
            )
        typer.echo("\n--- End of Plan ---")
        return

    # Execute the agent - use the full execute method
    def display_callback(message: str) -> None:
        typer.echo(message)

    interpreter = AgentInterpreter(display_callback=display_callback)

    start_task = resume_from or 1
    if start_task > 1:
        typer.echo(f"Resuming from task {start_task}")
        # Skip tasks before resume point
        agent_config.tasks = agent_config.tasks[start_task - 1 :]

    if step:
        typer.echo("Step mode: will prompt before each task")
        # Execute tasks one by one with user confirmation
        _execute_step_mode(agent_config, interpreter, display_callback)
        return

    try:
        report = interpreter.execute(agent_config)

        # Display results
        for result in report.task_results:
            status_str = "✓" if result.status.value == "completed" else "✗"
            typer.echo(
                f"  {status_str} Task {result.task_id}: {result.duration_ms:.1f}ms"
            )
            if result.error:
                typer.echo(f"      Error: {result.error}")

        if report.status == "completed":
            typer.echo("\n✓ Agent execution completed")
        elif report.status == "partial":
            typer.echo("\n⚠ Agent execution partially completed")
        else:
            typer.echo("\n✗ Agent execution failed")
            raise typer.Exit(1)

    except KeyboardInterrupt:
        typer.echo("\n⚠ Execution interrupted")
        raise typer.Exit(130)


@app.command("validate")
def validate_agent(
    agent_file: Path = typer.Argument(..., help="Path to proxima_agent.md file"),
) -> None:
    """Validate a proxima_agent.md file without executing."""
    from proxima.core.agent_interpreter import AgentFileParser

    if not agent_file.exists():
        typer.echo(f"Agent file not found: {agent_file}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Validating: {agent_file}")

    try:
        parser = AgentFileParser()
        agent_config = parser.parse_file(agent_file)

        errors = [
            str(issue)
            for issue in agent_config.validation_issues
            if issue.severity.value == "error"
        ]

        if errors:
            typer.echo("\n✗ Validation failed:")
            for error in errors:
                typer.echo(f"  - {error}")
            raise typer.Exit(1)

        typer.echo("\n✓ Agent file is valid")
        typer.echo(f"  Name: {agent_config.metadata.name}")
        typer.echo(f"  Version: {agent_config.metadata.version}")
        typer.echo(f"  Tasks: {len(agent_config.tasks)}")

        for i, task in enumerate(agent_config.tasks, 1):
            typer.echo(f"  [{i}] {task.name} ({task.type.value})")

    except Exception as e:
        typer.echo(f"\n✗ Parse error: {e}", err=True)
        raise typer.Exit(1)


@app.command("new")
def new_agent(
    output_file: Path = typer.Argument(
        Path("proxima_agent.md"), help="Output file path"
    ),
    name: str = typer.Option("My Agent", "--name", "-n", help="Agent name"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing file"),
) -> None:
    """Create a new proxima_agent.md template."""
    if output_file.exists() and not force:
        typer.echo(f"File already exists: {output_file}", err=True)
        typer.echo("Use --force to overwrite")
        raise typer.Exit(1)

    template = f"""# Proxima Agent Instructions

## Metadata
- name: {name}
- version: 1.0.0
- author: {typer.prompt("Author", default="Anonymous")}

## Configuration
- backend: auto
- shots: 1000
- output_format: xlsx

## Tasks

### Task 1: Bell State Preparation
Create and measure a Bell state circuit.

**Type:** simulation
**Backend:** auto

```quantum
H 0
CNOT 0 1
MEASURE ALL
```

### Task 2: Analyze Results
Generate insights from the simulation results.

**Type:** analysis
**Use LLM:** optional

Analyze the measurement distribution and provide insights.

## Output
- Format: XLSX
- Include: fidelity, execution time, insights
"""

    output_file.write_text(template)
    typer.echo(f"Created agent file: {output_file}")
    typer.echo(f"Edit the file and run with: proxima agent run {output_file}")


def _execute_step_mode(agent_config, interpreter, display_callback) -> None:
    """Execute tasks one by one with user confirmation between each task.

    Args:
        agent_config: The parsed agent configuration
        interpreter: The AgentInterpreter instance
        display_callback: Callback for displaying messages
    """
    import time

    from proxima.core.agent_interpreter import TaskResult, TaskStatus

    results: list[TaskResult] = []
    total_tasks = len(agent_config.tasks)

    for i, task in enumerate(agent_config.tasks, 1):
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"Task {i}/{total_tasks}: {task.name}")
        typer.echo(f"Type: {task.type.value}")
        typer.echo(
            f"Description: {task.description[:100]}..."
            if len(task.description) > 100
            else f"Description: {task.description}"
        )
        typer.echo("=" * 60)

        # Prompt user for action
        action = typer.prompt(
            "\n[R]un, [S]kip, or [A]bort?",
            default="r",
            show_default=True,
        ).lower()

        if action == "a":
            typer.echo("\n⚠ Execution aborted by user")
            break
        elif action == "s":
            typer.echo(f"⏭ Skipping task {i}")
            current_time = time.time()
            results.append(
                TaskResult(
                    task_id=str(i),
                    status=TaskStatus.SKIPPED,
                    start_time=current_time,
                    end_time=current_time,
                )
            )
            continue

        # Execute the task
        start = time.time()
        try:
            # Execute single task using interpreter's internal method
            task_result = interpreter._execute_task(task, i)
            end = time.time()
            elapsed_ms = (end - start) * 1000

            results.append(
                TaskResult(
                    task_id=str(i),
                    status=TaskStatus.COMPLETED,
                    start_time=start,
                    end_time=end,
                    result=task_result,
                )
            )
            typer.echo(f"✓ Task {i} completed in {elapsed_ms:.1f}ms")

        except Exception as e:
            end = time.time()
            elapsed_ms = (end - start) * 1000
            results.append(
                TaskResult(
                    task_id=str(i),
                    status=TaskStatus.FAILED,
                    start_time=start,
                    end_time=end,
                    error=str(e),
                )
            )
            typer.echo(f"✗ Task {i} failed: {e}")

            # Ask if user wants to continue
            if i < total_tasks:
                continue_exec = typer.confirm(
                    "Continue with remaining tasks?", default=True
                )
                if not continue_exec:
                    typer.echo("\n⚠ Execution stopped by user")
                    break

    # Summary
    completed = sum(1 for r in results if r.status == TaskStatus.COMPLETED)
    failed = sum(1 for r in results if r.status == TaskStatus.FAILED)
    skipped = sum(1 for r in results if r.status == TaskStatus.SKIPPED)

    typer.echo(f"\n{'=' * 60}")
    typer.echo("Step Mode Execution Summary")
    typer.echo(f"  Completed: {completed}/{total_tasks}")
    typer.echo(f"  Failed: {failed}")
    typer.echo(f"  Skipped: {skipped}")
    typer.echo("=" * 60)


@app.command("list-tasks")
def list_tasks(
    agent_file: Path = typer.Argument(..., help="Path to proxima_agent.md file"),
) -> None:
    """List tasks in an agent file."""
    from proxima.core.agent_interpreter import AgentFileParser

    if not agent_file.exists():
        typer.echo(f"Agent file not found: {agent_file}", err=True)
        raise typer.Exit(1)

    try:
        parser = AgentFileParser()
        agent_config = parser.parse_file(agent_file)

        typer.echo(f"\nTasks in {agent_file}:\n")
        for i, task in enumerate(agent_config.tasks, 1):
            typer.echo(f"[{i}] {task.name}")
            typer.echo(f"    Type: {task.type.value}")
            backend_name = task.parameters.get("backend")
            if backend_name:
                typer.echo(f"    Backend: {backend_name}")
            typer.echo()

    except Exception as e:
        typer.echo(f"Failed to parse agent file: {e}", err=True)
        raise typer.Exit(1)


@app.command("status")
def agent_status(
    agent_file: Path = typer.Argument(..., help="Path to proxima_agent.md file"),
) -> None:
    """Show execution status of an agent file.
    
    Displays whether the agent has been run, its last execution status,
    and any checkpoint information available.
    """
    from proxima.core.session import get_session_manager
    
    if not agent_file.exists():
        typer.echo(f"Agent file not found: {agent_file}", err=True)
        raise typer.Exit(1)
    
    session_manager = get_session_manager()
    agent_name = agent_file.stem
    
    # Look for sessions related to this agent
    sessions = session_manager.list_sessions()
    agent_sessions = [s for s in sessions if agent_name in s.get("name", "")]
    
    if not agent_sessions:
        typer.echo(f"\nNo execution history for: {agent_file}")
        typer.echo("Run with: proxima agent run " + str(agent_file))
        return
    
    typer.echo(f"\nExecution History for: {agent_file}\n")
    for session in agent_sessions[-5:]:  # Show last 5
        status = session.get("status", "unknown")
        timestamp = session.get("timestamp", "unknown")
        typer.echo(f"  [{timestamp}] Status: {status}")
        if session.get("error"):
            typer.echo(f"      Error: {session['error'][:50]}...")


@app.command("export")
def export_agent(
    agent_file: Path = typer.Argument(..., help="Path to proxima_agent.md file"),
    output_format: str = typer.Option(
        "json", "--format", "-f", help="Output format (json, yaml)"
    ),
    output_file: Path = typer.Option(
        None, "--output", "-o", help="Output file (default: stdout)"
    ),
) -> None:
    """Export agent file as structured data (JSON or YAML).
    
    Converts the agent.md file to a structured format for
    integration with other tools or programmatic access.
    """
    import json
    
    from proxima.core.agent_interpreter import AgentFileParser
    
    if not agent_file.exists():
        typer.echo(f"Agent file not found: {agent_file}", err=True)
        raise typer.Exit(1)
    
    try:
        parser = AgentFileParser()
        agent_config = parser.parse_file(agent_file)
        
        # Convert to dictionary
        data = {
            "metadata": {
                "name": agent_config.metadata.name,
                "version": agent_config.metadata.version,
                "author": getattr(agent_config.metadata, "author", None),
            },
            "configuration": agent_config.configuration,
            "tasks": [
                {
                    "name": task.name,
                    "type": task.type.value,
                    "description": task.description,
                    "parameters": task.parameters,
                }
                for task in agent_config.tasks
            ],
        }
        
        if output_format.lower() == "yaml":
            try:
                import yaml
                output = yaml.dump(data, default_flow_style=False, sort_keys=False)
            except ImportError:
                typer.echo("YAML export requires PyYAML: pip install pyyaml", err=True)
                raise typer.Exit(1)
        else:
            output = json.dumps(data, indent=2)
        
        if output_file:
            output_file.write_text(output)
            typer.echo(f"Exported to: {output_file}")
        else:
            typer.echo(output)
            
    except Exception as e:
        typer.echo(f"Export failed: {e}", err=True)
        raise typer.Exit(1)


@app.command("import")
def import_agent(
    input_file: Path = typer.Argument(..., help="Path to JSON/YAML file"),
    output_file: Path = typer.Option(
        Path("proxima_agent.md"), "--output", "-o", help="Output agent.md file"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing file"),
) -> None:
    """Import structured data as agent.md file.
    
    Converts a JSON or YAML file to proxima_agent.md format.
    """
    import json
    
    if not input_file.exists():
        typer.echo(f"Input file not found: {input_file}", err=True)
        raise typer.Exit(1)
    
    if output_file.exists() and not force:
        typer.echo(f"Output file exists: {output_file}", err=True)
        typer.echo("Use --force to overwrite")
        raise typer.Exit(1)
    
    try:
        content = input_file.read_text()
        
        if input_file.suffix in [".yaml", ".yml"]:
            try:
                import yaml
                data = yaml.safe_load(content)
            except ImportError:
                typer.echo("YAML import requires PyYAML: pip install pyyaml", err=True)
                raise typer.Exit(1)
        else:
            data = json.loads(content)
        
        # Generate agent.md content
        metadata = data.get("metadata", {})
        tasks = data.get("tasks", [])
        
        md_lines = [
            "# Proxima Agent Instructions",
            "",
            "## Metadata",
            f"- name: {metadata.get('name', 'Imported Agent')}",
            f"- version: {metadata.get('version', '1.0.0')}",
        ]
        
        if metadata.get("author"):
            md_lines.append(f"- author: {metadata['author']}")
        
        md_lines.extend(["", "## Tasks", ""])
        
        for i, task in enumerate(tasks, 1):
            md_lines.extend([
                f"### Task {i}: {task.get('name', f'Task {i}')}",
                task.get("description", ""),
                "",
                f"**Type:** {task.get('type', 'simulation')}",
            ])
            
            if task.get("parameters", {}).get("backend"):
                md_lines.append(f"**Backend:** {task['parameters']['backend']}")
            
            md_lines.append("")
        
        output_file.write_text("\n".join(md_lines))
        typer.echo(f"Created agent file: {output_file}")
        
    except Exception as e:
        typer.echo(f"Import failed: {e}", err=True)
        raise typer.Exit(1)


@app.command("diff")
def diff_agents(
    file1: Path = typer.Argument(..., help="First agent file"),
    file2: Path = typer.Argument(..., help="Second agent file"),
) -> None:
    """Compare two agent files and show differences.
    
    Useful for reviewing changes between agent file versions.
    """
    from proxima.core.agent_interpreter import AgentFileParser
    
    if not file1.exists():
        typer.echo(f"File not found: {file1}", err=True)
        raise typer.Exit(1)
    
    if not file2.exists():
        typer.echo(f"File not found: {file2}", err=True)
        raise typer.Exit(1)
    
    try:
        parser = AgentFileParser()
        config1 = parser.parse_file(file1)
        config2 = parser.parse_file(file2)
        
        typer.echo(f"\nComparing: {file1} vs {file2}\n")
        
        # Compare metadata
        if config1.metadata.name != config2.metadata.name:
            typer.echo(f"Name: {config1.metadata.name} → {config2.metadata.name}")
        if config1.metadata.version != config2.metadata.version:
            typer.echo(f"Version: {config1.metadata.version} → {config2.metadata.version}")
        
        # Compare task counts
        typer.echo(f"\nTasks: {len(config1.tasks)} → {len(config2.tasks)}")
        
        # Compare individual tasks
        max_tasks = max(len(config1.tasks), len(config2.tasks))
        for i in range(max_tasks):
            task1 = config1.tasks[i] if i < len(config1.tasks) else None
            task2 = config2.tasks[i] if i < len(config2.tasks) else None
            
            if task1 and task2:
                if task1.name != task2.name or task1.type != task2.type:
                    typer.echo(f"\n[Task {i+1}]")
                    if task1.name != task2.name:
                        typer.echo(f"  Name: {task1.name} → {task2.name}")
                    if task1.type != task2.type:
                        typer.echo(f"  Type: {task1.type.value} → {task2.type.value}")
            elif task1:
                typer.echo(f"\n[Task {i+1}] REMOVED: {task1.name}")
            elif task2:
                typer.echo(f"\n[Task {i+1}] ADDED: {task2.name}")
        
    except Exception as e:
        typer.echo(f"Diff failed: {e}", err=True)
        raise typer.Exit(1)


@app.command("watch")
def watch_agent(
    agent_file: Path = typer.Argument(..., help="Path to proxima_agent.md file"),
    interval: float = typer.Option(2.0, "--interval", "-i", help="Watch interval in seconds"),
) -> None:
    """Watch an agent file and re-run on changes.
    
    Monitors the agent file for changes and automatically
    re-executes when modifications are detected.
    """
    import time
    
    from proxima.core.agent_interpreter import AgentFileParser, AgentInterpreter
    
    if not agent_file.exists():
        typer.echo(f"Agent file not found: {agent_file}", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"Watching: {agent_file}")
    typer.echo(f"Interval: {interval}s")
    typer.echo("Press Ctrl+C to stop\n")
    
    last_mtime = agent_file.stat().st_mtime
    
    def execute_agent():
        try:
            parser = AgentFileParser()
            agent_config = parser.parse_file(agent_file)
            
            def display_callback(message: str) -> None:
                typer.echo(f"  {message}")
            
            interpreter = AgentInterpreter(display_callback=display_callback)
            report = interpreter.execute(agent_config)
            
            status = "✓" if report.status == "completed" else "✗"
            typer.echo(f"\n{status} Execution {report.status}")
            
        except Exception as e:
            typer.echo(f"\n✗ Execution failed: {e}")
    
    try:
        while True:
            time.sleep(interval)
            
            current_mtime = agent_file.stat().st_mtime
            if current_mtime != last_mtime:
                last_mtime = current_mtime
                typer.echo(f"\n[{time.strftime('%H:%M:%S')}] File changed, re-running...")
                execute_agent()
                
    except KeyboardInterrupt:
        typer.echo("\n\nWatch stopped.")
