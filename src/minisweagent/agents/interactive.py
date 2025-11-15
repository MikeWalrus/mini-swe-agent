"""A small generalization of the default agent that puts the user in the loop.

There are three modes:
- human: commands issued by the user are executed immediately
- confirm: commands issued by the LM but not whitelisted are confirmed by the user
- yolo: commands issued by the LM are executed immediately without confirmation
"""

import re
from dataclasses import dataclass, field
from typing import Literal

from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console
from rich.rule import Rule

from minisweagent import global_config_dir
from minisweagent.agents.default import AgentConfig, DefaultAgent, LimitsExceeded, NonTerminatingException, Submitted

console = Console(highlight=False)
prompt_session = PromptSession(history=FileHistory(global_config_dir / "interactive_history.txt"))


@dataclass
class SnapshotInfo:
    name: str
    message_count: int
    step_number: int


@dataclass
class InteractiveAgentConfig(AgentConfig):
    mode: Literal["human", "confirm", "yolo"] = "confirm"
    """Whether to confirm actions."""
    whitelist_actions: list[str] = field(default_factory=list)
    """Never confirm actions that match these regular expressions."""
    confirm_exit: bool = True
    """If the agent wants to finish, do we ask for confirmation from user?"""


class InteractiveAgent(DefaultAgent):
    _MODE_COMMANDS_MAPPING = {"/u": "human", "/c": "confirm", "/y": "yolo"}

    def __init__(self, *args, config_class=InteractiveAgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)
        self.cost_last_confirmed = 0.0
        self.snapshots: list[SnapshotInfo] = []
        self.step_number = 0

    def add_message(self, role: str, content: str, **kwargs):
        # Extend supermethod to print messages
        super().add_message(role, content, **kwargs)
        if role == "assistant":
            console.print(
                f"\n[red][bold]mini-swe-agent[/bold] (step [bold]{self.model.n_calls}[/bold], [bold]${self.model.cost:.2f}[/bold]):[/red]\n",
                end="",
                highlight=False,
            )
        else:
            console.print(f"\n[bold green]{role.capitalize()}[/bold green]:\n", end="", highlight=False)
        console.print(content, highlight=False, markup=False)

    def query(self) -> dict:
        # Extend supermethod to handle human mode
        if self.config.mode == "human":
            match command := self._prompt_and_handle_special("[bold yellow]>[/bold yellow] "):
                case "/y" | "/c":  # Just go to the super query, which queries the LM for the next action
                    pass
                case _:
                    msg = {"content": f"\n```bash\n{command}\n```"}
                    self.add_message("assistant", msg["content"])
                    return msg
        try:
            with console.status("Waiting for the LM to respond..."):
                return super().query()
        except LimitsExceeded:
            console.print(
                f"Limits exceeded. Limits: {self.config.step_limit} steps, ${self.config.cost_limit}.\n"
                f"Current spend: {self.model.n_calls} steps, ${self.model.cost:.2f}."
            )
            self.config.step_limit = int(input("New step limit: "))
            self.config.cost_limit = float(input("New cost limit: "))
            return super().query()

    def step(self) -> dict:
        # Override the step method to handle user interruption
        try:
            console.print(Rule())
            return super().step()
        except KeyboardInterrupt:
            # We always add a message about the interrupt and then just proceed to the next step
            interruption_message = self._prompt_and_handle_special(
                "\n\n[bold yellow]Interrupted.[/bold yellow] "
                "[green]Type a comment/command[/green] (/h for available commands)"
                "\n[bold yellow]>[/bold yellow] "
            ).strip()
            if not interruption_message or interruption_message in self._MODE_COMMANDS_MAPPING:
                interruption_message = "Temporary interruption caught."
            raise NonTerminatingException(f"Interrupted by user: {interruption_message}")

    def execute_action(self, action: dict) -> dict:
        # Override the execute_action method to handle user confirmation
        if self.should_ask_confirmation(action["action"]):
            self.ask_confirmation()
        self._create_snapshot()
        return super().execute_action(action)

    def should_ask_confirmation(self, action: str) -> bool:
        return self.config.mode == "confirm" and not any(re.match(r, action) for r in self.config.whitelist_actions)

    def ask_confirmation(self) -> None:
        prompt = (
            "[bold yellow]Execute?[/bold yellow] [green][bold]Enter[/bold] to confirm[/green], "
            "or [green]Type a comment/command[/green] (/h for available commands)\n"
            "[bold yellow]>[/bold yellow] "
        )
        match user_input := self._prompt_and_handle_special(prompt).strip():
            case "" | "/y":
                pass  # confirmed, do nothing
            case "/u":  # Skip execution action and get back to query
                raise NonTerminatingException("Command not executed. Switching to human mode")
            case _:
                raise NonTerminatingException(
                    f"Command not executed. The user rejected your command with the following message: {user_input}"
                )

    def _prompt_and_handle_special(self, prompt: str) -> str:
        """Prompts the user, takes care of /h (followed by requery) and sets the mode. Returns the user input."""
        console.print(prompt, end="")
        user_input = prompt_session.prompt("")
        if user_input == "/h":
            console.print(
                f"Current mode: [bold green]{self.config.mode}[/bold green]\n"
                f"[bold green]/y[/bold green] to switch to [bold yellow]yolo[/bold yellow] mode (execute LM commands without confirmation)\n"
                f"[bold green]/c[/bold green] to switch to [bold yellow]confirmation[/bold yellow] mode (ask for confirmation before executing LM commands)\n"
                f"[bold green]/u[/bold green] to switch to [bold yellow]human[/bold yellow] mode (execute commands issued by the user)\n"
                f"[bold green]/r[/bold green] to list available snapshots\n"
                f"[bold green]/r N[/bold green] to rollback to step N\n"
            )
            return self._prompt_and_handle_special(prompt)
        if user_input.startswith("/r"):
            return self._handle_rollback(user_input)
        if user_input in self._MODE_COMMANDS_MAPPING:
            if self.config.mode == self._MODE_COMMANDS_MAPPING[user_input]:
                return self._prompt_and_handle_special(
                    f"[bold red]Already in {self.config.mode} mode.[/bold red]\n{prompt}"
                )
            self.config.mode = self._MODE_COMMANDS_MAPPING[user_input]
            console.print(f"Switched to [bold green]{self.config.mode}[/bold green] mode.")
            return user_input
        return user_input

    def _create_snapshot(self) -> None:
        """Create a btrfs snapshot before executing a step."""
        if hasattr(self.env, "create_snapshot") and self.env.config.cwd:
            snapshot_name = f"step-{self.step_number}"
            try:
                self.env.create_snapshot(snapshot_name)
                self.snapshots.append(SnapshotInfo(snapshot_name, len(self.messages), self.step_number))
                self.step_number += 1
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to create snapshot: {e}[/yellow]")

    def _handle_rollback(self, user_input: str) -> str:
        """Handle /r and /r N commands for rollback."""
        parts = user_input.split()
        if len(parts) == 1:
            if not self.snapshots:
                console.print("[yellow]No snapshots available[/yellow]")
            else:
                console.print("[bold green]Available snapshots:[/bold green]")
                for snapshot in self.snapshots:
                    console.print(f"  Step {snapshot.step_number}: {snapshot.name} ({snapshot.message_count} messages)")
            return self._prompt_and_handle_special("[bold yellow]>[/bold yellow] ")
        
        try:
            target_step = int(parts[1])
            snapshot_info = next((s for s in self.snapshots if s.step_number == target_step), None)
            if not snapshot_info:
                console.print(f"[red]Snapshot for step {target_step} not found[/red]")
                return self._prompt_and_handle_special("[bold yellow]>[/bold yellow] ")
            
            if hasattr(self.env, "rollback_snapshot"):
                self.env.rollback_snapshot(snapshot_info.name)
                self.messages = self.messages[:snapshot_info.message_count]
                self.snapshots = [s for s in self.snapshots if s.step_number <= target_step]
                self.step_number = target_step + 1
                console.print(f"[bold green]Rolled back to step {target_step}[/bold green]")
            else:
                console.print("[red]Environment does not support rollback[/red]")
        except (ValueError, IndexError):
            console.print("[red]Invalid rollback command. Use /r to list snapshots or /r N to rollback to step N[/red]")
        
        return self._prompt_and_handle_special("[bold yellow]>[/bold yellow] ")

    def has_finished(self, output: dict[str, str]):
        try:
            return super().has_finished(output)
        except Submitted as e:
            if self.config.confirm_exit:
                console.print(
                    "[bold green]Agent wants to finish.[/bold green] "
                    "[green]Type a comment to give it a new task or press enter to quit.\n"
                    "[bold yellow]>[/bold yellow] ",
                    end="",
                )
                if new_task := self._prompt_and_handle_special("").strip():
                    raise NonTerminatingException(f"The user added a new task: {new_task}")
            raise e
