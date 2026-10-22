"""Silivra's training, local assistant, tools, and private memory command line."""

import argparse
import json
import sys
from pathlib import Path


def parser():
    root = argparse.ArgumentParser(description="Silivra: from-scratch AI training and a local assistant runtime.")
    from silivra import __version__
    root.add_argument("--version", action="version", version=f"Silivra {__version__}")
    root.add_argument("--workspace", dest="startup_workspace", type=Path, default=Path.cwd(), help="Workspace for the interactive terminal")
    root.add_argument("--model", dest="startup_model", type=Path, help="Local model run/checkpoint for the interactive terminal")
    root.add_argument("--user", dest="startup_user", default="local", help="Local user label for the interactive terminal")
    sub = root.add_subparsers(dest="command", required=False)
    sub.add_parser("doctor", help="Inspect non-identifying hardware and MLX Metal support")
    prep = sub.add_parser("prepare", help="Download bounded story subsets and train a new BPE tokenizer")
    prep.add_argument("--data", type=Path, default=Path("data/tinystories"))
    prep.add_argument("--train-stories", type=int, default=30000)
    prep.add_argument("--valid-stories", type=int, default=1000)
    prep.add_argument("--vocab-size", type=int, default=4096)
    trainer = sub.add_parser("train", help="Pretrain random weights, or resume a local run")
    trainer.add_argument("--data", type=Path, default=Path("data/tinystories"))
    trainer.add_argument("--run", type=Path, default=Path("runs/silivra-tiny"))
    trainer.add_argument("--config", type=Path, help="JSON training settings; resume uses saved settings by default")
    trainer.add_argument("--max-minutes", type=float, default=15.0, help="Checkpoint and stop after this session budget")
    trainer.add_argument("--resume", action="store_true")
    exporter = sub.add_parser("export", help="Bundle trained weights, source, licenses, and provenance")
    exporter.add_argument("--run", type=Path, default=Path("runs/silivra-tiny"))
    exporter.add_argument("--which", choices=("latest", "best"), default="best")
    exporter.add_argument("--output", type=Path, default=Path("exports/silivra-tiny-v0.1"))
    exporter.add_argument("--archive", action="store_true", help="Also write a shareable ZIP archive")
    for name, help_text in [("generate", "Complete text offline using trained weights"),
                            ("benchmark", "Measure warm Metal generation speed"),
                            ("evaluate", "Measure held-out next-token validation loss")]:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--run", type=Path, default=Path("runs/silivra-tiny"))
        command.add_argument("--which", choices=("latest", "best"), default="best")
        if name == "generate":
            command.add_argument("prompt", nargs="?", default="Once upon a time, a little fox")
            command.add_argument("--max-new-tokens", type=int, default=128)
            command.add_argument("--temperature", type=float, default=0.7)
            command.add_argument("--top-k", type=int, default=40)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--json", action="store_true")
        elif name == "benchmark":
            command.add_argument("--trials", type=int, default=3)
            command.add_argument("--max-new-tokens", type=int, default=128)
            command.add_argument("--output", type=Path)
        else:
            command.add_argument("--data", type=Path, default=Path("data/tinystories"))
            command.add_argument("--batches", type=int, default=32)
    _assistant_parsers(sub)
    return root


def _assistant_parsers(sub):
    def workspace(command, *, memory=False, permissions=False):
        command.add_argument("--workspace", type=Path, default=Path.cwd())
        if memory:
            command.add_argument("--memory", type=Path, help="Private SQLite path; defaults to WORKSPACE/.silivra/memory.sqlite")
            command.add_argument("--user", default="local", help="Local user scope label (not authentication)")
        if permissions:
            command.add_argument("--allow-write", action="store_true", help="Permit workspace file writes")
            command.add_argument("--allow-network", action="store_true", help="Permit public web search and fetch")
            command.add_argument("--allow-exec", action="store_true", help="Permit trusted local commands; NOT a sandbox")

    listing = sub.add_parser("tools", help="List enabled tools and JSON argument schemas")
    workspace(listing, permissions=True)
    tool = sub.add_parser("tool", help="Execute one explicit tool without relying on model tool selection")
    tool.add_argument("name")
    tool.add_argument("--arguments", default="{}", help="One JSON object")
    workspace(tool, permissions=True)

    chat = sub.add_parser("chat", help="Experimental own-checkpoint assistant with bounded JSON tool actions")
    chat.add_argument("prompt", nargs="?", help="Omit for an interactive session; /exit quits and /new starts a session")
    chat.add_argument("--run", type=Path, default=None, help="Run/checkpoint; interactive mode also remembers /model")
    chat.add_argument("--which", choices=("latest", "best"), default=None)
    chat.add_argument("--session", help="Resume a session in this user/project scope")
    chat.add_argument("--tool", dest="selected_tools", action="append", help="Expose selected enabled tools; repeat as needed (default: none)")
    chat.add_argument("--all-tools", action="store_true", help="Expose every enabled tool; needs a sufficiently long-context checkpoint")
    chat.add_argument("--no-tools", action="store_true")
    chat.add_argument("--max-new-tokens", type=int, help="Output reservation (interactive: saved preference; one-shot default: 96)")
    chat.add_argument("--max-steps", type=int, help="Turn action limit (interactive: saved preference; one-shot default: 6)")
    chat.add_argument("--max-seconds", type=float, help="Turn time budget (interactive: saved preference; one-shot default: 120)")
    chat.add_argument("--history-messages", type=int, help="Previous messages (interactive: saved preference; one-shot default: 8)")
    chat.add_argument("--memory-chars", type=int, help="Retrieved memory budget (interactive: saved preference; one-shot default: 1500)")
    chat.add_argument("--json", action="store_true")
    workspace(chat, memory=True, permissions=True)

    memory = sub.add_parser("memory", help="Inspect and explicitly curate private, user/project-scoped memory")
    workspace(memory, memory=True)
    operations = memory.add_subparsers(dest="operation", required=True)
    remember = operations.add_parser("remember")
    remember.add_argument("key")
    remember.add_argument("content")
    remember.add_argument("--category", default="fact")
    remember.add_argument("--importance", type=float, default=0.5)
    recall = operations.add_parser("recall")
    recall.add_argument("query")
    recall.add_argument("--limit", type=int, default=8)
    operations.add_parser("list")
    forget = operations.add_parser("forget")
    forget.add_argument("key")
    revisions = operations.add_parser("revisions")
    revisions.add_argument("key")
    operations.add_parser("sessions")
    history = operations.add_parser("history")
    history.add_argument("session")
    history.add_argument("--limit", type=int, default=40)
    delete = operations.add_parser("delete-session")
    delete.add_argument("session")
    export = operations.add_parser("export")
    export.add_argument("--output", type=Path, required=True)
    backup = operations.add_parser("backup", help="Consistent backup of the entire private database, all scopes")
    backup.add_argument("--output", type=Path, required=True)

    prep = sub.add_parser("prepare-instructions", help="Validate licensed JSONL and build assistant-only loss masks")
    prep.add_argument("--train", type=Path, required=True)
    prep.add_argument("--valid", type=Path, required=True)
    prep.add_argument("--tokenizer", type=Path, required=True)
    prep.add_argument("--output", type=Path, default=Path("data/instructions"))
    prep.add_argument("--max-seq-len", type=int, default=512)
    tune = sub.add_parser("finetune", help="Instruction-tune a verified local checkpoint into a separate run")
    tune.add_argument("--data", type=Path, default=Path("data/instructions"))
    tune.add_argument("--base", type=Path, default=Path("runs/silivra-tiny"))
    tune.add_argument("--run", type=Path, default=Path("runs/silivra-instruct"))
    tune.add_argument("--config", type=Path)
    tune.add_argument("--resume", action="store_true")
    tune.add_argument("--max-minutes", type=float, default=15)

    evaluation = sub.add_parser("evaluate-assistant", help="Measure JSON protocol and held-out task responses without executing tools")
    evaluation.add_argument("--run", type=Path, default=Path("runs/silivra-instruct"))
    evaluation.add_argument("--which", choices=("latest", "best"), default="best")
    evaluation.add_argument("--suite", type=Path, default=Path("examples/evaluations/assistant.jsonl"))
    evaluation.add_argument("--max-new-tokens", type=int, default=96)
    evaluation.add_argument("--output", type=Path)


def _assistant_command(args):
    if args.command in ("tools", "tool"):
        from silivra.tools import ToolRegistry
        registry = ToolRegistry(args.workspace, allow_write=args.allow_write,
                                allow_exec=args.allow_exec, allow_network=args.allow_network)
        if args.command == "tools":
            return registry.schemas()
        from silivra.terminal import _json
        arguments = _json(args.arguments)
        result = registry.execute(args.name, arguments)
        if not result.get("ok"):
            print(json.dumps(result, indent=2))
            raise SystemExit(1)
        return result
    if args.command in ("memory", "chat"):
        if args.command == "chat" and args.prompt is None and not args.json and sys.stdin.isatty():
            from silivra.terminal import run_terminal
            if sum((bool(args.no_tools), bool(args.selected_tools), bool(args.all_tools))) > 1:
                raise ValueError("Choose --no-tools, --all-tools, or --tool selections.")
            selection = None if args.all_tools else ([] if args.no_tools else args.selected_tools or ())
            raise SystemExit(run_terminal(workspace=args.workspace, user=args.user, run=args.run, which=args.which,
                                          memory=args.memory, session_id=args.session, tool_names=selection,
                                          allow_write=args.allow_write, allow_network=args.allow_network,
                                          allow_exec=args.allow_exec, max_new_tokens=args.max_new_tokens,
                                          config_overrides={key: getattr(args, key) for key in
                                                            ("max_steps", "max_seconds", "history_messages", "memory_chars")
                                                            if getattr(args, key) is not None}))
        from silivra.agent import project_scope
        from silivra.memory import MemoryStore
        scope = project_scope(args.workspace, args.user)
        database = args.memory or args.workspace / ".silivra" / "memory.sqlite"
        with MemoryStore(database) as store:
            if args.command == "memory":
                if args.operation == "remember":
                    return store.remember(scope, args.key, args.content, category=args.category,
                                          source="user", importance=args.importance)
                if args.operation == "recall":
                    return store.recall(scope, args.query, limit=args.limit)
                if args.operation == "list":
                    return store.list_memories(scope)
                if args.operation == "forget":
                    return {"forgotten": store.forget(scope, args.key)}
                if args.operation == "revisions":
                    return store.revisions(scope, args.key)
                if args.operation == "sessions":
                    return store.list_sessions(scope)
                if args.operation == "history":
                    return store.recent_messages(scope, args.session, limit=args.limit)
                if args.operation == "delete-session":
                    store.delete_session(scope, args.session)
                    return {"deleted_session": args.session}
                if args.operation == "backup":
                    return {"backup": store.backup(args.output), "includes_all_scopes": True}
                import os
                with os.fdopen(os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w", encoding="utf-8") as output:
                    json.dump(store.export(scope), output, indent=2, ensure_ascii=False)
                    output.write("\n")
                return {"export": str(args.output)}
            from silivra.agent import Agent, AgentConfig, MLXBackend
            from silivra.tools import ToolRegistry
            if sum((bool(args.no_tools), bool(args.selected_tools), bool(args.all_tools))) > 1:
                raise ValueError("Choose --no-tools, --all-tools, or --tool selections.")
            tools = ToolRegistry(args.workspace, allow_write=args.allow_write,
                                 allow_exec=args.allow_exec, allow_network=args.allow_network)
            from silivra.terminal import default_model_run
            if args.prompt is None and not sys.stdin.isatty():
                args.prompt = sys.stdin.read(32_001)
            if args.prompt is not None and (not args.prompt.strip() or len(args.prompt) > 32_000):
                raise ValueError("prompt must contain 1–32,000 characters.")
            config = AgentConfig(max_steps=args.max_steps if args.max_steps is not None else 6,
                                 max_seconds=args.max_seconds if args.max_seconds is not None else 120,
                                 history_messages=args.history_messages if args.history_messages is not None else 8,
                                 memory_chars=args.memory_chars if args.memory_chars is not None else 1500)
            backend = MLXBackend(args.run or default_model_run(args.workspace), which=args.which or "best",
                                 max_new_tokens=args.max_new_tokens if args.max_new_tokens is not None else 96)
            agent = Agent(backend, tools, store, scope,
                          config=config,
                          tool_names=None if args.all_tools else (args.selected_tools or []))
            print("Experimental local checkpoint: tool-following ability must be evaluated separately.", file=sys.stderr)
            session = args.session
            while True:
                try:
                    prompt = args.prompt if args.prompt is not None else input("you> ")
                except (EOFError, KeyboardInterrupt):
                    return None
                if prompt.strip() == "/exit":
                    return None
                if prompt.strip() == "/new":
                    session = None
                    if args.prompt is not None:
                        return None
                    continue
                if not prompt.strip():
                    if args.prompt is not None:
                        raise ValueError("prompt cannot be empty.")
                    continue
                result = agent.run(prompt, session_id=session)
                session = result["session_id"]
                if args.json:
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                else:
                    print(result["answer"] or f"Stopped: {result['status']}. {result.get('error', '')}")
                    print(f"session: {session}", file=sys.stderr)
                if args.prompt is not None:
                    if result["status"] != "completed":
                        raise SystemExit(1)
                    return None
    if args.command == "prepare-instructions":
        from silivra.instructions import prepare_instructions
        return prepare_instructions(args.train, args.valid, args.tokenizer, args.output, max_seq_len=args.max_seq_len)
    if args.command == "finetune":
        from silivra.finetune import FinetuneConfig, finetune
        config = FinetuneConfig(**json.loads(args.config.read_text())) if args.config else None
        return finetune(args.data, args.base, args.run, config, resume=args.resume, max_minutes=args.max_minutes)
    if args.command == "evaluate-assistant":
        from silivra.agent import MLXBackend
        from silivra.evaluation import evaluate_assistant
        backend = MLXBackend(args.run, which=args.which, max_new_tokens=args.max_new_tokens)
        result = evaluate_assistant(backend, args.suite)
        result.update(checkpoint=str(backend.checkpoint), training_mode=backend.state.get("training_mode", "pretraining"))
        from silivra.checkpoint import sha256
        result.update(model_sha256=sha256(backend.checkpoint / "model.safetensors"),
                      tokenizer_sha256=sha256(backend.checkpoint / "tokenizer.json"),
                      generation={"temperature": 0, "top_k": 0, "max_new_tokens": args.max_new_tokens})
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as output:
                json.dump(result, output, indent=2, ensure_ascii=False)
                output.write("\n")
        return result
    raise ValueError(f"Unknown assistant command: {args.command}")


def main():
    args = parser().parse_args()
    try:
        if args.command is None:
            if sys.stdin.isatty():
                from silivra.terminal import run_terminal
                raise SystemExit(run_terminal(workspace=args.startup_workspace, user=args.startup_user, run=args.startup_model))
            prompt = sys.stdin.read(32_001)
            if not prompt.strip():
                parser().print_help()
                return
            chat_args = ["chat", "--workspace", str(args.startup_workspace), "--user", args.startup_user, "--", prompt]
            if args.startup_model:
                chat_args[1:1] = ["--run", str(args.startup_model)]
            args = parser().parse_args(chat_args)
            result = _assistant_command(args)
            if result is not None:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            return
        if args.command == "doctor":
            from silivra.runtime import hardware_info
            result = hardware_info()
        elif args.command == "prepare":
            from silivra.data import prepare_data
            result = prepare_data(args.data, train_stories=args.train_stories,
                                  valid_stories=args.valid_stories, vocab_size=args.vocab_size)
        elif args.command == "train":
            from silivra.train import TrainConfig, train
            if args.config:
                settings = json.loads(args.config.read_text())
            elif args.resume:
                from silivra.checkpoint import resolve_checkpoint, verify_checkpoint
                settings = verify_checkpoint(resolve_checkpoint(args.run))["training_config"]
            else:
                settings = {}
            train(args.data, args.run, TrainConfig(**settings), resume=args.resume, max_minutes=args.max_minutes)
            return
        elif args.command == "export":
            from silivra.export import export_model
            result = export_model(args.run, args.output, which=args.which, archive=args.archive)
        elif args.command in ("chat", "tools", "tool", "memory", "prepare-instructions", "finetune", "evaluate-assistant"):
            result = _assistant_command(args)
            if result is None:
                return
        else:
            from silivra.runtime import require_apple_gpu
            from silivra.checkpoint import load_model, sha256, write_json
            require_apple_gpu()
            model, tokenizer, state, checkpoint = load_model(args.run, args.which)
            if args.command == "generate":
                from silivra.inference import complete
                result = complete(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens,
                                  temperature=args.temperature, top_k=args.top_k, seed=args.seed)
                if not args.json:
                    print(args.prompt + result["text"])
                    print(f"\n{result['generated_tokens']} tokens · {result['tokens_per_second']:.1f} tokens/s · "
                          f"MLX peak {result['mlx_peak_gib']:.2f} GiB", file=sys.stderr)
                    return
            elif args.command == "benchmark":
                from silivra.inference import benchmark
                result = benchmark(model, tokenizer, trials=args.trials, max_new_tokens=args.max_new_tokens)
                result.update(checkpoint=str(checkpoint), training_step=state["step"],
                              model_sha256=sha256(checkpoint / "model.safetensors"))
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    write_json(args.output, result)
            else:
                import math
                if not 1 <= args.batches <= 10000:
                    raise ValueError("batches must be between 1 and 10000.")
                if sha256(args.data / "manifest.json") != state["data_manifest_sha256"]:
                    raise ValueError("Evaluation data manifest differs from the training run.")
                if state.get("training_mode") == "instruction_sft":
                    from silivra.instructions import InstructionDataset
                    from silivra.finetune import evaluate_loss, _manifest
                    manifest = _manifest(args.data.resolve())
                    dataset = InstructionDataset(args.data / manifest["splits"]["valid"]["path"],
                                                 model.config.max_seq_len, model.config.vocab_size)
                    loss = evaluate_loss(model, dataset, state["training_config"]["batch_size"], args.batches)
                else:
                    from silivra.data import TokenDataset
                    from silivra.train import evaluate_loss
                    dataset = TokenDataset(args.data / "valid.npy", state["training_config"]["seq_len"],
                                           vocab_size=model.config.vocab_size)
                    loss = evaluate_loss(model, dataset, state["training_config"]["batch_size"], args.batches)
                result = {"validation_loss": loss, "perplexity": math.exp(min(loss, 80)),
                          "batches": args.batches, "checkpoint": str(checkpoint),
                          "training_mode": state.get("training_mode", "pretraining")}
        print(json.dumps(result, indent=2, sort_keys=True))
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"silivra: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
