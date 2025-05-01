import argparse
import subprocess
import os
import sys


def get_project_root():
    """Gets the absolute path of the directory where this script resides."""
    return os.path.dirname(os.path.abspath(__file__))


def run_script(script_path: str, project_dir: str, script_args: list):
    """Runs a Python script in the specified project directory."""
    command = [sys.executable, script_path] + script_args
    print(f"Executing: {' '.join(command)} in directory: {project_dir}")
    try:
        # Ensure the script path is absolute or relative to the project_dir
        if not os.path.isabs(script_path):
            script_path_abs = os.path.join(project_dir, script_path)
        else:
            script_path_abs = script_path  # Already absolute? Maybe log warning if not within project_dir

        # Construct command with the potentially adjusted script path
        command = [sys.executable, script_path_abs] + script_args

        result = subprocess.run(
            command, cwd=project_dir, check=True, text=True, capture_output=True
        )
        print(f"--- Script Output ({os.path.basename(script_path)}) ---")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("--- Errors/Warnings ---")
            print(result.stderr)
        print("--- End Script Output ---")
        print(f"Successfully executed: {os.path.basename(script_path)}")
    except FileNotFoundError:
        print(
            f"Error: Script not found at '{script_path_abs}'. Make sure the path is correct relative to '{project_dir}'."
        )
    except subprocess.CalledProcessError as e:
        print(f"Error executing script: {os.path.basename(script_path)}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("--- Output ---")
            print(e.stdout)
        if e.stderr:
            print("--- Errors ---")
            print(e.stderr)
    except Exception as e:
        print(
            f"An unexpected error occurred while running {os.path.basename(script_path)}: {e}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Launcher for RAG and ZeroShot projects."
    )
    parser.add_argument(
        "project",
        choices=["rag", "zeroshot"],
        help="The project to launch tasks for.",
        metavar="PROJECT",
    )
    parser.add_argument(
        "task",
        help="The task to run within the selected project.",
        metavar="TASK",
        # Choices are dynamic based on project, validated later
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass directly to the target script (e.g., --model-name xyz).",
        metavar="SCRIPT_ARGS",
    )

    args = parser.parse_args()

    project_root = get_project_root()
    rag_dir = os.path.join(project_root, "p_llm_manual", "RAG")
    zeroshot_dir = os.path.join(project_root, "ZeroShot")  # Corrected path

    # --- Task Definitions ---
    rag_tasks = {
        "run_tests": "rag_tester.py",  # Assumes rag_tester.py exists
        "visualize": os.path.join(
            "visualization", "plot_scripts", "main_visualization.py"
        ),
        "ask": "ask_question.py",  # Assumes ask_question.py exists (from notebook)
        "create_db": "create_databases.py",  # Assumes create_databases.py exists
        # Add other RAG tasks as needed
    }

    zeroshot_tasks = {
        "run_tests": "llm_tester.py",
        "evaluate": "llm_tester.py",  # Evaluation is often part of the main test script run
        "reformat_results": os.path.join("utils", "reformat_zeroshot_results.py"),
        "check_integrity": os.path.join("utils", "results_integrity_checker.py"),
        # Add other ZeroShot tasks as needed
    }

    # --- Select Project and Task ---
    script_rel_path = None
    project_dir = None

    if args.project == "rag":
        project_dir = rag_dir
        if args.task in rag_tasks:
            script_rel_path = rag_tasks[args.task]
        else:
            print(f"Error: Unknown task '{args.task}' for project 'rag'.")
            print(f"Available RAG tasks: {', '.join(rag_tasks.keys())}")
            sys.exit(1)

    elif args.project == "zeroshot":
        project_dir = zeroshot_dir
        if args.task in zeroshot_tasks:
            script_rel_path = zeroshot_tasks[args.task]
            # Special handling if evaluate needs specific flags (though current llm_tester doesn't support it)
            if args.task == "evaluate":
                print(
                    "Note: 'evaluate' task for ZeroShot currently runs the full 'llm_tester.py'."
                )
                # If llm_tester.py were modified to accept an --evaluate flag:
                # args.script_args.insert(0, '--evaluate')
                pass

        else:
            print(f"Error: Unknown task '{args.task}' for project 'zeroshot'.")
            print(f"Available ZeroShot tasks: {', '.join(zeroshot_tasks.keys())}")
            sys.exit(1)

    else:
        # Should not happen due to argparse choices, but good practice
        print(f"Error: Unknown project '{args.project}'.")
        sys.exit(1)

    # --- Execute Script ---
    if script_rel_path and project_dir:
        script_abs_path = os.path.join(project_dir, script_rel_path)
        run_script(script_abs_path, project_dir, args.script_args)
    else:
        # Should already be handled by earlier checks, but acts as a safeguard
        print("Error: Could not determine script path or project directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
