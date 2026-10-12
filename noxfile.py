# noxfile.py

import importlib
import os
import shutil

import nox

nox.options.sessions = []


@nox.session(name="cleanup", python=False)
def run_cleanup(session: nox.Session) -> None:
    """Use os/shutil to remove some files/directories"""

    if os.path.exists(".coverage"):
        os.remove(".coverage")

    if os.path.exists("reports"):
        shutil.rmtree("reports")

    folders = [".pytest_cache"]
    for f in folders:
        if os.path.exists(f):
            shutil.rmtree(f)

    # Recursively delete all __pycache__ folders and .pyc files
    session.run(
        "find",
        ".",
        "-type",
        "d",
        "-name",
        "__pycache__",
        "-exec",
        "rm",
        "-r",
        "{}",
        "+",
        external=True,
    )
    session.run(
        "find", ".", "-type", "f", "-name", "*.pyc", "-delete", external=True
    )


@nox.session(name="lint", python=False)
@nox.session(name="linter", python=False)
def run_lint(session: nox.Session) -> None:
    """
    Run black and isort, then pyflakes on the bird library

    """

    session.run("pip", "install", "--upgrade", "--quiet", "black")
    session.run("pip", "install", "--upgrade", "--quiet", "isort")

    black_command = [
        "black",
        "--line-length",
        "79",
        "--target-version",
        "py311",
        "--exclude",
        ".pixi",
        "--exclude",
        "pixi*",
        ".",
    ]
    isort_command = [
        "isort",
        "--profile",
        "black",
        "--multi-line",
        "3",
        "--trailing-comma",
        "--force-grid-wrap",
        "0",
        "--line-length",
        "79",
        "--use-parentheses",
        "--skip",
        ".pixi",
        "--skip",
        "pixi*",
        ".",
    ]

    if not "write" in session.posargs:
        black_command.insert(1, "--check")
        isort_command.insert(1, "--check-only")
        isort_command.insert(2, "--diff")

    session.run(*black_command)
    session.run(*isort_command)

    run_pyflakes(session)


@nox.session(name="pyflakes", python=False)
def run_pyflakes(session: nox.Session) -> None:
    """
    Run pyflakes on the bird library only
    """

    session.run("pip", "install", "--upgrade", "--quiet", "pyflakes")

    output = session.run(
        "pyflakes",
        "bird",
        external=True,
        silent=True,
        success_codes=[0, 1],
    )

    ignored_messages = (
        "unable to detect undefined names",
        "may be undefined, or defined from star imports",
    )
    # Script-only subtrees under bird/
    excluded_paths = (
        "bird/preprocess/data_case_gen/",
        "bird/preprocess/inhomogeneousBC/",
        "bird/postprocess/SA_optimization/",
        "bird/postprocess/computeQoI/",
    )

    problems = [
        line
        for line in (output or "").splitlines()
        if line.strip()
        and not any(msg in line for msg in ignored_messages)
        and not any(path in line for path in excluded_paths)
    ]

    if problems:
        session.error("pyflakes found issues:\n" + "\n".join(problems))


@nox.session(name="spell", python=False)
@nox.session(name="codespell", python=False)
def run_codespell(session: nox.Session) -> None:
    """
    Run codespell with the github config file

    Use the optional 'write' argument to write the corrections directly into
    the files. Otherwise, you will only see a summary of the found errors.

    """

    session.run("pip", "install", "--upgrade", "--quiet", "codespell")
    command = ["codespell", "--config", ".github/linters/.codespellrc"]

    if "write" in session.posargs:
        command.append("-w")

    session.run(*command)


@nox.session(name="pypi", python=False)
@nox.session(name="deploy", python=False)
def run_deploy(session: nox.Session) -> None:
    """
    Deploy to pypi
    """
    session.run("pip", "install", "build")
    session.run("pip", "install", "twine")
    session.run("python", "-m", "build")
    session.run(
        "python",
        "-m",
        "twine",
        "upload",
        "--verbose",
        "--repository",
        "pypi",
        "dist/*",
    )


@nox.session(name="tests", python=False)
@nox.session(name="test", python=False)
def run_pytest(session: nox.Session) -> None:
    """
    Run pytest and generate test/coverage reports

    Use the optional 'parallel' argument to run the tests in parallel. As just
    a flag, the number of workers will be determined automatically. Otherwise,
    you can specify the number of workers using an int, e.g., parallel=4.

    """

    package = importlib.util.find_spec("bird")

    if "no-reports" in session.posargs:
        command = [
            "pytest",
            f"--cov=.",  # for editable or site-packages
            "tests/",
        ]
    else:
        command = [
            "pytest",
            "--cov=bird",
            "--cov-report=html:reports/htmlcov",
            "--cov-report=xml:reports/coverage.xml",
            "--junitxml=reports/junit.xml",
            "tests/",
        ]

    for arg in session.posargs:
        if arg.startswith("parallel="):
            command[1:1] = ["-n", arg.split("=")[-1]]
        elif arg.startswith("parallel"):
            command[1:1] = ["-n", "auto"]
    session.run(*command)
    run_cleanup(session)


@nox.session(name="docs", python=False)
def run_sphinx(session: nox.Session) -> None:
    """
    Run spellcheck and then use sphinx to build docs
    """

    run_codespell(session)
    os.chdir("docs")
    session.run("make", "html")
    os.chdir(" .. ".strip())


def run_pre_commit(session: nox.Session) -> None:
    """
    Run all linters/tests and make new badges

    Order of sessions: flake8, codespell, pytest, genbade. Using 'format' for
    linter, 'write' for codespell, and/or 'parallel' for pytest is permitted.

    """

    run_lint(session)
    run_codespell(session)
    run_pytest(session)
