#!/bin/bash

set -eu

setup_python() {
    declare -r venv="./env"
    python3 -m venv "$venv"
    source "$venv/bin/activate"

    pip install -r requirements.txt
}

setup_graphviz() {
    brew install graphviz
}

main() {
    setup_python
    setup_graphviz
}

main "$@"