#!/bin/bash
set -e

if [ -z "$LOCAL_UID" ]; then
    echo "Running as root."
    exec "$@" # no local set continue as root
else
    echo "Running as user: $LOCAL_UID"
    for DIR in /workspace/data /workspace/models /workspace/results; do
        if [ -d "$DIR" ]; then
            chown -R "$LOCAL_UID" "$DIR"
        fi
    done

    exec gosu "$LOCAL_UID" "$@" # dropping root
fi