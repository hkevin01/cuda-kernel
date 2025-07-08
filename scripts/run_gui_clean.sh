#!/bin/bash

# Simple wrapper to filter out Qt SharedSignalPool warnings
# while preserving all other stderr output

exec 2> >(grep -v "Warning: Resource leak detected by SharedSignalPool" >&2)

# Execute the GUI with the original arguments
exec "$@"
