#!/bin/bash

# Script to synchronize bioast_dataset1 from remote server to local machine
# Usage: ./sync_bioast_dataset.sh [remote_user@remote_host] [remote_path] [local_destination_path]

# Default values
REMOTE_USER_HOST=${1:-"user@remote_server"}
REMOTE_PATH=${2:-"/path/to/rtx/bioastModel/bioast_dataset1"}
LOCAL_DEST=${3:-"./local_bioast_dataset1"}

# Create local destination directory if it doesn't exist
mkdir -p "$LOCAL_DEST"

echo "Starting synchronization of bioast_dataset1 from $REMOTE_USER_HOST:$REMOTE_PATH to $LOCAL_DEST..."

# Use rsync for efficient synchronization (install with: sudo apt-get install rsync)
# -a: archive mode (preserves permissions, timestamps, etc.)
# -v: verbose
# -z: compress data during transfer
# --progress: show progress during transfer
echo "Synchronizing dataset using rsync..."
rsync -avz --progress "$REMOTE_USER_HOST:$REMOTE_PATH/" "$LOCAL_DEST/"

# Alternative: Use scp if rsync is not available
# echo "Synchronizing dataset using scp..."
# scp -r "$REMOTE_USER_HOST:$REMOTE_PATH/" "$LOCAL_DEST/"

# Copy the training_completion_summary.md file from remote if it exists there
echo "Copying training_completion_summary.md..."
scp "$REMOTE_USER_HOST:/path/to/rtx/bioastModel/training_completion_summary.md" "$LOCAL_DEST/" || echo "Could not copy summary file from remote, using local copy instead" && cp "training_completion_summary.md" "$LOCAL_DEST/" 2>/dev/null || echo "No summary file found"

echo "Synchronization complete!"
echo "Dataset synchronized to: $LOCAL_DEST"
echo ""
echo "Usage examples:"
echo "  ./sync_bioast_dataset.sh username@server.com /home/username/rtx/bioastModel/bioast_dataset1 ./my_local_dataset"
echo "  ./sync_bioast_dataset.sh"
