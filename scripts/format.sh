#!/usr/bin/env bash

echo "Applying black..."
./scripts/fix-black.sh
echo "Done"

echo "Applying isort..."
./scripts/fix-isort.sh
echo "Done"

echo "Finished formatting"
