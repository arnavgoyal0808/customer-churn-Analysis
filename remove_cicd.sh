#!/bin/bash

# Script to remove CI/CD pipeline from the project
# Run this if you prefer to remove CI/CD entirely

echo "🗑️  Removing CI/CD pipeline..."

# Remove GitHub workflows directory
if [ -d ".github/workflows" ]; then
    rm -rf .github/workflows
    echo "✅ Removed .github/workflows directory"
else
    echo "ℹ️  .github/workflows directory not found"
fi

# Remove the entire .github directory if it's empty
if [ -d ".github" ] && [ -z "$(ls -A .github)" ]; then
    rm -rf .github
    echo "✅ Removed empty .github directory"
fi

# Remove development dependencies file
if [ -f "requirements-dev.txt" ]; then
    rm requirements-dev.txt
    echo "✅ Removed requirements-dev.txt"
fi

# Remove CI/CD documentation
if [ -f "CICD_FIXES.md" ]; then
    rm CICD_FIXES.md
    echo "✅ Removed CICD_FIXES.md"
fi

echo "🎉 CI/CD pipeline has been completely removed from the project!"
echo "📝 You may want to update your README.md to remove any CI/CD badges or references."
