#!/bin/bash

# Project Organization Verification Script
# Verifies that all files have been properly organized

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_status "Verifying project organization..."

# Check that no loose files remain in root
print_status "Checking for loose files in project root..."
loose_files=$(find . -maxdepth 1 -type f -name "*.sh" ! -name "run.sh" | wc -l)
loose_logs=$(find . -maxdepth 1 -type f -name "*.log" | wc -l)
loose_md=$(find . -maxdepth 1 -type f -name "*STATUS*.md" -o -name "*SUMMARY*.md" | wc -l)

if [ "$loose_files" -eq 0 ]; then
    print_success "No loose .sh files in root (except run.sh)"
else
    print_error "Found $loose_files loose .sh files in root"
    find . -maxdepth 1 -type f -name "*.sh" ! -name "run.sh"
fi

if [ "$loose_logs" -eq 0 ]; then
    print_success "No loose .log files in root"
else
    print_error "Found $loose_logs loose .log files in root"
    find . -maxdepth 1 -type f -name "*.log"
fi

if [ "$loose_md" -eq 0 ]; then
    print_success "No loose status/summary .md files in root"
else
    print_error "Found $loose_md loose status/summary .md files in root"
    find . -maxdepth 1 -type f -name "*STATUS*.md" -o -name "*SUMMARY*.md"
fi

# Check organized directories
print_status "Verifying organized directories..."

# Check logs directory
if [ -d "logs" ]; then
    log_count=$(find logs/ -name "*.log" | wc -l)
    print_success "Found logs/ directory with $log_count log files"
else
    print_error "logs/ directory not found"
fi

# Check scripts organization
if [ -d "scripts" ]; then
    total_scripts=$(find scripts/ -name "*.sh" | wc -l)
    build_scripts=$(find scripts/build/ -name "*.sh" 2>/dev/null | wc -l)
    test_scripts=$(find scripts/testing/ -name "*.sh" 2>/dev/null | wc -l)
    gui_scripts=$(find scripts/gui/ -name "*.sh" 2>/dev/null | wc -l)
    verify_scripts=$(find scripts/verification/ -name "*.sh" 2>/dev/null | wc -l)
    
    print_success "Found scripts/ directory with $total_scripts total scripts"
    print_success "  - Build scripts: $build_scripts"
    print_success "  - Test scripts: $test_scripts"
    print_success "  - GUI scripts: $gui_scripts"
    print_success "  - Verification scripts: $verify_scripts"
else
    print_error "scripts/ directory not found"
fi

# Check docs directory
if [ -d "docs" ]; then
    doc_count=$(find docs/ -name "*.md" | wc -l)
    print_success "Found docs/ directory with $doc_count documentation files"
else
    print_error "docs/ directory not found"
fi

# Verify main launcher exists
if [ -f "run.sh" ]; then
    print_success "Main launcher run.sh exists in root"
else
    print_error "Main launcher run.sh not found in root"
fi

# Check essential directories
essential_dirs=("src" "gui" "tests" ".vscode")
for dir in "${essential_dirs[@]}"; do
    if [ -d "$dir" ]; then
        print_success "Essential directory $dir exists"
    else
        print_warning "Essential directory $dir not found"
    fi
done

print_status "Organization verification complete!"

# Summary
echo ""
echo "=== PROJECT ORGANIZATION SUMMARY ==="
echo "Root directory files: $(find . -maxdepth 1 -type f | wc -l)"
echo "Root directory folders: $(find . -maxdepth 1 -type d ! -name "." | wc -l)"
echo "Total scripts organized: $(find scripts/ -name "*.sh" 2>/dev/null | wc -l)"
echo "Total logs organized: $(find logs/ -name "*.log" 2>/dev/null | wc -l)"
echo "Total docs: $(find docs/ -name "*.md" 2>/dev/null | wc -l)"
echo "=================================="
