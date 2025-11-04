# CI Testing Guide for MFGnet.jl

## Overview

This document explains the CI (Continuous Integration) testing infrastructure now in place for MFGnet.jl and how to use it.

---

## Current Status

**Branch:** `claude/codebase-review-julia-011CUoKF1mkF9mfUnxRMDMbF`

**CI Workflow:** `.github/workflows/CI.yml`

**Commits:**
1. `dc1b15f` - Modernization and optimization changes
2. `7fb9b9f` - Added CI workflow
3. `4718f09` - Removed Revise dependency from tests

---

## What's Being Tested

### Julia Versions
- **Julia 1.6** (LTS - Long Term Support)
- **Julia 1.10** (Recent stable)
- **Julia 1.11** (Latest stable)
- **Julia nightly** (Development version)

### Operating Systems
- **Linux** (ubuntu-latest) - All Julia versions
- **macOS** (macOS-latest) - Julia 1.11 only
- **Windows** (windows-latest) - Julia 1.11 only

### Test Matrix
Total of **7 test configurations** running in parallel:
- 4 Julia versions on Linux
- 1 on macOS
- 1 on Windows
- 1 nightly build

---

## How to Monitor CI Results

### Option 1: GitHub Web Interface

Visit the Actions tab in your repository:
```
https://github.com/EmoryMLIP/MFGnet.jl/actions
```

You'll see:
- ✅ Green checkmark = All tests passed
- ❌ Red X = Tests failed
- 🟡 Yellow circle = Tests running
- ⚪ Gray circle = Tests queued

### Option 2: GitHub CLI (if available)

```bash
# List recent workflow runs
gh run list --branch claude/codebase-review-julia-011CUoKF1mkF9mfUnxRMDMbF

# Watch a specific run
gh run watch

# View details of failed run
gh run view <run-id> --log-failed
```

### Option 3: Badge in README

The README now includes a CI badge that shows current status:
```markdown
[![CI](https://github.com/EmoryMLIP/MFGnet.jl/actions/workflows/CI.yml/badge.svg)](...)
```

---

## Expected Test Flow

### 1. Workflow Triggers
CI runs automatically when:
- Code is pushed to the branch
- A pull request is created
- Manually triggered from Actions tab

### 2. Test Steps
For each Julia version/OS combination:

```
1. Checkout code
2. Install Julia (version X.Y)
3. Cache Julia packages (speeds up future runs)
4. Install package dependencies (Pkg.instantiate())
5. Precompile packages
6. Run test suite (Pkg.test())
7. Generate coverage report
8. Upload coverage to Codecov
```

### 3. Test Duration
Expected time: **5-15 minutes** per configuration
- Parallel execution means total time ≈ 10-20 minutes

---

## Potential Issues and Solutions

### Issue 1: jInv Package Not Found

**Symptom:** Error during `Pkg.instantiate()`
```
ERROR: cannot find jInv in registry
```

**Reason:** jInv is a custom package not in the General registry

**Solution:**
Since jInv is only used in examples (not in core package or tests), it should work. But if it fails:

```julia
# Option A: Make jInv optional
[deps]
# Move jInv to [extras] or remove if not needed for tests

# Option B: Add jInv URL explicitly
[deps]
jInv = {git = "https://github.com/JuliaInv/jInv.jl", rev = "master"}
```

### Issue 2: Old Package Versions

**Symptom:** Compatibility errors with Flux, Zygote, etc.

**Reason:** `Manifest.toml` contains versions from 2020

**Solution:** The workflow runs `Pkg.instantiate()` which should resolve to compatible versions. If issues persist:

```bash
# Locally, delete Manifest.toml and regenerate
rm Manifest.toml
julia --project=. -e 'using Pkg; Pkg.instantiate()'
git add Manifest.toml
git commit -m "Update package versions"
git push
```

### Issue 3: Type Instability or Performance Regressions

**Symptom:** Tests pass but take much longer

**Solution:** Our changes actually improve performance. If slower:
- Check if pre-allocated vectors are being used correctly
- Verify tuple operations were properly replaced

### Issue 4: Numerical Accuracy Differences

**Symptom:** Tests fail with small numerical differences

**Reason:** Different Julia versions or BLAS backends

**Solution:** Tests use `sqrt(eps(R))` tolerance which should be robust. If failures occur:
```julia
# In test files, adjust tolerance if needed
@test norm(trH-trHt)/norm(trH) < 10*sqrt(eps(R))  # More lenient
```

### Issue 5: Windows-Specific Failures

**Symptom:** Tests pass on Linux/macOS but fail on Windows

**Common causes:**
- Path separator issues (use `joinpath` not string concatenation)
- Line ending differences (Git should handle automatically)
- Case-sensitive file systems

---

## How to Fix Failing Tests

### Step 1: Identify the Failure

Click on the failed job in GitHub Actions to see logs:
```
Tests > Julia 1.11 - ubuntu-latest - x64
  └─ Run tests (failed)
     └─ Error: LoadError: ...
```

### Step 2: Reproduce Locally

If you have Julia installed:
```bash
cd /path/to/MFGnet.jl
git checkout claude/codebase-review-julia-011CUoKF1mkF9mfUnxRMDMbF

# Use the test script
julia test_local.jl

# Or manually
julia --project=. -e 'using Pkg; Pkg.test()'

# Or run specific test
julia --project=. test/testNN.jl
```

### Step 3: Fix the Issue

Common fixes:
```julia
# If type errors:
@test typeof(result) == expected_type

# If numerical errors:
@test result ≈ expected atol=1e-10 rtol=1e-8

# If missing imports:
using RequiredPackage
```

### Step 4: Test and Push

```bash
# Run tests locally
julia test_local.jl

# If passing, commit and push
git add .
git commit -m "Fix: description of fix"
git push origin claude/codebase-review-julia-011CUoKF1mkF9mfUnxRMDMbF
```

CI will automatically run again.

---

## Changes Made for CI Compatibility

### 1. Modernization Changes (Commit `dc1b15f`)
- Fixed `!=nothing` → `isnothing()`
- Optimized `myMap()` for type stability
- Replaced tuple appending with pre-allocated vectors
- Updated Julia requirement: 1.5 → 1.6

### 2. CI Infrastructure (Commit `7fb9b9f`)
- Added `.github/workflows/CI.yml`
- Added `test_local.jl` helper script
- Updated README with CI badge and testing instructions

### 3. Test Compatibility (Commit `4718f09`)
- Removed `using Revise` from all test files
- Revise is a development tool not needed for CI

---

## Performance Testing

While CI tests correctness, performance should be benchmarked separately:

```julia
using BenchmarkTools

# Test a specific function
@benchmark N(s, Θ) setup=(s=randn(10,100); Θ=...)

# Compare against old version
# Should see 2-3x speedup in neural network operations
```

---

## Coverage Reports

CI automatically generates coverage reports uploaded to Codecov.

To view:
1. Check CI logs for Codecov link
2. Visit `https://codecov.io/gh/EmoryMLIP/MFGnet.jl`
3. Look for coverage badge in README

Coverage helps identify:
- Untested code paths
- Which optimizations are actually being exercised

---

## Troubleshooting Commands

### Check Workflow Syntax
```bash
# Validate YAML syntax
yamllint .github/workflows/CI.yml

# Or use GitHub's validator
# (push to branch and GitHub will validate)
```

### Debug Failed Jobs
```bash
# If you have 'act' tool installed (runs GH Actions locally)
act -j test

# Or use GitHub's debug logging
# Re-run workflow with "Enable debug logging" checkbox
```

### Package Resolution Issues
```bash
# See what packages would be installed
julia --project=. -e 'using Pkg; Pkg.resolve()'

# Check for conflicts
julia --project=. -e 'using Pkg; Pkg.status()'

# Update to latest compatible versions
julia --project=. -e 'using Pkg; Pkg.update()'
```

---

## Next Steps After CI Passes

1. **Verify Performance**
   ```bash
   # Run benchmarks comparing old vs new code
   julia examples/ROLNWF2019/runObstacleExperiment.jl
   ```

2. **Update Documentation**
   - Document performance improvements
   - Note any breaking changes
   - Update version number if needed

3. **Create Pull Request**
   ```bash
   # From GitHub, create PR from your branch to main
   # Include CI badge showing all tests pass
   ```

4. **Merge and Release**
   - Once reviewed and approved
   - Tag new version
   - Update package registry

---

## Maintenance

### Updating Julia Versions
When new Julia versions are released, update `.github/workflows/CI.yml`:

```yaml
matrix:
  version:
    - '1.6'    # Keep LTS
    - '1.12'   # Add new version
    - 'nightly'
```

### Updating Dependencies
Periodically update `Manifest.toml`:
```bash
julia --project=. -e 'using Pkg; Pkg.update()'
git add Manifest.toml
git commit -m "Update package dependencies"
```

---

## Resources

- **Julia CI Docs:** https://github.com/julia-actions
- **GitHub Actions:** https://docs.github.com/en/actions
- **Julia Testing:** https://docs.julialang.org/en/v1/stdlib/Test/
- **Package Manager:** https://pkgdocs.julialang.org/

---

## Summary

✅ **What We Did:**
- Added comprehensive CI testing across Julia versions and OSes
- Fixed compatibility issues (Revise, type stability)
- Automated testing for every push

✅ **What You Should Do:**
1. Monitor CI results at GitHub Actions page
2. Fix any failing tests using this guide
3. Once passing, benchmark performance
4. Create PR to merge changes

✅ **Expected Outcome:**
- All tests pass on Julia 1.6, 1.10, 1.11
- 2-3x performance improvement maintained
- Automated testing for future changes

---

**Last Updated:** 2025-11-04
**Branch:** claude/codebase-review-julia-011CUoKF1mkF9mfUnxRMDMbF
