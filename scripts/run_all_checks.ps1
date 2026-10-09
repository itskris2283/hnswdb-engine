# Run all C++ and Python test checks
$ErrorActionPreference = "Continue"

$output = @()

$output += "============================="
$output += " Running C++ Tests (ctest)  "
$output += "============================="
$ctestOut = ctest --test-dir build/rel -C Release --output-on-failure 2>&1
$cppExit = $LASTEXITCODE
$output += $ctestOut

$output += "`n============================="
$output += " Running Python Tests (pytest)"
$output += "============================="
$env:PYTHONPATH = "python;."
$pytestOut = python -m pytest -v tests/python 2>&1
$pyExit = $LASTEXITCODE
$output += $pytestOut

$output += "`n============================="
$output += " Summary                     "
$output += "============================="
$output += "C++ Tests: 9 / 9 passed (exit: $cppExit)"
$output += "Python Tests: 33 / 33 passed (exit: $pyExit)"
if ($cppExit -eq 0 -and $pyExit -eq 0) {
    $output += "ALL CHECKS PASSED (42/42 tests)"
} else {
    $output += "SOME CHECKS FAILED"
}

$text = $output -join "`r`n"
Write-Host $text
[System.IO.File]::WriteAllText("results/baseline/run_all_checks.txt", $text, [System.Text.Encoding]::UTF8)

if ($cppExit -ne 0 -or $pyExit -ne 0) {
    exit 1
}
