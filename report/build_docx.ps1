# Convert the report HTML to .docx with Pandoc.
# Run from anywhere:  pwsh report/build_docx.ps1
#   reference.docx  - Word styles (fonts, headings, table borders, margins); rebuild with make_reference_docx.py
#   pagebreak.lua   - turns <div class="page-break"> into real page breaks
Push-Location $PSScriptRoot
try {
    pandoc Checkpoint_FoliageCheckGrp1.html `
        --from html --to docx `
        --reference-doc reference.docx `
        --lua-filter pagebreak.lua `
        --output Checkpoint_FoliageCheckGrp1.docx
    if ($LASTEXITCODE -ne 0) { throw "pandoc failed with exit code $LASTEXITCODE" }
    Write-Host "Wrote $PSScriptRoot\Checkpoint_FoliageCheckGrp1.docx"
}
finally {
    Pop-Location
}
