# Exam Cheat Sheet

This cheat sheet is written in Typst and can be compiled to PDF.

## Setup

### Install Typst CLI

Install Typst CLI (e.g. via brew):
```bash
brew install typst
```

You can then verify it with:
```bash
typst --version
```

### Syntax Highlighting in Cursor/VS Code

For syntax highlighting and better Typst support:

1. **Install the Tinymist extension:**
   - Open the Extensions view in Cursor (Cmd+Shift+X)
   - Search for "Typst" or "Tinymist"
   - Install the **"Typst"** extension by Myriad-Dreamin (or search for `tinymist.tinymist`)
   - Alternatively, install via command line:
     ```bash
     code --install-extension myriad-dreamin.tinymist
     ```

2. **Note:** Cursor may have some compatibility issues with VS Code extensions. If the extension doesn't work perfectly, you can still use basic syntax highlighting by ensuring `.typ` files are recognized.

The extension provides:
- ✅ Syntax highlighting
- ✅ Code completion
- ✅ Error checking
- ✅ Live preview (if configured)

## Usage

### ⚡ Watch mode (RECOMMENDED - Fastest!)
**Use this for development!** It uses incremental compilation - only recompiles what changed, making subsequent compiles much faster.
```bash
make watch
# or directly:
typst watch cheat_sheet.typ cheat_sheet.pdf
```
Keep this running in a terminal while editing. The PDF will auto-update on save!

### Compile to PDF (one-time)
```bash
make compile
# or directly:
typst compile cheat_sheet.typ cheat_sheet.pdf
```

### Fast compile (optimized)
```bash
make fast
```

### Preview (compile and open)
```bash
make preview
```

### Clean generated files
```bash
make clean
```

## Typst Resources

- [Typst Documentation](https://typst.app/docs/)
- [Typst Tutorial](https://typst.app/docs/tutorial/)
- [Typst Reference](https://typst.app/docs/reference/)

## Tips for Cheat Sheets

1. **Small font sizes**: Use `#set text(size: 8pt)` or smaller for maximum content
2. **Minimal margins**: Use `#set page(margin: (x: 0.5cm, y: 0.5cm))`
3. **Compact spacing**: Reduce leading and paragraph spacing
4. **Columns**: Use `#grid()` for multi-column layouts
5. **Tables**: Great for formulas, definitions, or structured info
6. **Equations**: Use `$ ... $` for inline math, `$ block(..) $` for display math

The CLI approach (using `make watch`) works great - just save your `.typ` file and the PDF updates automatically!

