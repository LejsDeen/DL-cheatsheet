# Exam Cheat Sheet

This cheat sheet is written in Typst and can be compiled to PDF.

## Setup

Typst CLI is already installed. You can verify it with:
```bash
typst --version
```

## Usage

### Compile to PDF
```bash
make compile
# or directly:
typst compile cheat_sheet.typ cheat_sheet.pdf
```

### Watch mode (auto-recompile on changes)
```bash
make watch
# or directly:
typst watch cheat_sheet.typ cheat_sheet.pdf
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

