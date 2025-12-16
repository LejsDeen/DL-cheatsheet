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

## GitHub Setup

This repository is already initialized with git. To connect it to GitHub:

1. **Create a new repository on GitHub** (don't initialize it with README, .gitignore, or license)

2. **Connect your local repo to GitHub:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

3. **For future updates:**
   ```bash
   git add .
   git commit -m "Your commit message"
   git push
   ```

## Collaboration

### For Collaborators

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. **Install Typst CLI** (if not already installed):
   ```bash
   # macOS
   brew install typst
   
   # Linux
   # Follow instructions at https://github.com/typst/typst
   ```

3. **Make changes and compile:**
   - Edit `cheat_sheet.typ`
   - Run `make watch` to auto-compile
   - Preview the PDF to verify changes

4. **Contribute changes:**
   ```bash
   git add .
   git commit -m "Description of your changes"
   git push
   ```

### Collaboration Tips

- **Use branches for major changes:**
  ```bash
  git checkout -b feature/new-section
  # Make your changes
  git push -u origin feature/new-section
  # Then create a Pull Request on GitHub
  ```

- **Resolve merge conflicts:** If multiple people edit the same section, you may need to manually merge changes in `cheat_sheet.typ`

- **Compile before committing:** Make sure your changes compile successfully (`make compile`)

