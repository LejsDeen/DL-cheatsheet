.PHONY: compile watch preview clean fast

# Compile the cheat sheet to PDF
compile:
	typst compile cheat_sheet.typ cheat_sheet.pdf

# Fast compile (skip font compilation cache)
fast:
	typst compile --format pdf cheat_sheet.typ cheat_sheet.pdf

# Watch mode - automatically recompile on changes (FASTEST - incremental compilation)
# Use this for development! Only recompiles what changed.
watch:
	typst watch cheat_sheet.typ cheat_sheet.pdf

# Preview (compile and open PDF)
preview: compile
	open cheat_sheet.pdf

# Clean generated files and cache
clean:
	rm -f cheat_sheet.pdf
	rm -rf .typst/

