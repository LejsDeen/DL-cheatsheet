.PHONY: compile watch preview clean

# Compile the cheat sheet to PDF
compile:
	typst compile cheat_sheet.typ cheat_sheet.pdf

# Watch mode - automatically recompile on changes
watch:
	typst watch cheat_sheet.typ cheat_sheet.pdf

# Preview in browser (opens PDF automatically)
preview: compile
	open cheat_sheet.pdf

# Clean generated files
clean:
	rm -f cheat_sheet.pdf

