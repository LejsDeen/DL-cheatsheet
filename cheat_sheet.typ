#set page(margin: (x: 0.5cm, y: 0.5cm))
#set text(size: 8pt, font: "Times New Roman")
#set par(justify: false, leading: 0.65em)
#set heading(numbering: none)

#let exam-title = "Exam Cheat Sheet"
#let course = "Course Name"

#pagebreak()
#text(weight: "bold", size: 10pt)[
  #exam-title
]
#text(size: 8pt)[
  #course - Date: _Date here_
]

#pagebreak()

// Section 1
#align(center)[
  #text(weight: "bold", size: 9pt)[Section 1: Topic Name]
]

Your content goes here. Use:
- Bullet points for quick reference
- Equations: $ x = (-b ± sqrt(b^2 - 4 a c)) / (2 a) $
- Code snippets: `inline code`
- Tables for structured data

#pagebreak()

// Section 2
#align(center)[
  #text(weight: "bold", size: 9pt)[Section 2: Another Topic]
]

More content here...

// Add more sections as needed
#pagebreak()

// Useful Commands Reference
#align(center)[
  #text(weight: "bold", size: 9pt)[Quick Reference]
]

#grid(
  columns: 2,
  gutter: 0.3cm,
  [
    // Column 1
    [Command 1: Description]
    [Command 2: Description]
    [Command 3: Description]
  ],
  [
    // Column 2
    [Command 4: Description]
    [Command 5: Description]
    [Command 6: Description]
  ]
)

