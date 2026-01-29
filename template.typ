#let project(title: "", authors: (), date: none, body) = {
  // Set the document's basic properties.
  set document(author: authors.map(a => a.name), title: title)
  set page(numbering: "1", number-align: center, flipped: true, margin: 0.8em)

  set text(font: "Libertinus Serif")

  // Set paragraph spacing.

  set heading(numbering: "1.1")
  set par(leading: 0.48em)
  
  show: columns.with(5, gutter: 0.8em)

  // Main body.
  set par(justify: true)
  // set text(size:0.92em)
  set text(size: 0.72em)

  // Display title, author, and date
  align(center)[
    #if title != "" {
      text(size: 1em, weight: "bold")[#title]
      linebreak()
    }
    
    #if authors.len() > 0 {
      text(size: 1em)[
        #authors.map(a => a.name).join(", ")
      ]
      linebreak()
    }
    
    #if date != none {
      text(size: 0.9em, style: "italic")[#date]
    }
  ]
  
  v(1em)

  body
}

#let colorbox(title: none, inline: true, breakable: true, color: blue, content) = {
  let colorOutset = 3pt
  let titleContent = if title != none {
    box(
      fill: silver,
      outset: (left: colorOutset - 1pt, rest: colorOutset),
      width: if inline { auto } else { 100% },
      radius: if inline { (bottom-right: 4pt) } else { 0pt },
      [*#title*]) + if inline { h(6pt) }
  }

  block(
    stroke: (left: 2pt + color),
    outset: colorOutset, 
    fill: silver.lighten(60%), 
    breakable: breakable,
    width: 100%,
    
    titleContent + content)
}

// Inline "fit width": scale content so it stays on the current row (no new line).
// Uses a fixed scale factor. Typst's layout() forces block-level layout, so we
// avoid it; true "fit to remaining line width" isn't possible inline.
#let fitWidth(content, factor: 88%) = scale(reflow: true, x: factor, y: factor, content)