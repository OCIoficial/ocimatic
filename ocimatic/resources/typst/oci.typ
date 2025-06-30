#let parbox(width: auto, content) = {
  box(width: width, baseline: 20%, height: 1em, align(horizon + center, content))
}

#let year = datetime.today().year()

#let read-input(key) = {
  sys.inputs.at(key, default: "").trim()
}

#let render-phase(with-year: false) = {
  let phase = read-input("OCIMATIC_PHASE")
  if phase != "" {
    phase
    if with-year {
      [ #year]
    }
  } else {
    [`<`#parbox(`filled by ocimatic`)`>`]
  }
}

#let base(doc) = {
 let logo = box(image(height: 34pt, "logo.png"))
  set page(
    "us-letter",
    margin: (top: 1.77in),
    header-ascent: 0pt,
    header: stack(
      [#logo#h(1fr)#render-phase(with-year: true)],
      v(0.5em),
      line(length: 100%, stroke: rgb(30%,30%,30%)),
      v(0.3in)
    ),
  )
  set par(leading: 0.55em, spacing: 1.1em, justify: true)
  set text(font: "New Computer Modern", lang: "es", region: "cl", size: 11pt)
  show emph: it => {
    set text(font: "New Computer Modern", style: "italic")
    it.body
  }
  show heading: set block(above: 1.4em, below: 1em)
  doc
}

#let titlepage(doc) = {
  set text(font: "New Computer Modern", lang: "es", region: "cl", size: 11pt)
  doc
}

#let general(doc) = {
  show: base
  doc
}

#let problem(title: "", doc) = {
  show: base

  let sf(content) = {
    set text(font: "New Computer Modern Sans")
    content
  }

  let problem-number = {
    let num = read-input("OCIMATIC_PROBLEM_NUMBER")
    if num != "" {
      num
    } else {
      [`<`#parbox(width: 1.5em, text(size: 7pt, `filled by ocimatic`))`>`]
    }
  }

  let codename = {
    let codename = read-input("OCIMATIC_CODENAME" )
    if codename != "" {
      raw(codename)
    } else {
      [`<`#parbox(text(size: 9pt, `filled by ocimatic`))`>`]
    }
  }

  {
    set align(center)
    stack(
      spacing: 0.8em,
      sf(text(size: 20pt)[Problema #problem-number]),
      sf(text(size: 17.2pt, title)),
      [_nombre clave:_ #codename]
    )
    v(1.2em)
  }
  doc
}

#let problemDescription(content) = {
  content
}

#let inputDescription(content) = [
  == Entrada
  #content
]

#let outputDescription(content) = [
  == Salida
  #content
]

#let scoreDescription(content) = [
  == Subtareas y puntaje
  #content
]

#let st = counter("subtask")
#let subtask(score, content) = {
  let bullet = box(move(dy: -1.5pt, sym.triangle.filled.r))
  st.step()
  grid(
    columns: (auto, auto),
    row-gutter: 0.8em,
    column-gutter: 0.5em,
    bullet, [*Subtarea #context st.display() (#score puntos)*],
    [], content
  )
}

#let sampleIO(name) = {
  let input = read(name + ".in")
  let output = read(name + ".sol")

  v(0.5em)
  box(
    stroke: 0.5pt +rgb(30%,30%,30%),
    inset: 10pt,
    grid(
      columns: (1fr, 1fr),
      stack(
        [*Entrada de ejemplo*],
        spacing: 1.2em,
        raw(input),
      ),
      stack(
        spacing: 1.2em,
        [*Salida de ejemplo*],
        raw(output),
      )
    )
  )
}

#let sampleDescription(content) = [
  == Ejemplos de entrada y salida
  #content
]
