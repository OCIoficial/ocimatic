#import "oci.typ": *

#show: problem.with(
  title: "Suma de ejemplo"
)

#problemDescription[
  Este es un problema de ejemplo para mostrar cómo usar ocimatic.
  El problema es muy simple.
  Te dan dos enteros y tienes que imprimir su suma.
  Pero ten cuidado, !la suma podría no caber en un entero de 32 bits con signo!
]

#inputDescription[
  La entrada consiste en una sola línea con dos enteros $a$ y $b$ ($-3*10^9 <= a, b <= 3*10^9$).
]

#outputDescription[
  La salida debe contener un único entero correspondiente a la suma de $a$ y $b$.
]

#scoreDescription[
  #subtask(50)[
    Se probarán varios casos de prueba donde $-10^9 <= a, b <= 10^9$.
  ]

  #subtask(50)[
    Se probarán varios casos de prueba sin restricciones adicionales
  ]
]

#sampleDescription[
  #sampleIO("sample-1")
  #sampleIO("sample-2")
]
