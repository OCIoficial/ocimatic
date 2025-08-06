#import "oci.typ": *

#show: general
#set enum(spacing: 1.6em)

== Información General

Esta página muestra información general que se aplica a todos los problemas.

== Envío de una solución

+ Los participantes deben enviar *un solo archivo* con el código fuente de su solución.
+ El nombre del archivo debe tener la extensión `.cpp` o
  `.java` dependiendo de si la solución está escrita en
  `C++` o `Java` respectivamente.
  Para enviar una solución en Java hay que seguir algunos pasos adicionales. Ver detalles más abajo.

== Casos de prueba, subtareas y puntaje
+ La solución enviada por los participantes será ejecutada varias veces con
   distintos casos de prueba.
+ A menos que se indique lo contrario, cada problema define diferentes
  subtareas que lo restringen. Se asignará puntaje de acuerdo a la
  cantidad de subtareas que se logre solucionar de manera correcta.
+ A menos que se indique lo contrario, para obtener el puntaje en una
  subtarea se debe tener correctos todos los casos de prueba incluídos en ella.
+ Una solución puede resolver al mismo tiempo más de una subtarea.
+ La solución es ejecutada con cada caso de prueba de manera independiente y
   por tanto puede fallar en algunas subtareas sin influir en la ejecución de
   otras.

== Entrada
+ Toda lectura debe ser hecha desde la *entrada estándar* usando, por
  ejemplo, las funciones `scanf` o `std::cin` en C++ o la clase
  `BufferedReader` en Java.
+ La entrada corresponde a un solo caso de prueba, el cual está descrito en
  varias líneas dependiendo del problema.
+ *Se garantiza que la entrada sigue el formato descrito* en el
  enunciado de cada problema.

#pagebreak()

== Salida
+ Toda escritura debe ser hecha hacia la *salida estándar* usando, por
  ejemplo, las funciones `printf`, `std::cout` en C++  o
  `System.out.println` en Java.
+ El formato de salida es explicado en el enunciado de cada problema.
+ *La salida del programa debe cumplir estrictamente con el formato indicado*,
  considerando los espacios, las mayúsculas y minúsculas.
+ Toda línea, incluyendo la última, debe terminar con un salto de línea.

== Envío de una solución en Java

+ Cada problema tiene un _nombre clave_ que será especificado en el
  enunciado.
  Este nombre clave será también utilizado en el sistema de evaluación
  para identificar al problema.
+ Para enviar correctamente una solución en Java, el archivo debe contener
  una clase llamada igual que el nombre clave del problema.
  Esta clase debe contener también el método `main`.
  Por ejemplo, si el nombre clave es `marraqueta`, el archivo con la
  solución debe llamarse `marraqueta.java` y tener la siguiente
  estructura:

  ```
    public class marraqueta {
        public static void main (String[] args) {
            // tu solución va aquí
        }
    }
  ```
+ Si el archivo no contiene la clase con el nombre correcto, el sistema de
  evaluación reportará un error de compilación.

+ La clase no debe estar contenida dentro de un _package_.
    Hay que tener cuidado pues algunos entornos de desarrollo como Eclipse
  incluyen las clases en un _package_ por defecto.
+ Si la clase está contenida dentro de un package, el sistema reportará un
  error de compilación.
