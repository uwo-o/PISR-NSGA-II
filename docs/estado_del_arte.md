# Estado del arte: soluciones simbólicas de ecuaciones diferenciales sin datos

Revisión para posicionar PISR-NSGA-II en la memoria. Fecha: noviembre 2026.

**Nivel de verificación:** las referencias bibliográficas (autores, revista, año, DOI) están confirmadas
en fuentes primarias o en la GP Bibliography. Los detalles de método vienen de resúmenes y de páginas
de las propias publicaciones. **No leí los textos completos**: antes de citar un detalle fino (función
de pérdida exacta, lista de benchmarks), hay que revisarlo en el PDF.

---

## 1. Resumen del posicionamiento

Encontrar la solución analítica de una EDO/EDP con programación genética, usando solo el residuo de la
ecuación y las condiciones de borde (sin datos de la solución), **no es una idea nueva**: existe desde
Tsoulos & Lagaris (2006). Usar dos objetivos (error y tamaño del árbol) con ajuste de constantes también
existe (Rockett, 2025–2026). Además hay una línea reciente de aprendizaje profundo con el mismo objetivo
(SSDE, ICML 2025; SES, 2026).

Lo que PISR-NSGA-II puede defender como propio, según esta revisión:

1. **Evaluación de interpretabilidad**: extrapolación fuera del dominio contra un sustituto con la misma
   precisión interior, y extracción de cantidades físicas desde la fórmula (energía, pendientes, número de
   onda, dispersión, radio de Lane–Emden). No encontré trabajos de esta línea que evalúen así; todos
   reportan error contra la solución.
2. **Cobertura**: campos complejos (Schrödinger estacionario y no estacionario), problemas singulares
   (Lane–Emden, Thomas–Fermi), orden 4 (Navier–Stokes en función de corriente) y 2D con frontera exacta
   por interpolación de Coons. Los trabajos previos de GP se concentran en EDO y EDP lineales simples.
3. **Tres objetivos y representación híbrida**: residuo, frontera y complejidad en NSGA-II, con
   individuos con ansatz exacto e individuos directos compitiendo en la misma población.
4. **Priors derivados del operador** (simetrías, modos exponenciales, relación de dispersión), sondeados
   sobre el operador y nunca sobre la solución.
5. **Selección del campeón sin la referencia** (validación solo con información de la EDP) y oráculo de
   coherencia antes de cada experimento.

Los puntos 2–4 son diferencias de grado; el punto 1 es la diferencia de enfoque y coincide con la tesis
central de la memoria (interpretabilidad a cambio de precisión).

### Dos familias de métodos para obtener una fórmula sin datos

| | En dos etapas (destilación) | De punta a punta |
|---|---|---|
| Idea | Resolver con una red (PINN) y ajustar una fórmula a su salida con regresión simbólica | Buscar la fórmula optimizando directamente contra el residuo y el borde |
| Trabajos | SymTorch 2026, StruSR 2025, PINN+SR (Neural Comput. Appl. 2025); DeSyR 2026 como híbrido | Tsoulos & Lagaris 2006, Oh 2023, Rockett 2025–2026, SSDE 2025, SES 2026, **PISR-NSGA-II** |
| Crítica principal | Los errores del PINN se propagan: suele acertar el esqueleto y fallar las constantes (SSDE) | Búsqueda más difícil: el residuo es el único guía |

La crítica a la primera familia viene del artículo de SSDE, que compara contra PINN+DSR (casi idéntico a
nuestro PINN+PySR). Es el argumento de fondo de la memoria: una búsqueda guiada por la ecuación premia
directamente la forma exacta (residuo cero), mientras que la destilación ajusta una aproximación numérica
y puede agregar términos que solo modelan su error. **Casi todas las cifras de estos trabajos son de los
propios autores, con benchmarks distintos; no encontré una comparación independiente entre familias.**

---

## 2. Tabla comparativa

| Trabajo | Método | Sin datos | Objetivos | Frontera | Complejos | Evalúa interpretabilidad | Problemas |
|---|---|---|---|---|---|---|---|
| Tsoulos & Lagaris 2006 (GPEM) | Evolución gramatical | Sí | 1 (residuo + penalización) | Penalización | No | No | EDO, EDO no lineales, sistemas, EDP |
| Sóbester, Nair & Keane 2008 (IEEE TEC) | GP + gradient boosting + RBF | Sí | 1 (mínimos cuadrados por colocación) | RBF para dominios irregulares | No | No | EDP elípticas |
| Seaton, Brown & Miller 2010 (EuroGP) | GP cartesiana (grafos) | Sí | 1 | — | No | No | EDO, EDP, sistemas |
| Oh et al. 2023 (arXiv; versión en revista por confirmar) | GP con regularización física | Sí* | 1 | Términos de pérdida | No | No | Dos EDs con solución conocida |
| Rockett 2025 (UKCI) / 2026 (GPEM) | GP + ajuste de constantes | Sí | **2 (pérdida, nodos), selección por rango de Pareto** | Restricciones duras | No | No | EDO de benchmark |
| Wei et al. 2025, SSDE (ICML) | RL: RNN + gradiente de política buscadora de riesgo | Sí | 1 | Restricciones físicas en la recompensa | No | No | EDO y EDP variadas |
| SES 2026 (arXiv) | Modelo simbólico diferenciable | Sí | 1 | Residuos de condiciones | No consta | No | Algebraicas, EDO, EDP |
| SymTorch 2026 (arXiv) | PINN entrenada y luego destilada con PySR | Indirecto (vía PINN) | Pareto de PySR | La de la PINN | No | Parcial (forma funcional) | Ecuación de calor 1D (demo) |
| PINN+SR, Neural Comput. Appl. 2025 | PINN y luego regresión simbólica | Indirecto (vía PINN) | 1 | La de la PINN | No | No | Burgers, evolución de 5.º orden |
| DeSyR 2026 (arXiv) | PINN guía la estructura; coeficientes refinados con la física | Indirecto (vía PINN) | 2 etapas | La de la PINN | No consta | No consta | EDP (revisar PDF) |
| KAN / Symbolic-KAN 2024–2026 | Red de Kolmogorov-Arnold física, funciones convertibles a símbolos | Sí | 1 | Términos de pérdida | No | Parcial | EDP de demostración |
| PhySO, Tenachi et al. 2023 | RL con unidades físicas | **No** (ajusta datos) | 1 | — | No | Coherencia de unidades | Feynman / SRBench |
| **PISR-NSGA-II** | GP + NSGA-II + jets AD | Sí | **3 (residuo, frontera, nodos)** | **Ansatz exacto (1D, IVP, Coons 2D)** e híbrido | **Sí** | **Sí: extrapolación y cantidades físicas** | 21 problemas: lineales, no lineales, complejos, NS, singulares |

\* Oh et al. reportan que con un conjunto de operadores grande necesitaron unos pocos puntos de datos
para guiar la búsqueda.

---

## 3. Notas por trabajo

### Tsoulos & Lagaris (2006): el antecedente directo
*Solving differential equations with genetic programming*, Genetic Programming and Evolvable Machines
7, 33–54. Usan evolución gramatical para generar soluciones de prueba en forma analítica, con derivadas
calculadas por reglas de diferenciación sobre la expresión, y funciones `+ − × ÷ sin cos log exp`. Cubren
EDO, EDO no lineales, sistemas y EDP. **Hay que citarlo como origen de la idea.** Sus problemas de prueba
son un benchmark estándar que también usa la línea posterior (por ejemplo Rockett).

### Sóbester, Nair & Keane (2008)
*Genetic Programming Approaches for Solving Elliptic Partial Differential Equations*, IEEE TEC 12(4),
469–478, DOI 10.1109/TEVC.2007.908467. Colocación por mínimos cuadrados como función objetivo, mejoras
con gradient boosting, y una red RBF para imponer la frontera en dominios irregulares. Es el antecedente
de "imponer la frontera por construcción", que en PISR-NSGA-II hace el ansatz (Coons en 2D).

### Seaton, Brown & Miller (2010)
*Analytic Solutions to Differential Equations under Graph-based Genetic Programming*, EuroGP 2010,
LNCS 6021, 232–243, DOI 10.1007/978-3-642-12148-7_20. Usan GP cartesiana porque las soluciones analíticas
repiten subexpresiones. Útil para discutir la representación.

### Oh et al. (2023)
*Genetic Programming Based Symbolic Regression for Analytical Solutions to Differential Equations*,
arXiv:2302.03175 (Oh, Amici, Bomarito, Zhe, Kirby, Hochhalter). Un listado indica una versión en
Engineering with Computers (DOI 10.1007/s00366-023-01915-7); **está sin confirmar**. Observación
relevante: con más operadores la búsqueda sin datos se vuelve difícil. Es la misma tensión que en
PISR-NSGA-II resuelven los priors.

### Rockett (2025, 2026): el más cercano
- *The Solution of Ordinary Differential Equations using Genetic Programming with Constant Tuning*, UKCI 2025.
- *Solving ordinary differential equations with genetic programming with hard initial/boundary value
  constraints*, GPEM 27(1), 2026, DOI 10.1007/s10710-026-09536-x.

Colocación, ajuste numérico de constantes, y **dos objetivos (pérdida y número de nodos) con selección por
rango de Pareto**. Reporta como problema abierto que la GP **converge a la solución trivial y = 0** en
varias EDO (ODE-4, 6, 7, 8 de su conjunto), incluso con restricciones duras.

**Conexión directa con PISR-NSGA-II.** Vimos esa misma familia de fallas:
- Troesch, atascado en u ≡ 0 con el modo híbrido.
- La "trampa del pico" en Lane-Emden y Thomas-Fermi: u ≡ 0 en el interior más un pico que fija el dato
  de borde.

PISR-NSGA-II tiene tres respuestas concretas, que se pueden presentar como aporte:
- el ansatz exacto, que hace infactible u = 0 cuando el dato de borde no es nulo;
- la frontera exigida como **límite lateral**, que cierra la trampa del pico;
- la eliminación de duplicados, que evita que el clon trivial llene el frente.

Hay que verificar en el texto de Rockett qué remedio propone, para no atribuirle algo incorrecto.

### SSDE: Wei et al. (2025)
*Closed-form Solutions: A New Perspective on Solving Differential Equations*, ICML 2025, PMLR 267,
66062–66083. Una RNN genera expresiones con un gradiente de política buscador de riesgo y un evaluador
basado en restricciones físicas. Reporta superar a otros métodos de ML en EDO y EDP. En versiones del
artículo el método aparece como **SymPDE**. Compara contra la vía en dos etapas (PINN+DSR) y reporta
que esta suele recuperar el esqueleto correcto pero falla en las constantes, porque el error del PINN se
amplifica en la regresión simbólica; es la cita para la crítica a PINN+PySR. **Es el rival
moderno más fuerte**: misma tarea (forma cerrada sin datos) con aprendizaje profundo. Si su código es
público, es la línea base a correr.

### SES (2026)
*A Data-Free Symbolic Regression Approach for Solving Equations*, arXiv:2606.07152 (cs.NE). Modelo
simbólico diferenciable optimizado directamente con el residuo y las condiciones. No encontré la lista
de autores ni los benchmarks; revisar el PDF.

### SymTorch (2026) y StruSR (2025)
SymTorch (arXiv:2602.21307) destila una red en expresiones con PySR. En su demo de una PINN recupera la
forma de la solución de la ecuación de calor 1D. StruSR (arXiv:2510.06635) guía la regresión simbólica
con expansiones de Taylor de una PINN. Representan la vía alternativa **PINN y después regresión
simbólica**: es la línea base natural para "interpretabilidad desde la física". **Está implementada**
como competidor PINN+PySR (`competitors/`, `run_competitors.sh`), con la misma referencia y la misma
métrica que PISR-NSGA-II.

### PINN + regresión simbólica (Neural Computing and Applications, 2025)
*Physics-informed neural network with symbolic regression for deriving analytical approximate solutions
to nonlinear partial differential equations*, DOI 10.1007/s00521-025-11450-9. Misma vía en dos etapas,
aplicada a Burgers y a ecuaciones de evolución de quinto orden; reporta buena concordancia (error
absoluto medio) entre la PINN y la expresión simbólica. **Autores sin confirmar**; revisar el PDF.

### DeSyR (2026)
*A decoupled symbolic recovery framework for PDEs with PINN-guided structure discovery and physics-based
coefficient refinement*, arXiv:2609.00530. Híbrido de las dos familias: la PINN propone la estructura y
los coeficientes se refinan contra la física. Existe precisamente porque la destilación pura falla en las
constantes, lo que respalda la crítica de SSDE. **Pregunta probable en la defensa**: "¿y si se refinan
las constantes de PySR con la física?". Respuesta: es lo que hace DeSyR, y confirma que la ecuación debe
guiar la búsqueda; PISR-NSGA-II lo hace desde el principio (estructura y constantes contra el residuo).

### Redes de Kolmogorov-Arnold (KAN)
Liu et al. (2024, arXiv:2404.19756) muestran que una KAN puede reemplazar al perceptrón de una PINN y
que sus funciones univariadas aprendidas se pueden convertir en símbolos. Variantes: Symbolic-KAN
(arXiv:2603.23854) y GINN-KAN (arXiv:2408.14780). Es una tercera vía hacia la interpretabilidad, pero la
extracción simbólica sigue siendo un agregado y no una característica estándar. Sirve para la sección de
trabajo relacionado y como trabajo futuro.

### PySR con pérdida física
PySR acepta una función de pérdida propia en Julia (`loss_function`), así que en principio podría
minimizar el residuo de una EDP. No encontré ningún ejemplo publicado, y su documentación advierte que
una pérdida con manipulación simbólica puede romper `.sympy()` y `.predict()`. Construirla sería un
PISR alternativo dentro de PySR; queda como trabajo futuro, no como línea base estándar.

### PhySO: Tenachi et al. (2023)
*Deep symbolic regression for physics guided by units constraints*, arXiv:2303.03192. Regresión
simbólica **con datos**, que impone coherencia de unidades por construcción. No resuelve ecuaciones
diferenciales, así que no es comparable en la tarea principal. Sirve para la discusión de priors
físicos y para la comparación de eficiencia de datos.

---

## 4. Plan de comparación propuesto

1. **Benchmarks de Tsoulos & Lagaris** (los que también usa Rockett): correr PISR-NSGA-II sobre ese
   conjunto da una comparación directa con 20 años de literatura de GP, sin reimplementar a nadie.
2. **SSDE**: si el código es público, correrlo sobre una intersección de la suite (Poisson, oscilador,
   Airy, Lane–Emden, Schrödinger). Si no lo es, comparar contra las cifras de su artículo en los
   problemas compartidos.
3. **PINN y después PySR** (estilo SymTorch). **Implementado**: PINN en PyTorch con los mismos residuos
   y datos de borde, y PySR sobre su predicción (`run_competitors.sh`, 30 semillas). La tabla de
   interpretabilidad del reporte (`report/interpretability.py`) compara error, tamaño de la fórmula,
   recuperación exacta y extrapolación.
4. **Métricas de comparación**, alineadas con la tesis:
   - tasa de recuperación exacta en 20 semillas (equivalencia simbólica con SymPy);
   - error de extrapolación (sección 5 del reporte);
   - consistencia física (sección 6 del reporte);
   - complejidad de la expresión;
   - el MSE, como métrica secundaria.
5. **La falla trivial (y = 0)** como experimento propio: contar en cuántas semillas cada método cae en
   la solución trivial, en los problemas donde existe (Troesch, Lane–Emden). Responde directamente al
   problema abierto de Rockett. **Observación de las 30 semillas**: en los problemas donde una constante
   resuelve la ecuación (Airy, Fisher, Ginzburg–Landau, Lane–Emden, Troesch, NS, Schrödinger no
   estacionario), u ≡ 0 aparece en el frente de Pareto como extremo de residuo mínimo, pero nunca es
   elegida campeona, porque la selección por validación suma el error de frontera.

---

## 5. Fuentes

- Tsoulos & Lagaris 2006: [GP Bibliography](https://gpbib.cs.ucl.ac.uk/gp-html/Tsoulos_2006_GPEM.html), [PDF de los autores](https://www.cs.uoi.gr/~lagaris/papers/DEGen.pdf)
- Sóbester, Nair & Keane 2008: [GP Bibliography](https://gpbib.cs.ucl.ac.uk/gp-html/Sobester_2008_TEC.html), [Southampton ePrints](https://eprints.soton.ac.uk/64449)
- Seaton, Brown & Miller 2010: [GP Bibliography](https://gpbib.cs.ucl.ac.uk/gp-html/Seaton_2010_EuroGP.html), [Springer](https://rd.springer.com/chapter/10.1007/978-3-642-12148-7_20)
- Oh et al. 2023: [arXiv:2302.03175](https://arxiv.org/abs/2302.03175)
- Rockett 2025: [White Rose ePrints](https://eprints.whiterose.ac.uk/id/eprint/229165/), [GP Bibliography](http://gpbib.cs.ucl.ac.uk/gp-html/Rockett_2025_UKCI.html)
- Rockett 2026: [Springer, GPEM](https://link.springer.com/article/10.1007/s10710-026-09536-x)
- Wei et al. 2025 (SSDE): [arXiv:2405.14620](https://arxiv.org/abs/2405.14620), [PMLR](https://proceedings.mlr.press/v267/wei25e.html)
- SES 2026: [arXiv:2606.07152](https://arxiv.org/abs/2606.07152)
- SymTorch 2026: [arXiv:2602.21307](https://arxiv.org/pdf/2602.21307); StruSR 2025: [arXiv:2510.06635](https://arxiv.org/html/2510.06635v1)
- PhySO 2023: [arXiv:2303.03192](https://arxiv.org/pdf/2303.03192)
- Kozax (NSGA-II con aptitud y tamaño, EDE estocásticas) 2026: [arXiv:2603.09597](https://arxiv.org/pdf/2603.09597)
- PINN + regresión simbólica 2025: [Springer, Neural Computing and Applications](https://link.springer.com/article/10.1007/s00521-025-11450-9)
- DeSyR 2026: [arXiv:2609.00530](https://arxiv.org/pdf/2609.00530)
- KAN 2024: [arXiv:2404.19756](https://arxiv.org/pdf/2404.19756); Symbolic-KAN 2026: [arXiv:2603.23854](https://arxiv.org/pdf/2603.23854); GINN-KAN: [arXiv:2408.14780](https://arxiv.org/html/2408.14780v2)
- Demo de SymTorch con PINN: [documentación](https://symtorch.readthedocs.io/en/latest/demos/pinns_demo.html)
