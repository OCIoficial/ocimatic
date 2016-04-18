%\usepackage[utf8]{inputenc}
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{oci}[2014/09/29 v1.0 Clase LaTeX para estandarizacion de problemas OCI]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[11pt]{article}

\RequirePackage{ifthen}
\RequirePackage{fancyhdr}
\RequirePackage{graphicx}
\RequirePackage{verbatim}
\RequirePackage{enumitem}



%% GEOMETRY
\oddsidemargin  0.0in
\evensidemargin 0.0in
\textwidth      6.7in
\textheight     8.2in
\voffset        -0.5in
%\headsep        0in
\parindent      0.0in


%% EXTRA
\addtolength{\parskip}{0.4\baselineskip}

%% HEADER
\global\let\@phase\@empty
\newcommand{\phase}[1]{\def\@phase{#1}}

\setlength\headheight{54.3pt} 
\pagestyle{fancy}
\lhead{\includegraphics[height=37.5pt]{logo.eps}}
\rhead{\@phase\ \the\year}
\renewcommand{\headrulewidth}{0.2pt}


%% TITLE
\newif\ifproblemcnt%
\problemcntfalse%
% \global\let\@filename\@empty
% \newcommand{\filename}[1]{\def\@filename{#1}}

\global\let\@problem\@empty
\newcommand{\problem}[1]{\def\@problem{#1}}
\newcounter{problemcnt}
\setcounter{problemcnt}{1}

\newcommand{\mymaketitle}{
    \begin{center}
      \ifproblemcnt%
      {\huge\bf \sf Problema \Alph{problemcnt}}\\
      \else
      {\huge\bf \sf Problema \@problem}\\
      \fi
    \LARGE{\sf \@title}\\[2mm]
    % \normalsize {\em archivo:} {\tt \@filename\{.c,.cpp,.pas\}}
    \vspace{6pt}
    \end{center}
    \stepcounter{problemcnt}
}

\renewcommand*{\cleardoublepage}{\clearpage\if@twoside \ifodd\c@page\else
\hbox{}%
\thispagestyle{empty}%
\newpage%
\if@twocolumn\hbox{}\newpage\fi\fi\fi}

%% Secciones de description
\renewcommand{\subsection}{\@startsection{subsection}{2}{\z@}%
                                     {-1.0ex\@plus -1ex \@minus -.2ex}%
                                     {1.0ex \@plus .2ex}%
                                   {\reset@font\large\bfseries}}
\newenvironment{problemDescription}{\cleardoublepage\mymaketitle}{}
\newenvironment{inputDescription}{\subsection*{Entrada}}{\par}
\newenvironment{outputDescription}{\subsection*{Salida}}{\par}


%% Ejemplo Input/Output

\newcommand{\sampleIO}[1]{%
  \begin{minipage}[c]{\textwidth}
    \begin{center}
      \begin{tabular}{|l|} 
        \hline
        \begin{minipage}[t]{0.45\textwidth}
            \vskip 0.3pt
            \bf{Entrada de ejemplo}
            \vskip 12pt
\verbatiminput{#1.in}
            \vskip 4pt
        \end{minipage}
        \begin{minipage}[t]{0.45\textwidth}
            \vskip 0.3pt
            \bf{Salida de ejemplo}
            \vskip 12pt
\verbatiminput{#1.sol}
            \vskip 4pt
        \end{minipage}
        \\
        \hline
      \end{tabular}
    \end{center}
  \end{minipage}
  \vskip 12pt
}


\newenvironment{scoreDescription}{\subsection*{Subtareas y Puntaje}\begin{description}[leftmargin=0pt]\itemsep 0pt}{\end{description}}
\newcommand{\score}[1]{\item[#1 puntos]}

\newenvironment{sampleDescription}{\subsection*{Ejemplos de Entrada y Salida}}{}

