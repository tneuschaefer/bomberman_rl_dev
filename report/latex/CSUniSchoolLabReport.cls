%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% University/School Laboratory Report
% LaTeX Class
% Version 4.0 (March 21, 2022)
%
% This class originates from:
% https://www.LaTeXTemplates.com
%
% Authors:
% Vel (vel@latextemplates.com)
% Linux and Unix Users Group at Virginia Tech Wiki 
%
% License:
% CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%----------------------------------------------------------------------------------------
%	CLASS CONFIGURATION
%----------------------------------------------------------------------------------------

\LoadClass[12pt]{article} % Load the base class

%----------------------------------------------------------------------------------------
%	REQUIRED PACKAGES AND MISC CONFIGURATIONS
%----------------------------------------------------------------------------------------

\usepackage{graphicx} % Required for including images
\graphicspath{{Figures/}{./}} % Specifies where to look for included images (trailing slash required)
\usepackage{subcaption}
\usepackage{titlesec}


\usepackage{float} % Allows more precisely positioning floats e.g. \begin{figure}[H]
\usepackage{tabularx}
\usepackage{algorithm}
\usepackage[noend]{algpseudocode}
\usepackage{amssymb,amsmath,amsfonts} % nice math rendering
\usepackage{braket} % needed for \Set
\DeclareCaptionFormat{myformat}{#3}
\captionsetup[algorithm]{format=myformat}
\DeclareMathOperator*{\argmax}{arg\,max}
\usepackage{algorithm}


\usepackage{amsmath} % Required for some math elements 
\usepackage{multirow}
\usepackage{hyperref}

	
\usepackage[usenames,dvipsnames,svgnames,table]{xcolor}
%\usepackage[dvipsnames]{xcolor}
%\definecolor{myYellow}{yellow}{0.3}

\newcolumntype{n}{>{\columncolor{yellow!20}}c}
\newcolumntype{o}{>{\columncolor{pink!20}}c}
\newcolumntype{e}{>{\columncolor{cyan!10}}c}
\newcolumntype{b}{>{\columncolor{teal!10}}c}

\newcommand{\AB}{\tiny Angelina Basova}




%----------------------------------------------------------------------------------------
%	MARGINS
%----------------------------------------------------------------------------------------

\usepackage[
	top=4.25cm, % Top margin
	bottom=4cm, % Bottom margin
	left=4.75cm, % Left margin
	right=4.75cm, % Right margin
	%showframe % Uncomment to show frames around the margins for debugging purposes
]{geometry}

\setcounter{tocdepth}{4}
\setcounter{secnumdepth}{4}
%----------------------------------------------------------------------------------------
%	FONTS
%----------------------------------------------------------------------------------------

\usepackage[T1]{fontenc} % Output font encoding for international characters
\usepackage[utf8]{inputenc} % Required for inputting international characters

%%%%%%%%%%%%%%%%%%%%%%ADD SUBSUBSUBSECTION 
\usepackage[scale=0.96]{XCharter} % Use the XCharter text font
\usepackage[xcharter]{newtxmath} % Set the math font
\titleformat{\paragraph}
{\normalfont\normalsize\bfseries}{\theparagraph}{1em}{}
\titlespacing*{\paragraph}
{0pt}{3.25ex plus 1ex minus .2ex}{1.5ex plus .2ex}
%----------------------------------------------------------------------------------------
%	BIBLIOGRAPHY
%----------------------------------------------------------------------------------------

\usepackage[ % Use the biber backend for compiling the bibliography
	citestyle=numeric, % In-text citation style
	bibstyle=numeric, % Bibliography style
	backend=bibtex,
]{biblatex}
