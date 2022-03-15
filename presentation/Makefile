AUXFILES = *.aux *.log *.nav *.out *.snm *.toc 
MAIN = main.tex
PDF = main.pdf
LATEX = pdflatex
VIEWER = /usr/bin/evince

.PHONY: clean compile view

all: pdf view

clean: 
	rm ${AUXFILES}

pdf:
	${LATEX} ${MAIN}

view:
	${VIEWER} ${PDF} &
