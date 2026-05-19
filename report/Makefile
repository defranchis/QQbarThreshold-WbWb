TEX  := ww_bfs_implementation
LATEX ?= pdflatex
LATEXOPTS := -interaction=nonstopmode -halt-on-error

.PHONY: all clean clobber

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex
	$(LATEX) $(LATEXOPTS) $(TEX).tex
	$(LATEX) $(LATEXOPTS) $(TEX).tex

clean:
	rm -f $(TEX).aux $(TEX).log $(TEX).out $(TEX).toc $(TEX).synctex.gz

clobber: clean
	rm -f $(TEX).pdf
