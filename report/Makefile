TEX  := ww_bfs_implementation
LATEX ?= pdflatex
LATEXOPTS := -interaction=nonstopmode -halt-on-error

.PHONY: all publish clean clobber

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex
	$(LATEX) $(LATEXOPTS) $(TEX).tex
	$(LATEX) $(LATEXOPTS) $(TEX).tex

# Build + mirror to EOS (https://mdefranc.web.cern.ch/WW_threshold/report/).
# Run after every .tex edit so the published PDF stays in sync with the source.
publish:
	./publish.sh

clean:
	rm -f $(TEX).aux $(TEX).log $(TEX).out $(TEX).toc $(TEX).synctex.gz

clobber: clean
	rm -f $(TEX).pdf
