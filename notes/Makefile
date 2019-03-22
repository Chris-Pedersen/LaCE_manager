#
# Top-level Makefile 
#
# Use this shell to interpret shell commands, & pass its value to sub-make
#
export SHELL = /bin/sh
#
# This is like doing 'make -w' on the command line.  This tells make to
# print the directory it is in.
#
MAKEFLAGS = w
#
# This is a list of subdirectories that make should descend into.  Right now
# this only applies to the 'make clean' command.
#
SUBDIRS =
#
# This line helps prevent make from getting confused in the case where you
# have a file named 'clean'.
#
.PHONY: clean
#
# This will compile the proposal
#
all:
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex
#
# GNU make pre-defines $(RM).  The - in front of $(RM) causes make to
# ignore any errors produced by $(RM).
#
clean:
	- $(RM) -r *.aux *.toc *.out *.log *.bbl *.blg *~ main.pdf
	@ for f in $(SUBDIRS); do $(MAKE) -C $$f clean ; done
