PYSETUP_FLAGS=
PYMSES_BASE=pymses

# This compiles all modules for an inplace use
inplace_build:
	python setup.py build_src build_ext --inplace $(PYSETUP_FLAGS)

# This generates the C sources from Cython sources
cython:
	find . -name "*.pyx" | xargs cython -3
	# This make also the inplace_build:
	make

clean:
	find . -type f -name "*.pyc" -exec rm {} \;
	find . -type f -name "*.so" -exec rm {} \;
	rm -rf build

# This runs fast unit tests
test:
	nosetests -v -w $(PYMSES_BASE)

# This runs all tests
all_tests:
	nosetests -v -w .

