DEBUG=0
CC:=$(shell command -v nvcc || echo hipcc)
FLAG:=-I include/

ifneq ($(CC), hipcc)
	FLAG:=$(FLAG) -x cu
endif

ifeq ($(DEBUG),1)
	ifeq ($(CC), hipcc)
		FLAG:=$(FLAG) -g -O0
	else
		FLAG:=$(FLAG) -G
	endif

endif

all:
	$(CC) $(FLAG) main.cpp

clean: 
	rm ./a.out
