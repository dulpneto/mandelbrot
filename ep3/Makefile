CC=gcc
CCFLAGS=-Wall -O0 
LDFLAGS=-lm -lpthread
FILE=pi_process

ifeq ($(DEBUG), 1)
		CCFLAGS += -DDEBUG_MSG
endif

.PHONY: all
all:

	$(CC) $(CCFLAGS) $(FILE).c -o $(FILE) $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(FILE)
