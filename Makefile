#Makefile for lightcrawl
#Author: Utkarsh Desai

CC=gcc
CPPFLAGS=
LDFLAGS=

PROG = nninc
HDRS = src/config.h src/data.h src/trainer.h src/testinh.h
SRCS = src/main.c src/config.c src/data.c src/trainer.c src/testing.c

#Object files have same name as sources except with a .o
OBJS = $(SRCS:.c=.o)

#Build the program using object files (default rule)
$(PROG) : $(OBJS)
	$(CC) -o $(PROG) $(OBJS) -lm

#Rules for source files
main.o : src/main.c src/config.c src/data.c src/trainer.c

config.o : src/config.c

data.o: src/data.c

trainer.o: src/trainer.c src/data.c src/config.c

testing.o: src/testing.c src/data/c src/trainer.c

#Clean target
clean:
	rm -f $(PROG) $(OBJS)
