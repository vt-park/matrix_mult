CC = gcc
FLAGS = -Wall -Werror -std=gnu99
APP = matrix_mult

main: $(APP).c main.c
	$(CC) $(FLAGS) $^ -o $@ -lpthread

clean:
	rm -f main
