CC=gcc
CFLAGS=-g -Wall -Wunused-variable
LDFLAGS=-L$(HOME)/local/lib

UTIL_OBJ=bayesian.o bimatrix.o matrix.o polymatrix.o graph.o util.o
PM_OBJ=pm-gen.o coord_zero.o strict_comp.o weight_coop.o $(UTIL_OBJ)
BLOTTO_OBJ=blotto_gen.o blotto.o adj_winner.o partition.o $(UTIL_OBJ)
ITEM_OBJ=item_gen.o valuation.o repetition.o item_bidding.o multi_unit.o $(UTIL_OBJ)
OBJ=$(PM_OBJ) $(UTIL_OBJ) $(BLOTTO_OBJ) $(ITEM_OBJ)
EXEC=pm-gen item-gen blotto-gen

%.o : %.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@
	
all: $(EXEC)

pm-gen: $(PM_OBJ)
	$(CC) $(CFLAGS) $(PM_OBJ) $(LDFLAGS) -o $@ -lgmp -lm

blotto-gen: $(BLOTTO_OBJ)
	$(CC) $(CFLAGS) $(BLOTTO_OBJ) $(LDFLAGS) -o $@ -lgmp -lm

item-gen: $(ITEM_OBJ)
	$(CC) $(CFLAGS) $(ITEM_OBJ) $(LDFLAGS) -o $@ -lgmp -lm

clean:
	rm $(OBJ) $(EXEC)
