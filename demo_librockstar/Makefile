CC=mpicxx
CFLAGS=-fopenmp
LDFLAGS=../libmpi_rockstar.a -lm -ltirpc -fopenmp
HDF5_LDFLAGS=../libmpi_rockstar_hdf5.a -lm -ltirpc -fopenmp -lhdf5 -lz

example: example.o ../libmpi_rockstar.a
	$(CC) -o $@ example.o $(LDFLAGS)

example_hdf5: example.o ../libmpi_rockstar_hdf5.a
	$(CC) -o $@ example.o $(HDF5_LDFLAGS)

example.o: example.c
	$(CC) $(CFLAGS) -c -o $@ example.c

clean:
	rm -f example.o example example_hdf5
