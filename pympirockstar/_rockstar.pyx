# cython: language_level=3
# distutils: language=c++
from libc.stdlib cimport malloc, free
from libc.string cimport strdup

cdef extern from *:
    """
    #include "mpi_rockstar.h"
    #include "error.h"

    static int rockstar_run_wrap(int argc, char **argv,
                                 int *code, const char **file, int *line) {
        try {
            rockstar_mpi_main(argc, argv);
            return 0;
        } catch (rockstar_error &e) {
            if (code) *code = e.code;
            if (file) *file = e.file;
            if (line) *line = e.line;
            return -1;
        }
    }
    """
    int rockstar_run_wrap(int argc, char **argv,
                          int *code, const char **file, int *line)

cdef class RockstarError(Exception):
    cdef public int code
    cdef public str file
    cdef public int line

    def __init__(self, int c, const char *f, int l):
        self.code = c
        self.file = f.decode('utf-8')
        self.line = l
        Exception.__init__(self, f"Rockstar error {c} at {self.file}:{self.line}")

cdef int _rockstar_run(int argc, char **cargs) except *:
    cdef int code
    cdef const char *file
    cdef int line
    cdef int status = rockstar_run_wrap(argc, cargs, &code, &file, &line)
    if status != 0:
        raise RockstarError(code, file, line)
    return status


def run(args):
    """Run MPI-Rockstar with a list of command line arguments.

    Parameters
    ----------
    args : list of str
        Arguments as would be passed on the command line. The program name
        should be the first element.
    """
    cdef int argc = len(args)
    cdef char **cargs = <char **>malloc((argc + 1) * sizeof(char *))
    if cargs == NULL:
        raise MemoryError()

    cdef Py_ssize_t i
    for i in range(argc):
        arg = args[i].encode('utf-8')
        cargs[i] = strdup(arg)
    cargs[argc] = NULL

    cdef int exit_code
    try:
        exit_code = _rockstar_run(argc, cargs)
    finally:
        for i in range(argc):
            free(cargs[i])
        free(cargs)
    return exit_code
