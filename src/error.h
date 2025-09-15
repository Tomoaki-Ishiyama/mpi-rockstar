#ifndef ROCKSTAR_ERROR_H
#define ROCKSTAR_ERROR_H

#include <mpi.h>

#ifdef __cplusplus
#include <exception>
extern "C" {
#endif

typedef void (*rockstar_error_handler)(int code, const char *file, int line);

void rockstar_set_error_handler(rockstar_error_handler handler);
void rockstar_default_error_handler(int code, const char *file, int line);
void rockstar_abort_impl(int code, const char *file, int line);

#define rockstar_abort(code) rockstar_abort_impl((code), __FILE__, __LINE__)

/* Replace any direct call to exit() inside the code base. */
#define exit(code) rockstar_abort(code)

#ifdef __cplusplus
}

struct rockstar_error : public std::exception {
    int code;
    const char *file;
    int line;
    rockstar_error(int c, const char *f, int l) : code(c), file(f), line(l) {}
    const char *what() const noexcept override { return "rockstar error"; }
};
#endif

#endif /* ROCKSTAR_ERROR_H */
