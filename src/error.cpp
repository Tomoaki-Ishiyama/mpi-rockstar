#include "error.h"

static rockstar_error_handler current_handler = rockstar_default_error_handler;

void rockstar_set_error_handler(rockstar_error_handler handler)
{
    if (handler) {
        current_handler = handler;
    } else {
        current_handler = rockstar_default_error_handler;
    }
}

void rockstar_default_error_handler(int code, const char *file, int line)
{
    throw rockstar_error(code, file, line);
}

void rockstar_abort_impl(int code, const char *file, int line)
{
    current_handler(code, file, line);
}
