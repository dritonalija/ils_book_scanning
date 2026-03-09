"""Exact JIT-accelerated evaluation for the Book Scanning solver."""

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(fastmath=False, cache=False):
        def decorator(func):
            return func
        return decorator


@njit(fastmath=True, cache=True)
def fast_evaluate(order, libs_signup, libs_rate, lib_num_books, books_by_score,
                  book_libs_flat, book_libs_offsets, book_libs_lengths,
                  book_scores, total_days):
    """Exact score using the same global assignment rule as the Python solver."""
    num_libs = len(libs_signup)
    active = np.zeros(num_libs, dtype=np.uint8)
    remaining_capacity = np.zeros(num_libs, dtype=np.int32)
    remaining_candidates = np.zeros(num_libs, dtype=np.int32)
    positions = np.zeros(num_libs, dtype=np.int32)
    day = 0

    for position in range(len(order)):
        lib_id = order[position]
        signup = libs_signup[lib_id]
        if day + signup >= total_days:
            continue
        day += signup
        remaining_days = total_days - day
        capacity = remaining_days * libs_rate[lib_id]
        if capacity <= 0:
            continue
        active[lib_id] = 1
        remaining_capacity[lib_id] = min(capacity, lib_num_books[lib_id])
        remaining_candidates[lib_id] = lib_num_books[lib_id]
        positions[lib_id] = position

    score = np.int64(0)
    for idx in range(len(books_by_score)):
        book_id = books_by_score[idx]
        best_lib = -1
        best_key1 = -1
        best_key2 = -1
        best_key3 = -2147483647
        offset = book_libs_offsets[book_id]
        length = book_libs_lengths[book_id]

        for k in range(length):
            lib_id = book_libs_flat[offset + k]
            if active[lib_id] == 0 or remaining_capacity[lib_id] <= 0:
                continue
            key1 = remaining_candidates[lib_id]
            if remaining_capacity[lib_id] < key1:
                key1 = remaining_capacity[lib_id]
            key2 = remaining_capacity[lib_id]
            key3 = -positions[lib_id]
            if (
                key1 > best_key1
                or (key1 == best_key1 and key2 > best_key2)
                or (key1 == best_key1 and key2 == best_key2 and key3 > best_key3)
            ):
                best_key1 = key1
                best_key2 = key2
                best_key3 = key3
                best_lib = lib_id

        if best_lib == -1:
            continue

        score += book_scores[book_id]
        remaining_capacity[best_lib] -= 1

        for k in range(length):
            lib_id = book_libs_flat[offset + k]
            if active[lib_id] == 1:
                remaining_candidates[lib_id] -= 1

    return score


@njit(fastmath=True, cache=True)
def fast_evaluate_detailed(
    order,
    libs_signup,
    libs_rate,
    lib_num_books,
    books_by_score,
    book_libs_flat,
    book_libs_offsets,
    book_libs_lengths,
    book_scores,
    total_days,
    out_selected,
    out_books,
    out_book_libs,
):
    """Exact detailed evaluation with assignment output for reconstruction."""
    num_libs = len(libs_signup)
    active = np.zeros(num_libs, dtype=np.uint8)
    remaining_capacity = np.zeros(num_libs, dtype=np.int32)
    remaining_candidates = np.zeros(num_libs, dtype=np.int32)
    positions = np.zeros(num_libs, dtype=np.int32)
    day = 0
    score = np.int64(0)
    selected_count = 0
    assigned_count = 0

    for position in range(len(order)):
        lib_id = order[position]
        signup = libs_signup[lib_id]
        if day + signup >= total_days:
            continue
        day += signup
        remaining_days = total_days - day
        capacity = remaining_days * libs_rate[lib_id]
        if capacity <= 0:
            continue
        active[lib_id] = 1
        remaining_capacity[lib_id] = min(capacity, lib_num_books[lib_id])
        remaining_candidates[lib_id] = lib_num_books[lib_id]
        positions[lib_id] = position
        out_selected[selected_count] = lib_id
        selected_count += 1

    for idx in range(len(books_by_score)):
        book_id = books_by_score[idx]
        best_lib = -1
        best_key1 = -1
        best_key2 = -1
        best_key3 = -2147483647
        offset = book_libs_offsets[book_id]
        length = book_libs_lengths[book_id]

        for k in range(length):
            lib_id = book_libs_flat[offset + k]
            if active[lib_id] == 0 or remaining_capacity[lib_id] <= 0:
                continue
            key1 = remaining_candidates[lib_id]
            if remaining_capacity[lib_id] < key1:
                key1 = remaining_capacity[lib_id]
            key2 = remaining_capacity[lib_id]
            key3 = -positions[lib_id]
            if (
                key1 > best_key1
                or (key1 == best_key1 and key2 > best_key2)
                or (key1 == best_key1 and key2 == best_key2 and key3 > best_key3)
            ):
                best_key1 = key1
                best_key2 = key2
                best_key3 = key3
                best_lib = lib_id

        if best_lib == -1:
            continue

        score += book_scores[book_id]
        out_books[assigned_count] = book_id
        out_book_libs[assigned_count] = best_lib
        assigned_count += 1
        remaining_capacity[best_lib] -= 1

        for k in range(length):
            lib_id = book_libs_flat[offset + k]
            if active[lib_id] == 1:
                remaining_candidates[lib_id] -= 1

    return score, selected_count, assigned_count


def warmup_jit(flat):
    """Trigger Numba compilation before timed runs."""
    if not HAS_NUMBA:
        return
    dummy = np.array([0], dtype=np.int32)
    fast_evaluate(dummy, flat['libs_signup'], flat['libs_rate'],
                  flat['lib_num_books'], flat['books_by_score'],
                  flat['book_libs_flat'], flat['book_libs_offsets'],
                  flat['book_libs_lengths'], flat['book_scores'],
                  flat['total_days'])
    out_selected = np.zeros(1, dtype=np.int32)
    out_books = np.zeros(max(1, flat['n_books']), dtype=np.int32)
    out_book_libs = np.zeros(max(1, flat['n_books']), dtype=np.int32)
    fast_evaluate_detailed(dummy, flat['libs_signup'], flat['libs_rate'],
                           flat['lib_num_books'], flat['books_by_score'],
                           flat['book_libs_flat'], flat['book_libs_offsets'],
                           flat['book_libs_lengths'], flat['book_scores'],
                           flat['total_days'],
                           out_selected, out_books, out_book_libs)
