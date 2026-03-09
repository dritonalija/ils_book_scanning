import numpy as np


class Solution:
    __slots__ = ('signed_libraries', 'unsigned_libraries',
                 'scanned_books_per_library', 'scanned_books',
                 'fitness_score', 'initial_score')

    def __init__(
        self,
        signed_libs=None,
        unsigned_libs=None,
        scanned_books_per_library=None,
        scanned_books=None,
        fitness_score=0,
    ):
        self.signed_libraries = signed_libs or []
        self.unsigned_libraries = unsigned_libs or []
        self.scanned_books_per_library = scanned_books_per_library or {}
        self.scanned_books = scanned_books or set()
        self.fitness_score = fitness_score
        self.initial_score = 0

    @classmethod
    def from_order(cls, order, data):
        solution = cls()
        solution.rebuild_from_order(order, data)
        return solution

    def clone(self):
        s = Solution(
            self.signed_libraries.copy(),
            self.unsigned_libraries.copy(),
            {k: v.copy() for k, v in self.scanned_books_per_library.items()},
            self.scanned_books.copy(),
            self.fitness_score,
        )
        s.initial_score = self.initial_score
        return s

    def ordered_libraries(self):
        return self.signed_libraries + self.unsigned_libraries

    def library_contributions(self, data):
        contributions = []
        for position, lib_id in enumerate(self.signed_libraries):
            books = self.scanned_books_per_library.get(lib_id, [])
            raw_score = sum(data.scores[book_id] for book_id in books)
            contributions.append({
                "lib_id": lib_id,
                "position": position,
                "score": raw_score,
                "book_count": len(books),
                "score_per_signup": raw_score / max(1, data.libs[lib_id].signup_days),
            })
        return contributions

    def export(self, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"{len(self.signed_libraries)}\n")
            for library in self.signed_libraries:
                books = self.scanned_books_per_library.get(library, [])
                f.write(f"{library} {len(books)}\n")
                f.write(" ".join(map(str, books)) + "\n")

    def calculate_fitness_score(self, scores):
        self.fitness_score = sum(scores[book] for book in self.scanned_books)
        return self.fitness_score

    def rebuild_from_order(self, order, data):
        flat = data.to_flat_arrays() if hasattr(data, "to_flat_arrays") else None
        if flat is not None:
            return self._rebuild_fast(order, data, flat)
        return self._rebuild_python(order, data)

    def _rebuild_fast(self, order, data, flat):
        from models.evaluation import fast_evaluate_detailed

        out_selected = np.zeros(max(1, len(order)), dtype=np.int32)
        out_books = np.zeros(max(1, flat["n_books"]), dtype=np.int32)
        out_book_libs = np.zeros(max(1, flat["n_books"]), dtype=np.int32)

        signed_arr = np.array(order, dtype=np.int32)
        score, selected_count, assigned_count = fast_evaluate_detailed(
            signed_arr,
            flat["libs_signup"],
            flat["libs_rate"],
            flat["lib_num_books"],
            flat["books_by_score"],
            flat["book_libs_flat"],
            flat["book_libs_offsets"],
            flat["book_libs_lengths"],
            flat["book_scores"],
            flat["total_days"],
            out_selected,
            out_books,
            out_book_libs,
        )

        scanned_per_lib = {}
        scanned_books = set()
        for idx in range(assigned_count):
            lib_id = int(out_book_libs[idx])
            book_id = int(out_books[idx])
            scanned_per_lib.setdefault(lib_id, []).append(book_id)
            scanned_books.add(book_id)

        signed_libraries = []
        for idx in range(selected_count):
            lib_id = int(out_selected[idx])
            if scanned_per_lib.get(lib_id):
                signed_libraries.append(lib_id)

        signed_set = set(signed_libraries)
        self.signed_libraries = signed_libraries
        self.unsigned_libraries = [lid for lid in order if lid not in signed_set]
        self.scanned_books_per_library = scanned_per_lib
        self.scanned_books = scanned_books
        self.fitness_score = int(score)
        return self

    def _rebuild_python(self, order, data):
        """Pure Python fallback rebuild."""
        selected, capacities, positions = self._feasible_libraries(order, data)
        scanned_per_lib, scanned_books, fitness_score = self._assign_books_global(
            selected, capacities, positions, data)

        signed_libraries = [lid for lid in selected if scanned_per_lib.get(lid)]
        signed_set = set(signed_libraries)
        self.signed_libraries = signed_libraries
        self.unsigned_libraries = [lid for lid in order if lid not in signed_set]
        self.scanned_books_per_library = scanned_per_lib
        self.scanned_books = scanned_books
        self.fitness_score = fitness_score
        return self

    def _feasible_libraries(self, order, data):
        day = 0
        selected = []
        capacities = {}
        positions = {}

        for position, lib_id in enumerate(order):
            signup_days = data.lib_signup_days[lib_id]
            if day + signup_days >= data.num_days:
                continue
            day += signup_days
            remaining_days = data.num_days - day
            capacity = remaining_days * data.lib_books_per_day[lib_id]
            if capacity <= 0:
                continue
            selected.append(lib_id)
            capacities[lib_id] = min(capacity, data.lib_num_books[lib_id])
            positions[lib_id] = position

        return selected, capacities, positions

    def _assign_books_global(self, selected, capacities, positions, data):
        scanned_per_lib = {lib_id: [] for lib_id in selected}
        if not selected:
            return scanned_per_lib, set(), 0

        num_libs = data.num_libs
        active = bytearray(num_libs)
        remaining_capacity = [0] * num_libs
        remaining_candidates = [0] * num_libs
        lib_positions = [0] * num_libs
        fitness_score = 0

        for lib_id in selected:
            active[lib_id] = 1
            remaining_capacity[lib_id] = capacities[lib_id]
            remaining_candidates[lib_id] = data.lib_num_books[lib_id]
            lib_positions[lib_id] = positions[lib_id]

        for book_id in data.books_by_score:
            best_lib = None
            best_key = None
            for lib_id in data.book_libs[book_id]:
                if not active[lib_id] or remaining_capacity[lib_id] <= 0:
                    continue
                key = (
                    min(remaining_candidates[lib_id], remaining_capacity[lib_id]),
                    remaining_capacity[lib_id],
                    -lib_positions[lib_id],
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_lib = lib_id

            if best_lib is None:
                continue

            scanned_per_lib[best_lib].append(book_id)
            remaining_capacity[best_lib] -= 1
            fitness_score += data.scores[book_id]

            for lib_id in data.book_libs[book_id]:
                if active[lib_id]:
                    remaining_candidates[lib_id] -= 1

        scanned_books = {
            book_id
            for books in scanned_per_lib.values()
            for book_id in books
        }
        return scanned_per_lib, scanned_books, fitness_score
