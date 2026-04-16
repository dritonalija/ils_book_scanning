import numpy as np


class InstanceData:
    def __init__(self, num_books, num_libs, num_days, scores, libs):
        self.num_books = num_books
        self.num_libs = num_libs
        self.num_days = num_days
        self.scores = scores
        self.libs = libs
        self.book_libs = [[] for _ in range(num_books)]
        self.lib_signup_days = [0] * num_libs
        self.lib_books_per_day = [0] * num_libs
        self.lib_num_books = [0] * num_libs
        self.lib_book_ids = [[] for _ in range(num_libs)]
        for i, lib in enumerate(libs):
            self.lib_signup_days[i] = lib.signup_days
            self.lib_books_per_day[i] = lib.books_per_day
            self.lib_num_books[i] = lib.num_books
            book_ids = [book.id for book in lib.books]
            self.lib_book_ids[i] = book_ids
            for book in lib.books:
                self.book_libs[book.id].append(i)
        self.books_by_score = sorted(
            range(num_books), key=lambda bid: scores[bid], reverse=True)
        self.book_freq = [len(self.book_libs[bid]) for bid in range(num_books)]
        self.effective_scores = [
            scores[bid] * (1.0 + 0.35 / (self.book_freq[bid] ** 0.5 if self.book_freq[bid] > 0 else 1.0))
            for bid in range(num_books)
        ]
        self.top_raw_potential = [0.0] * num_libs
        self.top_rare_potential = [0.0] * num_libs
        self.cap_raw_potential = [0.0] * num_libs
        self.cap_rare_potential = [0.0] * num_libs
        self._build_potentials()

        self._flat = None
        self._rebuild_workspace = None

    def _build_potentials(self):
        k_limit = 1000
        for lib_id in range(self.num_libs):
            book_ids = self.lib_book_ids[lib_id]
            top_limit = min(len(book_ids), k_limit)
            if top_limit > 0:
                top_slice = book_ids[:top_limit]
                self.top_raw_potential[lib_id] = float(sum(self.scores[bid] for bid in top_slice))
                self.top_rare_potential[lib_id] = float(sum(self.effective_scores[bid] for bid in top_slice))

            signup = self.lib_signup_days[lib_id]
            rate = self.lib_books_per_day[lib_id]
            cap_limit = min(len(book_ids), max(0, self.num_days - signup) * rate, k_limit)
            if cap_limit > 0:
                cap_slice = book_ids[:cap_limit]
                self.cap_raw_potential[lib_id] = float(sum(self.scores[bid] for bid in cap_slice))
                self.cap_rare_potential[lib_id] = float(sum(self.effective_scores[bid] for bid in cap_slice))

    def potential_array(self, mode):
        if mode == "top_raw":
            return self.top_raw_potential
        if mode == "top_rare":
            return self.top_rare_potential
        if mode == "cap_raw":
            return self.cap_raw_potential
        if mode == "cap_rare":
            return self.cap_rare_potential
        raise ValueError(f"Unknown potential mode: {mode}")

    def to_flat_arrays(self):
        """
        Convert the instance representation to flat NumPy arrays for
        Numba-accelerated evaluation. The result is cached after the first call.
        """
        if self._flat is not None:
            return self._flat

        book_scores = np.array(self.scores, dtype=np.int32)
        libs_signup = np.array(self.lib_signup_days, dtype=np.int32)
        libs_rate = np.array(self.lib_books_per_day, dtype=np.int32)
        lib_num_books = np.array(self.lib_num_books, dtype=np.int32)
        books_by_score = np.array(self.books_by_score, dtype=np.int32)

        flat_books = []
        offsets = []
        lengths = []
        for i in range(self.num_libs):
            offsets.append(len(flat_books))
            bids = self.lib_book_ids[i]
            lengths.append(len(bids))
            flat_books.extend(bids)

        book_libs_flat = []
        book_libs_offsets = []
        book_libs_lengths = []
        for book_id in range(self.num_books):
            book_libs_offsets.append(len(book_libs_flat))
            libs = self.book_libs[book_id]
            book_libs_lengths.append(len(libs))
            book_libs_flat.extend(libs)

        self._flat = {
            'n_books': self.num_books,
            'n_libs': self.num_libs,
            'total_days': self.num_days,
            'book_scores': book_scores,
            'libs_signup': libs_signup,
            'libs_rate': libs_rate,
            'lib_num_books': lib_num_books,
            'books_by_score': books_by_score,
            'books_flat': np.array(flat_books, dtype=np.int32),
            'books_offsets': np.array(offsets, dtype=np.int32),
            'books_lengths': np.array(lengths, dtype=np.int32),
            'book_libs_flat': np.array(book_libs_flat, dtype=np.int32),
            'book_libs_offsets': np.array(book_libs_offsets, dtype=np.int32),
            'book_libs_lengths': np.array(book_libs_lengths, dtype=np.int32),
        }
        return self._flat

    def rebuild_workspace(self):
        """
        Allocate reusable NumPy buffers for `Solution.from_order` rebuilds.
        This avoids repeated allocation of O(num_books) arrays during search.
        """
        if self._rebuild_workspace is None:
            max_order = max(1, self.num_libs)
            max_books = max(1, self.num_books)
            self._rebuild_workspace = {
                "signed_buffer": np.empty(max_order, dtype=np.int32),
                "out_selected": np.empty(max_order, dtype=np.int32),
                "out_books": np.empty(max_books, dtype=np.int32),
                "out_book_libs": np.empty(max_books, dtype=np.int32),
                "out_selected_global": np.empty(max_order, dtype=np.int32),
                "out_books_global": np.empty(max_books, dtype=np.int32),
                "out_book_libs_global": np.empty(max_books, dtype=np.int32),
            }
        return self._rebuild_workspace

    def order_array_view(self, signed_order):
        workspace = self.rebuild_workspace()
        signed_buffer = workspace["signed_buffer"]
        order_len = len(signed_order)
        if order_len:
            signed_buffer[:order_len] = signed_order
        return signed_buffer[:order_len]

    def fast_evaluate(self, signed_order):
        """
        Evaluate a signed library ordering using Numba-accelerated code.
        A pure Python implementation is used when Numba is unavailable.
        """
        from models.evaluation import fast_evaluate, fast_evaluate_global
        flat = self.to_flat_arrays()
        signed_arr = self.order_array_view(signed_order)
        sequential_score = int(fast_evaluate(
            signed_arr, flat['libs_signup'], flat['libs_rate'],
            flat['books_flat'], flat['books_offsets'],
            flat['books_lengths'], flat['book_scores'],
            flat['total_days']))
        global_score = int(fast_evaluate_global(
            signed_arr, flat['libs_signup'], flat['libs_rate'],
            flat['lib_num_books'], flat['books_by_score'],
            flat['book_libs_flat'], flat['book_libs_offsets'],
            flat['book_libs_lengths'], flat['book_scores'],
            flat['total_days']))
        return max(sequential_score, global_score)

    def screen_evaluate(self, signed_order):
        """
        Cheaper screening score for local search.
        Uses the global assignment scorer only, which tracks the current exact
        objective better than the sequential proxy on dense instances while
        still avoiding the second evaluation pass.
        """
        from models.evaluation import fast_evaluate_global
        flat = self.to_flat_arrays()
        signed_arr = self.order_array_view(signed_order)
        return int(fast_evaluate_global(
            signed_arr,
            flat['libs_signup'],
            flat['libs_rate'],
            flat['lib_num_books'],
            flat['books_by_score'],
            flat['book_libs_flat'],
            flat['book_libs_offsets'],
            flat['book_libs_lengths'],
            flat['book_scores'],
            flat['total_days'],
        ))

    def screen_evaluate_sequential(self, signed_order):
        """
        Lowest-cost screening score used during direct intensification.
        It mirrors the proxy used in the intensification phase so that the
        search can evaluate substantially more moves per second.
        """
        from models.evaluation import fast_evaluate_sequential
        flat = self.to_flat_arrays()
        signed_arr = self.order_array_view(signed_order)
        return int(fast_evaluate_sequential(
            signed_arr,
            flat['libs_signup'],
            flat['libs_rate'],
            flat['books_flat'],
            flat['books_offsets'],
            flat['books_lengths'],
            flat['book_scores'],
            flat['total_days'],
        ))

    def describe(self):
        print(f"Instance: {self.num_books:,} books, "
              f"{self.num_libs:,} libs, {self.num_days:,} days")

    def calculate_upper_bound(self):
        """Sum of scores of all unique books across all libraries."""
        unique_books = set()
        for lib in self.libs:
            for book in lib.books:
                unique_books.add(book.id)
        return sum(self.scores[bid] for bid in unique_books)
