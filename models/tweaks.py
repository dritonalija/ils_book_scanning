import random

from models.solution import Solution


class Tweaks:
    DEFAULT_WEIGHTS = {
        "swap_signed": 1.0,
        "swap_signed_with_unsigned": 2.5,
        "move_signed": 1.5,
        "swap_neighbor_libraries": 1.0,
        "insert_library": 1.5,
        "remove_library": 0.6,
        "reverse_segment": 0.7,
        "replace_worst": 1.8,
        "targeted_reorder": 1.7,
        "block_reinsert": 1.4,
    }

    @staticmethod
    def get_tweak_methods():
        return [
            ("swap_signed", Tweaks.tweak_solution_swap_signed),
            ("swap_signed_with_unsigned", Tweaks.tweak_solution_swap_signed_with_unsigned),
            ("move_signed", Tweaks.tweak_solution_move_signed),
            ("swap_neighbor_libraries", Tweaks.tweak_solution_swap_neighbor_libraries),
            ("insert_library", Tweaks.tweak_solution_insert_library),
            ("remove_library", Tweaks.tweak_solution_remove_library),
            ("reverse_segment", Tweaks.tweak_solution_reverse_segment),
            ("replace_worst", Tweaks.tweak_solution_replace_worst),
            ("targeted_reorder", Tweaks.tweak_solution_targeted_reorder),
            ("block_reinsert", Tweaks.tweak_solution_block_reinsert),
        ]

    @staticmethod
    def choose_tweak_method(weights=None):
        weights = weights or Tweaks.DEFAULT_WEIGHTS
        methods = Tweaks.get_tweak_methods()
        labels = [label for label, _ in methods]
        funcs = [func for _, func in methods]
        probs = [weights.get(label, 0.0) for label in labels]
        return random.choices(funcs, weights=probs, k=1)[0]

    @staticmethod
    def _clone_order(solution):
        return solution.signed_libraries + solution.unsigned_libraries

    @staticmethod
    def _rebuilt(order, data):
        return Solution.from_order(order, data)

    @staticmethod
    def tweak_solution_swap_signed(solution, data):
        if len(solution.signed_libraries) < 2:
            return solution
        order = Tweaks._clone_order(solution)
        i, j = random.sample(range(len(solution.signed_libraries)), 2)
        order[i], order[j] = order[j], order[i]
        return Tweaks._rebuilt(order, data)

    @staticmethod
    def tweak_solution_swap_signed_with_unsigned(solution, data, bias_type=None, bias_ratio=2 / 3):
        signed_count = len(solution.signed_libraries)
        if signed_count == 0 or not solution.unsigned_libraries:
            return solution

        order = Tweaks._clone_order(solution)
        if bias_type == "favor_first_half" and signed_count > 1 and random.random() < bias_ratio:
            signed_idx = random.randint(0, max(0, signed_count // 2 - 1))
        elif bias_type == "favor_second_half" and signed_count > 1 and random.random() < bias_ratio:
            signed_idx = random.randint(signed_count // 2, signed_count - 1)
        else:
            signed_idx = random.randint(0, signed_count - 1)

        unsigned_idx = random.randint(signed_count, len(order) - 1)
        order[signed_idx], order[unsigned_idx] = order[unsigned_idx], order[signed_idx]
        return Tweaks._rebuilt(order, data)

    @staticmethod
    def tweak_solution_move_signed(solution, data):
        if len(solution.signed_libraries) < 2:
            return solution
        order = Tweaks._clone_order(solution)
        signed_count = len(solution.signed_libraries)
        i = random.randint(0, signed_count - 1)
        j = random.randint(0, signed_count - 1)
        if i == j:
            return solution
        lib_id = order.pop(i)
        order.insert(j, lib_id)
        return Tweaks._rebuilt(order, data)

    @staticmethod
    def tweak_solution_swap_neighbor_libraries(solution, data):
        if len(solution.signed_libraries) < 2:
            return solution
        order = Tweaks._clone_order(solution)
        pos = random.randint(0, len(solution.signed_libraries) - 2)
        order[pos], order[pos + 1] = order[pos + 1], order[pos]
        return Tweaks._rebuilt(order, data)

    @staticmethod
    def tweak_solution_insert_library(solution, data):
        signed_count = len(solution.signed_libraries)
        if not solution.unsigned_libraries:
            return solution
        order = Tweaks._clone_order(solution)
        unsigned_idx = random.randint(signed_count, len(order) - 1)
        lib_id = order.pop(unsigned_idx)
        insert_pos = random.randint(0, signed_count)
        order.insert(insert_pos, lib_id)
        return Tweaks._rebuilt(order, data)

    @staticmethod
    def tweak_solution_remove_library(solution, data):
        if len(solution.signed_libraries) == 0:
            return solution
        order = Tweaks._clone_order(solution)
        signed_count = len(solution.signed_libraries)
        idx = random.randint(0, signed_count - 1)
        lib_id = order.pop(idx)
        order.append(lib_id)
        return Tweaks._rebuilt(order, data)

    @staticmethod
    def tweak_solution_reverse_segment(solution, data):
        if len(solution.signed_libraries) < 3:
            return solution
        order = Tweaks._clone_order(solution)
        signed_count = len(solution.signed_libraries)
        i, j = sorted(random.sample(range(signed_count), 2))
        order[i : j + 1] = reversed(order[i : j + 1])
        return Tweaks._rebuilt(order, data)

    @staticmethod
    def tweak_solution_replace_worst(solution, data):
        signed = solution.signed_libraries
        unsigned = solution.unsigned_libraries
        if not signed or not unsigned:
            return solution

        scored = []
        for pos, lib_id in enumerate(signed):
            books = solution.scanned_books_per_library.get(lib_id, [])
            contribution = sum(data.scores[book_id] for book_id in books)
            scored.append((contribution, pos, lib_id))

        scored.sort()
        _, worst_pos, _ = scored[0]
        candidate_pool = unsigned[: min(32, len(unsigned))]
        if not candidate_pool:
            return solution

        best_candidate = None
        best_value = None
        for lib_id in candidate_pool:
            value = 0
            limit = min(data.lib_num_books[lib_id], max(10, data.lib_books_per_day[lib_id] * 5))
            for book_id in data.lib_book_ids[lib_id][:limit]:
                value += data.scores[book_id] / max(1, data.book_freq[book_id])
            if best_value is None or value > best_value:
                best_value = value
                best_candidate = lib_id

        if best_candidate is None:
            return solution

        order = Tweaks._clone_order(solution)
        replacement_idx = order.index(best_candidate)
        order[worst_pos], order[replacement_idx] = order[replacement_idx], order[worst_pos]
        return Tweaks._rebuilt(order, data)

    @staticmethod
    def tweak_solution_targeted_reorder(solution, data):
        contributions = solution.library_contributions(data)
        if len(contributions) < 2:
            return solution

        order = Tweaks._clone_order(solution)
        low = sorted(contributions, key=lambda item: (item["score"], item["score_per_signup"]))
        high = sorted(contributions, key=lambda item: (item["score"], item["score_per_signup"]), reverse=True)

        left = low[0]["position"]
        right = high[0]["position"]
        if left == right:
            return solution

        lib_id = order.pop(right)
        insert_at = min(left, len(order))
        order.insert(insert_at, lib_id)
        return Tweaks._rebuilt(order, data)

    @staticmethod
    def tweak_solution_block_reinsert(solution, data):
        signed_count = len(solution.signed_libraries)
        if signed_count < 3:
            return solution

        order = Tweaks._clone_order(solution)
        block_size = min(max(2, signed_count // 10), 8)
        start = random.randint(0, max(0, signed_count - block_size))
        block = order[start : start + block_size]
        del order[start : start + block_size]

        insert_at = random.randint(0, max(0, len(solution.signed_libraries) - block_size))
        order[insert_at:insert_at] = block
        return Tweaks._rebuilt(order, data)
