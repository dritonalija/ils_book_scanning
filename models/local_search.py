import time

from models.solution import Solution
from models.tweaks import Tweaks


class LocalSearch:
    @staticmethod
    def local_search(
        solution,
        data,
        time_limit=60.0,
        max_iterations=1000,
        no_improve_limit=250,
        tweak_weights=None,
    ):
        start_time = time.time()
        current_solution = solution.clone()
        best_solution = current_solution.clone()
        iterations = 0
        stagnant = 0

        while (time.time() - start_time < time_limit) and (iterations < max_iterations):
            tweak_method = Tweaks.choose_tweak_method(tweak_weights)
            new_solution = tweak_method(current_solution, data)
            if new_solution.fitness_score > current_solution.fitness_score:
                current_solution = new_solution
                stagnant = 0
                if current_solution.fitness_score > best_solution.fitness_score:
                    best_solution = current_solution.clone()
            else:
                stagnant += 1
                if stagnant >= no_improve_limit:
                    break
            iterations += 1

        deadline = start_time + time_limit
        if time.time() < deadline:
            best_solution = LocalSearch._polish_solution(best_solution, data, deadline)

        return best_solution

    @staticmethod
    def _polish_solution(solution, data, deadline):
        current = solution
        passes = 0
        while passes < 2 and time.time() < deadline:
            passes += 1
            signed_count = len(current.signed_libraries)
            prefix_limit = min(signed_count, 24 if data.num_libs >= 2000 else 36)
            if prefix_limit < 2:
                break

            base_order = current.ordered_libraries()
            best_candidate = current
            checks = 0

            for i in range(prefix_limit - 1):
                if time.time() >= deadline or checks >= 96:
                    break
                order = base_order.copy()
                order[i], order[i + 1] = order[i + 1], order[i]
                candidate = Solution.from_order(order, data)
                checks += 1
                if candidate.fitness_score > best_candidate.fitness_score:
                    best_candidate = candidate

            move_span = 3
            for i in range(prefix_limit):
                if time.time() >= deadline or checks >= 160:
                    break
                left = max(0, i - move_span)
                right = min(prefix_limit - 1, i + move_span)
                for j in range(left, right + 1):
                    if i == j or time.time() >= deadline or checks >= 160:
                        continue
                    order = base_order.copy()
                    lib_id = order.pop(i)
                    order.insert(j, lib_id)
                    candidate = Solution.from_order(order, data)
                    checks += 1
                    if candidate.fitness_score > best_candidate.fitness_score:
                        best_candidate = candidate

            if best_candidate.fitness_score > current.fitness_score:
                current = best_candidate
            else:
                break

        return current
