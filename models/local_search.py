import time

from models.solution import Solution
from models.tweaks import Tweaks


class LocalSearch:
    @staticmethod
    def _polish_profile(data):
        total_occurrences = sum(data.lib_num_books)
        if (
            data.num_libs >= 15000
            or total_occurrences >= 700000
            or data.num_days >= 50000
        ):
            return {
                "passes": 1,
                "prefix_limit": 18,
                "adjacent_checks": 48,
                "total_checks": 84,
                "move_span": 2,
            }
        if (
            data.num_libs >= 3000
            or total_occurrences >= 300000
            or data.num_days >= 10000
        ):
            return {
                "passes": 2,
                "prefix_limit": 24,
                "adjacent_checks": 72,
                "total_checks": 120,
                "move_span": 3,
            }
        if (
            data.num_libs >= 500
            or total_occurrences >= 100000
            or data.num_days >= 1000
        ):
            return {
                "passes": 2,
                "prefix_limit": 32,
                "adjacent_checks": 108,
                "total_checks": 180,
                "move_span": 3,
            }
        return {
            "passes": 2,
            "prefix_limit": 40,
            "adjacent_checks": 132,
            "total_checks": 220,
            "move_span": 4,
        }

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
        profile = LocalSearch._polish_profile(data)
        passes = 0
        while passes < profile["passes"] and time.time() < deadline:
            passes += 1
            signed_count = len(current.signed_libraries)
            prefix_limit = min(signed_count, profile["prefix_limit"])
            if prefix_limit < 2:
                break

            base_order = current.ordered_libraries()
            best_candidate = current
            checks = 0

            for i in range(prefix_limit - 1):
                if time.time() >= deadline or checks >= profile["adjacent_checks"]:
                    break
                order = base_order.copy()
                order[i], order[i + 1] = order[i + 1], order[i]
                candidate = Solution.from_order(order, data)
                checks += 1
                if candidate.fitness_score > best_candidate.fitness_score:
                    best_candidate = candidate

            move_span = profile["move_span"]
            for i in range(prefix_limit):
                if time.time() >= deadline or checks >= profile["total_checks"]:
                    break
                left = max(0, i - move_span)
                right = min(prefix_limit - 1, i + move_span)
                for j in range(left, right + 1):
                    if i == j or time.time() >= deadline or checks >= profile["total_checks"]:
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
