import random
import time

from models.solution import Solution
from models.tweaks import Tweaks


class LocalSearch:
    @staticmethod
    def _screen_score(data, order):
        if hasattr(data, "screen_evaluate_sequential"):
            return data.screen_evaluate_sequential(order)
        return data.screen_evaluate(order)

    @staticmethod
    def _weighted_methods(tweak_weights=None, data=None):
        methods = Tweaks.get_tweak_methods()
        weights = tweak_weights or Tweaks.DEFAULT_WEIGHTS
        if data is not None:
            weights = Tweaks.instance_adjusted_weights(weights, data)
        labels = [label for label, _ in methods]
        probs = [
            weights.get(label, 0.0)
            for label in labels
        ]
        if sum(probs) <= 0:
            raise ValueError("At least one local-search operator weight must be positive")
        return methods, probs

    @staticmethod
    def _polish_profile(data):
        total_occurrences = sum(data.lib_num_books)
        dense_small_library_instance = (
            data.num_libs <= 250 and data.num_days <= 1000
        )
        if (
            not dense_small_library_instance
            and (
            data.num_libs >= 15000
            or total_occurrences >= 700000
            or data.num_days >= 50000
            )
        ):
            return {
                "passes": 1,
                "prefix_limit": 18,
                "adjacent_checks": 48,
                "total_checks": 84,
                "move_span": 2,
                "boundary_signed": 5,
                "boundary_unsigned": 8,
                "boundary_checks": 30,
            }
        if (
            not dense_small_library_instance
            and (
            data.num_libs >= 3000
            or total_occurrences >= 300000
            or data.num_days >= 10000
            )
        ):
            return {
                "passes": 2,
                "prefix_limit": 24,
                "adjacent_checks": 72,
                "total_checks": 120,
                "move_span": 3,
                "boundary_signed": 6,
                "boundary_unsigned": 10,
                "boundary_checks": 48,
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
                "boundary_signed": 8,
                "boundary_unsigned": 12,
                "boundary_checks": 72,
            }
        return {
            "passes": 2,
            "prefix_limit": 40,
            "adjacent_checks": 132,
            "total_checks": 220,
            "move_span": 4,
            "boundary_signed": 10,
            "boundary_unsigned": 14,
            "boundary_checks": 96,
        }

    @staticmethod
    def local_search(
        solution,
        data,
        time_limit=60.0,
        max_iterations=None,
        no_improve_limit=250,
        tweak_weights=None,
        operator_stats=None,
    ):
        start_time = time.time()
        current_solution = solution.clone()
        best_solution = current_solution.clone()
        current_proxy_score = LocalSearch._screen_score(
            data,
            current_solution.ordered_libraries()
        )
        stagnant = 0
        methods, probs = LocalSearch._weighted_methods(tweak_weights, data)
        batch_profile = LocalSearch._batch_profile(data)
        search_deadline = start_time + time_limit * (1.0 - batch_profile["reserve"])

        # max_iterations is kept only for compatibility with older callers.
        # The active budget is time/stagnation based.
        while time.time() < search_deadline:
            label, tweak_method = random.choices(methods, weights=probs, k=1)[0]
            new_solution = None
            new_proxy_score = None
            if operator_stats is not None:
                stats = operator_stats.setdefault(
                    label,
                    {
                        "attempts": 0,
                        "proxy_improved": 0,
                        "exact_checked": 0,
                        "accepted": 0,
                        "rejected": 0,
                    },
                )
                stats["attempts"] += 1
            else:
                stats = None

            if label in Tweaks.FAST_ORDER_OPERATORS:
                order = Tweaks.build_candidate_order(label, current_solution, data)
                if order is not None:
                    proxy_score = LocalSearch._screen_score(data, order)
                    proxy_improved = proxy_score > current_proxy_score
                    near_proxy = (
                        current_proxy_score > 0
                        and proxy_score + max(
                            1,
                            int(current_proxy_score * batch_profile["proxy_slack_ratio"]),
                        ) >= current_proxy_score
                    )
                    should_check_exact = (
                        proxy_improved
                        or (
                            proxy_score == current_proxy_score
                            and random.random() < batch_profile["plateau_probe_rate"]
                        )
                        or (
                            near_proxy
                            and random.random() < batch_profile["near_proxy_probe_rate"]
                        )
                    )
                    if should_check_exact:
                        if stats is not None:
                            if proxy_improved:
                                stats["proxy_improved"] += 1
                        exact_score = data.fast_evaluate(order)
                        if stats is not None:
                            stats["exact_checked"] += 1
                        if exact_score > current_solution.fitness_score:
                            candidate = Solution.from_order(order, data)
                            if candidate.fitness_score > current_solution.fitness_score:
                                new_solution = candidate
                                new_proxy_score = proxy_score
            else:
                candidate = tweak_method(current_solution, data)
                if (
                    candidate is not None
                    and candidate.fitness_score > current_solution.fitness_score
                ):
                    new_solution = candidate

            if new_solution is not None:
                current_solution = new_solution
                current_proxy_score = (
                    new_proxy_score
                    if new_proxy_score is not None
                    else LocalSearch._screen_score(
                        data,
                        current_solution.ordered_libraries(),
                    )
                )
                stagnant = 0
                if stats is not None:
                    stats["accepted"] += 1
                if current_solution.fitness_score > best_solution.fitness_score:
                    best_solution = current_solution.clone()
            else:
                if stats is not None:
                    stats["rejected"] += 1
                stagnant += 1
                if stagnant >= no_improve_limit:
                    break

        deadline = start_time + time_limit
        if time.time() < deadline:
            best_solution = LocalSearch._batch_order_intensification(
                best_solution, data, deadline,
            )
        if time.time() < deadline:
            best_solution = LocalSearch._polish_solution(best_solution, data, deadline)

        return best_solution

    @staticmethod
    def _batch_profile(data):
        total_occurrences = sum(data.lib_num_books)
        if data.num_libs >= 15000 or total_occurrences >= 700000 or data.num_days >= 50000:
            return {
                "passes": 1,
                "checks": 80,
                "unsigned_sample": 32,
                "segment_cap": 8,
                "reserve": 0.15,
                "plateau_probe_rate": 0.03,
                "near_proxy_probe_rate": 0.01,
                "proxy_slack_ratio": 0.001,
            }
        if data.num_libs >= 3000 or total_occurrences >= 300000 or data.num_days >= 10000:
            return {
                "passes": 2,
                "checks": 140,
                "unsigned_sample": 48,
                "segment_cap": 10,
                "reserve": 0.20,
                "plateau_probe_rate": 0.04,
                "near_proxy_probe_rate": 0.015,
                "proxy_slack_ratio": 0.0015,
            }
        if data.num_libs >= 500 or total_occurrences >= 100000 or data.num_days >= 1000:
            return {
                "passes": 3,
                "checks": 220,
                "unsigned_sample": 72,
                "segment_cap": 12,
                "reserve": 0.35,
                "plateau_probe_rate": 0.06,
                "near_proxy_probe_rate": 0.02,
                "proxy_slack_ratio": 0.002,
            }
        return {
            "passes": 3,
            "checks": 180,
            "unsigned_sample": 48,
            "segment_cap": 10,
            "reserve": 0.20,
            "plateau_probe_rate": 0.08,
            "near_proxy_probe_rate": 0.03,
            "proxy_slack_ratio": 0.003,
        }

    @staticmethod
    def _batch_order_intensification(solution, data, deadline):
        """
        Evaluate many cheap order-only neighbors before rebuilding a Solution.

        This keeps the normal ILS architecture intact while borrowing the
        useful part of the fast order search: use the compiled exact scorer for
        lots of candidate library orders, and materialize only the best strict
        improvement found in a pass.
        """
        current = solution
        profile = LocalSearch._batch_profile(data)
        passes = 0
        while passes < profile["passes"] and time.time() < deadline:
            passes += 1
            base_order = current.ordered_libraries()
            signed_count = len(current.signed_libraries)
            if signed_count == 0 or len(base_order) < 2:
                break

            best_order = None
            best_score = current.fitness_score
            checks = 0
            while checks < profile["checks"] and time.time() < deadline:
                order = LocalSearch._sample_order_neighbor(
                    base_order,
                    signed_count,
                    profile,
                )
                checks += 1
                if order is None:
                    continue
                score = data.fast_evaluate(order)
                if score > best_score:
                    best_score = score
                    best_order = order

            if best_order is None:
                break
            candidate = Solution.from_order(best_order, data)
            if candidate.fitness_score <= current.fitness_score:
                break
            current = candidate
        return current

    @staticmethod
    def _sample_order_neighbor(base_order, signed_count, profile):
        order_len = len(base_order)
        unsigned_count = order_len - signed_count
        if signed_count <= 0:
            return None

        order = base_order.copy()
        move = random.random()

        if move < 0.40 and unsigned_count > 0:
            signed_idx = random.randrange(signed_count)
            unsigned_limit = min(unsigned_count, profile["unsigned_sample"])
            unsigned_idx = signed_count + random.randrange(unsigned_limit)
            order[signed_idx], order[unsigned_idx] = order[unsigned_idx], order[signed_idx]
            return order

        if move < 0.55 and signed_count >= 2:
            i, j = random.sample(range(signed_count), 2)
            order[i], order[j] = order[j], order[i]
            return order

        if move < 0.70 and signed_count >= 2:
            i, j = random.sample(range(signed_count), 2)
            lib_id = order.pop(i)
            order.insert(j, lib_id)
            return order

        if move < 0.82 and unsigned_count > 0:
            unsigned_limit = min(unsigned_count, profile["unsigned_sample"])
            unsigned_idx = signed_count + random.randrange(unsigned_limit)
            lib_id = order.pop(unsigned_idx)
            order.insert(random.randrange(signed_count + 1), lib_id)
            return order

        if move < 0.92 and signed_count >= 3:
            segment = random.randint(2, min(profile["segment_cap"], signed_count))
            start = random.randrange(0, signed_count - segment + 1)
            end = start + segment
            order[start:end] = reversed(order[start:end])
            return order

        if signed_count > 0:
            idx = random.randrange(signed_count)
            lib_id = order.pop(idx)
            order.append(lib_id)
            return order

        return None

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
                checks += 1
                if data.fast_evaluate(order) > best_candidate.fitness_score:
                    candidate = Solution.from_order(order, data)
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
                    checks += 1
                    if data.fast_evaluate(order) > best_candidate.fitness_score:
                        candidate = Solution.from_order(order, data)
                        if candidate.fitness_score > best_candidate.fitness_score:
                            best_candidate = candidate

            # Intensify the boundary between signed and unsigned libraries,
            # where small changes often determine the final feasible prefix.
            signed_tail = min(signed_count, profile["boundary_signed"])
            unsigned_limit = min(len(base_order) - signed_count, profile["boundary_unsigned"])
            boundary_checks = 0
            if signed_tail > 0 and unsigned_limit > 0 and time.time() < deadline:
                signed_start = signed_count - signed_tail
                unsigned_end = signed_count + unsigned_limit
                for i in range(signed_start, signed_count):
                    if time.time() >= deadline or boundary_checks >= profile["boundary_checks"]:
                        break
                    for j in range(signed_count, unsigned_end):
                        if time.time() >= deadline or boundary_checks >= profile["boundary_checks"]:
                            break
                        order = base_order.copy()
                        order[i], order[j] = order[j], order[i]
                        boundary_checks += 1
                        if data.fast_evaluate(order) > best_candidate.fitness_score:
                            candidate = Solution.from_order(order, data)
                            if candidate.fitness_score > best_candidate.fitness_score:
                                best_candidate = candidate

            if best_candidate.fitness_score > current.fitness_score:
                current = best_candidate
            else:
                break

        return current
