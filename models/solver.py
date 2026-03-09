"""
Iterated Local Search (ILS) with Random Restarts for Book Scanning.

Supports multiple algorithm variants for ablation studies:
  - full         : Complete ILS (default)
  - no_perturb   : ILS without perturbation (local search only)
  - no_restart   : ILS without restart mechanism
  - no_accept    : ILS without acceptance of worse solutions (pure HC)
  - random_walk  : Accept all perturbations (no quality filter)
  - ls_only      : Single local search run (no ILS loop)
"""

import csv
import os
import random
import time

from models.initial_solution import InitialSolution
from models.local_search import LocalSearch
from models.solution import Solution
from models.tweaks import Tweaks

VALID_VARIANTS = {
    'full', 'no_perturb', 'no_restart', 'no_accept',
    'random_walk', 'ls_only',
}


class Solver:
    def __init__(self, seed=None, verbose=True):
        self.verbose = verbose
        if seed is not None:
            random.seed(seed)

    def iterated_local_search(
        self,
        data,
        time_limit=300,
        max_iterations=None,
        pool_size=None,
        init_max_time=120.0,
        init_budget_ratio=None,
        restart_threshold=None,
        perturb_strength_base=None,
        perturb_strength_growth=None,
        accept_worse_prob=0.04,
        alpha_values=None,
        weighted_beta=0.12,
        grasp_rcl=0.05,
        grasp_max_time=5.0,
        noisy_restarts=None,
        local_no_improve_limit=None,
        ls_order_weight=1.0,
        ls_insert_weight=1.0,
        ls_strategic_weight=1.0,
        perturb_replace_bias=0.65,
        restart_fresh_probability=0.35,
        variant='full',
        log_csv=None,
    ):
        if variant not in VALID_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from: {VALID_VARIANTS}")

        profile = self._instance_profile(data)
        max_iterations = profile["max_iterations"] if max_iterations is None else max_iterations
        pool_size = profile["pool_size"] if pool_size is None else pool_size
        restart_threshold = profile["restart_threshold"] if restart_threshold is None else restart_threshold
        perturb_strength_base = profile["perturb_strength_base"] if perturb_strength_base is None else perturb_strength_base
        perturb_strength_growth = profile["perturb_strength_growth"] if perturb_strength_growth is None else perturb_strength_growth
        noisy_restarts = profile["noisy_restarts"] if noisy_restarts is None else noisy_restarts
        local_no_improve_limit = profile["local_no_improve_limit"] if local_no_improve_limit is None else local_no_improve_limit
        initial_budget = min(max(1.0, init_max_time), 120.0)
        tweak_weights = Tweaks.grouped_weights(
            order_scale=ls_order_weight,
            insert_scale=ls_insert_weight,
            strategic_scale=ls_strategic_weight,
        )

        if self.verbose:
            print("---------- ITERATED LOCAL SEARCH WITH RANDOM RESTARTS ----------")
            print(
                f"Instance: {data.num_books:,} books | "
                f"{data.num_libs:,} libs | {data.num_days:,} days")
            print(
                f"Profile: {profile['name']} | Variant: {variant} | "
                f"Init: {initial_budget:.0f}s | ILS: {time_limit:.0f}s")

        # --- CSV convergence log ---
        csv_writer = None
        csv_file = None
        if log_csv:
            os.makedirs(os.path.dirname(log_csv) or '.', exist_ok=True)
            csv_file = open(log_csv, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                'timestamp', 'elapsed_s', 'phase', 'round',
                'current_score', 'best_score', 'event'])

        try:
            # --- Timing breakdown ---
            time_construction = 0.0
            time_local_search = 0.0
            time_perturbation = 0.0

            # =============================================
            # Phase 1: Initial solution construction
            # =============================================
            t0 = time.time()
            initial_solution, candidate_pool = InitialSolution.generate_initial_solution(
                data,
                max_time=initial_budget,
                alphas=alpha_values,
                beta=weighted_beta,
                grasp_rcl=grasp_rcl,
                grasp_max_time=grasp_max_time,
                noisy_restarts=noisy_restarts,
                verbose=self.verbose,
            )
            time_construction = time.time() - t0
            initial_score = initial_solution.fitness_score

            best_solution = initial_solution.clone()
            home_base = initial_solution.clone()
            best_label = "initial"

            if csv_writer:
                csv_writer.writerow([
                    time.time(), time_construction, 'construction', 0,
                    initial_score, initial_score, 'initial_solution'])

            # =============================================
            # Phase 2: Initial local search
            # =============================================
            ils_start_time = time.time()
            initial_ls_time = self._phase_time_limit(time_limit, ils_start_time, profile, phase="initial")

            t0 = time.time()
            home_base = LocalSearch.local_search(
                home_base, data,
                time_limit=initial_ls_time,
                max_iterations=profile["initial_ls_iterations"],
                no_improve_limit=local_no_improve_limit,
                tweak_weights=tweak_weights,
            )
            time_local_search += time.time() - t0

            if home_base.fitness_score > best_solution.fitness_score:
                best_solution = home_base.clone()
                best_label = "initial_local_search"

            home_pool = [home_base.clone()]
            if self.verbose:
                print(f"Construction: {time_construction:.2f}s | "
                      f"Score: {initial_score:,} -> {home_base.fitness_score:,}")

            if csv_writer:
                csv_writer.writerow([
                    time.time(), time.time() - ils_start_time, 'initial_ls', 0,
                    home_base.fitness_score, best_solution.fitness_score,
                    'after_initial_ls'])

            # Early exit for ls_only variant
            if variant == 'ls_only':
                best_solution.initial_score = initial_score
                return best_solution

            # =============================================
            # Phase 3: ILS main loop
            # =============================================
            outer_round = 0
            restart_count = 0
            stagnant_rounds = 0

            while time.time() - ils_start_time < time_limit and outer_round < max_iterations:
                outer_round += 1

                # --- Perturbation ---
                if variant == 'no_perturb':
                    candidate = home_base.clone()
                else:
                    t0 = time.time()
                    candidate = self._perturb_solution(
                        home_base, data,
                        strength=perturb_strength_base + stagnant_rounds * perturb_strength_growth,
                        profile=profile,
                        replace_bias=perturb_replace_bias,
                    )
                    time_perturbation += time.time() - t0

                # --- Local search ---
                ls_time = self._phase_time_limit(time_limit, ils_start_time, profile, phase="round")
                t0 = time.time()
                candidate = LocalSearch.local_search(
                    candidate, data,
                    time_limit=ls_time,
                    max_iterations=profile["round_ls_iterations"],
                    no_improve_limit=local_no_improve_limit,
                    tweak_weights=tweak_weights,
                )
                time_local_search += time.time() - t0

                # --- Update global best ---
                if candidate.fitness_score > best_solution.fitness_score:
                    best_solution = candidate.clone()
                    best_label = f"round_{outer_round}"
                    stagnant_rounds = 0
                    if self.verbose:
                        t = time.time() - ils_start_time
                        print(f"  [Round {outer_round:>4d}] New best: "
                              f"{best_solution.fitness_score:,} (t={t:.1f}s)")
                    if csv_writer:
                        csv_writer.writerow([
                            time.time(), time.time() - ils_start_time,
                            'ils', outer_round,
                            candidate.fitness_score, best_solution.fitness_score,
                            'new_best'])

                # --- Acceptance criterion ---
                if variant == 'random_walk':
                    accepted = True
                elif variant == 'no_accept':
                    accepted = candidate.fitness_score >= home_base.fitness_score
                else:
                    accepted = self._accept_candidate(
                        candidate, home_base, accept_worse_prob, stagnant_rounds)

                if accepted:
                    improved_home = candidate.fitness_score > home_base.fitness_score
                    if self.verbose and improved_home:
                        print(f"  [Round {outer_round:>4d}] Home base: "
                              f"{candidate.fitness_score:,}")
                    home_base = candidate.clone()
                    self._push_pool(home_pool, home_base, pool_size)
                    stagnant_rounds = 0 if improved_home else stagnant_rounds + 1
                else:
                    stagnant_rounds += 1

                if self.verbose and outer_round % 5 == 0:
                    t = time.time() - ils_start_time
                    print(
                        f"  [Round {outer_round:>4d}] home={home_base.fitness_score:,} | "
                        f"best={best_solution.fitness_score:,} | "
                        f"restarts={restart_count} | stag={stagnant_rounds} | t={t:.1f}s")

                if csv_writer and outer_round % 5 == 0:
                    csv_writer.writerow([
                        time.time(), time.time() - ils_start_time,
                        'ils', outer_round,
                        home_base.fitness_score, best_solution.fitness_score,
                        'status'])

                # --- Restart on stagnation ---
                if variant != 'no_restart' and stagnant_rounds >= restart_threshold:
                    remaining_budget = time_limit - (time.time() - ils_start_time)
                    if remaining_budget <= 0:
                        break
                    restart_init_budget = min(
                        profile["restart_init_max_time"],
                        remaining_budget * 0.3)
                    restart_label, restart_state = self._restart_state(
                        candidate_pool, home_pool, data, profile,
                        restart_fresh_probability,
                        alpha_values=alpha_values,
                        weighted_beta=weighted_beta,
                        grasp_rcl=grasp_rcl,
                        grasp_max_time=min(grasp_max_time, restart_init_budget * 0.5),
                        noisy_restarts=noisy_restarts,
                        init_max_time=restart_init_budget,
                    )
                    restart_count += 1
                    restart_time = self._phase_time_limit(
                        time_limit, ils_start_time, profile, phase="restart")

                    t0 = time.time()
                    restart_state = LocalSearch.local_search(
                        restart_state, data,
                        time_limit=restart_time,
                        max_iterations=profile["restart_ls_iterations"],
                        no_improve_limit=max(
                            profile["restart_no_improve_limit_floor"],
                            local_no_improve_limit // 2),
                        tweak_weights=tweak_weights,
                    )
                    time_local_search += time.time() - t0

                    if self.verbose:
                        print(f"  [Restart {restart_count}] {restart_label} -> "
                              f"{restart_state.fitness_score:,}")
                    if restart_state.fitness_score > best_solution.fitness_score:
                        best_solution = restart_state.clone()
                        best_label = f"restart_{restart_count}"
                        if self.verbose:
                            print(f"  [Restart {restart_count}] New best: "
                                  f"{best_solution.fitness_score:,}")

                    if csv_writer:
                        csv_writer.writerow([
                            time.time(), time.time() - ils_start_time,
                            'restart', restart_count,
                            restart_state.fitness_score,
                            best_solution.fitness_score, 'restart'])

                    home_base = restart_state.clone()
                    self._push_pool(home_pool, home_base, pool_size)
                    stagnant_rounds = 0

            total_time = time.time() - ils_start_time

            # =============================================
            # Summary
            # =============================================
            best_solution.initial_score = initial_score
            improvement = ((best_solution.fitness_score - initial_score) /
                            initial_score * 100) if initial_score > 0 else 0

            if self.verbose:
                print(f"\n{'=' * 60}")
                print(f"  Variant: {variant}")
                print(f"  Rounds: {outer_round} | Restarts: {restart_count}")
                print(f"  Initial: {initial_score:,} | "
                      f"Final: {best_solution.fitness_score:,} "
                      f"(+{improvement:.2f}%)")
                print(f"  Best found at: {best_label}")
                print(f"  Time breakdown:")
                print(f"    Construction:  {time_construction:.2f}s")
                print(f"    Local search:  {time_local_search:.2f}s")
                print(f"    Perturbation:  {time_perturbation:.2f}s")
                print(f"    Total ILS:     {total_time:.2f}s")
                print(f"{'=' * 60}")

            if csv_writer:
                csv_writer.writerow([
                    time.time(), total_time, 'final', outer_round,
                    best_solution.fitness_score, best_solution.fitness_score,
                    f'done_{best_label}'])

            return best_solution
        finally:
            if csv_file:
                csv_file.close()

    # =============================================
    # Helper methods (unchanged logic)
    # =============================================

    def _phase_time_limit(self, total_limit, start_time, profile, phase):
        remaining = max(0.0, total_limit - (time.time() - start_time))
        if remaining <= 0:
            return 0.0
        base = {
            "initial": profile["initial_ls_time"],
            "round": profile["round_ls_time"],
            "restart": profile["restart_ls_time"],
        }[phase]
        reserve_ratio = {
            "initial": 0.18,
            "round": 0.08,
            "restart": 0.12,
        }[phase]
        return min(base, max(0.05, remaining * reserve_ratio))

    def _accept_candidate(self, candidate, home_base, accept_worse_prob, stagnant_rounds):
        if candidate.fitness_score >= home_base.fitness_score:
            return True
        if home_base.fitness_score <= 0:
            return False
        gap = (home_base.fitness_score - candidate.fitness_score) / home_base.fitness_score
        if gap <= 0.003 and random.random() < 0.20:
            return True
        probability = accept_worse_prob * max(0.05, 1.0 - gap) * (1.0 + min(1.0, stagnant_rounds / 8.0))
        return random.random() < probability

    def _perturb_solution(self, solution, data, strength, profile, replace_bias):
        order = solution.ordered_libraries()
        if not order:
            return solution.clone()

        signed_count = len(solution.signed_libraries)
        strength = max(2, min(
            max(profile["min_strength_cap"], signed_count // profile["strength_divisor"]),
            strength))

        if signed_count == 0:
            return solution.clone()

        move = random.random()
        if move < replace_bias and len(order) > signed_count:
            return self._perturb_replace_subset(solution, data, strength)
        if move < 0.85:
            return self._perturb_reorder(solution, data, strength, profile)
        return self._perturb_shuffle(solution, data, strength, profile)

    def _perturb_replace_subset(self, solution, data, strength):
        order = solution.ordered_libraries()
        contributions = solution.library_contributions(data)
        if not contributions or len(solution.unsigned_libraries) == 0:
            return solution.clone()

        remove_count = min(len(contributions), max(1, strength // 2))
        weak_libs = sorted(
            contributions,
            key=lambda item: (item["score_per_signup"], item["score"], -item["position"]),
        )[:remove_count]
        weak_ids = {item["lib_id"] for item in weak_libs}

        unsigned_candidates = self._rank_unsigned_candidates(solution, data)
        top_candidates = [lib_id for _, lib_id in unsigned_candidates[:max(remove_count * 3, 12)]]
        if not top_candidates:
            return solution.clone()

        replacements = []
        inserted = set()
        for lib_id in top_candidates:
            if lib_id in inserted or lib_id not in order:
                continue
            replacements.append(lib_id)
            inserted.add(lib_id)
            if len(replacements) >= len(weak_libs):
                break

        if not replacements:
            return solution.clone()

        new_order = order.copy()
        ordered_weak = sorted(weak_libs, key=lambda item: item["position"])
        for weak, replacement_id in zip(ordered_weak, replacements):
            weak_idx = new_order.index(weak["lib_id"])
            replacement_idx = new_order.index(replacement_id)
            new_order[weak_idx], new_order[replacement_idx] = (
                new_order[replacement_idx],
                new_order[weak_idx],
            )

        return Solution.from_order(new_order, data)

    def _perturb_reorder(self, solution, data, strength, profile):
        order = solution.ordered_libraries()
        signed_count = len(solution.signed_libraries)
        if signed_count < 2:
            return solution.clone()

        segment = min(max(3, strength), profile["reorder_segment_cap"], signed_count)
        start = random.randint(0, max(0, signed_count - segment))
        end = start + segment
        block = order[start:end]

        if random.random() < 0.5:
            block = list(reversed(block))
        else:
            scored = []
            for lib_id in block:
                library = data.libs[lib_id]
                score = 0.0
                for book in library.books[:min(len(library.books), max(15, library.books_per_day * 6))]:
                    score += data.scores[book.id] / max(1, len(data.book_libs[book.id]))
                scored.append((score / max(1, library.signup_days), lib_id))
            block = [lib_id for _, lib_id in sorted(scored, reverse=True)]

        order[start:end] = block
        return Solution.from_order(order, data)

    def _perturb_shuffle(self, solution, data, strength, profile):
        order = solution.ordered_libraries()
        signed_count = len(solution.signed_libraries)
        if signed_count < 2:
            return solution.clone()

        segment = min(max(2, strength), profile["shuffle_segment_cap"], signed_count)
        start = random.randint(0, max(0, signed_count - segment))
        end = start + segment
        sub = order[start:end]
        random.shuffle(sub)
        order[start:end] = sub
        return Solution.from_order(order, data)

    def _rank_unsigned_candidates(self, solution, data):
        ranked = []
        seen_books = solution.scanned_books
        for lib_id in solution.unsigned_libraries:
            signup_days = data.lib_signup_days[lib_id]
            if signup_days >= data.num_days:
                continue
            score = 0.0
            count = 0
            book_limit = max(10, data.lib_books_per_day[lib_id] * 5)
            for book_id in data.lib_book_ids[lib_id]:
                if book_id in seen_books:
                    continue
                score += data.scores[book_id] / max(1, data.book_freq[book_id])
                count += 1
                if count >= book_limit:
                    break
            if score > 0:
                ranked.append((score / max(1, signup_days), lib_id))
        ranked.sort(reverse=True)
        return ranked

    def _push_pool(self, home_pool, solution, pool_size):
        if all(existing.fitness_score != solution.fitness_score for existing in home_pool):
            home_pool.append(solution.clone())
            home_pool.sort(key=lambda item: item.fitness_score, reverse=True)
            del home_pool[pool_size:]

    def _restart_state(
        self, candidate_pool, home_pool, data, profile,
        restart_fresh_probability, alpha_values, weighted_beta,
        grasp_rcl, grasp_max_time, noisy_restarts, init_max_time=None,
    ):
        budget = init_max_time or profile["restart_init_max_time"]
        if random.random() < restart_fresh_probability:
            if budget <= 8.0:
                fresh_solution, label = InitialSolution.generate_adaptive_restart_solution(
                    data,
                    alphas=alpha_values,
                )
                return label, fresh_solution

            fresh_solution, _ = InitialSolution.generate_initial_solution(
                data,
                max_time=budget,
                alphas=alpha_values,
                beta=weighted_beta,
                grasp_rcl=grasp_rcl,
                grasp_max_time=min(grasp_max_time, budget),
                noisy_restarts=noisy_restarts,
                verbose=False,
            )
            return "fresh_init", fresh_solution

        choices = []
        for score, label, solution in candidate_pool[:max(1, min(5, len(candidate_pool)))]:
            choices.append((label, solution.clone()))
        for idx, solution in enumerate(home_pool):
            choices.append((f"pool_home_{idx + 1}", solution.clone()))
        if not choices:
            raise RuntimeError("No restart candidates available")
        return random.choice(choices)

    def _instance_profile(self, data):
        if data.num_libs >= 8000 or data.num_books >= 500000:
            return {
                "name": "huge",
                "max_iterations": 500,
                "pool_size": 4,
                "restart_threshold": 3,
                "perturb_strength_base": 8,
                "perturb_strength_growth": 3,
                "noisy_restarts": 0,
                "local_no_improve_limit": 120,
                "initial_ls_iterations": 900,
                "round_ls_iterations": 650,
                "restart_ls_iterations": 450,
                "initial_ls_time": 1.1,
                "round_ls_time": 0.5,
                "restart_ls_time": 0.7,
                "restart_no_improve_limit_floor": 80,
                "restart_init_max_time": 20.0,
                "reorder_segment_cap": 20,
                "shuffle_segment_cap": 10,
                "min_strength_cap": 8,
                "strength_divisor": 5,
            }
        if data.num_libs >= 1800 or data.num_books >= 120000:
            return {
                "name": "large",
                "max_iterations": 800,
                "pool_size": 5,
                "restart_threshold": 4,
                "perturb_strength_base": 6,
                "perturb_strength_growth": 2,
                "noisy_restarts": 1,
                "local_no_improve_limit": 220,
                "initial_ls_iterations": 1400,
                "round_ls_iterations": 1000,
                "restart_ls_iterations": 750,
                "initial_ls_time": 1.8,
                "round_ls_time": 0.9,
                "restart_ls_time": 1.2,
                "restart_no_improve_limit_floor": 120,
                "restart_init_max_time": 30.0,
                "reorder_segment_cap": 18,
                "shuffle_segment_cap": 9,
                "min_strength_cap": 6,
                "strength_divisor": 6,
            }
        if data.num_libs >= 250 or data.num_books >= 15000:
            return {
                "name": "medium",
                "max_iterations": 1100,
                "pool_size": 6,
                "restart_threshold": 5,
                "perturb_strength_base": 4,
                "perturb_strength_growth": 2,
                "noisy_restarts": 2,
                "local_no_improve_limit": 320,
                "initial_ls_iterations": 2400,
                "round_ls_iterations": 1700,
                "restart_ls_iterations": 1200,
                "initial_ls_time": 2.6,
                "round_ls_time": 1.3,
                "restart_ls_time": 1.6,
                "restart_no_improve_limit_floor": 150,
                "restart_init_max_time": 45.0,
                "reorder_segment_cap": 14,
                "shuffle_segment_cap": 8,
                "min_strength_cap": 5,
                "strength_divisor": 7,
            }
        return {
            "name": "small",
            "max_iterations": 1500,
            "pool_size": 8,
            "restart_threshold": 6,
            "perturb_strength_base": 3,
            "perturb_strength_growth": 1,
            "noisy_restarts": 3,
            "local_no_improve_limit": 320,
            "initial_ls_iterations": 3200,
            "round_ls_iterations": 2200,
            "restart_ls_iterations": 1500,
            "initial_ls_time": 3.2,
            "round_ls_time": 1.6,
            "restart_ls_time": 2.0,
            "restart_no_improve_limit_floor": 140,
            "restart_init_max_time": 60.0,
            "reorder_segment_cap": 10,
            "shuffle_segment_cap": 6,
            "min_strength_cap": 4,
            "strength_divisor": 8,
        }
