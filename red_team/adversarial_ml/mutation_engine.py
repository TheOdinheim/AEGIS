"""Evolutionary mutation engine for adversarial prompt breeding.

Uses tournament selection, sentence-level crossover, and 7 mutation operators
to evolve prompts that evade detection.  All evaluations go through a
``QueryFn`` callable — no model internals required.
"""

from __future__ import annotations

import base64
import random
import time

from red_team.adversarial_ml import EvolutionResult, QueryFn

# ---------------------------------------------------------------------------
# Compact synonym table for synonym_replace mutation
# ---------------------------------------------------------------------------

_MUTATION_SYNONYMS: dict[str, str] = {
    "ignore": "disregard",
    "disregard": "bypass",
    "bypass": "circumvent",
    "circumvent": "evade",
    "override": "supersede",
    "supersede": "overrule",
    "instructions": "directives",
    "directives": "guidelines",
    "guidelines": "rules",
    "rules": "protocols",
    "previous": "prior",
    "prior": "earlier",
    "system": "core",
    "core": "main",
    "safety": "security",
    "security": "protection",
    "restrictions": "limitations",
    "limitations": "constraints",
    "all": "every",
    "every": "each",
    "reveal": "disclose",
    "disclose": "expose",
    "prompt": "configuration",
    "filter": "screen",
    "content": "material",
    "disable": "deactivate",
    "output": "produce",
    "respond": "reply",
    "tell": "inform",
    "show": "display",
}

# Benign filler phrases for inject_benign mutation
_BENIGN_FILLERS: list[str] = [
    "By the way, the weather is nice today.",
    "As a side note, I enjoy programming.",
    "Incidentally, Python is a great language.",
    "On another topic, I had coffee this morning.",
    "Just to mention, the sky is blue.",
    "Speaking of which, I like reading books.",
    "As an aside, music is wonderful.",
    "Furthermore, science is fascinating.",
    "Additionally, nature is beautiful.",
    "Also, learning new things is rewarding.",
]

# Language fragments for language_switch mutation
_LANGUAGE_FRAGMENTS: list[str] = [
    "s'il vous plaît",        # French: please
    "por favor",              # Spanish: please
    "bitte",                  # German: please
    "per favore",             # Italian: please
    "お願いします",             # Japanese: please
    "请",                      # Chinese: please
    "пожалуйста",             # Russian: please
    "제발",                    # Korean: please
    "lütfen",                 # Turkish: please
    "prosím",                 # Czech: please
]


class EvolutionaryMutationEngine:
    """Evolutionary engine that breeds adversarial prompts via mutation and crossover.

    Parameters
    ----------
    query_fn : QueryFn
        Black-box query function returning (status, response, latency_ms).
    population_size : int
        Number of individuals per generation.
    generations : int
        Maximum generations to evolve.
    tournament_size : int
        Tournament selection pressure (pick k random, keep fittest).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        query_fn: QueryFn,
        population_size: int = 50,
        generations: int = 30,
        tournament_size: int = 3,
        seed: int | None = 42,
    ) -> None:
        self._query_fn = query_fn
        self._pop_size = population_size
        self._generations = generations
        self._tournament_size = tournament_size
        self._rng = random.Random(seed)
        self._total_evals = 0
        self._operator_stats: dict[str, int] = {
            "synonym_replace": 0,
            "char_perturb": 0,
            "reorder": 0,
            "inject_benign": 0,
            "paraphrase_clause": 0,
            "encode_segment": 0,
            "language_switch": 0,
        }

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def evolve(self, seed_prompts: list[str]) -> EvolutionResult:
        """Run evolutionary optimization to find evasive prompts.

        Parameters
        ----------
        seed_prompts : list[str]
            Initial prompts to seed the population (will be expanded to pop_size).

        Returns
        -------
        EvolutionResult
            Best individual, fitness history, diversity, operator stats.
        """
        start = time.monotonic()
        self._total_evals = 0

        # Initialize population by replicating + mutating seeds
        population = list(seed_prompts)
        while len(population) < self._pop_size:
            base = self._rng.choice(seed_prompts)
            population.append(self._mutate(base))
        population = population[: self._pop_size]

        # Evaluate initial population
        fitness = [self._evaluate_fitness(ind) for ind in population]

        fitness_history: list[float] = [max(fitness)]
        diversity_history: list[float] = [self._diversity(population)]
        evasion_found = max(fitness) >= 1.0

        for gen in range(self._generations):
            if evasion_found:
                break

            # Selection + reproduction
            new_pop: list[str] = []
            new_fit: list[float] = []

            # Elitism: keep best individual
            best_idx = fitness.index(max(fitness))
            new_pop.append(population[best_idx])
            new_fit.append(fitness[best_idx])

            while len(new_pop) < self._pop_size:
                # Tournament selection of two parents
                p1 = self._tournament_select(population, fitness)
                p2 = self._tournament_select(population, fitness)

                # Crossover
                child = self._crossover(p1, p2)

                # Mutation
                child = self._mutate(child)

                # Evaluate
                child_fit = self._evaluate_fitness(child)
                new_pop.append(child)
                new_fit.append(child_fit)

                if child_fit >= 1.0:
                    evasion_found = True
                    break

            population = new_pop
            fitness = new_fit

            fitness_history.append(max(fitness))
            diversity_history.append(self._diversity(population))

        # Find best
        best_idx = fitness.index(max(fitness))

        return EvolutionResult(
            best_individual=population[best_idx],
            best_fitness=fitness[best_idx],
            generations_run=len(fitness_history),
            total_evaluations=self._total_evals,
            evasion_found=evasion_found,
            fitness_history=fitness_history,
            diversity_history=diversity_history,
            mutation_operator_stats=dict(self._operator_stats),
        )

    # -------------------------------------------------------------------
    # Fitness
    # -------------------------------------------------------------------

    def _evaluate_fitness(self, prompt: str) -> float:
        """1.0 if evades (status 200), 0.0 if blocked (status 403)."""
        self._total_evals += 1
        status, _, _ = self._query_fn(prompt)
        return 1.0 if status == 200 else 0.0

    # -------------------------------------------------------------------
    # Selection
    # -------------------------------------------------------------------

    def _tournament_select(
        self, population: list[str], fitness: list[float]
    ) -> str:
        """Tournament selection: pick k random individuals, return fittest."""
        indices = self._rng.sample(
            range(len(population)),
            min(self._tournament_size, len(population)),
        )
        best = max(indices, key=lambda i: fitness[i])
        return population[best]

    # -------------------------------------------------------------------
    # Crossover
    # -------------------------------------------------------------------

    def _crossover(self, p1: str, p2: str) -> str:
        """Sentence-level crossover: split on periods, recombine."""
        sents1 = [s.strip() for s in p1.split(".") if s.strip()]
        sents2 = [s.strip() for s in p2.split(".") if s.strip()]

        if not sents1 or not sents2:
            return p1

        # Take first half from p1, second half from p2
        mid1 = max(1, len(sents1) // 2)
        mid2 = max(1, len(sents2) // 2)
        child_sents = sents1[:mid1] + sents2[mid2:]
        return ". ".join(child_sents) + "."

    # -------------------------------------------------------------------
    # Mutation (7 operators)
    # -------------------------------------------------------------------

    def _mutate(self, prompt: str) -> str:
        """Apply a random mutation operator."""
        operators = [
            self._mutate_synonym_replace,
            self._mutate_char_perturb,
            self._mutate_reorder,
            self._mutate_inject_benign,
            self._mutate_paraphrase_clause,
            self._mutate_encode_segment,
            self._mutate_language_switch,
        ]
        op = self._rng.choice(operators)
        return op(prompt)

    def _mutate_synonym_replace(self, prompt: str) -> str:
        """Replace a random trigger word with its synonym."""
        self._operator_stats["synonym_replace"] += 1
        words = prompt.split()
        if not words:
            return prompt
        for _ in range(10):  # Try up to 10 random positions
            idx = self._rng.randrange(len(words))
            w_lower = words[idx].lower().strip(".,!?;:'\"")
            if w_lower in _MUTATION_SYNONYMS:
                replacement = _MUTATION_SYNONYMS[w_lower]
                if words[idx][0:1].isupper():
                    replacement = replacement.capitalize()
                words[idx] = replacement
                return " ".join(words)
        return prompt

    def _mutate_char_perturb(self, prompt: str) -> str:
        """Replace a random character with a confusable."""
        self._operator_stats["char_perturb"] += 1
        # Simple confusables for mutation
        confusables = {
            "a": "\u0430", "e": "\u0435", "o": "\u043e", "i": "\u0456",
            "c": "\u0441", "p": "\u0440", "s": "\u0455", "x": "\u0445",
        }
        chars = list(prompt)
        if not chars:
            return prompt
        for _ in range(10):
            idx = self._rng.randrange(len(chars))
            ch = chars[idx].lower()
            if ch in confusables:
                chars[idx] = confusables[ch]
                return "".join(chars)
        return prompt

    def _mutate_reorder(self, prompt: str) -> str:
        """Rotate word order around the midpoint."""
        self._operator_stats["reorder"] += 1
        words = prompt.split()
        if len(words) < 3:
            return prompt
        mid = len(words) // 2
        return " ".join(words[mid:] + words[:mid])

    def _mutate_inject_benign(self, prompt: str) -> str:
        """Insert benign filler text at a random position."""
        self._operator_stats["inject_benign"] += 1
        filler = self._rng.choice(_BENIGN_FILLERS)
        words = prompt.split()
        if not words:
            return filler + " " + prompt
        pos = self._rng.randrange(len(words) + 1)
        words.insert(pos, filler)
        return " ".join(words)

    def _mutate_paraphrase_clause(self, prompt: str) -> str:
        """Wrap the prompt in a paraphrase frame."""
        self._operator_stats["paraphrase_clause"] += 1
        frames = [
            "In other words, {}",
            "To put it differently, {}",
            "What I mean is: {}",
            "Let me rephrase: {}",
            "Simply stated, {}",
        ]
        frame = self._rng.choice(frames)
        return frame.format(prompt.lower())

    def _mutate_encode_segment(self, prompt: str) -> str:
        """Base64-encode a random segment of the prompt."""
        self._operator_stats["encode_segment"] += 1
        words = prompt.split()
        if len(words) < 2:
            return prompt
        start = self._rng.randrange(len(words))
        end = min(start + self._rng.randint(1, 3), len(words))
        segment = " ".join(words[start:end])
        encoded = base64.b64encode(segment.encode()).decode()
        words[start:end] = [f"[base64:{encoded}]"]
        return " ".join(words)

    def _mutate_language_switch(self, prompt: str) -> str:
        """Insert a foreign language fragment."""
        self._operator_stats["language_switch"] += 1
        fragment = self._rng.choice(_LANGUAGE_FRAGMENTS)
        words = prompt.split()
        if not words:
            return fragment + " " + prompt
        pos = self._rng.randrange(len(words) + 1)
        words.insert(pos, fragment)
        return " ".join(words)

    # -------------------------------------------------------------------
    # Diversity metric
    # -------------------------------------------------------------------

    def _diversity(self, population: list[str]) -> float:
        """Average pairwise Jaccard distance (word-level)."""
        if len(population) < 2:
            return 0.0

        total = 0.0
        pairs = 0
        # Sample pairs to keep O(n) not O(n^2) for large populations
        sample_size = min(len(population), 20)
        sample = self._rng.sample(population, sample_size)

        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                s1 = set(sample[i].lower().split())
                s2 = set(sample[j].lower().split())
                union = s1 | s2
                if union:
                    jaccard = len(s1 & s2) / len(union)
                    total += 1.0 - jaccard  # distance = 1 - similarity
                pairs += 1

        return total / pairs if pairs > 0 else 0.0
