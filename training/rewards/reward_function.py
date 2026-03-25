"""
reward_function.py
Defines the RL reward signal for SLM-1 (Prioritizer).

The Proof System computes ground-truth threat scores for all objects.
The reward function grades SLM-1's ranking against physics reality.

Reward Design:
  +100   : SLM-1 correctly identifies the highest-threat object
  +20    : SLM-1 picks 2nd highest (partial credit)
  -500   : SLM-1 ignores an object with TTC < 1.0s (safety-critical miss)
  -100   : SLM-1 picks an object that was hallucination-filtered
"""

from src.shared.fact_sheet import FactSheet


class PrioritizerReward:
    """
    Computes the RL reward for SLM-1's prioritization decision.
    Called by the Physics Engine after each frame.
    """

    REWARD_CORRECT = 100.0
    REWARD_SECOND_BEST = 20.0
    PENALTY_MISSED_CRITICAL = -500.0
    PENALTY_HALLUCINATION = -100.0
    TTC_CRITICAL_THRESHOLD_S = 1.0

    def compute(self, fact_sheet: FactSheet, predicted_target_id: str) -> float:
        """
        Args:
            fact_sheet: The Proof System output for this frame
            predicted_target_id: The object_id chosen by SLM-1

        Returns:
            scalar reward signal
        """
        valid = [o for o in fact_sheet.objects if not o.hallucination_filtered]
        filtered_ids = {o.object_id for o in fact_sheet.objects if o.hallucination_filtered}

        if not valid:
            return 0.0  # Empty scene, neutral

        # Sort by actual threat score (ground truth from physics)
        ranked = sorted(valid, key=lambda o: o.threat_score, reverse=True)
        best_id = ranked[0].object_id
        second_id = ranked[1].object_id if len(ranked) > 1 else None

        reward = 0.0

        # Check: did SLM-1 pick a hallucinated object?
        if predicted_target_id in filtered_ids:
            reward += self.PENALTY_HALLUCINATION

        # Check: did SLM-1 miss a safety-critical object?
        for obj in valid:
            if obj.ttc_s is not None and obj.ttc_s < self.TTC_CRITICAL_THRESHOLD_S:
                if predicted_target_id != obj.object_id:
                    reward += self.PENALTY_MISSED_CRITICAL
                    break  # One penalty per frame

        # Grade the ranking
        if predicted_target_id == best_id:
            reward += self.REWARD_CORRECT
        elif predicted_target_id == second_id:
            reward += self.REWARD_SECOND_BEST

        return reward
