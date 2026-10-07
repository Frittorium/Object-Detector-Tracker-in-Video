from dataclasses import dataclass

from detection.models import TrackObservation


@dataclass
class Candidate:
    tracker_id: int
    first_seen_frame: int
    consecutive_observations: int
    latest_observation: TrackObservation


class TargetSelector:
    """First-N policy. Receives only observations not already mapped to a target."""

    def __init__(self, confirmation_frames: int):
        self._n = confirmation_frames
        self._cands: dict[int, Candidate] = {}

    def evaluate(self, observations, frame_index, slots_left):
        if slots_left <= 0:
            self._cands.clear()
            return []
        seen, confirmed = set(), []
        for o in observations:
            seen.add(o.tracker_id)
            c = self._cands.get(o.tracker_id)
            if c is None:
                c = self._cands[o.tracker_id] = Candidate(o.tracker_id, frame_index, 1, o)
            else:
                c.consecutive_observations += 1
                c.latest_observation = o
            if c.consecutive_observations >= self._n:
                confirmed.append(c)
        for tid in [t for t in self._cands if t not in seen]:
            del self._cands[tid]          # not consecutive -> reset
        confirmed.sort(key=lambda c: (c.latest_observation.detection.x1, c.tracker_id))
        return confirmed[:slots_left]

    def discard(self, tracker_id):
        self._cands.pop(tracker_id, None)