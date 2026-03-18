from typing import List, Union
import re

# Listen duration for Coarse Sphere / Distance Vision (chart reading) phases
COARSE_SPHERE_LISTEN_SECONDS = 10
COARSE_SPHERE_SECONDS_PER_LETTER = 1.2
COARSE_SPHERE_MIN_SECONDS = 5
COARSE_SPHERE_MAX_SECONDS = 25


def listen_seconds(chart_letters: Union[str, List[str], None]) -> float:
    """Compute listen duration for Coarse Sphere based on number of letters in the chart line."""
    if not chart_letters:
        return COARSE_SPHERE_LISTEN_SECONDS
    n = len(chart_letters) if isinstance(chart_letters, str) else sum(len(s) for s in chart_letters)
    sec = COARSE_SPHERE_MIN_SECONDS + COARSE_SPHERE_SECONDS_PER_LETTER * n
    return min(COARSE_SPHERE_MAX_SECONDS, max(COARSE_SPHERE_MIN_SECONDS, sec))


CHART_LETTERS_MAP = {
    "400": ["E"],

    "200_150": "ENHSLC",
    "100_80": "HBVPHT",
    "70_60_50": "VLNEADAOFCEGNDH",
    "40_30_25": "FZBDEOFLCTAPEOF",
    "20_15_10": "TZVECOHPNTVLFTH",
    "20_20_20": "EVOTLTBGABHNFZC",
}

class ChartReadingDetector:
    """
    Detects whether text represents chart reading (Snellen-style letter sequences)
    and parses them into per-line segments. Also matches utterances to chart lines.
    """

    def __init__(
        self,
        min_letters: int = 3,
        max_letters_per_line: int = 6):
        self.min_letters = min_letters
        self.max_letters_per_line = max_letters_per_line

    def _normalize_chart_utterance(self, text: str) -> str:
        """Keep only letters and digits, uppercase, order preserved."""
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    def _lcs_length(self, a: str, b: str) -> int:
        """Length of longest common subsequence of a and b (order preserved)."""
        na, nb = len(a), len(b)
        prev = [0] * (nb + 1)
        for i in range(1, na + 1):
            curr = [0] * (nb + 1)
            for j in range(1, nb + 1):
                if a[i - 1] == b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev = curr
        return prev[nb]

    def utterance_matches_chart(
        self, utterance: str, one_chart_letters: str
    ) -> tuple[str, float]:
        """
        Return match: READABLE / BLURRY / NOT_READABLE.
        Uses longest common subsequence (LCS) of chart and utterance: fraction of
        chart letters that appear in the utterance in order. Extra letters in the
        utterance (insertions) do not penalize.
        """
        u = self._normalize_chart_utterance(utterance)
        c = self._normalize_chart_utterance(one_chart_letters)
        n = len(c)
        if n == 0:
            return "READABLE"
        correct = self._lcs_length(c, u)
        percentage_correct = correct / n
        if percentage_correct >= 0.8:
            return "READABLE"
        elif percentage_correct >= 0.5:
            return "BLURRY"
        else:
            return "NOT_READABLE"


    def preprocess_text(self, text: str) -> str:
        """Remove special characters and spaces."""
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = text.replace(".", "")
        text = text.replace(" ", "")
        return text.upper()
   
    def get_chart_intend(self, options: List[str], text: str, chart_letters: str) -> str:
        """
        Return the chart intention for the given text.
        Maps intent to options: "able to see" -> READABLE, "blurry" -> BLURRY.
        Returns the matched option from options if present, otherwise the mapped value.
        """
        utterance  = self.preprocess_text(text)
        intent = self.utterance_matches_chart(utterance, chart_letters)
        return intent
