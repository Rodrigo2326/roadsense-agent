"""RoadSense -- an evidence-seeking visual agent.

Decider 2B chooses which evidence to gather next; Rekognition gathers it; Bedrock
writes an answer limited to what was gathered. The reusable half of that -- choosing
a tool from an evidence state, with a budget and a way to say "not enough" -- lives
in :mod:`roadsense.pattern`.
"""

__version__ = "0.1.0"
