"""Board and zone layout, in metres, in the boards' own frame (same in every room)."""

BOARD_XY = {"tx": (0.0, 0.0), "rx1": (3.0, 0.0), "rx2": (3.0, 3.0), "rx3": (0.0, 3.0)}
# six zones in a 3 x 2 grid inside the square; the taped spot is the zone centre
ZONE_XY = {"1": (0.75, 1.0), "2": (1.5, 1.0), "3": (2.25, 1.0),
           "4": (0.75, 2.0), "5": (1.5, 2.0), "6": (2.25, 2.0)}
ZONES = list(ZONE_XY)
ACTIVITIES = ["still", "walk", "jump"]  # all done in place, on the zone's spot
