"""
cue.py - full-screen cue window for record.py (--screen).

Red = wait (shows the next action and a countdown to GO), green = do the action now
(countdown to STOP). record.py starts this as a child process and sends one line per change on
stdin:  state <TAB> label <TAB> end_time (epoch s, 0 = no countdown) <TAB> info
Esc or q closes the window; recording carries on. It closes itself 5 s after record.py ends.
"""
import sys, threading, time
import tkinter as tk

HOW = {
    "empty": "leave the area, stand still behind the laptop",
    "walk": "walk back and forth across both lines",
    "sit": "sit down on the chair, stand up once",
    "wave": "stand on the spot, wave one arm only",
}
COLORS = {"wait": "#b3261e", "go": "#1e8e3e", "idle": "#3c4043", "done": "#3c4043"}

state = {"state": "idle", "label": "warming up", "end": 0.0, "info": ""}


def read_stdin():
    for line in sys.stdin:
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 4:
            state.update(state=parts[0], label=parts[1], end=float(parts[2]), info=parts[3])
    state.update(state="done", label="done", end=0.0, closed=time.time())


root = tk.Tk()
root.title("CSI cue")
root.attributes("-fullscreen", True)
root.bind("<Escape>", lambda e: root.destroy())
root.bind("q", lambda e: root.destroy())

top = tk.Label(root, font=("Helvetica", 40), fg="white")
big = tk.Label(root, font=("Helvetica", 150, "bold"), fg="white")
how = tk.Label(root, font=("Helvetica", 36), fg="white")
count = tk.Label(root, font=("Helvetica", 220, "bold"), fg="white")
info = tk.Label(root, font=("Helvetica", 28), fg="white")
for w in (top, big, how, count, info):
    w.pack(expand=True)


def refresh():
    s, lab = state["state"], state["label"]
    bg = COLORS.get(s, "#3c4043")
    root.configure(bg=bg)
    for w in (top, big, how, count, info):
        w.configure(bg=bg)
    top.configure(text={"wait": "WAIT - next:", "go": "GO - do it now", "idle": "", "done": ""}.get(s, ""))
    big.configure(text=lab.upper())
    how.configure(text=HOW.get(lab, "") if s in ("wait", "go") else "")
    left = state["end"] - time.time()
    count.configure(text=f"{max(0, int(left) + 1)}" if state["end"] and left > 0 else "")
    info.configure(text=state["info"])
    if time.time() - state.get("closed", time.time()) > 5:   # record.py finished: close after 5 s
        root.destroy(); return
    root.after(100, refresh)


threading.Thread(target=read_stdin, daemon=True).start()
refresh()
root.mainloop()
