"""The control path, as operators see it (rpps.profile.control)

A header step (a physical header, a link header) and a frame sync publish what they decoded, with the absolute stream
position it applies from; a step that drops or inserts items anchors positions. They live on `meta.stream.control`.
What a profile does with them (tables, controls, runners) is rpps.profile.control.

Both directions:

    publish(meta, op, {"modcod": 12, "pilots": 1}, at=<input position>, out=<index in this call's output>, span=None,
            group=None)
        at: absolute position in the op's input from which the fields apply. out: where that is in what the op gives in
        this call (None: nothing it gives is affected). span: input items the fields hold for, after which they go back
        to the controls' defaults (a PLCP rate holds for its payload only). group: publications made from one header
        share it (default: `at`), so a runner that re-runs keeps them together
    anchor(meta, op, at, out, rate=None)
        a step that drops or inserts items says which output item an input position became, so positions translate
        exactly through it
    emit(meta, op, kind, fields, bits=None, ok=None, at=<input position>)
        a decoded frame, for whoever reads the profile's messages (rpps.profile.message): its fields, its payload bits,
        whether its check passed, from absolute position `at` in the op's input (where the frame starts)
"""

class Control:
    """The control path of one stream (`stream.control`)

    published, anchors, messages: what operators reported since the runner last collected them (the runner empties these)
    log: every change the runner accepted, in order: {"step", "fields", "at", "targets"}
    reports: what did not go as announced: late changes re-run or dropped, restarts, fields with no table row
    rerun: re-runs done; buffer: the re-run buffer's current size, in items at its first step
    """
    __slots__ = ("published", "anchors", "messages", "log", "reports", "rerun", "buffer")
    def __init__(self):
        self.published, self.anchors, self.messages, self.log, self.reports = [], [], [], [], []
        self.rerun, self.buffer = 0, 0

    def fields(self, step=None):
        """The field sets published so far, oldest first: [(step id, fields, position)]"""
        return [(e["step"], e["fields"], e["at"]) for e in self.log if step is None or e["step"] == step]

def of(stream):
    """The stream's Control, made on first use"""
    if stream.control is None:
        stream.control = Control()
    return stream.control

def publish(meta, op, fields, at, out=None, span=None, group=None):
    """Fields an op decoded (or was given, transmitting), applying from absolute input position `at` onward (see module)"""
    of(meta.stream).published.append((op, dict(fields), int(at), None if out is None else int(out),
                                      None if span is None else int(span), int(at) if group is None else group))

def anchor(meta, op, at, out, rate=None):
    """Absolute input position `at` became item `out` of this call's output (a step that drops or inserts items)"""
    of(meta.stream).anchors.append((op, int(at), int(out), rate))

def emit(meta, op, kind, fields, bits=None, ok=None, at=0):
    """A decoded frame (see module): kind None takes the profile step's id"""
    of(meta.stream).messages.append((op, kind, dict(fields or {}), None if bits is None else bits, ok, int(at)))
